import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
import gc

# ============================================
# IMPORTS (use fixed modules)
# ============================================


from model.backbone.model import GCNetWithDWSA
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1),
            num_classes=C
        ).permute(0, 3, 1, 2).float()
        
        mask = (targets != self.ignore_index).float().unsqueeze(1)
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask
        
        probs = probs.reshape(B, C, -1)
        targets_one_hot = targets_one_hot.reshape(B, C, -1)
        
        intersection = (probs * targets_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss_per_class = 1.0 - dice_per_class
        
        return dice_loss_per_class.mean() if self.reduction == 'mean' else dice_loss_per_class


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        B, C, H, W = logits.shape
        
        log_probs = log_probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        
        valid_mask = targets_flat != self.ignore_index
        log_probs = log_probs[valid_mask]
        targets_flat = targets_flat[valid_mask]
        
        if targets_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        probs = log_probs.exp()
        targets_probs = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - targets_probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

class HybridLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.0,
        ignore_index=255,
        class_weights=None,
        focal_alpha=0.25,
        focal_gamma=2.0,
        dice_smooth=1.0
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.dice_loss = DiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(self, logits, targets):
        losses = {}
        
        losses['ce'] = self.ce_loss(logits, targets) if self.ce_weight > 0 else 0
        losses['dice'] = self.dice_loss(logits, targets) if self.dice_weight > 0 else 0
        losses['focal'] = self.focal_loss(logits, targets) if self.focal_weight > 0 else 0
        
        losses['total'] = (
            self.ce_weight * losses['ce'] +
            self.dice_weight * losses['dice'] +
            self.focal_weight * losses['focal']
        )
        
        return losses

# ============================================
# MEMORY UTILITIES (unchanged)
# ============================================

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def print_memory_usage(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix} GPU Memory - Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")

def setup_memory_efficient_training():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class TrainingSchedule:
    """
    Multi-stage training schedule for from-scratch training
    """
    
    @staticmethod
    def get_schedule_from_scratch(total_epochs=200):
        """
        Progressive training strategy:
        
        Stage 1 (0-50): High LR, learn basic features
        Stage 2 (50-150): Stable training
        Stage 3 (150-200): Fine-tuning
        """
        return {
            # Stage 1: Warm-up and initial learning
            "stage1": {
                "epochs": (0, 50),
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "aux_weight": 0.4,
                "img_size": (384, 768),  # Smaller for faster iteration
                "batch_size": 8,
                "accumulation": 2
            },
            # Stage 2: Main training
            "stage2": {
                "epochs": (50, 150),
                "lr": 5e-4,
                "weight_decay": 5e-5,
                "aux_weight": 0.3,
                "img_size": (512, 1024),  # Full resolution
                "batch_size": 4,
                "accumulation": 4
            },
            # Stage 3: Fine-tuning
            "stage3": {
                "epochs": (150, 200),
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "aux_weight": 0.2,
                "img_size": (512, 1024),
                "batch_size": 4,
                "accumulation": 4
            }
        }
# ============================================
# ✅ UPDATED MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config_from_scratch():
        """
        ✅ Config cho training từ scratch
        
        Key changes:
        - Simpler architecture (avoid overfitting)
        - Higher regularization
        - Conservative DWSA usage
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,  # Base channels
                "ppm_channels": 96,
                "num_blocks_per_stage": [3, 3, [4, 3], [4, 3], [2, 2]],
                "dwsa_stages": ['bottleneck'],  # ✅ ONLY bottleneck for stability
                "dwsa_num_heads": 8,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                # ✅ CRITICAL FIX: Match backbone output!
                "in_channels": 64,  # c5 = channels * 2 = 32 * 2 = 64
                "channels": 128,
                "decoder_channels": 128,
                "dropout_ratio": 0.15,  # ✅ Increased from 0.1 for regularization
                "align_corners": False
            },
            "aux_head": {
                # ✅ CRITICAL FIX: Match backbone output!
                "in_channels": 128,  # c4 = channels * 4 = 32 * 4 = 128
                "channels": 64,
                "dropout_ratio": 0.15,
                "align_corners": False,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "loss": {
                # ✅ Optimized for scratch training
                "ce_weight": 1.0,
                "dice_weight": 1.0,  # Equal weight helps convergence
                "focal_weight": 0.0,  # Disabled - adds instability
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "dice_smooth": 1.0
            }
        }
    @staticmethod
    def get_lightweight_config():
        """
        ✅ FIXED: Proper channel alignment
        
        Backbone output channels:
        - c1: 32 (H/2)
        - c2: 32 (H/4)  
        - c3: 64 (H/8)
        - c4: 128 (H/16) ← Used by aux_head
        - c5: 64 (H/8)   ← Used by main head
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,  # Base channels
                "ppm_channels": 96,
                "num_blocks_per_stage": [3, 3, [4, 3], [4, 3], [2, 2]],
                "dwsa_stages": ['bottleneck'],  # ✅ Only bottleneck initially
                "dwsa_num_heads": 8,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                # ✅ FIX: Match backbone c5 output
                "in_channels": 64,  # channels * 2 = 32 * 2 = 64
                "channels": 128,
                "decoder_channels": 128,
                "dropout_ratio": 0.1,
                "align_corners": False
            },
            "aux_head": {
                # ✅ FIX: Match backbone c4 output  
                "in_channels": 128,  # channels * 4 = 32 * 4 = 128
                "channels": 64,
                "dropout_ratio": 0.1,
                "align_corners": False,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "loss": {
                # ✅ FIX: Optimized for Cityscapes
                "ce_weight": 1.0,
                "dice_weight": 1.0,      # ✅ Increased from 0.5
                "focal_weight": 0.0,     # ✅ Disabled initially
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "dice_smooth": 1.0
            }
        }
    
    @staticmethod
    def get_config_standard():
        """
        ✅ Standard config (if you get pretrained weights later)
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 48,
                "ppm_channels": 112,
                "num_blocks_per_stage": [3, 3, [4, 3], [4, 3], [2, 2]],
                "dwsa_stages": ['stage3', 'bottleneck'],  # More DWSA when pretrained
                "dwsa_num_heads": 8,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                "in_channels": 96,   # 48 * 2
                "channels": 128,
                "decoder_channels": 128,
                "dropout_ratio": 0.1,
                "align_corners": False
            },
            "aux_head": {
                "in_channels": 192,  # 48 * 4
                "channels": 96,
                "dropout_ratio": 0.1,
                "align_corners": False,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "loss": {
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "focal_weight": 0.0,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "dice_smooth": 1.0
            }
        }

class TrainingSchedule:
    """
    Multi-stage training schedule for from-scratch training
    """
    
    @staticmethod
    def get_schedule_from_scratch(total_epochs=200):
        """
        Progressive training strategy:
        
        Stage 1 (0-50): High LR, learn basic features
        Stage 2 (50-150): Stable training
        Stage 3 (150-200): Fine-tuning
        """
        return {
            # Stage 1: Warm-up and initial learning
            "stage1": {
                "epochs": (0, 50),
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "aux_weight": 0.4,
                "img_size": (384, 768),  # Smaller for faster iteration
                "batch_size": 8,
                "accumulation": 2
            },
            # Stage 2: Main training
            "stage2": {
                "epochs": (50, 150),
                "lr": 5e-4,
                "weight_decay": 5e-5,
                "aux_weight": 0.3,
                "img_size": (512, 1024),  # Full resolution
                "batch_size": 4,
                "accumulation": 4
            },
            # Stage 3: Fine-tuning
            "stage3": {
                "epochs": (150, 200),
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "aux_weight": 0.2,
                "img_size": (512, 1024),
                "batch_size": 4,
                "accumulation": 4
            }
        }
# ============================================
# SEGMENTOR (unchanged)
# ============================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)

    def forward_train(self, x):
        feats = self.backbone(x)
        outputs = {"main": self.decode_head(feats)}
        if self.aux_head is not None:
            outputs["aux"] = self.aux_head(feats)
        return outputs

# ============================================
# ✅ UPDATED TRAINER
# ============================================

class:
    """
    ✅ Updated with hybrid loss
    """
    
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        args,
        class_weights=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        

        
        loss_cfg = args.loss_config
        self.criterion = HybridLoss(
            ce_weight=loss_cfg['ce_weight'],
            dice_weight=loss_cfg['dice_weight'],
            focal_weight=loss_cfg['focal_weight'],
            ignore_index=args.ignore_index,
            class_weights=class_weights.to(device) if class_weights is not None else None,
            focal_alpha=loss_cfg['focal_alpha'],
            focal_gamma=loss_cfg['focal_gamma'],
            dice_smooth=loss_cfg['dice_smooth']
        )
        
        # Mixed precision
        self.scaler = GradScaler(enabled=args.use_amp)
        
        # Tracking
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        
        self.best_miou = 0.0
        self.start_epoch = 0
        self.global_step = 0
        
        # Save config
        self.save_config()
        
        print(f"\n{'='*70}")
        print("⚙️  Trainer Configuration")
        print(f"{'='*70}")
        print(f"📦 Batch size: {args.batch_size}")
        print(f"🔁 Gradient accumulation: {args.accumulation_steps}")
        print(f"📊 Effective batch size: {args.batch_size * args.accumulation_steps}")
        print(f"⚡ Mixed precision: {args.use_amp}")
        print(f"✂️  Gradient clipping: {args.grad_clip}")
        print(f"📉 Loss: CE({loss_cfg['ce_weight']}) + Dice({loss_cfg['dice_weight']}) + Focal({loss_cfg['focal_weight']})")
        print(f"{'='*70}\n")

    def save_config(self):
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, loader, epoch):
        """✅ Updated to log hybrid loss components"""
        self.model.train()
        
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        total_focal = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            # Forward with AMP
            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)
                logits = outputs["main"]
                
                # Interpolate to mask size
                logits = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                # ✅ Main loss (hybrid)
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total']
                
                # Auxiliary loss
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = nn.functional.interpolate(
                        outputs["aux"],
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    aux_loss_dict = self.criterion(aux_logits, masks)
                    loss = loss + self.args.aux_weight * aux_loss_dict['total']
                
                # Scale for gradient accumulation
                loss = loss / self.args.accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler and self.args.scheduler_type == 'onecycle':
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Track loss components
            total_loss += loss.item() * self.args.accumulation_steps
            total_ce += loss_dict['ce'].item()
            total_dice += loss_dict['dice'].item()
            total_focal += loss_dict['focal'].item()
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ce': f'{loss_dict["ce"].item():.4f}',
                'dice': f'{loss_dict["dice"].item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Clear cache
            if batch_idx % 50 == 0:
                clear_gpu_memory()
            
            # TensorBoard
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/total_loss', loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ce_loss', loss_dict['ce'].item(), self.global_step)
                self.writer.add_scalar('train/dice_loss', loss_dict['dice'].item(), self.global_step)
                self.writer.add_scalar('train/focal_loss', loss_dict['focal'].item(), self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)

        avg_loss = total_loss / len(loader)
        avg_ce = total_ce / len(loader)
        avg_dice = total_dice / len(loader)
        avg_focal = total_focal / len(loader)
        
        return {
            'loss': avg_loss,
            'ce': avg_ce,
            'dice': avg_dice,
            'focal': avg_focal
        }

    @torch.no_grad()
    def validate(self, loader, epoch):
        """Validation (unchanged)"""
        self.model.eval()
        total_loss = 0.0
        
        num_classes = self.args.num_classes
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        pbar = tqdm(loader, desc=f"Val Epoch {epoch+1}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits = self.model(imgs)
                logits = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total']
            
            total_loss += loss.item()
            
            # Metrics on CPU
            pred = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            
            mask = (target >= 0) & (target < num_classes)
            label = num_classes * target[mask].astype('int') + pred[mask]
            count = np.bincount(label, minlength=num_classes**2)
            confusion_matrix += count.reshape(num_classes, num_classes)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        # mIoU
        intersection = np.diag(confusion_matrix)
        union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        
        acc = intersection.sum() / (confusion_matrix.sum() + 1e-10)
        
        avg_loss = total_loss / len(loader)
        
        return {
            'loss': avg_loss,
            'miou': miou,
            'accuracy': acc,
            'per_class_iou': iou
        }

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'best_miou': self.best_miou,
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, self.save_dir / "last.pth")
        
        if is_best:
            torch.save(checkpoint, self.save_dir / "best.pth")
            print(f"✅ Best model saved! mIoU: {metrics['miou']:.4f}")
        
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(checkpoint, self.save_dir / f"epoch_{epoch+1}.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if self.args.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint['best_miou']
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
        print(f"   Best mIoU: {self.best_miou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Optimized GCNet Training")
    
    # Dataset
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    
    # Training strategy
    parser.add_argument("--from_scratch", action="store_true", default=True,
                        help="Training from scratch (no pretrained weights)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total epochs (200 recommended for scratch)")
    
    # Model
    parser.add_argument("--model_size", default="lightweight",
                        choices=["lightweight", "standard"])
    
    # Optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.4)
    
    # Image size (will be adjusted by stage)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    
    # System
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints_optimized")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"🚀 Optimized GCNet Training")
    print(f"{'='*70}")
    print(f"📱 Device: {device}")
    print(f"🎯 Training: {'FROM SCRATCH' if args.from_scratch else 'WITH PRETRAINED'}")
    print(f"📐 Initial image size: {args.img_h}x{args.img_w}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"⚠️  Expected mIoU: {'65-70%' if args.from_scratch else '76-78%'}")
    print(f"{'='*70}\n")
    
    # ✅ Load FIXED config
    if args.from_scratch:
        cfg = OptimizedConfig.get_config_from_scratch()
        print("✅ Using FROM-SCRATCH config with FIXED channels")
    else:
        cfg = OptimizedConfig.get_config_standard()
        print("✅ Using STANDARD config with FIXED channels")
    
    # Verify channels
    print(f"\n{'='*70}")
    print("🔍 Channel Verification")
    print(f"{'='*70}")
    print(f"Backbone base channels: {cfg['backbone']['channels']}")
    print(f"Expected c5 (main): {cfg['backbone']['channels'] * 2}")
    print(f"Expected c4 (aux):  {cfg['backbone']['channels'] * 4}")
    print(f"Head expects:       {cfg['head']['in_channels']} ✅")
    print(f"Aux expects:        {cfg['aux_head']['in_channels']} ✅")
    print(f"{'='*70}\n")
    
    # Store loss config
    args.loss_config = cfg["loss"]
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=False,
        dataset_type=args.dataset_type
    )
    
    # Create model
    print(f"\n{'='*70}")
    print("🏗️  Building Model with FIXED Channels...")
    print(f"{'='*70}\n")
    
    model = Segmentor(
        backbone=GCNetWithDWSA(**cfg["backbone"]),
        head=GCNetHead(num_classes=args.num_classes, **cfg["head"]),
        aux_head=GCNetAuxHead(num_classes=args.num_classes, **cfg["aux_head"])
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Test forward pass
    model = model.to(device)
    with torch.no_grad():
        sample = torch.randn(1, 3, args.img_h, args.img_w).to(device)
        try:
            outputs = model.forward_train(sample)
            print(f"✅ Forward pass successful!")
            print(f"   Main output: {outputs['main'].shape}")
            if 'aux' in outputs:
                print(f"   Aux output:  {outputs['aux'].shape}")
        except Exception as e:
            print(f"❌ Forward pass FAILED: {e}")
            print(f"⚠️  FIX CHANNELS BEFORE TRAINING!")
            return
    
    print(f"\n{'='*70}")
    print("✅ Model verification passed! Ready to train.")
    print(f"{'='*70}\n")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler - Polynomial (better for segmentation)
    def poly_lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** 0.9
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    
    # Criterion
    criterion = HybridLoss(
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        focal_weight=cfg["loss"]["focal_weight"],
        ignore_index=args.ignore_index,
        class_weights=class_weights.to(device) if class_weights is not None else None
    )
    
    # Training imports
    trainer = MemoryEfficientTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        class_weights=class_weights
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training loop
    print(f"\n{'='*70}")
    print("🚀 Starting Optimized Training!")
    print(f"{'='*70}\n")
    
    for epoch in range(trainer.start_epoch, args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Logging
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"CE: {train_metrics['ce']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"mIoU: {val_metrics['miou']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"{'='*70}\n")
        
        # Save checkpoint
        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
    
    print(f"\n{'='*70}")
    print("✅ Training Completed!")
    print(f"🏆 Best mIoU: {trainer.best_miou:.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
