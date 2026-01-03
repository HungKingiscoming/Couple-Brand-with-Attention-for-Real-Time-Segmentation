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
import warnings

warnings.filterwarnings('ignore')

# ============================================
# IMPORTS
# ============================================

from model.backbone.model import GCNetWithDWSA
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# LOSS FUNCTIONS
# ============================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255, reduction='mean'):  # ✅ Added 'smooth'
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        
        # ✅ Step 1: Create mask FIRST
        valid_mask = (targets != self.ignore_index).float()
        
        # ✅ Step 2: One-hot with mask applied
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1), num_classes=C
        ).permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)
        
        # ✅ Step 3: Softmax with mask
        probs = F.softmax(logits, dim=1) * valid_mask.unsqueeze(1)
        
        # ✅ Step 4: THEN reshape
        probs_flat = probs.reshape(B, C, -1)
        targets_flat = targets_one_hot.reshape(B, C, -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean(dim=1)
        
        return dice_loss.mean()

class FocalLoss(nn.Module):
    """Focal Loss for hard example mining"""
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
    """Combined loss: CE + Dice + Focal"""
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
        
        losses['ce'] = self.ce_loss(logits, targets) if self.ce_weight > 0 else torch.tensor(0.0, device=logits.device)
        losses['dice'] = self.dice_loss(logits, targets) if self.dice_weight > 0 else torch.tensor(0.0, device=logits.device)
        losses['focal'] = self.focal_loss(logits, targets) if self.focal_weight > 0 else torch.tensor(0.0, device=logits.device)
        
        losses['total'] = (
            self.ce_weight * losses['ce'] +
            self.dice_weight * losses['dice'] +
            self.focal_weight * losses['focal']
        )
        
        return losses


# ============================================
# MEMORY UTILITIES
# ============================================

def clear_gpu_memory():
    """Clear GPU cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def print_memory_usage(prefix=""):
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix} GPU Memory - Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")


def setup_memory_efficient_training():
    """Enable memory efficient training"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    """Model configuration for different training scenarios"""
    
    @staticmethod
    def get_config_from_scratch():
        """Config for training from scratch (lightweight, conservative)"""
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,
                "ppm_channels": 96,
                "num_blocks_per_stage": [3, 3, [4, 3], [4, 3], [2, 2]],
                "dwsa_stages": ['bottleneck'],  # Only bottleneck for stability
                "dwsa_num_heads": 8,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                "in_channels": 64,  # c5 = channels * 2 = 32 * 2
                "decoder_channels": 128,
                "dropout_ratio": 0.15,
                "align_corners": False
            },
            "aux_head": {
                "in_channels": 128,  # c4 = channels * 4 = 32 * 4
                "channels": 64,
                "dropout_ratio": 0.15,
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
    
    @staticmethod
    def get_config_lightweight():
        """Lightweight config for fast training"""
        return ModelConfig.get_config_from_scratch()
    
    @staticmethod
    def get_config_standard():
        """Standard config (when pretrained weights available)"""
        return {
            "backbone": {
                "in_channels": 3,
                "decoder_channels": 48,
                "ppm_channels": 112,
                "num_blocks_per_stage": [3, 3, [4, 3], [4, 3], [2, 2]],
                "dwsa_stages": ['stage3', 'bottleneck'],
                "dwsa_num_heads": 8,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                "in_channels": 96,   # 48 * 2
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


# ============================================
# SEGMENTOR MODEL
# ============================================

class Segmentor(nn.Module):
    """Segmentation model with backbone + head + optional auxiliary head"""
    
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        """Inference mode (no aux head)"""
        feats = self.backbone(x)
        return self.decode_head(feats)

    def forward_train(self, x):
        """Training mode (with aux head)"""
        feats = self.backbone(x)
        outputs = {"main": self.decode_head(feats)}
        if self.aux_head is not None:
            outputs["aux"] = self.aux_head(feats)
        return outputs


# ============================================
# TRAINER
# ============================================

class Trainer:
    """Main training class with logging and checkpointing"""
    
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
        
        # Loss function
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
        
        # Print config
        self._print_config(loss_cfg)

    def _print_config(self, loss_cfg):
        """Print training configuration"""
        print(f"\n{'='*70}")
        print("⚙️  TRAINER CONFIGURATION")
        print(f"{'='*70}")
        print(f"📦 Batch size: {self.args.batch_size}")
        print(f"🔁 Gradient accumulation: {self.args.accumulation_steps}")
        print(f"📊 Effective batch size: {self.args.batch_size * self.args.accumulation_steps}")
        print(f"⚡ Mixed precision: {self.args.use_amp}")
        print(f"✂️  Gradient clipping: {self.args.grad_clip}")
        print(f"📉 Loss: CE({loss_cfg['ce_weight']}) + Dice({loss_cfg['dice_weight']}) + Focal({loss_cfg['focal_weight']})")
        print(f"💾 Save dir: {self.args.save_dir}")
        print(f"{'='*70}\n")

    def save_config(self):
        """Save training config to JSON"""
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, loader, epoch):
        """Train one epoch"""
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
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                # Main loss
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total']
                
                # Auxiliary loss
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = F.interpolate(
                        outputs["aux"],
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    aux_loss_dict = self.criterion(aux_logits, masks)
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    loss = loss + aux_weight * aux_loss_dict['total']
                    
                # Scale for gradient accumulation
                loss = loss / self.args.accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                # ✅ CORRECT ORDER: unscale → clip → step → update
                self.scaler.unscale_(self.optimizer)
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                self.global_step += 1
            
            # ✅ Step scheduler EVERY iteration (not just after accumulation)
            if self.scheduler and self.args.scheduler == 'onecycle':
                self.scheduler.step()
            
            # Track loss components
            total_loss += loss.item() * self.args.accumulation_steps
            
            ce_val = loss_dict['ce'].item() if isinstance(loss_dict['ce'], torch.Tensor) else loss_dict['ce']
            dice_val = loss_dict['dice'].item() if isinstance(loss_dict['dice'], torch.Tensor) else loss_dict['dice']
            focal_val = loss_dict['focal'].item() if isinstance(loss_dict['focal'], torch.Tensor) else loss_dict['focal']
            
            total_ce += ce_val
            total_dice += dice_val
            total_focal += focal_val
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ce': f'{ce_val:.4f}',
                'dice': f'{dice_val:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Clear cache
            if batch_idx % 50 == 0:
                clear_gpu_memory()
            
            # TensorBoard logging
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/total_loss', loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ce_loss', ce_val, self.global_step)
                self.writer.add_scalar('train/dice_loss', dice_val, self.global_step)
                self.writer.add_scalar('train/focal_loss', focal_val, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)

        # Step other schedulers
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

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
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        num_classes = self.args.num_classes
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        pbar = tqdm(loader, desc=f"Validation")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits = self.model(imgs)
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total']
            
            total_loss += loss.item()
            
            # Metrics
            pred = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            
            mask = (target >= 0) & (target < num_classes)
            label = num_classes * target[mask].astype('int') + pred[mask]
            count = np.bincount(label, minlength=num_classes**2)
            confusion_matrix += count.reshape(num_classes, num_classes)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        # Compute mIoU
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
        """Save checkpoint"""
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
        """Load checkpoint - reset epoch for new phase"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # ✅ RESET FOR NEW PHASE
        self.start_epoch = 0  # Start from epoch 0
        self.best_miou = 0.0  # Reset best metric
        self.global_step = 0  # Reset step counter
        
        print(f"✅ Weights loaded from epoch {checkpoint['epoch']}")
        print(f"   Starting new phase from epoch 0 (scheduler will be rebuilt)")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="🚀 Optimized GCNet Training from Scratch")
    
    # Dataset
    parser.add_argument("--train_txt", required=True, help="Path to training list")
    parser.add_argument("--val_txt", required=True, help="Path to validation list")
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    
    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs (200 for scratch)")
    parser.add_argument("--model_size", default="lightweight", choices=["lightweight", "standard"])
    parser.add_argument("--from_scratch", action="store_true", default=True)
    
    # Optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3, help="Max LR for OneCycleLR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=1.0, help="Increased for faster learning")
    parser.add_argument("--scheduler", default="onecycle", choices=["onecycle", "poly", "cosine"])
    
    # Data
    parser.add_argument("--img_h", type=int, default=256, help="Start with small images for fast learning")
    parser.add_argument("--img_w", type=int, default=512)
    
    # System
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print header
    print(f"\n{'='*70}")
    print(f"🚀 OPTIMIZED GCNET TRAINING - CURRICULUM LEARNING")
    print(f"{'='*70}")
    print(f"📱 Device: {device}")
    print(f"🎯 Training: FROM SCRATCH")
    print(f"🖼️  Initial image size: {args.img_h}x{args.img_w} (start small for fast learning)")
    print(f"📊 Epochs: {args.epochs}")
    print(f"⚡ Scheduler: {args.scheduler}")
    print(f"💡 Auxiliary weight: {args.aux_weight} (increased for faster backbone learning)")
    print(f"{'='*70}\n")
    
    # Load config
    if args.model_size == "lightweight":
        cfg = ModelConfig.get_config_lightweight()
    else:
        cfg = ModelConfig.get_config_standard()
    
    # Store loss config
    args.loss_config = cfg["loss"]
    
    # Create dataloaders
    print(f"📂 Creating dataloaders...")
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
    print(f"✅ Dataloaders created\n")
    
    # Create model
    print(f"{'='*70}")
    print("🏗️  BUILDING MODEL")
    print(f"{'='*70}\n")
    
    model = Segmentor(
        backbone=GCNetWithDWSA(**cfg["backbone"]),
        head=GCNetHead(num_classes=args.num_classes, **cfg["head"]),
        aux_head=GCNetAuxHead(num_classes=args.num_classes, **cfg["aux_head"])
    )
    
    # ====================================================
    # ✅ CRITICAL: Apply model optimizations
    # ====================================================    
    print("\n🔧 Applying Model Optimizations...")
    print("   ├─ Converting BatchNorm → GroupNorm")
    model = replace_bn_with_gn(model)
    
    print("   └─ Applying Kaiming Initialization")
    model.apply(init_weights)
    
    check_model_health(model)
    print()
    # ====================================================
    
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
            return
    
    print(f"{'='*70}\n")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * args.epochs  # Total training iterations
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,  # ✅ Use total_steps instead
            pct_start=0.05,  # 5% warmup (standard for segmentation)
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,
            final_div_factor=100000,
        )
    elif args.scheduler == 'poly':
        print(f"✅ Using Polynomial LR decay")
        def poly_lr_lambda(epoch):
            return (1 - epoch / args.epochs) ** 0.9
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    else:
        print(f"✅ Using Cosine Annealing LR")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    
    # Trainer
    trainer = Trainer(
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
    print("🚀 STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(trainer.start_epoch, args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader, epoch)
        
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
        
        # TensorBoard logging
        trainer.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        trainer.writer.add_scalar('val/miou', val_metrics['miou'], epoch)
        trainer.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        
        # Save checkpoint
        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
    
    trainer.writer.close()
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"🏆 Best mIoU: {trainer.best_miou:.4f}")
    print(f"💾 Checkpoints saved to: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
