# ============================================
# OPTIMIZED train.py - MAXIMUM TRANSFER LEARNING + v2 BACKBONE
# ============================================
# Key changes:
# 1. Import GCNetWithDWSA_v2 (optimized for 80%+ transfer learning)
# 2. Backbone config: channels=32, dwsa_stages=['stage4','stage5','stage6']
# 3. Load pretrained weights with shape-based filtering
# 4. Progressive unfreezing schedule
# 5. Discriminative learning rates (backbone_lr_factor=0.1)

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

from model.backbone.model import (
    GCNetWithEnhance,
    GCNetCore,      # nếu cần
    GCBlock,
    DWSABlock,
    MultiScaleContextModule,
)
from model.head.segmentation_head import (
    GCNetHead,
    GCNetAuxHead,
)
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health

def load_pretrained_gcnet_core(model, ckpt_path):
    """
    Load GCNet pretrained weights vào model. với matching theo shape.
    - Không giả định checkpoint có prefix '.'
    - Bỏ qua các layer mới (DWSA/DCN/MultiScale), chỉ load phần trùng tên + shape.
    """
    print(f"📥 Loading pretrained GCNet weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for ckpt_key, ckpt_val in state.items():
        # Bỏ một số prefix hay gặp: '.', 'model.', 'module.'
        k = ckpt_key
        for pref in ['.', 'model.', 'module.']:
            if k.startswith(pref):
                k = k[len(pref):]

        # 1) Thử match trực tiếp
        if k in model_state and model_state[k].shape == ckpt_val.shape:
            compatible[k] = ckpt_val
            continue

        # 2) Thử match theo hậu tố (endswith), phòng trường hợp tên hơi khác
        matched = False
        for mk in model_state.keys():
            if mk.endswith(k) and model_state[mk].shape == ckpt_val.shape:
                compatible[mk] = ckpt_val
                matched = True
                break

        if not matched:
            skipped.append(ckpt_key)

    loaded = len(compatible)
    total = len(model_state)
    rate = 100 * loaded / total if total > 0 else 0.0

    print(f"Loaded {loaded}/{total} params into  ({rate:.1f}%).")
    if loaded == 0:
        print("⚠️  WARNING: 0 params loaded. Check checkpoint format and key names.")

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Missing keys in loaded dict: {len(missing)}")
    print(f"Unexpected keys in loaded dict: {len(unexpected)}\n")

    return rate


# ============================================
# LOSS FUNCTIONS
# ============================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        
        valid_mask = (targets != self.ignore_index).float()
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1), num_classes=C
        ).permute(0, 3, 1, 2).float()
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)
        
        probs = F.softmax(logits, dim=1) * valid_mask.unsqueeze(1)
        
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
        dice_smooth=1e-5
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
# UTILITIES
# ============================================

def clear_gpu_memory():
    """Clear GPU cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_memory_efficient_training():
    """Enable memory efficient training"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ============================================
# FREEZE/UNFREEZE UTILITIES FOR TRANSFER LEARNING
# ============================================

def freeze_(model):
    """Freeze toàn bộ """
    for param in model.parameters():
        param.requires_grad = False
    print("🔒  FROZEN - chỉ head được train")


def unfreeze__progressive(model, stage_names):
    """
    Unfreeze các stage cụ thể của 
    stage_names: list như ['stage1', 'stage2'] hoặc string như 'stage4'
    """
    if isinstance(stage_names, str):
        stage_names = [stage_names]
    
    unfrozen_count = 0
    for stage_name in stage_names:
        for name, module in model.named_modules():
            if stage_name in name:
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
    
    print(f"🔓 Unfrozen stages: {stage_names} ({unfrozen_count} parameters)")


def count_trainable_params(model):
    """Đếm và hiển thị số parameters trainable/frozen"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    _total = sum(p.numel() for p in model.parameters())
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    head_total = sum(p.numel() for p in model.decode_head.parameters())
    head_trainable = sum(p.numel() for p in model.decode_head.parameters() if p.requires_grad)
    
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        aux_total = sum(p.numel() for p in model.aux_head.parameters())
        aux_trainable = sum(p.numel() for p in model.aux_head.parameters() if p.requires_grad)
    else:
        aux_total = aux_trainable = 0
    
    print(f"\n{'='*70}")
    print("📊 PARAMETER STATISTICS")
    print(f"{'='*70}")
    print(f"Total:        {total:>15,} | 100%")
    print(f"Trainable:    {trainable:>15,} | {100*trainable/total:.1f}%")
    print(f"Frozen:       {frozen:>15,} | {100*frozen/total:.1f}%")
    print(f"{'-'*70}")
    print(f":     {_trainable:>15,} / {_total:,} | {100*_trainable/_total:.1f}%")
    print(f"Head:         {head_trainable:>15,} / {head_total:,} | {100*head_trainable/head_total:.1f}%")
    if aux_total > 0:
        print(f"Aux Head:     {aux_trainable:>15,} / {aux_total:,} | {100*aux_trainable/aux_total:.1f}%")
    print(f"{'='*70}\n")
    
    return trainable, frozen


def setup_discriminative_lr(model, base_lr, _lr_factor=0.1, weight_decay=1e-4):
    """
    Tạo optimizer với LR khác nhau cho  vs head
    _lr = base_lr * _lr_factor
    head_lr = base_lr
    """
    _params = [p for n, p in model.named_parameters() 
                      if '' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() 
                  if '' not in n and p.requires_grad]
    
    if len(_params) == 0:
        optimizer = torch.optim.AdamW(head_params, lr=base_lr, weight_decay=weight_decay)
        print(f"⚙️  Optimizer: AdamW (lr={base_lr}) - chỉ head")
    else:
        _lr = base_lr * _lr_factor
        param_groups = [
            {'params': _params, 'lr': _lr, 'name': ''},
            {'params': head_params, 'lr': base_lr, 'name': 'head'}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        print(f"⚙️  Optimizer: AdamW (Discriminative LR)")
        print(f"   ├─  LR: {_lr:.2e} ({len(_params):,} params)")
        print(f"   └─ Head LR:     {base_lr:.2e} ({len(head_params):,} params)")
    
    return optimizer


# ============================================
# MODEL CONFIG - OPTIMIZED v2  FOR TRANSFER LEARNING
# ============================================

class ModelConfig:
    """✅ Optimized Config: GCNetWithDWSA_v2 + Transfer Learning"""
    
    @staticmethod
    def get_config():
        """
        Config để tối đa transfer learning từ GCNet Cityscapes:
        - channels=32 (khớp GCNet gốc)
        - dwsa_stages=['stage4','stage5','stage6'] (chỉ deep layers)
        - use_dcn_in_stage5_6=True (ở nơi cần nhất)
        Expected: 80%+ params reuse từ Cityscapes pretrained
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,  # ✅ Giữ nguyên = GCNet gốc
                "ppm_channels": 128,
                "num_blocks_per_stage": [3, 3, [3, 2], [3, 2], [2, 2]],  # ✅ Giữ nguyên
                "dwsa_stages": [ 'stage6'],  # ✅ Chỉ ở cuối
                "dwsa_num_heads": 4,
                "use_multi_scale_context": True,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                "in_channels": 64,  # channels * 2 = 32 * 2 (sẽ override bằng detect__channels)
                "decoder_channels": 128,
                "dropout_ratio": 0.1,
                "align_corners": False,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "aux_head": {
                "in_channels": 128,  # channels * 4 = 32 * 4 (sẽ override bằng detect_backbone_channels)
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
                "dice_smooth": 1e-5
            }
        }


# ============================================
# SEGMENTOR MODEL
# ============================================

class Segmentor(nn.Module):
    """Segmentation model with backbone + head + auxiliary head"""
    
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        """Inference mode"""
        feats = self.backbone(x)
        return self.decode_head(feats)

    def forward_train(self, x):
        """Training mode with auxiliary head"""
        feats = self.backbone(x)
        outputs = {"main": self.decode_head(feats)}
        if self.aux_head is not None:
            outputs["aux"] = self.aux_head(feats)
        return outputs


# ============================================
# TRAINER
# ============================================

class Trainer:
    """Training class with logging and checkpointing"""
    
    def __init__(self, model, optimizer, scheduler, device, args, class_weights=None):
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
        
        self.save_config()
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
        """Save training config"""
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

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

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)
                logits = outputs["main"]
                
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total']
                
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = outputs["aux"]
                    aux_logits = F.interpolate(aux_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    aux_loss_dict = self.criterion(aux_logits, masks)
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    loss = loss + aux_weight * aux_loss_dict['total']
                    
                loss = loss / self.args.accumulation_steps

            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            
            if self.scheduler and self.args.scheduler == 'onecycle':
                self.scheduler.step()
            
            total_loss += loss.item() * self.args.accumulation_steps
            
            ce_val = loss_dict['ce'].item() if isinstance(loss_dict['ce'], torch.Tensor) else loss_dict['ce']
            dice_val = loss_dict['dice'].item() if isinstance(loss_dict['dice'], torch.Tensor) else loss_dict['dice']
            focal_val = loss_dict['focal'].item() if isinstance(loss_dict['focal'], torch.Tensor) else loss_dict['focal']
            
            total_ce += ce_val
            total_dice += dice_val
            total_focal += focal_val
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ce': f'{ce_val:.4f}',
                'dice': f'{dice_val:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            if batch_idx % 50 == 0:
                clear_gpu_memory()
            
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/total_loss', loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ce_loss', ce_val, self.global_step)
                self.writer.add_scalar('train/dice_loss', dice_val, self.global_step)
                self.writer.add_scalar('train/focal_loss', focal_val, self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)

        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_ce = total_ce / len(loader)
        avg_dice = total_dice / len(loader)
        avg_focal = total_focal / len(loader)
        
        return {'loss': avg_loss, 'ce': avg_ce, 'dice': avg_dice, 'focal': avg_focal}

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
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total']
            
            total_loss += loss.item()
            
            pred = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            
            mask = (target >= 0) & (target < num_classes)
            label = num_classes * target[mask].astype('int') + pred[mask]
            count = np.bincount(label, minlength=num_classes**2)
            confusion_matrix += count.reshape(num_classes, num_classes)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        intersection = np.diag(confusion_matrix)
        union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        
        acc = intersection.sum() / (confusion_matrix.sum() + 1e-10)
        avg_loss = total_loss / len(loader)
        
        return {'loss': avg_loss, 'miou': miou, 'accuracy': acc, 'per_class_iou': iou}

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

    def load_checkpoint(self, checkpoint_path, reset_epoch=True):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        if reset_epoch:
            self.start_epoch = 0
            self.best_miou = 0.0
            self.global_step = 0
            print(f"✅ Weights loaded from epoch {checkpoint['epoch']}, starting new phase from epoch 0")
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            if self.scheduler and checkpoint.get('scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"✅ Checkpoint loaded, resuming from epoch {self.start_epoch}")


def detect_backbone_channels(backbone, device, img_size=(512, 1024)):
    """Automatically detect backbone output channels"""
    backbone.eval()
    with torch.no_grad():
        sample = torch.randn(1, 3, *img_size).to(device)
        feats = backbone(sample)
        
        channels = {}
        for key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            if key in feats:
                channels[key] = feats[key].shape[1]
        
        print(f"\n{'='*70}")
        print("🔍 BACKBONE CHANNEL DETECTION")
        print(f"{'='*70}")
        for key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            if key in channels:
                print(f"   {key}: {channels[key]} channels")
        print(f"{'='*70}\n")
        
        return channels


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="🚀 GCNetWithDWSA_v2 Training - Maximum Transfer Learning")
    
    # Transfer Learning Arguments
    parser.add_argument("--pretrained_weights", type=str, default=None,
                       help="Path to pretrained GCNet weights (Cityscapes)")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                       help="Freeze backbone từ epoch 0")
    parser.add_argument("--unfreeze_schedule", type=str, default="",
                       help="Các epoch sẽ unfreeze (VD: '10,20,30,40' hoặc rỗng để không unfreeze)")
    parser.add_argument("--use_discriminative_lr", action="store_true", default=True,
                       help="Dùng LR khác nhau cho backbone vs head")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1,
                       help="Backbone LR = head_lr * factor")
    
    # Dataset
    parser.add_argument("--train_txt", required=True, help="Path to training list")
    parser.add_argument("--val_txt", required=True, help="Path to validation list")
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    
    # Optimization
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4, help="Max LR (for head)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=1.0, help="Auxiliary head weight (decays over epochs)")
    parser.add_argument("--scheduler", default="onecycle", choices=["onecycle", "poly", "cosine"])
    
    # Data
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    
    # System
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_mode", type=str, default="transfer", 
                    choices=["transfer", "continue"],
                    help="transfer: start from epoch 0, continue: resume from saved epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"🚀 GCNetWithDWSA_v2 Training - Maximum Transfer Learning")
    print(f"{'='*70}")
    print(f"📱 Device: {device}")
    print(f"🖼️  Image size: {args.img_h}x{args.img_w}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"⚡ Scheduler: {args.scheduler}")
    print(f"❄️  Freeze backbone: {args.freeze_backbone}")
    if args.unfreeze_schedule:
        print(f"📅 Unfreeze schedule: {args.unfreeze_schedule}")
    print(f"🔀 Discriminative LR: {args.use_discriminative_lr} (backbone_factor={args.backbone_lr_factor})")
    print(f"{'='*70}\n")
    
    # Config
    cfg = ModelConfig.get_config()
    args.loss_config = cfg["loss"]
    
    # Dataloaders
    print(f"📂 Creating dataloaders..")
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
    
    # Model
    print(f"{'='*70}")
    print("🏗️  BUILDING GCNetWithDWSA_v2 WITH TRANSFER LEARNING")
    print(f"{'='*70}\n")
    
    # Build backbone (v2)
    backbone = GCNetWithEnhance(**cfg["backbone"]).to(device)
    
    # Auto-detect backbone channels
    detected_channels = detect_backbone_channels(backbone, device, (args.img_h, args.img_w))
    
    # Build head config
    head_cfg = {
        **cfg["head"],
        "in_channels": detected_channels['c5'],
        "c1_channels": detected_channels['c1'],
        "c2_channels": detected_channels['c2'],
        "num_classes": args.num_classes,
    }
    
    aux_head_cfg = {
        **cfg["aux_head"],
        "in_channels": detected_channels['c4'],
        "num_classes": args.num_classes,
    }
    
    # Build Segmentor
    model = Segmentor(
        backbone=backbone,
        head=GCNetHead(**head_cfg),
        aux_head=GCNetAuxHead(**aux_head_cfg),
    )
    
    print("\n🔧 Applying Model Optimizations..")
    print("   ├─ Converting BatchNorm → GroupNorm")
    model = replace_bn_with_gn(model)
    
    print("   ├─ Applying Kaiming Initialization")
    model.apply(init_weights)
    
    print("   └─ Checking Model Health")
    check_model_health(model)
    print()
    
    # ===================== TRANSFER LEARNING SETUP =====================
    print(f"{'='*70}")
    print("🔄 TRANSFER LEARNING SETUP")
    print(f"{'='*70}\n")
    
    # Load pretrained weights
    # Load pretrained weights with EXACT KEY MAPPING (stem.stageX → stageX)
    if args.pretrained_weights:
        load_pretrained_gcnet_core(model, args.pretrained_weights)

    
    # Freeze backbone if requested
    if args.freeze_backbone:
        freeze_backbone(model)
        print()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    count_trainable_params(model)
    
    # ===================== END TRANSFER LEARNING SETUP =====================
    
    # Test forward pass
    model = model.to(device)
    with torch.no_grad():
        sample = torch.randn(1, 3, args.img_h, args.img_w).to(device)
        try:
            outputs = model.forward_train(sample)
            print(f"✅ Forward pass successful!")
            print(f"   Main head output:  {outputs['main'].shape}")
            if 'aux' in outputs:
                print(f"   Aux head output:   {outputs['aux'].shape}\n")
        except Exception as e:
            print(f"❌ Forward pass FAILED: {e}\n")
            return
    
    # Optimizer with discriminative LR
    if args.use_discriminative_lr:
        optimizer = setup_discriminative_lr(
            model,
            base_lr=args.lr,
            backbone_lr_factor=args.backbone_lr_factor,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        print(f"⚙️  Optimizer: AdamW (lr={args.lr})")
    
    # Scheduler
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * args.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,
            final_div_factor=100000,
        )
        print(f"✅ Using OneCycleLR scheduler (total_steps={total_steps})")
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
        reset_epoch = (args.resume_mode == "transfer")
        trainer.load_checkpoint(args.resume, reset_epoch=reset_epoch)
    
    # Training loop
    print(f"\n{'='*70}")
    print("🚀 STARTING TRAINING")
    print(f"{'='*70}\n")
    
    # Parse unfreeze schedule
    unfreeze_epochs = []
    if args.unfreeze_schedule:
        try:
            unfreeze_epochs = [int(e) for e in args.unfreeze_schedule.split(',')]
        except:
            unfreeze_epochs = []
    
    for epoch in range(trainer.start_epoch, args.epochs):
        # Progressive unfreezing
        if epoch in unfreeze_epochs:
            stage_to_unfreeze = f"stage{4 + len([e for e in unfreeze_epochs if e <= epoch])}"
            unfreeze_backbone_progressive(model, stage_to_unfreeze)
            print()
        
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)
        
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
        
        trainer.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        trainer.writer.add_scalar('val/miou', val_metrics['miou'], epoch)
        trainer.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        
        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
    
    trainer.writer.close()
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"🏆 Best mIoU: {trainer.best_miou:.4f}")
    print(f"💾 Checkpoints saved to: {args.save_dir}")
    print(f"📊 Tensorboard logs at: {args.save_dir}/tensorboard")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
