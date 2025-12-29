# ============================================
# ADVANCED GCNET TRAINING PIPELINE
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, Optional
import json
import time
from datetime import datetime

# ============================================
# IMPORTS
# ============================================

from model.backbone.model import GCNetWithDWSA
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders

# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config(base_channels=32):
        """
        Model configuration with flexible base channels
        
        Args:
            base_channels: Base channel multiplier (default: 32)
                - 32: Lightweight model (~2-3M params)
                - 64: Standard model (~8-10M params)
                - 128: Large model (~30M+ params)
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": base_channels,
                "ppm_channels": min(128, base_channels * 4),
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
                "deploy": False
            },
            "head": {
                "in_channels": base_channels * 2,   # c5
                "channels": min(128, base_channels * 4),
                "decode_enabled": False,
                "skip_channels": [base_channels * 2, base_channels, base_channels],
                "dropout_ratio": 0.1,
                "align_corners": False
            },
            "aux_head": {
                "in_channels": base_channels * 4,   # c4
                "channels": min(64, base_channels * 2),
                "dropout_ratio": 0.1,
                "align_corners": False
            }
        }

# ============================================
# SEGMENTOR
# ============================================

class Segmentor(nn.Module):
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
        """Training mode with auxiliary outputs"""
        feats = self.backbone(x)
        outputs = {"main": self.decode_head(feats)}
        if self.aux_head is not None:
            outputs["aux"] = self.aux_head(feats)
        return outputs

# ============================================
# LOSS FUNCTIONS
# ============================================

class SegmentationLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""
    
    def __init__(self, ignore_index=255, class_weights=None, ce_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights
        )
        self.ce_weight = ce_weight
        self.dice_weight = 1.0 - ce_weight
        self.ignore_index = ignore_index

    def dice_loss(self, logits, targets, eps=1e-6):
        """Soft Dice Loss"""
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # Create one-hot encoded targets
        targets_oh = torch.nn.functional.one_hot(
            targets, num_classes
        ).permute(0, 3, 1, 2).float()

        # Compute Dice coefficient
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        union = torch.sum(probs + targets_oh, dims)
        dice = (2 * intersection + eps) / (union + eps)
        
        return 1 - dice.mean()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        # Mask out ignore_index for Dice loss
        mask = targets != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# ============================================
# EMA (EXPONENTIAL MOVING AVERAGE)
# ============================================

class EMAModel:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        """Update EMA weights"""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v * (1 - self.decay))

    def apply(self, model):
        """Apply EMA weights to model"""
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        """Restore original weights"""
        if self.backup:
            model.load_state_dict(self.backup, strict=False)

# ============================================
# METRICS
# ============================================

class SegMetrics:
    """Segmentation metrics calculator"""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset confusion matrix"""
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        """Update confusion matrix"""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Remove ignore index
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.cm[int(t), int(p)] += 1

    def miou(self):
        """Mean Intersection over Union"""
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        iou = inter / (union + 1e-10)
        return np.nanmean(iou)
    
    def accuracy(self):
        """Pixel Accuracy"""
        return np.diag(self.cm).sum() / (self.cm.sum() + 1e-10)
    
    def per_class_iou(self):
        """IoU for each class"""
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        return inter / (union + 1e-10)

# ============================================
# TRAINER
# ============================================

class Trainer:
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
        self.criterion = SegmentationLoss(
            ignore_index=args.ignore_index,
            class_weights=class_weights.to(device) if class_weights is not None else None,
            ce_weight=args.ce_weight
        )
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=args.use_amp)
        
        # EMA
        self.ema = EMAModel(model, decay=args.ema_decay) if args.use_ema else None
        
        # Tracking
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        
        self.best_miou = 0.0
        self.start_epoch = 0
        self.global_step = 0
        
        # Save config
        self.save_config()

    def save_config(self):
        """Save training configuration"""
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved to {self.save_dir / 'config.json'}")

    def train_epoch(self, loader, epoch):
        """Train one epoch"""
        self.model.train()
        metrics = SegMetrics(num_classes=self.args.num_classes)
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).long()
            
            # Remove channel dimension if present
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)
                logits = outputs["main"]
                
                # Interpolate to target size
                logits = nn.functional.interpolate(
                    logits, 
                    size=masks.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                
                # Main loss
                main_loss = self.criterion(logits, masks)
                loss = main_loss
                
                # Auxiliary loss
                aux_loss = torch.tensor(0.0, device=self.device)
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = nn.functional.interpolate(
                        outputs["aux"], 
                        size=masks.shape[-2:], 
                        mode="bilinear", 
                        align_corners=False
                    )
                    aux_loss = self.criterion(aux_logits, masks)
                    loss += self.args.aux_weight * aux_loss

            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.grad_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            if self.scheduler and self.args.scheduler_type == 'onecycle':
                self.scheduler.step()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

            # Compute metrics
            pred = logits.argmax(1)
            metrics.update(pred, masks)
            
            # Track losses
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            if aux_loss.item() > 0:
                total_aux_loss += aux_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # TensorBoard logging
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1

        # Epoch metrics
        avg_loss = total_loss / len(loader)
        train_miou = metrics.miou()
        train_acc = metrics.accuracy()
        
        return {
            'loss': avg_loss,
            'main_loss': total_main_loss / len(loader),
            'aux_loss': total_aux_loss / len(loader) if total_aux_loss > 0 else 0,
            'miou': train_miou,
            'accuracy': train_acc
        }

    @torch.no_grad()
    def validate(self, loader, epoch):
        """Validate model"""
        # Apply EMA if enabled
        if self.ema:
            self.ema.apply(self.model)

        self.model.eval()
        metrics = SegMetrics(num_classes=self.args.num_classes)
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
        
        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            # Forward pass
            logits = self.model(imgs)
            logits = nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            # Compute loss
            loss = self.criterion(logits, masks)
            total_loss += loss.item()
            
            # Compute metrics
            pred = logits.argmax(1)
            metrics.update(pred, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Restore original weights if EMA was used
        if self.ema:
            self.ema.restore(self.model)

        avg_loss = total_loss / len(loader)
        val_miou = metrics.miou()
        val_acc = metrics.accuracy()
        per_class_iou = metrics.per_class_iou()

        return {
            'loss': avg_loss,
            'miou': val_miou,
            'accuracy': val_acc,
            'per_class_iou': per_class_iou
        }

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'ema': self.ema.shadow if self.ema else None,
            'best_miou': self.best_miou,
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.save_dir / "last.pth")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / "best.pth")
            print(f"✅ Best model saved! mIoU: {metrics['miou']:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(checkpoint, self.save_dir / f"epoch_{epoch+1}.pth")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if self.args.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        if self.ema and checkpoint['ema']:
            self.ema.shadow = checkpoint['ema']
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint['best_miou']
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
        print(f"   Best mIoU: {self.best_miou:.4f}")

# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GCNet Training Pipeline")
    
    # Dataset
    parser.add_argument("--train_txt", required=True, help="Path to train txt file")
    parser.add_argument("--val_txt", required=True, help="Path to val txt file")
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"],
                        help="Dataset type: normal or foggy Cityscapes")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 1024],
                        help="Image size [H, W]")
    parser.add_argument("--compute_class_weights", action="store_true",
                        help="Compute class weights for balanced training")
    
    # Model
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base channels (32/64/128)")
    parser.add_argument("--aux_weight", type=float, default=0.4,
                        help="Auxiliary loss weight")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping (0 to disable)")
    
    # Loss
    parser.add_argument("--ce_weight", type=float, default=0.5,
                        help="Cross Entropy weight in loss (1-ce_weight for Dice)")
    
    # Scheduler
    parser.add_argument("--scheduler_type", default="cosine",
                        choices=["cosine", "onecycle", "step"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs")
    
    # Optimization
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use Automatic Mixed Precision")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use Exponential Moving Average")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate")
    
    # Logging & Saving
    parser.add_argument("--save_dir", default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N batches")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # System
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"🚀 GCNet Training Pipeline")
    print(f"{'='*70}")
    print(f"📱 Device: {device}")
    print(f"🎯 Dataset: {args.dataset_type.upper()} Cityscapes")
    print(f"📐 Image size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"{'='*70}\n")
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        compute_class_weights=args.compute_class_weights,
        dataset_type=args.dataset_type
    )
    
    # Create model
    print(f"\n{'='*70}")
    print("🏗️  Building Model...")
    print(f"{'='*70}\n")
    
    cfg = ModelConfig.get_config(base_channels=args.base_channels)
    model = Segmentor(
        backbone=GCNetWithDWSA(**cfg["backbone"]),
        head=GCNetHead(num_classes=args.num_classes, **cfg["head"]),
        aux_head=GCNetAuxHead(num_classes=args.num_classes, **cfg["aux_head"])
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = None
    if args.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    elif args.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        class_weights=class_weights
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training loop
    print(f"\n{'='*70}")
    print("🚀 Starting Training...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(trainer.start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader, epoch)
        
        # Update scheduler
        if scheduler and args.scheduler_type != 'onecycle':
            scheduler.step()
        
        # Logging
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch+1}/{args.epochs} Summary (Time: {epoch_time:.2f}s)")
        print(f"{'='*70}")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['miou']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"{'='*70}\n")
        
        # TensorBoard logging
        trainer.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        trainer.writer.add_scalars('mIoU', {
            'train': train_metrics['miou'],
            'val': val_metrics['miou']
        }, epoch)
        trainer.writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        
        # Save checkpoint
        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ Training Completed!")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best validation mIoU: {trainer.best_miou:.4f}")
    print(f"💾 Checkpoints saved to: {args.save_dir}")
    print(f"{'='*70}\n")
    
    trainer.writer.close()

if __name__ == "__main__":
    main()
