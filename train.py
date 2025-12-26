import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path
import wandb
from typing import Dict, Optional, Tuple

# Import your modules
from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import CityscapesCustomDataset, create_dataloaders
from model.distillation_final import (
    GCNetWithDistillation,
    create_distillation_model,
    get_default_configs
)

# ============================================
# METRICS
# ============================================

class SegmentationMetrics:
    """Calculate mIoU, pixel accuracy, etc."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: (B, H, W) predicted class indices
            target: (B, H, W) ground truth
        """
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Remove ignore index
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_iou(self) -> np.ndarray:
        """Compute IoU per class"""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        iou = intersection / (union + 1e-10)
        return iou
    
    def compute_miou(self) -> float:
        """Compute mean IoU"""
        iou = self.compute_iou()
        return np.nanmean(iou)
    
    def compute_pixel_acc(self) -> float:
        """Compute pixel accuracy"""
        acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        return acc
    
    def get_results(self) -> Dict[str, float]:
        """Get all metrics"""
        iou_per_class = self.compute_iou()
        return {
            'mIoU': self.compute_miou(),
            'pixel_acc': self.compute_pixel_acc(),
            'iou_per_class': iou_per_class
        }


# ============================================
# TRAINER WITH DISTILLATION SUPPORT
# ============================================

class DistillationTrainer:
    """
    Enhanced trainer with distillation support
    
    Features:
    - Standard training (no distillation)
    - Distillation training (with teacher)
    - Mixed precision training
    - Auto-save & resume
    - W&B logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = 'cuda',
        num_classes: int = 19,
        use_distillation: bool = False,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        save_dir: str = './checkpoints',
        use_wandb: bool = False,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.use_distillation = use_distillation
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Auto-save
        self.last_autosave_time = time.time()
        self.autosave_interval = 15 * 60  # 15 minutes
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Loss function (for non-distillation mode)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255
        ) if not use_distillation else None
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=num_classes)
        
        # Best model tracking
        self.best_miou = 0.0
        self.current_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"Trainer initialized:")
        print(f"  Mode: {'Distillation' if use_distillation else 'Standard'}")
        print(f"  Device: {device}")
        print(f"  AMP: {use_amp}")
        print(f"  Gradient clipping: {grad_clip}")
        print(f"  Save dir: {save_dir}")
        print(f"{'='*60}\n")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze(1).long()  # (B, H, W)
            
            # Forward & backward
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                if self.use_distillation:
                    # Distillation mode
                    losses = self.model.forward_train(images, targets.unsqueeze(1))
                    loss = losses['loss']
                    
                    # Track all loss components
                    for k, v in losses.items():
                        if k != 'loss':
                            loss_components[k] = loss_components.get(k, 0.0) + v.item()
                else:
                    # Standard mode
                    logits = self.model(images)
                    
                    # Resize if needed
                    if logits.shape[-2:] != targets.shape[-2:]:
                        logits = nn.functional.interpolate(
                            logits,
                            size=targets.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    loss = self.criterion(logits, targets)
                    losses = {'loss': loss}
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                if self.use_distillation:
                    logits = self.model.forward_test(images)
                else:
                    logits = self.model(images)
                
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = nn.functional.interpolate(
                        logits,
                        size=targets.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                pred = logits.argmax(dim=1)
                self.metrics.update(pred, targets)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Auto-save
            if time.time() - self.last_autosave_time > self.autosave_interval:
                self.save_checkpoint(
                    filename='autosave.pth',
                    epoch=self.current_epoch,
                    metrics={'iter': batch_idx}
                )
                self.last_autosave_time = time.time()
                print(f"\n✓ Auto-saved at epoch {self.current_epoch}, iter {batch_idx}")
        
        # Compute epoch metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            **{f'train_{k}': v for k, v in self.metrics.get_results().items() if k != 'iou_per_class'}
        }
        
        # Add loss components if distillation
        if self.use_distillation and loss_components:
            for k, v in loss_components.items():
                epoch_metrics[f'train_{k}'] = v / num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate"""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze(1).long()
            
            # Forward
            with autocast(enabled=self.use_amp):
                if self.use_distillation:
                    logits = self.model.forward_test(images)
                else:
                    logits = self.model(images)
                
                # Resize if needed
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = nn.functional.interpolate(
                        logits,
                        size=targets.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits,
                    targets,
                    ignore_index=255
                )
            
            # Update metrics
            pred = logits.argmax(dim=1)
            self.metrics.update(pred, targets)
            
            total_loss += loss.item()
        
        # Compute metrics
        results = self.metrics.get_results()
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_mIoU': results['mIoU'],
            'val_pixel_acc': results['pixel_acc']
        }
        
        # Per-class IoU (optional, for detailed logging)
        if self.use_wandb:
            iou_per_class = results['iou_per_class']
            for i, iou in enumerate(iou_per_class):
                if not np.isnan(iou):
                    val_metrics[f'val_iou_class_{i}'] = iou
        
        return val_metrics
    
    def train(self, num_epochs: int):
        """Main training loop"""
        start_epoch = self.current_epoch
        
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Combine metrics
            all_metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': current_lr
            }
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs-1}")
            print(f"{'='*60}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Train mIoU: {train_metrics['train_mIoU']:.4f} | "
                  f"Train Acc: {train_metrics['train_pixel_acc']:.4f}")
            print(f"Val Loss:   {val_metrics['val_loss']:.4f} | "
                  f"Val mIoU:   {val_metrics['val_mIoU']:.4f} | "
                  f"Val Acc:   {val_metrics['val_pixel_acc']:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # Print distillation loss components
            if self.use_distillation:
                distill_losses = [k for k in train_metrics.keys() if 'loss_' in k and k != 'train_loss']
                if distill_losses:
                    print("\nDistillation losses:")
                    for k in distill_losses:
                        print(f"  {k}: {train_metrics[k]:.4f}")
            
            print(f"{'='*60}\n")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(all_metrics, step=epoch)
            
            # Save best model
            if val_metrics['val_mIoU'] > self.best_miou:
                self.best_miou = val_metrics['val_mIoU']
                self.save_checkpoint('best_model.pth', epoch, all_metrics)
                print(f"✓ Saved best model at epoch {epoch} (mIoU: {self.best_miou:.4f})")
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth', epoch, all_metrics)
                print(f"✓ Saved checkpoint at epoch {epoch}")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_miou': self.best_miou,
            'use_distillation': self.use_distillation
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        print(f"✓ Resumed at epoch {self.current_epoch}, best mIoU = {self.best_miou:.4f}")


# ============================================
# CONFIGURATION HELPERS
# ============================================

def get_model_config(mode: str = 'standard') -> Tuple[Dict, Dict, Dict]:
    """
    Get model configurations for different modes
    
    Args:
        mode: 'standard', 'small', 'large', or 'distillation'
    
    Returns:
        backbone_cfg, head_cfg, aux_head_cfg
    
    Channel calculation from backbone:
        channels = base_channels
        c1 = channels      (H/2)
        c2 = channels      (H/4)
        c3 = channels * 2  (H/8)
        c4 = channels * 4  (H/16)
        c5 = channels * 2  (H/8)
    """
    if mode == 'small':
        # Small model (for student or fast training)
        base_channels = 16
        
        backbone_cfg = {
            'in_channels': 3,
            'channels': base_channels,  # 16
            'ppm_channels': 64,
            'num_blocks_per_stage': [2, 2, [3, 2], [3, 2], [2, 1]],
            'use_flash_attention': False,
            'use_se': True,
            'deploy': False
        }
        
        # Skip connections: [c3, c2, c1]
        # c3 = 16*2 = 32, c2 = 16, c1 = 16
        head_cfg = {
            'in_channels': base_channels * 2,  # c5 = 32
            'channels': 64,
            'decode_enabled': True,
            'decoder_channels': 64,
            'skip_channels': [base_channels * 2, base_channels, base_channels],  # [32, 16, 16]
            'use_gated_fusion': True,
            'dropout_ratio': 0.1,
            'align_corners': False
        }
        
        aux_head_cfg = {
            'in_channels': base_channels * 4,  # c4 = 64
            'channels': 32,
            'dropout_ratio': 0.1,
            'align_corners': False
        }
    
    elif mode == 'standard':
        # Standard model (balanced)
        base_channels = 32
        
        backbone_cfg = {
            'in_channels': 3,
            'channels': base_channels,  # 32
            'ppm_channels': 128,
            'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
            'use_flash_attention': True,
            'flash_attn_stage': 4,
            'flash_attn_layers': 2,
            'flash_attn_heads': 8,
            'use_se': True,
            'deploy': False
        }
        
        # Skip connections: [c3, c2, c1]
        # c3 = 32*2 = 64, c2 = 32, c1 = 32
        head_cfg = {
            'in_channels': base_channels * 2,  # c5 = 64
            'channels': 128,
            'decode_enabled': True,
            'decoder_channels': 128,
            'skip_channels': [base_channels * 2, base_channels, base_channels],  # [64, 32, 32]
            'use_gated_fusion': True,
            'dropout_ratio': 0.1,
            'align_corners': False
        }
        
        aux_head_cfg = {
            'in_channels': base_channels * 4,  # c4 = 128
            'channels': 64,
            'dropout_ratio': 0.1,
            'align_corners': False
        }
    
    elif mode == 'large':
        # Large model (for teacher)
        base_channels = 48
        
        backbone_cfg = {
            'in_channels': 3,
            'channels': base_channels,  # 48
            'ppm_channels': 192,
            'num_blocks_per_stage': [5, 5, [6, 5], [6, 5], [3, 3]],
            'use_flash_attention': True,
            'flash_attn_stage': 4,
            'flash_attn_layers': 3,
            'flash_attn_heads': 12,
            'use_se': True,
            'deploy': False
        }
        
        # Skip connections: [c3, c2, c1]
        # c3 = 48*2 = 96, c2 = 48, c1 = 48
        head_cfg = {
            'in_channels': base_channels * 2,  # c5 = 96
            'channels': 192,
            'decode_enabled': True,
            'decoder_channels': 192,
            'skip_channels': [base_channels * 2, base_channels, base_channels],  # [96, 48, 48]
            'use_gated_fusion': True,
            'dropout_ratio': 0.1,
            'align_corners': False
        }
        
        aux_head_cfg = {
            'in_channels': base_channels * 4,  # c4 = 192
            'channels': 96,
            'dropout_ratio': 0.1,
            'align_corners': False
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return backbone_cfg, head_cfg, aux_head_cfg


# ============================================
# MAIN TRAINING FUNCTIONS
# ============================================

def train_standard(args):
    """Train without distillation"""
    print("\n" + "="*60)
    print("STANDARD TRAINING MODE")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model_size}_standard",
            config=vars(args)
        )
    
    # Get model config
    backbone_cfg, head_cfg, aux_head_cfg = get_model_config(args.model_size)
    
    # Create model
    from model.backbone.model import GCNetImproved
    from model.head.segmentation_head import GCNetHead, GCNetAuxHead
    
    backbone = GCNetImproved(**backbone_cfg)
    head = GCNetHead(
        in_channels=head_cfg['in_channels'],
        channels=head_cfg['channels'],
        num_classes=args.num_classes,
        **{k: v for k, v in head_cfg.items() if k not in ['in_channels', 'channels']}
    )
    aux_head = GCNetAuxHead(
        in_channels=aux_head_cfg['in_channels'],
        channels=aux_head_cfg['channels'],
        num_classes=args.num_classes,
        **{k: v for k, v in aux_head_cfg.items() if k not in ['in_channels', 'channels']}
    ) if args.use_aux_head else None
    
    # Wrap in simple container
    class SimpleSegmentor(nn.Module):
        def __init__(self, backbone, head, aux_head=None):
            super().__init__()
            self.backbone = backbone
            self.decode_head = head
            self.auxiliary_head = aux_head
        
        def forward(self, x):
            feats = self.backbone(x)
            return self.decode_head(feats)
    
    model = SimpleSegmentor(backbone, head, aux_head)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        compute_class_weights=args.use_class_weights
    )
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_lr
    )
    
    # Trainer
    trainer = DistillationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=args.num_classes,
        use_distillation=False,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        class_weights=class_weights.to(device) if class_weights is not None else None
    )
    
    # Resume if needed
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.num_epochs)
    
    return trainer.best_miou


def train_with_distillation(args):
    """Train with distillation"""
    print("\n" + "="*60)
    print("DISTILLATION TRAINING MODE")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"distill_{args.student_size}_from_{args.teacher_size}",
            config=vars(args)
        )
    
    # Student config
    student_backbone_cfg, student_head_cfg, student_aux_cfg = get_model_config(args.student_size)
    
    student_config = {
        'backbone': student_backbone_cfg,
        'head': student_head_cfg,
        'aux_head': student_aux_cfg if args.use_aux_head else None
    }
    
    # Teacher config
    teacher_backbone_cfg, teacher_head_cfg, _ = get_model_config(args.teacher_size)
    
    teacher_config = {
        'backbone': teacher_backbone_cfg,
        'head': teacher_head_cfg,
        'pretrained': args.teacher_checkpoint
    }
    
    # Distillation config
    distillation_config = {
        'temperature': args.temperature,
        'alpha': args.alpha,
        'logit_weight': args.logit_weight,
        'feature_weight': args.feature_weight,
        'attention_weight': args.attention_weight,
        'feature_stages': ['c3', 'c4', 'c5'],
        'student_channels': {
            'c3': student_backbone_cfg['channels'] * 2,
            'c4': student_backbone_cfg['channels'] * 4,
            'c5': student_backbone_cfg['channels'] * 2
        },
        'teacher_channels': {
            'c3': teacher_backbone_cfg['channels'] * 2,
            'c4': teacher_backbone_cfg['channels'] * 4,
            'c5': teacher_backbone_cfg['channels'] * 2
        }
    }
    
    # Create model with distillation
    model = create_distillation_model(
        student_config=student_config,
        teacher_config=teacher_config,
        distillation_config=distillation_config,
        num_classes=args.num_classes
    )
    
    print(f"Student parameters: {sum(p.numel() for p in model.backbone.parameters()) / 1e6:.2f}M")
    print(f"Teacher parameters: {sum(p.numel() for p in model.teacher_backbone.parameters()) / 1e6:.2f}M")
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        compute_class_weights=args.use_class_weights
    )
    
    # Optimizer & scheduler (only for student)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.min_lr
    )
    
    # Trainer
    trainer = DistillationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=args.num_classes,
        use_distillation=True,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        class_weights=class_weights.to(device) if class_weights is not None else None
    )
    
    # Resume if needed
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.num_epochs)
    
    return trainer.best_miou


# ============================================
# MAIN SCRIPT
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train GCNet with optional distillation')
    
    # Data
    parser.add_argument('--train_txt', type=str, required=True)
    parser.add_argument('--val_txt', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 1024])
    
    # Training mode
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'distillation'],
                        help='Training mode')
    
    # Model configs
    parser.add_argument('--model_size', type=str, default='standard',
                        choices=['small', 'standard', 'large'],
                        help='Model size for standard training')
    parser.add_argument('--student_size', type=str, default='small',
                        choices=['small', 'standard'],
                        help='Student model size for distillation')
    parser.add_argument('--teacher_size', type=str, default='large',
                        choices=['standard', 'large'],
                        help='Teacher model size for distillation')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to pretrained teacher checkpoint')
    parser.add_argument('--use_aux_head', action='store_true',
                        help='Use auxiliary head')
    
    # Distillation hyperparameters
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Balance between CE and KD loss')
    parser.add_argument('--logit_weight', type=float, default=1.0,
                        help='Logit distillation weight')
    parser.add_argument('--feature_weight', type=float, default=0.5,
                        help='Feature distillation weight')
    parser.add_argument('--attention_weight', type=float, default=0.3,
                        help='Attention distillation weight')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume')
    parser.add_argument('--log_interval', type=int, default=10)
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='gcnet-improved',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'distillation' and args.teacher_checkpoint is None:
        raise ValueError("--teacher_checkpoint is required for distillation mode")
    
    # Run training
    if args.mode == 'standard':
        best_miou = train_standard(args)
    else:
        best_miou = train_with_distillation(args)
    
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
