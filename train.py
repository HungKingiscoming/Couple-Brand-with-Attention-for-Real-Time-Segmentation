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
from typing import Dict, Optional

# Import your modules
from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import CityscapesCustomDataset, create_dataloaders
from model.distillation import DistillationWrapper

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
        return {
            'mIoU': self.compute_miou(),
            'pixel_acc': self.compute_pixel_acc()
        }


# ============================================
# MODEL DEFINITION
# ============================================

class GCNetSegmentor(nn.Module):
    """Complete segmentation model with backbone + head"""
    
    def __init__(
        self,
        num_classes: int,
        backbone_cfg: Dict,
        head_cfg: Dict,
        aux_head_cfg: Optional[Dict] = None
    ):
        super().__init__()
        
        # Backbone
        self.backbone = GCNetImproved(**backbone_cfg)
        
        # Main decode head
        self.decode_head = GCNetHead(
            in_channels=head_cfg['in_channels'],
            channels=head_cfg['channels'],
            num_classes=num_classes,
            decode_enabled=head_cfg.get('decode_enabled', True),
            decoder_channels=head_cfg.get('decoder_channels', 128),
            skip_channels=head_cfg.get('skip_channels', [64, 32, 32]),
            use_gated_fusion=head_cfg.get('use_gated_fusion', True),
            dropout_ratio=head_cfg.get('dropout_ratio', 0.1),
            align_corners=head_cfg.get('align_corners', False)
        )
        
        # Auxiliary head (optional, for deep supervision)
        self.auxiliary_head = None
        if aux_head_cfg is not None:
            self.auxiliary_head = GCNetAuxHead(
                in_channels=aux_head_cfg['in_channels'],
                channels=aux_head_cfg['channels'],
                num_classes=num_classes,
                dropout_ratio=aux_head_cfg.get('dropout_ratio', 0.1),
                align_corners=aux_head_cfg.get('align_corners', False)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes, H, W)
        """
        # Backbone
        features = self.backbone(x)  # Dict: {c1, c2, c3, c4, c5}
        
        # Main head
        logits = self.decode_head(features)
        
        return logits
    
    def forward_train(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training forward with auxiliary outputs
        
        Returns:
            Dict with 'main' and 'aux' logits
        """
        features = self.backbone(x)
        
        outputs = {}
        outputs['main'] = self.decode_head(features)
        
        if self.auxiliary_head is not None:
            outputs['aux'] = self.auxiliary_head(features)
        
        return outputs


# ============================================
# LOSS FUNCTIONS
# ============================================

class SegmentationLoss(nn.Module):
    """Combined loss for segmentation with class weighting"""
    
    def __init__(
        self,
        aux_weight: float = 0.4,
        ignore_index: int = 255,
        use_ohem: bool = False,
        ohem_thresh: float = 0.7,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.use_ohem = use_ohem
        self.ohem_thresh = ohem_thresh
        
        # ✅ Use class weights if provided
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        if class_weights is not None:
            print(f"✓ Using class weights for loss computation")
            print(f"  Weights: {class_weights.tolist()}")
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict with 'main' and optionally 'aux'
            target: (B, H, W)
        
        Returns:
            Dict of losses
        """
        losses = {}
        
        # Main loss
        main_logits = outputs['main']
        
        # Resize if needed
        if main_logits.shape[-2:] != target.shape[-2:]:
            main_logits = nn.functional.interpolate(
                main_logits,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        loss_main = self.ce_loss(main_logits, target)
        losses['loss_main'] = loss_main
        
        # Auxiliary loss
        if 'aux' in outputs:
            aux_logits = outputs['aux']
            
            # Resize aux to target size
            if aux_logits.shape[-2:] != target.shape[-2:]:
                aux_logits = nn.functional.interpolate(
                    aux_logits,
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            loss_aux = self.ce_loss(aux_logits, target)
            losses['loss_aux'] = loss_aux * self.aux_weight
        
        # Total loss
        losses['loss_total'] = sum(losses.values())
        
        return losses


# ============================================
# TRAINER
# ============================================

class Trainer:
    """Training engine"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = 'cuda',
        num_classes: int = 19,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        save_dir: str = './checkpoints',
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.last_autosave_time = time.time()
        self.autosave_interval = 15 * 60  # 15 phút
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=num_classes)
        
        # Best model tracking
        self.best_miou = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        loss_dict_sum = {}
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device).long()
            
            # Forward
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model.forward_train(images)
                losses = self.criterion(outputs, targets)
                loss = losses['loss_total']
            
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
                pred = outputs['main'].argmax(dim=1)
                self.metrics.update(pred, targets)
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in losses.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v.item()
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            if time.time() - self.last_autosave_time > self.autosave_interval:
                self.save_checkpoint(
                    filename='autosave.pth',
                    epoch=self.current_epoch,
                    metrics={'iter': batch_idx}
                )
                self.last_autosave_time = time.time()
        
        # Compute epoch metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in loss_dict_sum.items()},
            **{f'train_{k}': v for k, v in self.metrics.get_results().items()}
        }
        
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
            targets = targets.to(self.device).long()
            
            # Forward
            with autocast(enabled=self.use_amp):
                outputs = self.model.forward_train(images)
                losses = self.criterion(outputs, targets)
                loss = losses['loss_total']
            
            # Update metrics
            pred = outputs['main'].argmax(dim=1)
            self.metrics.update(pred, targets)
            
            total_loss += loss.item()
        
        # Compute metrics
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            **{f'val_{k}': v for k, v in self.metrics.get_results().items()}
        }
        
        return val_metrics
    
    def train(self, num_epochs: int):
        start_epoch = self.current_epoch  # 🔥 LẤY EPOCH RESUME
    
        print(f"[Trainer] Start training from epoch {start_epoch}")
    
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
    
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
    
            if self.scheduler is not None:
                self.scheduler.step()
    
            all_metrics = {**train_metrics, **val_metrics}
    
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {all_metrics['train_loss']:.4f} | "
                  f"Train mIoU: {all_metrics['train_mIoU']:.4f}")
            print(f"Val Loss: {all_metrics['val_loss']:.4f} | "
                  f"Val mIoU: {all_metrics['val_mIoU']:.4f}")
    
            if self.use_wandb:
                wandb.log(all_metrics, step=epoch)
    
            if val_metrics['val_mIoU'] > self.best_miou:
                self.best_miou = val_metrics['val_mIoU']
                self.save_checkpoint('best_model.pth', epoch, all_metrics)
                print(f"✓ Saved best model at epoch {epoch}")
    
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth', epoch, all_metrics)
    
        
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_miou': self.best_miou
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
    
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
    
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved to {save_path}")



# ============================================
# MAIN TRAINING SCRIPT
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train GCNet Improved')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--train_txt', type=str, required=True,
                        help='Path to train txt file')
    parser.add_argument('--val_txt', type=str, required=True,
                        help='Path to validation txt file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of workers')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='gcnet-improved',
                        help='W&B project name')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # ============================================
    # MODEL CONFIG
    # ============================================
    
    num_classes = 19  # Cityscapes
    
    backbone_cfg = {
        'in_channels': 3,
        'channels': 32,
        'ppm_channels': 128,
        'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
        'use_flash_attention': True,
        'flash_attn_stage': 4,
        'flash_attn_layers': 2,
        'flash_attn_heads': 8,
        'use_se': True,
        'deploy': False
    }
    
    head_cfg = {
        'in_channels': 64,  # channels * 2 from backbone
        'channels': 128,
        'decode_enabled': True,
        'decoder_channels': 128,
        'skip_channels': [64, 32, 32],  # c3, c2, c1
        'use_gated_fusion': True,
        'dropout_ratio': 0.1,
        'align_corners': False
    }
    
    aux_head_cfg = {
        'in_channels': 128,  # channels * 4 from c4
        'channels': 64,
        'dropout_ratio': 0.1,
        'align_corners': False
    }
    
    # ============================================
    # CREATE MODEL
    # ============================================
    
    model = GCNetSegmentor(
        num_classes=num_classes,
        backbone_cfg=backbone_cfg,
        head_cfg=head_cfg,
        aux_head_cfg=aux_head_cfg
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ============================================
    # CREATE DATALOADERS
    # ============================================
    
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(512, 1024),  # Adjust as needed
        compute_class_weights=True
    )
    
    # ============================================
    # LOSS & OPTIMIZER
    # ============================================
    
    criterion = SegmentationLoss(
        aux_weight=0.4,
        ignore_index=255,
        class_weights=class_weights.to(device)
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # ============================================
    # TRAINER
    # ============================================
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        use_amp=True,
        grad_clip=1.0,
        log_interval=10,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb
    )
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False   # 🔥 QUAN TRỌNG
        )
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        if trainer.scaler is not None and 'scaler_state_dict' in checkpoint:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_miou = checkpoint.get('best_miou', 0.0)
    
        print(
            f"✓ Resumed at epoch {trainer.current_epoch}, "
            f"best mIoU = {trainer.best_miou:.4f}"
    )
    # ============================================
    # TRAIN
    # ============================================
    
    trainer.train(num_epochs=args.num_epochs)
    
    print("\n✓ Training completed!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")


if __name__ == '__main__':
    main()
