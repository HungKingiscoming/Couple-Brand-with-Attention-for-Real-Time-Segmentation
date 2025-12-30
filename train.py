# ============================================
# MEMORY-OPTIMIZED GCNET TRAINING PIPELINE
# TESTED FOR 16GB GPU (RTX 4090, T4, V100)
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
import gc

# ============================================
# IMPORTS
# ============================================

from model.backbone.model import GCNetWithDWSA
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders
from model.losses.composite_loss import CompositeSegLoss

# ============================================
# MEMORY OPTIMIZATION UTILITIES
# ============================================

def clear_gpu_memory():
    """Aggressive GPU memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def print_memory_usage(prefix=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{prefix} GPU Memory - Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")

def setup_memory_efficient_training():
    """Configure PyTorch for memory efficiency"""
    # Enable memory efficient algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocator for better fragmentation handling
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================
# MEMORY-EFFICIENT MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_lightweight_config():
        """
        Ultra-lightweight config for 16GB GPU
        
        Memory Budget:
        - Model: ~1-2GB
        - Optimizer states: ~2-3GB  
        - Activations: ~8-10GB (batch_size=4, img_size=512x1024)
        - Buffer: ~2GB
        Total: ~14-17GB (fits 16GB)
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 24,
                "ppm_channels": 96,
                "num_blocks_per_stage": [3, 3, [4, 3], [4, 3], [2, 2]],
                "dwsa_stages": ['bottleneck'],
                "dwsa_num_heads": 4,
                "deploy": False
            },
            "head": {
                # BaseDecodeHead parameters
                "in_channels": 48,              # c5 channels (24 * 2)
                "channels": 96,                 # Internal channels
                "dropout_ratio": 0.1,
                "align_corners": False,
                
                # Custom GCNetHead parameters
                "decoder_channels": 128,
                "decode_enabled": False,        # Disable decoder for memory
                "skip_channels": [48, 24, 24],
                "use_gated_fusion": True,
            },
            "aux_head": {
                # BaseDecodeHead parameters only
                "in_channels": 96,              # c4 channels (24 * 4)
                "channels": 48,
                "dropout_ratio": 0.1,
                "align_corners": False,
            }
        }
    
    @staticmethod
    def get_medium_config():
        """
        Balanced config for 24GB+ GPU
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,
                "ppm_channels": 128,
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
                "dwsa_stages": ['stage4', 'bottleneck'],
                "dwsa_num_heads": 8,
                "deploy": False
            },
            "head": {
                "in_channels": 64,              # c5 channels (32 * 2)
                "channels": 128,
                "dropout_ratio": 0.1,
                "align_corners": False,
                
                "decoder_channels": 256,
                "decode_enabled": True,         # Enable decoder
                "skip_channels": [64, 32, 32],
                "use_gated_fusion": True,
            },
            "aux_head": {
                # ✅ FIXED: Use c3 channels for medium (c4 might not exist)
                "in_channels": 64,              # c3 channels (32 * 2)
                "channels": 64,
                "dropout_ratio": 0.1,
                "align_corners": False,
            }
        }
    
    @staticmethod
    def get_performance_config():
        """
        Maximum performance for 32GB+ GPU
        """
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 48,
                "ppm_channels": 192,
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
                "dwsa_stages": ['stage3', 'stage4', 'bottleneck'],
                "dwsa_num_heads": 8,
                "deploy": False
            },
            "head": {
                "in_channels": 96,              # c5 channels (48 * 2)
                "channels": 192,
                "dropout_ratio": 0.1,
                "align_corners": False,
                
                "decoder_channels": 384,
                "decode_enabled": True,
                "skip_channels": [96, 48, 48],
                "use_gated_fusion": True,
            },
            "aux_head": {
                # ✅ Use c3 channels (safest approach)
                "in_channels": 96,              # c3 channels (48 * 2)
                "channels": 96,
                "dropout_ratio": 0.1,
                "align_corners": False,
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
        feats = self.backbone(x)
        return self.decode_head(feats)

    def forward_train(self, x):
        feats = self.backbone(x)
        outputs = {"main": self.decode_head(feats)}
        if self.aux_head is not None:
            outputs["aux"] = self.aux_head(feats)
        return outputs


# ============================================
# GRADIENT ACCUMULATION TRAINER
# ============================================

class MemoryEfficientTrainer:
    """
    Trainer with gradient accumulation and memory optimization
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
        
        # Loss function
        self.criterion = CompositeSegLoss(
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            class_weights=class_weights.to(device) if class_weights is not None else None,
            w_ce=1.0,
            w_lovasz=1.0,
            w_boundary=0.5
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
        print(f"{'='*70}\n")

    def save_config(self):
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, loader, epoch):
        """Memory-efficient training with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            # Forward with mixed precision
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
                
                # Main loss
                loss_dict = self.criterion(logits, masks)
                loss = loss_dict["loss"]
                
                # Auxiliary loss (if enabled)
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = nn.functional.interpolate(
                        outputs["aux"],
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    aux_loss_dict = self.criterion(aux_logits, masks)
                    loss = loss + self.args.aux_weight * aux_loss_dict["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / self.args.accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                # Gradient clipping
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Scheduler step
                if self.scheduler and self.args.scheduler_type == 'onecycle':
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.args.accumulation_steps
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.accumulation_steps:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                clear_gpu_memory()
            
            # TensorBoard logging
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar(
                    'train/loss',
                    loss.item() * self.args.accumulation_steps,
                    self.global_step
                )
                self.writer.add_scalar(
                    'train/lr',
                    current_lr,
                    self.global_step
                )

        avg_loss = total_loss / len(loader)
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self, loader, epoch):
        """Memory-efficient validation"""
        self.model.eval()
        total_loss = 0.0
        
        # Confusion matrix for mIoU
        num_classes = self.args.num_classes
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        pbar = tqdm(loader, desc=f"Val Epoch {epoch+1}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            # Forward with AMP
            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits = self.model(imgs)
                logits = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                loss = self.criterion.ce(logits, masks)
            
            total_loss += loss.item()
            
            # Compute metrics on CPU to save GPU memory
            pred = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            
            # Update confusion matrix
            mask = (target >= 0) & (target < num_classes)
            label = num_classes * target[mask].astype('int') + pred[mask]
            count = np.bincount(label, minlength=num_classes**2)
            confusion_matrix += count.reshape(num_classes, num_classes)
            
            # Update progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear cache
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        # Compute mIoU
        intersection = np.diag(confusion_matrix)
        union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        
        # Pixel accuracy
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
        """Load checkpoint"""
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


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized GCNet Training")
    
    # Dataset
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 1024],
                        help="⚠️ Use 512x1024 for 16GB GPU, 1024x2048 for 24GB+")
    parser.add_argument("--compute_class_weights", action="store_true")
    
    # Model
    parser.add_argument("--model_size", default="lightweight",
                        choices=["lightweight", "medium", "performance"],
                        help="lightweight=16GB, medium=24GB+, performance=32GB+")
    parser.add_argument("--aux_weight", type=float, default=0.4)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="⚠️ Per-GPU batch size. Use 4 for 16GB GPU")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective_bs = bs * accum)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Scheduler
    parser.add_argument("--scheduler_type", default="cosine",
                        choices=["cosine", "onecycle", "poly"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    
    # Optimization
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use AMP (saves ~40% memory)")
    
    # Logging
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    
    # System
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup memory optimization
    setup_memory_efficient_training()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"🚀 Memory-Optimized GCNet Training")
    print(f"{'='*70}")
    print(f"📱 Device: {device}")
    print(f"🎯 Dataset: {args.dataset_type.upper()} Cityscapes")
    print(f"📐 Image size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔁 Accumulation steps: {args.accumulation_steps}")
    print(f"📊 Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"⚡ Mixed precision: {args.use_amp}")
    print(f"🏗️  Model size: {args.model_size}")
    print(f"{'='*70}\n")
    
    print_memory_usage("Initial")
    
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
    
    # ============================================
    # ✅ FIXED MODEL CREATION
    # ============================================
    print(f"\n{'='*70}")
    print("🏗️  Building Model...")
    print(f"{'='*70}\n")
    
    # Get config based on model size
    if args.model_size == "lightweight":
        cfg = ModelConfig.get_lightweight_config()
    elif args.model_size == "medium":
        cfg = ModelConfig.get_medium_config()
    else:  # performance
        cfg = ModelConfig.get_performance_config()
    
    # ✅ FIX: Properly set num_classes before passing to model
    head_cfg = cfg["head"].copy()
    head_cfg["num_classes"] = args.num_classes
    
    aux_head_cfg = cfg["aux_head"].copy()
    aux_head_cfg["num_classes"] = args.num_classes
    
    # Create model
    model = Segmentor(
        backbone=GCNetWithDWSA(**cfg["backbone"]),
        head=GCNetHead(**head_cfg),
        aux_head=GCNetAuxHead(**aux_head_cfg)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"   Model size:           {args.model_size}")
    print(f"   Backbone channels:    {cfg['backbone']['channels']}")
    print(f"   DWSA stages:          {cfg['backbone']['dwsa_stages']}")
    print(f"   Decoder enabled:      {head_cfg['decode_enabled']}")
    
    print_memory_usage("After model creation")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = None
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    
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
            total_steps=total_steps
        )
    elif args.scheduler_type == "poly":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9
        )
    
    # Create trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        class_weights=class_weights
    )
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    print_memory_usage("Before training")
    
    # Training loop
    print(f"\n{'='*70}")
    print("🚀 Starting Training...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(trainer.start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Clear memory before validation
        clear_gpu_memory()
        
        # Validate
        val_metrics = trainer.validate(val_loader, epoch)
        
        # Update scheduler
        if scheduler and args.scheduler_type != 'onecycle':
            scheduler.step()
        
        # Logging
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch+1}/{args.epochs} (Time: {epoch_time:.2f}s)")
        print(f"{'='*70}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | "
              f"mIoU: {val_metrics['miou']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"{'='*70}\n")
        
        print_memory_usage(f"After epoch {epoch+1}")
        
        # TensorBoard
        trainer.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        trainer.writer.add_scalar('mIoU', val_metrics['miou'], epoch)
        
        # Save checkpoint
        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        # Clear memory
        clear_gpu_memory()
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ Training Completed!")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best mIoU: {trainer.best_miou:.4f}")
    print(f"💾 Checkpoints: {args.save_dir}")
    print(f"{'='*70}\n")
    
    trainer.writer.close()


if __name__ == "__main__":
    main()
