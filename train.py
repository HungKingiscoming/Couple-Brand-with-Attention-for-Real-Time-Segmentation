# ============================================
# COMPLETE DISTILLATION WORKFLOW
# Train Teacher → Distillation → Deploy
# ============================================

import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import your modules
from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders

# ============================================
# CONFIGURATION SYSTEM
# ============================================

class ModelConfig:
    """Centralized model configuration"""
    
    @staticmethod
    def get_teacher_config():
        """Large teacher model (channels=48)"""
        return {
            'backbone': {
                'in_channels': 3,
                'channels': 48,  # Large
                'ppm_channels': 192,
                'num_blocks_per_stage': [5, 5, [6, 5], [6, 5], [3, 3]],
                'use_flash_attention': False,  # Auto-detect
                'flash_attn_stage': 4,
                'flash_attn_layers': 3,
                'flash_attn_heads': 12,
                'use_se': True,
                'deploy': False
            },
            'head': {
                'in_channels': 96,  # channels * 2
                'channels': 192,
                'decode_enabled': False,  # Simple fusion for stability
                'dropout_ratio': 0.1,
                'align_corners': False
            },
            'aux_head': {
                'in_channels': 192,  # channels * 4
                'channels': 96,
                'dropout_ratio': 0.1,
                'align_corners': False
            }
        }
    
    @staticmethod
    def get_student_config():
        """Standard student model (channels=32)"""
        return {
            'backbone': {
                'in_channels': 3,
                'channels': 32,  # Standard
                'ppm_channels': 128,
                'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
                'use_flash_attention': False,
                'flash_attn_stage': 4,
                'flash_attn_layers': 2,
                'flash_attn_heads': 8,
                'use_se': True,
                'deploy': False
            },
            'head': {
                'in_channels': 64,  # channels * 2
                'channels': 128,
                'decode_enabled': False,  # Simple fusion
                'dropout_ratio': 0.1,
                'align_corners': False
            },
            'aux_head': {
                'in_channels': 128,  # channels * 4
                'channels': 64,
                'dropout_ratio': 0.1,
                'align_corners': False
            }
        }
    
    @staticmethod
    def get_distillation_config():
        """Distillation hyperparameters"""
        return {
            'temperature': 4.0,
            'alpha': 0.5,  # 50% CE + 50% KD
            'logit_weight': 1.0,
            'feature_weight': 0.5,
            'attention_weight': 0.3,
            'feature_stages': ['c3', 'c4', 'c5'],
            'student_channels': {
                'c3': 64,   # 32 * 2
                'c4': 128,  # 32 * 4
                'c5': 64    # 32 * 2
            },
            'teacher_channels': {
                'c3': 96,   # 48 * 2
                'c4': 192,  # 48 * 4
                'c5': 96    # 48 * 2
            }
        }


# ============================================
# SIMPLE SEGMENTOR
# ============================================

class SimpleSegmentor(nn.Module):
    """Complete segmentation model"""
    
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.auxiliary_head = aux_head
    
    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)
    
    def forward_train(self, x):
        feats = self.backbone(x)
        outputs = {'main': self.decode_head(feats)}
        if self.auxiliary_head is not None:
            outputs['aux'] = self.auxiliary_head(feats)
        return outputs


# ============================================
# METRICS
# ============================================

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_miou(self):
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / (union + 1e-10)
        return np.nanmean(iou)
    
    def compute_pixel_acc(self):
        acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        return acc


# ============================================
# DISTILLATION LOSSES
# ============================================

class KLDivLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits):
        T = self.temperature
        student_log_softmax = torch.log_softmax(student_logits / T, dim=1)
        teacher_softmax = torch.softmax(teacher_logits / T, dim=1)
        loss = torch.nn.functional.kl_div(
            student_log_softmax,
            teacher_softmax,
            reduction='batchmean'
        ) * (T ** 2)
        return loss


class FeatureDistillationLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.align = nn.Identity()
    
    def forward(self, student_feat, teacher_feat):
        student_feat = self.align(student_feat)
        
        # Resize if needed
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = torch.nn.functional.interpolate(
                student_feat, size=teacher_feat.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        # Normalize and compute loss
        student_feat = torch.nn.functional.normalize(student_feat, dim=1)
        teacher_feat = torch.nn.functional.normalize(teacher_feat, dim=1)
        return torch.nn.functional.mse_loss(student_feat, teacher_feat)


# ============================================
# STAGE 1: TEACHER TRAINER
# ============================================

class TeacherTrainer:
    """Train large teacher model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device='cuda',
        num_classes=19,
        save_dir='./checkpoints/teacher',
        class_weights=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255
        )
        
        # AMP
        self.scaler = GradScaler('cuda')
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes)
        self.best_miou = 0.0
        self.current_epoch = 0
    
    def train_epoch(self):
        self.model.train()
        self.metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Teacher Epoch {self.current_epoch}')
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze(1).long()
            
            self.optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = self.model.forward_train(images)
                main_logits = outputs['main']
                
                # Resize
                if main_logits.shape[-2:] != targets.shape[-2:]:
                    main_logits = torch.nn.functional.interpolate(
                        main_logits, size=targets.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                loss = self.criterion(main_logits, targets)
                
                # Aux loss
                if 'aux' in outputs:
                    aux_logits = outputs['aux']
                    if aux_logits.shape[-2:] != targets.shape[-2:]:
                        aux_logits = torch.nn.functional.interpolate(
                            aux_logits, size=targets.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    loss += 0.4 * self.criterion(aux_logits, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            with torch.no_grad():
                pred = main_logits.argmax(dim=1)
                self.metrics.update(pred, targets)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_mIoU': self.metrics.compute_miou(),
            'train_pixel_acc': self.metrics.compute_pixel_acc()
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        
        for images, targets in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze(1).long()
            
            with autocast('cuda'):
                logits = self.model(images)
                
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=targets.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                loss = self.criterion(logits, targets)
            
            pred = logits.argmax(dim=1)
            self.metrics.update(pred, targets)
            total_loss += loss.item()
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_mIoU': self.metrics.compute_miou(),
            'val_pixel_acc': self.metrics.compute_pixel_acc()
        }
    
    def train(self, num_epochs):
        print("\n" + "="*60)
        print("STAGE 1: TRAINING TEACHER MODEL (channels=48)")
        print("="*60 + "\n")
        
        start_epoch = self.current_epoch  # Resume từ đây
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train: Loss={train_metrics['train_loss']:.4f}, mIoU={train_metrics['train_mIoU']:.4f}")
            print(f"Val:   Loss={val_metrics['val_loss']:.4f}, mIoU={val_metrics['val_mIoU']:.4f}")
            
            # Save best
            if val_metrics['val_mIoU'] > self.best_miou:
                self.best_miou = val_metrics['val_mIoU']
                self.save_checkpoint('best_teacher.pth', epoch, val_metrics)
                print(f"✓ Saved best teacher (mIoU: {self.best_miou:.4f})")
            
            # Periodic save
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'teacher_epoch_{epoch}.pth', epoch, val_metrics)
        
        print(f"\n✓ Teacher training completed! Best mIoU: {self.best_miou:.4f}")
        return self.best_miou
    
    def load_checkpoint(self, checkpoint_path):
        """Resume từ checkpoint"""
        print(f"Loading teacher checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        print(f"✓ Resumed at epoch {self.current_epoch}, best mIoU = {self.best_miou:.4f}")
    
    def save_checkpoint(self, filename, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'best_miou': self.best_miou,
            'config': 'teacher_48'
        }
        torch.save(checkpoint, self.save_dir / filename)


# ============================================
# STAGE 2: DISTILLATION TRAINER
# ============================================

class DistillationTrainer:
    """Train student with teacher distillation"""
    
    def __init__(
        self,
        student_model,
        teacher_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        distill_cfg,
        device='cuda',
        num_classes=19,
        save_dir='./checkpoints/student'
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Distillation config
        self.temperature = distill_cfg['temperature']
        self.alpha = distill_cfg['alpha']
        self.logit_weight = distill_cfg['logit_weight']
        self.feature_weight = distill_cfg['feature_weight']
        self.feature_stages = distill_cfg['feature_stages']
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.kd_loss = KLDivLoss(temperature=self.temperature)
        
        # Feature distillation
        self.feature_distill = nn.ModuleDict()
        for stage in self.feature_stages:
            self.feature_distill[stage] = FeatureDistillationLoss(
                student_channels=distill_cfg['student_channels'][stage],
                teacher_channels=distill_cfg['teacher_channels'][stage]
            ).to(device)
        
        # AMP
        self.scaler = GradScaler('cuda')
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes)
        self.best_miou = 0.0
        self.current_epoch = 0
    
    def train_epoch(self):
        self.student.train()
        self.metrics.reset()
        
        total_loss = 0.0
        loss_components = {'ce': 0.0, 'kd': 0.0, 'feat': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f'Distill Epoch {self.current_epoch}')
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze(1).long()
            
            self.optimizer.zero_grad()
            
            with autocast('cuda'):
                # Student forward
                student_feats = self.student.backbone(images)
                student_logits = self.student.decode_head(student_feats)
                
                # Teacher forward
                with torch.no_grad():
                    teacher_feats = self.teacher.backbone(images)
                    teacher_logits = self.teacher.decode_head(teacher_feats)
                
                # Resize
                if student_logits.shape[-2:] != targets.shape[-2:]:
                    student_logits_resized = torch.nn.functional.interpolate(
                        student_logits, size=targets.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                else:
                    student_logits_resized = student_logits
                
                if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
                    teacher_logits = torch.nn.functional.interpolate(
                        teacher_logits, size=student_logits.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                # 1. CE Loss
                loss_ce = self.ce_loss(student_logits_resized, targets)
                
                # 2. KD Loss
                loss_kd = self.kd_loss(student_logits, teacher_logits) * self.logit_weight
                
                # 3. Feature Distillation
                loss_feat = 0.0
                for stage in self.feature_stages:
                    if stage in student_feats and stage in teacher_feats:
                        loss_feat += self.feature_distill[stage](
                            student_feats[stage],
                            teacher_feats[stage]
                        )
                loss_feat = (loss_feat / len(self.feature_stages)) * self.feature_weight
                
                # Total loss
                loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd + loss_feat
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            with torch.no_grad():
                pred = student_logits_resized.argmax(dim=1)
                self.metrics.update(pred, targets)
            
            total_loss += loss.item()
            loss_components['ce'] += loss_ce.item()
            loss_components['kd'] += loss_kd.item()
            loss_components['feat'] += loss_feat.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        n = len(self.train_loader)
        return {
            'train_loss': total_loss / n,
            'train_loss_ce': loss_components['ce'] / n,
            'train_loss_kd': loss_components['kd'] / n,
            'train_loss_feat': loss_components['feat'] / n,
            'train_mIoU': self.metrics.compute_miou(),
            'train_pixel_acc': self.metrics.compute_pixel_acc()
        }
    
    @torch.no_grad()
    def validate(self):
        self.student.eval()
        self.metrics.reset()
        total_loss = 0.0
        
        for images, targets in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze(1).long()
            
            with autocast('cuda'):
                logits = self.student(images)
                
                if logits.shape[-2:] != targets.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=targets.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                
                loss = self.ce_loss(logits, targets)
            
            pred = logits.argmax(dim=1)
            self.metrics.update(pred, targets)
            total_loss += loss.item()
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_mIoU': self.metrics.compute_miou(),
            'val_pixel_acc': self.metrics.compute_pixel_acc()
        }
    
    def train(self, num_epochs):
        print("\n" + "="*60)
        print("STAGE 2: DISTILLATION (Teacher 48 → Student 32)")
        print("="*60 + "\n")
        
        start_epoch = self.current_epoch  # Resume từ đây
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train: Loss={train_metrics['train_loss']:.4f} "
                  f"(CE={train_metrics['train_loss_ce']:.4f}, "
                  f"KD={train_metrics['train_loss_kd']:.4f}, "
                  f"Feat={train_metrics['train_loss_feat']:.4f})")
            print(f"       mIoU={train_metrics['train_mIoU']:.4f}")
            print(f"Val:   Loss={val_metrics['val_loss']:.4f}, mIoU={val_metrics['val_mIoU']:.4f}")
            
            # Save best
            if val_metrics['val_mIoU'] > self.best_miou:
                self.best_miou = val_metrics['val_mIoU']
                self.save_checkpoint('best_student.pth', epoch, val_metrics)
                print(f"✓ Saved best student (mIoU: {self.best_miou:.4f})")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'student_epoch_{epoch}.pth', epoch, val_metrics)
        
        print(f"\n✓ Distillation completed! Best mIoU: {self.best_miou:.4f}")
        return self.best_miou
    
    def load_checkpoint(self, checkpoint_path):
        """Resume từ checkpoint"""
        print(f"Loading student checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        print(f"✓ Resumed at epoch {self.current_epoch}, best mIoU = {self.best_miou:.4f}")
    
    def save_checkpoint(self, filename, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'best_miou': self.best_miou,
            'config': 'student_32_distilled'
        }
        torch.save(checkpoint, self.save_dir / filename)


# ============================================
# MAIN WORKFLOW
# ============================================

def create_model_from_config(config, num_classes):
    """Create model from config dict"""
    backbone = GCNetImproved(**config['backbone'])
    head = GCNetHead(
        in_channels=config['head']['in_channels'],
        channels=config['head']['channels'],
        num_classes=num_classes,
        **{k: v for k, v in config['head'].items() 
           if k not in ['in_channels', 'channels']}
    )
    aux_head = GCNetAuxHead(
        in_channels=config['aux_head']['in_channels'],
        channels=config['aux_head']['channels'],
        num_classes=num_classes,
        **{k: v for k, v in config['aux_head'].items() 
           if k not in ['in_channels', 'channels']}
    )
    return SimpleSegmentor(backbone, head, aux_head)


def main():
    parser = argparse.ArgumentParser(description='Distillation Workflow')
    parser.add_argument('--stage', type=str, required=True,
                        choices=['teacher', 'distill', 'both'],
                        help='Training stage')
    parser.add_argument('--train_txt', type=str, required=True)
    parser.add_argument('--val_txt', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher checkpoint (for distill stage or resume)')
    parser.add_argument('--student_checkpoint', type=str, default=None,
                        help='Path to student checkpoint (for resume)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # ✅ NEW: Resume options
    parser.add_argument('--resume_teacher', type=str, default=None,
                        help='Resume teacher training from checkpoint')
    parser.add_argument('--resume_student', type=str, default=None,
                        help='Resume student training from checkpoint')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(512, 1024),
        compute_class_weights=True
    )
    
    # ========== STAGE 1: TRAIN TEACHER ==========
    if args.stage in ['teacher', 'both']:
        print("\n" + "="*80)
        print("STAGE 1: TRAINING TEACHER (Large Model, channels=48)")
        print("="*80)
        
        teacher_config = ModelConfig.get_teacher_config()
        teacher_model = create_model_from_config(teacher_config, args.num_classes)
        
        print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")
        
        optimizer = optim.AdamW(teacher_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
        teacher_trainer = TeacherTrainer(
            model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=args.num_classes,
            save_dir=f'{args.save_dir}/teacher',
            class_weights=class_weights.to(device) if class_weights is not None else None
        )
        
        # ✅ RESUME TEACHER if checkpoint provided
        if args.resume_teacher:
            teacher_trainer.load_checkpoint(args.resume_teacher)
            print(f"✓ Resumed teacher from epoch {teacher_trainer.current_epoch}")
        
        teacher_miou = teacher_trainer.train(num_epochs=args.num_epochs)
        print(f"\n✓ Teacher training completed! Best mIoU: {teacher_miou:.4f}")
        
        # Auto-set teacher checkpoint for distillation
        args.teacher_checkpoint = f'{args.save_dir}/teacher/best_teacher.pth'
    
    # ========== STAGE 2: DISTILLATION ==========
    if args.stage in ['distill', 'both']:
        # ✅ AUTO-DETECT teacher checkpoint for "both" stage
        if args.stage == 'both' and args.teacher_checkpoint is None:
            args.teacher_checkpoint = f'{args.save_dir}/teacher/best_teacher.pth'
        
        if args.teacher_checkpoint is None:
            raise ValueError("--teacher_checkpoint is required for distillation stage")
        
        if not os.path.exists(args.teacher_checkpoint):
            raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
        
        print("\n" + "="*80)
        print("STAGE 2: DISTILLATION (Teacher 48 → Student 32)")
        print("="*80)
        
        # Load teacher
        teacher_config = ModelConfig.get_teacher_config()
        teacher_model = create_model_from_config(teacher_config, args.num_classes)
        
        print(f"Loading teacher from {args.teacher_checkpoint}")
        checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        teacher_metrics = checkpoint.get('metrics', {})
        teacher_miou = teacher_metrics.get('val_mIoU', checkpoint.get('best_miou', 0.0))
        print(f"✓ Teacher loaded (mIoU: {teacher_miou:.4f})")
        
        # Create student
        student_config = ModelConfig.get_student_config()
        student_model = create_model_from_config(student_config, args.num_classes)
        
        print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M")
        print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")
        print(f"Compression ratio: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.2f}x")
        
        # Optimizer
        optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
        # Distillation config
        distill_cfg = ModelConfig.get_distillation_config()
        
        distill_trainer = DistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            distill_cfg=distill_cfg,
            device=device,
            num_classes=args.num_classes,
            save_dir=f'{args.save_dir}/student'
        )
        
        # ✅ RESUME STUDENT if checkpoint provided
        if args.resume_student:
            distill_trainer.load_checkpoint(args.resume_student)
            print(f"✓ Resumed student from epoch {distill_trainer.current_epoch}")
        
        student_miou = distill_trainer.train(num_epochs=args.num_epochs)
        print(f"\n✓ Distillation completed! Best mIoU: {student_miou:.4f}")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED!")
    print("="*80)
    print("\nModel checkpoints saved:")
    if args.stage in ['teacher', 'both']:
        print(f"  Teacher: {args.save_dir}/teacher/best_teacher.pth")
    if args.stage in ['distill', 'both']:
        print(f"  Student: {args.save_dir}/student/best_student.pth")
    print("\nDeploy student model for production! 🚀")


if __name__ == '__main__':
    main()


# ============================================
# USAGE EXAMPLES
# ============================================

"""
# ============================================
# RESUME EXAMPLES
# ============================================

# 1. RESUME TEACHER TRAINING (bị gián đoạn ở epoch 45)
python train_distillation.py \
    --stage teacher \
    --train_txt data/train.txt \
    --val_txt data/val.txt \
    --num_epochs 100 \
    --resume_teacher checkpoints/teacher/teacher_epoch_44.pth \
    --save_dir checkpoints

# 2. RESUME STUDENT TRAINING (bị gián đoạn ở epoch 60)
python train_distillation.py \
    --stage distill \
    --teacher_checkpoint checkpoints/teacher/best_teacher.pth \
    --train_txt data/train.txt \
    --val_txt data/val.txt \
    --num_epochs 100 \
    --resume_student checkpoints/student/student_epoch_59.pth \
    --save_dir checkpoints

# 3. RESUME BOTH (Teacher xong rồi, resume Student)
python train_distillation.py \
    --stage both \
    --train_txt data/train.txt \
    --val_txt data/val.txt \
    --num_epochs 100 \
    --resume_teacher checkpoints/teacher/best_teacher.pth \
    --resume_student checkpoints/student/student_epoch_40.pth \
    --save_dir checkpoints

# 4. FULL WORKFLOW WITHOUT RESUME
python train_distillation.py \
    --stage both \
    --train_txt data/train.txt \
    --val_txt data/val.txt \
    --num_epochs 100 \
    --batch_size 8 \
    --save_dir checkpoints

# 5. ONLY DISTILLATION (Teacher đã train xong)
python train_distillation.py \
    --stage distill \
    --teacher_checkpoint checkpoints/teacher/best_teacher.pth \
    --train_txt data/train.txt \
    --val_txt data/val.txt \
    --num_epochs 100 \
    --save_dir checkpoints
"""
