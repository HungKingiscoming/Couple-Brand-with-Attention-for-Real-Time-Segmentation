# ============================================
# FIXED train.py - Production Ready
# ============================================
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
import gc
import warnings
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

warnings.filterwarnings('ignore')

# ============================================
# IMPORTS
# ============================================
from model.backbone.model import (
    GCNetWithEnhance,
    DWSABlock,
    MultiScaleContextModule,
)
from model.head.segmentation_head import (
    GCNetHead,
    GCNetAuxHead,
)
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# CHECKPOINT UTILS (FROM ARTIFACT)
# ============================================

def save_checkpoint_with_correct_norm(
    model, optimizer, scheduler, scaler,
    epoch, metrics, save_path,
    global_step=0, best_miou=0.0
):
    """✅ Save checkpoint với verify norm type"""
    
    # Detect norm type
    has_bn = any(isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) 
                for m in model.modules())
    has_gn = any(isinstance(m, nn.GroupNorm) 
                for m in model.modules())
    
    print(f"\n📦 Saving checkpoint...")
    print(f"   Norm type: {'BatchNorm' if has_bn else 'GroupNorm' if has_gn else 'Unknown'}")
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Verify BN running stats if using BN
    if has_bn:
        bn_stats_count = sum(1 for k in state_dict.keys() 
                            if 'running_mean' in k or 'running_var' in k)
        print(f"   BN running stats: {bn_stats_count} tensors")
        
        if bn_stats_count == 0:
            print(f"   ⚠️  WARNING: BatchNorm detected but NO running stats!")
    
    checkpoint = {
        'epoch': epoch,
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict(),
        'best_miou': best_miou,
        'metrics': metrics,
        'global_step': global_step,
        'norm_type': 'BatchNorm' if has_bn else 'GroupNorm' if has_gn else 'Mixed',
        'has_running_stats': bn_stats_count > 0 if has_bn else False,
    }
    
    torch.save(checkpoint, save_path)
    print(f"   ✅ Saved: {save_path}")


# ============================================
# PRETRAINED LOADER
# ============================================

def load_pretrained_gcnet_core(model, ckpt_path, strict_match=False):
    print(f"🔥 Loading pretrained weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    model_state = model.backbone.state_dict()
    compatible = {}
    skipped = []

    model_key_map = {}
    for mk in model_state.keys():
        normalized = mk
        for pref in ['backbone.', 'model.', 'module.']:
            if normalized.startswith(pref):
                normalized = normalized[len(pref):]
        model_key_map[normalized] = mk

    for ckpt_key, ckpt_val in state.items():
        normalized_ckpt = ckpt_key
        for pref in ['backbone.', 'model.', 'module.']:
            if normalized_ckpt.startswith(pref):
                normalized_ckpt = normalized_ckpt[len(pref):]

        matched = False
        if normalized_ckpt in model_key_map:
            mk = model_key_map[normalized_ckpt]
            if model_state[mk].shape == ckpt_val.shape:
                compatible[mk] = ckpt_val
                matched = True

        if not matched and not strict_match:
            for norm_model, mk in model_key_map.items():
                if norm_model.endswith(normalized_ckpt) or normalized_ckpt.endswith(norm_model):
                    if model_state[mk].shape == ckpt_val.shape:
                        compatible[mk] = ckpt_val
                        matched = True
                        break

        if not matched:
            skipped.append(ckpt_key)

    loaded = len(compatible)
    total = len(model_state)
    rate = 100 * loaded / total if total > 0 else 0.0

    print(f"\n{'='*70}")
    print("📊 WEIGHT LOADING SUMMARY")
    print(f"{'='*70}")
    print(f"Loaded:   {loaded:>5} / {total} params ({rate:.1f}%)")
    print(f"Skipped:  {len(skipped):>5} params from checkpoint")
    print(f"{'='*70}")

    model.backbone.load_state_dict(compatible, strict=False)
    return rate


# ============================================
# LOSS FUNCTIONS
# ============================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        valid_mask = (targets != self.ignore_index).float()
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1), num_classes=C
        ).permute(0, 3, 1, 2).float() * valid_mask.unsqueeze(1)
        
        probs = F.softmax(logits, dim=1) * valid_mask.unsqueeze(1)
        probs_flat = probs.reshape(B, C, -1)
        targets_flat = targets_one_hot.reshape(B, C, -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return (1.0 - dice.mean(dim=1)).mean()


# ============================================
# UTILITIES
# ============================================

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_memory_efficient_training():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("🔒 Backbone FROZEN")


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 70)
    print("📊 PARAMETER STATISTICS")
    print("=" * 70)
    print(f"Total:        {total:15,} | 100%")
    print(f"Trainable:    {trainable:15,} | {100*trainable/total:.1f}%")
    print(f"Frozen:       {total - trainable:15,} | {100*(total-trainable)/total:.1f}%")
    print("=" * 70)
    
    return trainable, total - trainable


def setup_discriminative_lr(model, base_lr, backbone_lr_factor=0.1, weight_decay=1e-4):
    backbone_params = [p for n, p in model.named_parameters() 
                      if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() 
                  if 'backbone' not in n and p.requires_grad]
    
    if len(backbone_params) == 0:
        return torch.optim.AdamW(head_params, lr=base_lr, weight_decay=weight_decay)
    
    backbone_lr = base_lr * backbone_lr_factor
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': head_params, 'lr': base_lr, 'name': 'head'}
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_base_config():
        return {
            'backbone': {
                'in_channels': 3,
                'channels': 32,
                'ppm_channels': 128,
                'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
                'dwsa_stages': ['stage5', 'stage6'],
                'dwsa_num_heads': 4,
                'dwsa_reduction': 4,
                'dwsa_qk_sharing': True,
                'dwsa_groups': 4,
                'dwsa_drop': 0.1,
                'dwsa_alpha': 0.1,
                'use_multi_scale_context': True,
                'ms_scales': (1, 2),
                'ms_branch_ratio': 8,
                'ms_alpha': 0.1,
                'align_corners': False,
                'deploy': False
            },
            'head': {
                'decoder_channels': 128,
                'dropout_ratio': 0.1,
                'use_gated_fusion': True,
                'norm_cfg': dict(type='BN', requires_grad=True),
                'act_cfg': dict(type='ReLU', inplace=False),
                'align_corners': False,
            },
            'auxhead': {
                'channels': 96,
                'dropout_ratio': 0.1,
                'norm_cfg': dict(type='BN', requires_grad=True),
                'act_cfg': dict(type='ReLU', inplace=False),
                'align_corners': False,
            },
            'loss': {
                'ce_weight': 1.0,
                'dice_weight': 0.2,
                'dice_smooth': 1e-5
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
# TRAINER
# ============================================

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, args, class_weights=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.best_miou = 0.0
        self.start_epoch = 0
        self.global_step = 0

        loss_cfg = args.loss_config
        self.dice = DiceLoss(smooth=loss_cfg['dice_smooth'], ignore_index=args.ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            ignore_index=args.ignore_index
        )
        
        self.ce_weight = loss_cfg['ce_weight']
        self.dice_weight = loss_cfg['dice_weight']
        self.scaler = GradScaler(enabled=args.use_amp)
        
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)
    
            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)
                logits = F.interpolate(outputs["main"], size=masks.shape[-2:], 
                                      mode="bilinear", align_corners=False)
                
                loss = self.ce_weight * self.ce(logits, masks) + \
                       self.dice_weight * self.dice(logits, masks)
                
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = F.interpolate(outputs["aux"], size=masks.shape[-2:],
                                              mode="bilinear", align_corners=False)
                    aux_loss = self.ce(aux_logits, masks) + self.dice(aux_logits, masks)
                    loss += self.args.aux_weight * aux_loss
                
                loss = loss / self.args.accumulation_steps
    
            if torch.isnan(loss):
                continue
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            
            total_loss += loss.item() * self.args.accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * self.args.accumulation_steps:.4f}'})
    
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()
    
        return {'loss': total_loss / len(loader)}

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        confusion_matrix = np.zeros((self.args.num_classes, self.args.num_classes), dtype=np.int64)
        
        for imgs, masks in tqdm(loader, desc="Validation"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            
            logits = self.model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            pred = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            
            mask = (target >= 0) & (target < self.args.num_classes)
            label = self.args.num_classes * target[mask].astype('int') + pred[mask]
            count = np.bincount(label, minlength=self.args.num_classes**2)
            confusion_matrix += count.reshape(self.args.num_classes, self.args.num_classes)
    
        intersection = np.diag(confusion_matrix)
        union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        
        return {'miou': miou, 'accuracy': intersection.sum() / confusion_matrix.sum()}

    def save_checkpoint(self, epoch, metrics, is_best=False):
        save_checkpoint_with_correct_norm(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            metrics=metrics,
            save_path=self.save_dir / "last.pth",
            global_step=self.global_step,
            best_miou=self.best_miou
        )
        
        if is_best:
            save_checkpoint_with_correct_norm(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                metrics=metrics,
                save_path=self.save_dir / "best.pth",
                global_step=self.global_step,
                best_miou=self.best_miou
            )
            print(f"✅ Best model saved! mIoU: {metrics['miou']:.4f}")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--scheduler", default="onecycle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig.get_base_config()
    args.loss_config = cfg['loss']
    
    # Data
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=False
    )
    
    # Model
    backbone = GCNetWithEnhance(**cfg['backbone']).to(device)
    
    cfg['head'].update({'in_channels': 128, 'c1_channels': 32, 'c2_channels': 64, 'num_classes': args.num_classes})
    cfg['auxhead'].update({'in_channels': 128, 'num_classes': args.num_classes})
    
    model = Segmentor(
        backbone=backbone,
        head=GCNetHead(**cfg['head']),
        aux_head=GCNetAuxHead(**cfg['auxhead'])
    )
    
    model = replace_bn_with_gn(model)
    model.apply(init_weights)
    
    if args.pretrained_weights:
        load_pretrained_gcnet_core(model, args.pretrained_weights)
    
    if args.freeze_backbone:
        freeze_backbone(model)
    
    count_trainable_params(model)
    
    # Optimizer
    optimizer = setup_discriminative_lr(model, args.lr, 0.1, args.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps
    )
    
    # Train
    trainer = Trainer(model, optimizer, scheduler, device, args, class_weights)
    
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, Val mIoU={val_metrics['miou']:.4f}")
        
        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        
        trainer.save_checkpoint(epoch, val_metrics, is_best)
    
    print(f"Training done! Best mIoU: {trainer.best_miou:.4f}")


if __name__ == "__main__":
    main()
