# ============================================
# FIXED train.py - Proper Gradient Clipping + Monitoring
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
import time
import gc
import warnings

warnings.filterwarnings('ignore')

# ============================================
# IMPORTS
# ============================================

from model.backbone.model import (
    GCNetWithEnhance,
    GCNetCore,
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

def load_pretrained_gcnet_core(model, ckpt_path, strict_match=False):
    print(f"Loading pretrained weights from: {ckpt_path}")
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
    print("WEIGHT LOADING SUMMARY")
    print(f"{'='*70}")
    print(f"Loaded:   {loaded:>5} / {total} params ({rate:.1f}%)")
    print(f"Skipped:  {len(skipped):>5} params from checkpoint")
    print(f"{'='*70}")

    if rate < 50:
        print(" WARNING: Less than 50% params loaded!")
        print(f"   First 5 skipped keys: {skipped[:5]}")

    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)

    if missing:
        print(f"\nâš ï¸  Missing keys in model ({len(missing)}):")
        for key in missing[:10]:
            print(f"   - {key}")
        if len(missing) > 10:
            print(f"   ... and {len(missing)-10} more")
    print()

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
    print("’ Backbone FROZEN")
def print_backbone_structure(model):
    """In ra  backbone debug"""
    print(f"\n{'='*70}")
    print(" BACKBONE STRUCTURE")
    print(f"{'='*70}")
    
    for name, module in model.backbone.named_children():
        print(f" {name}: {type(module).__name__}")
        
        # If ModuleList
        if isinstance(module, nn.ModuleList):
            for i, submodule in enumerate(module):
                print(f" [{i}]: {type(submodule).__name__}")
    
    print(f"{'='*70}\n")

def unfreeze_backbone_progressive(model, stage_names):
    """
    Unfreeze specific stages - FIXED to work with nested GCNetCore
    """
    if isinstance(stage_names, str):
        stage_names = [stage_names]

    unfrozen_params = 0
    unfrozen_modules = []

    for stage_name in stage_names:
        module = None
        found_path = None
        
        # Strategy 1: Direct lookup at model.backbone level
        if hasattr(model.backbone, stage_name):
            attr = getattr(model.backbone, stage_name)
            if attr is not None:
                module = attr
                found_path = f"backbone.{stage_name}"
        
        # Strategy 2: Lookup at model.backbone.backbone level (GCNetCore)
        if module is None and hasattr(model.backbone, 'backbone'):
            if hasattr(model.backbone.backbone, stage_name):
                attr = getattr(model.backbone.backbone, stage_name)
                if attr is not None:
                    module = attr
                    found_path = f"backbone.backbone.{stage_name}"
        
        # Strategy 3: Parse dotted names like 'semantic_branch_layers.0'
        if module is None and '.' in stage_name:
            parts = stage_name.split('.')
            base_name = parts[0]  # e.g., 'semantic_branch_layers'
            index = parts[1] if len(parts) > 1 else None
            
            # Try model.backbone.backbone.<base_name>[index]
            if hasattr(model.backbone.backbone, base_name):
                base_module = getattr(model.backbone.backbone, base_name)
                
                if index is not None and index.isdigit():
                    try:
                        module = base_module[int(index)]
                        found_path = f"backbone.backbone.{stage_name}"
                    except (IndexError, TypeError):
                        pass
                else:
                    module = base_module
                    found_path = f"backbone.backbone.{base_name}"
        
        # If still not found, skip
        if module is None:
            print(f"Module '{stage_name}' not found")
            continue
        
        # Unfreeze parameters
        param_count = 0
        for p in module.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                unfrozen_params += 1
                param_count += 1
        
        if param_count > 0:
            unfrozen_modules.append((found_path, param_count))
            print(f"Unfrozen: {found_path} ({param_count:,} params)")

    # Summary
    if unfrozen_modules:
        print(f"\nTotal: {len(unfrozen_modules)} modules, {unfrozen_params:,} params unfrozen")
    else:
        print(f"\nWARNING: No modules were unfrozen!")
    
    return unfrozen_params
def print_available_modules(model):
    """Debug helper - print all available modules in backbone"""
    print(f"\n{'='*70}")
    print(" AVAILABLE BACKBONE MODULES")
    print(f"{'='*70}")
    
    print("\nAt model.backbone level:")
    for name, module in model.backbone.named_children():
        if module is not None:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"   {name}: {type(module).__name__} ({param_count:,} params)")
    
    print("\n At model.backbone.backbone level (GCNetCore):")
    if hasattr(model.backbone, 'backbone'):
        for name, module in model.backbone.backbone.named_children():
            if isinstance(module, nn.ModuleList):
                print(f"{name}: ModuleList[{len(module)}]")
                for i, submodule in enumerate(module):
                    param_count = sum(p.numel() for p in submodule.parameters())
                    print(f"   [{i}]: {type(submodule).__name__} ({param_count:,} params)")
            elif module is not None:
                param_count = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {type(module).__name__} ({param_count:,} params)")
    
    print(f"{'='*70}\n")

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    
    head_total = sum(p.numel() for p in model.decode_head.parameters())
    head_trainable = sum(p.numel() for p in model.decode_head.parameters() if p.requires_grad)
    
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        aux_total = sum(p.numel() for p in model.aux_head.parameters())
        aux_trainable = sum(p.numel() for p in model.aux_head.parameters() if p.requires_grad)
    else:
        aux_total = aux_trainable = 0
    
    print(f"\n{'='*70}")
    print("PARAMETER STATISTICS")
    print(f"{'='*70}")
    print(f"Total:        {total:>15,} | 100%")
    print(f"Trainable:    {trainable:>15,} | {100*trainable/total:.1f}%")
    print(f"Frozen:       {frozen:>15,} | {100*frozen/total:.1f}%")
    print(f"{'-'*70}")
    print(f"Backbone:     {backbone_trainable:>15,} / {backbone_total:,} | {100*backbone_trainable/backbone_total:.1f}%")
    print(f"Head:         {head_trainable:>15,} / {head_total:,} | {100*head_trainable/head_total:.1f}%")
    if aux_total > 0:
        print(f"Aux Head:     {aux_trainable:>15,} / {aux_total:,} | {100*aux_trainable/aux_total:.1f}%")
    print(f"{'='*70}\n")
    
    return trainable, frozen


def setup_discriminative_lr(model, base_lr, backbone_lr_factor=0.1, weight_decay=1e-4):
    backbone_params = [p for n, p in model.named_parameters() 
                      if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() 
                  if 'backbone' not in n and p.requires_grad]
    
    if len(backbone_params) == 0:
        optimizer = torch.optim.AdamW(head_params, lr=base_lr, weight_decay=weight_decay)
        print(f"âš™ï¸  Optimizer: AdamW (lr={base_lr}) - head only")
    else:
        backbone_lr = base_lr * backbone_lr_factor
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': base_lr, 'name': 'head'}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        print(f"Optimizer: AdamW (Discriminative LR)")
        print(f"Backbone LR: {backbone_lr:.2e} ({len(backbone_params):,} params)")
        print(f"Head LR:     {base_lr:.2e} ({len(head_params):,} params)")
    
    return optimizer


# FIX: Monitor gradients
def check_gradients(model, threshold=10.0):
    """Monitor gradient norms"""
    max_grad = 0.0
    max_grad_name = ""
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2
            if grad_norm > max_grad:
                max_grad = grad_norm
                max_grad_name = name
    
    total_norm = total_norm ** 0.5
    
    if max_grad > threshold:
        print(f"Large gradient detected: {max_grad_name[:50]}... = {max_grad:.2f}")
    
    return max_grad, total_norm


# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config():
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,
                "ppm_channels": 128,
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
                "dwsa_stages": ['stage4','stage5', 'stage6'],
                "dwsa_num_heads": 4,
                "dwsa_reduction": 4,
                "dwsa_qk_sharing": True,
                "dwsa_groups": 4,
                "dwsa_drop": 0.1,  
                "dwsa_alpha": 0.1, 
                "use_multi_scale_context": True,
                "ms_alpha": 0.1, 
                "align_corners": False,
                "deploy": False
            },
            "head": {
                "in_channels": 64,
                "decoder_channels": 128,
                "dropout_ratio": 0.1,
                "align_corners": False,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "aux_head": {
                "in_channels": 128,
                "channels": 96,
                "dropout_ratio": 0.1,
                "align_corners": False,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "loss": {
                "ce_weight": 1.0,
                "dice_weight": 0.0,
                "focal_weight": 0.0,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "dice_smooth": 1e-5
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
# TRAINER - FIXED VERSION
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
        self.class_weights = class_weights.to(device) if class_weights is not None else None

        loss_cfg = args.loss_config
        self.dice = DiceLoss(
            smooth=loss_cfg['dice_smooth'],
            ignore_index=args.ignore_index,
            reduction='mean'
        )
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            ignore_index=args.ignore_index,
            reduction='mean'
        )
        
        self.ce_weight = loss_cfg['ce_weight']
        self.dice_weight = loss_cfg['dice_weight']
        self.base_loss_cfg = loss_cfg
        self.loss_phase = 'full'
        
        self.scaler = GradScaler(enabled=args.use_amp)
        
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        
        self.save_config()
        self._print_config(loss_cfg)
    
    def set_loss_phase(self, phase: str):
        if phase == self.loss_phase:
            return
    
        if phase == 'ce_only':
            self.dice_weight = 0.0
        elif phase == 'full':
            self.dice_weight = self.base_loss_cfg['dice_weight']
    
        self.loss_phase = phase
        print(f"Loss phase: {phase} (CE={self.ce_weight}, Dice={self.dice_weight})")
    
    def _print_config(self, loss_cfg):
        print(f"\n{'='*70}")
        print("TRAINER CONFIGURATION")
        print(f"{'='*70}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.accumulation_steps}")
        print(f"Effective batch: {self.args.batch_size * self.args.accumulation_steps}")
        print(f"Mixed precision: {self.args.use_amp}")
        print(f"Gradient clipping: {self.args.grad_clip}")
        print(f"Loss: CE({loss_cfg['ce_weight']}) + Dice({loss_cfg['dice_weight']})")
        print(f"{'='*70}\n")

    def save_config(self):
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def train_epoch(self, loader, epoch):
        self.model.train()
        
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        max_grad_epoch = 0.0
        max_grad = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            
            if masks.dim() == 4:
                masks = masks.squeeze(1)
    
            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)
                logits = outputs["main"]
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
    
                ce_loss = self.ce(logits, masks)
    
                if self.dice_weight > 0:
                    dice_loss = self.dice(logits, masks)
                else:
                    dice_loss = torch.tensor(0.0, device=logits.device)
    
                loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
    
                if "aux" in outputs and self.args.aux_weight > 0:
                    aux_logits = outputs["aux"]
                    aux_logits = F.interpolate(
                        aux_logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    aux_ce_loss = self.ce(aux_logits, masks)
                    if self.dice_weight > 0:
                        aux_dice_loss = self.dice(aux_logits, masks)
                    else:
                        aux_dice_loss = torch.tensor(0.0, device=logits.device)
    
                    aux_total = self.ce_weight * aux_ce_loss + self.dice_weight * aux_dice_loss
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    loss = loss + aux_weight * aux_total
    
                loss = loss / self.args.accumulation_steps
    
            # FIX 1: Check NaN BEFORE backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nNaN/Inf loss at epoch {epoch}, batch {batch_idx}")
                print(f"   CE: {ce_loss.item():.4f}, Dice: {dice_loss.item():.4f}")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            
            self.scaler.scale(loss).backward()
            
            # FIX 2: ALWAYS clip gradients (khÃ´ng phá»¥ thuá»™c accumulation)
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                
                # FIX 3: Monitor gradients
                max_grad, total_norm = check_gradients(self.model, threshold=10.0)
                max_grad_epoch = max(max_grad_epoch, max_grad)
                
                # FIX 4: ALWAYS apply gradient clipping
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.args.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            
            if self.scheduler and self.args.scheduler == 'onecycle':
                self.scheduler.step()
            
            total_loss += loss.item() * self.args.accumulation_steps
            total_ce += ce_loss.item()
            total_dice += dice_loss.item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'dice': f'{dice_loss.item():.4f}',
                'lr': f'{current_lr:.6f}',
                'max_grad': f'{max_grad:.2f}'  # â† Monitor
            })
            
            if batch_idx % 50 == 0:
                clear_gpu_memory()
            
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/total_loss', loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ce_loss', ce_loss.item(), self.global_step)
                self.writer.add_scalar('train/dice_loss', dice_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
                self.writer.add_scalar('train/max_grad', max_grad, self.global_step)  # â† Log
    
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()
    
        avg_loss = total_loss / len(loader)
        avg_ce = total_ce / len(loader)
        avg_dice = total_dice / len(loader)
        
        print(f"\nEpoch {epoch+1} Summary: Max Gradient = {max_grad_epoch:.2f}")
        
        return {'loss': avg_loss, 'ce': avg_ce, 'dice': avg_dice, 'focal': 0.0}

    @torch.no_grad()
    def validate(self, loader, epoch):
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
            
                ce_loss = self.ce(logits, masks)
                if self.dice_weight > 0:
                    dice_loss = self.dice(logits, masks)
                else:
                    dice_loss = torch.tensor(0.0, device=logits.device)
            
                loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
            
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
            print(f"Best model saved! mIoU: {metrics['miou']:.4f}")
        
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(checkpoint, self.save_dir / f"epoch_{epoch+1}.pth")

    def load_checkpoint(self, checkpoint_path, reset_epoch=True, load_optimizer=True, reset_best_metric=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model'])

        if load_optimizer and checkpoint.get('optimizer') is not None:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                print(f"Optimizer state not loaded: {e}")
        else:
            print("Skipping optimizer state loading.")

        if 'scaler' in checkpoint and checkpoint['scaler'] is not None and load_optimizer:
            try:
                self.scaler.load_state_dict(checkpoint['scaler'])
            except Exception as e:
                print(f"AMP scaler state not loaded: {e}")

        if reset_epoch:
            self.start_epoch = 0
            self.global_step = 0
            if reset_best_metric:
                self.best_miou = 0.0
            else:
                self.best_miou = checkpoint.get('best_miou', 0.0)
            print(f"Weights loaded from epoch {checkpoint['epoch']}, starting from epoch 0")
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            if self.scheduler and checkpoint.get('scheduler') and load_optimizer:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                except Exception as e:
                    print(f"Scheduler state not loaded: {e}")
            print(f"Checkpoint loaded, resuming from epoch {self.start_epoch}")


def detect_backbone_channels(backbone, device, img_size=(512, 1024)):
    backbone.eval()
    with torch.no_grad():
        sample = torch.randn(1, 3, *img_size).to(device)
        feats = backbone(sample)
        
        channels = {}
        for key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            if key in feats:
                channels[key] = feats[key].shape[1]
        
        print(f"\n{'='*70}")
        print("BACKBONE CHANNEL DETECTION")
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
    parser = argparse.ArgumentParser(description="ðŸš€ GCNetWithEnhance Training - FIXED")
    
    # Transfer Learning
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--unfreeze_schedule", type=str, default="")
    parser.add_argument("--use_discriminative_lr", action="store_true", default=True)
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--use_class_weights", action="store_true")
    
    # Dataset
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--reset_best_metric", action="store_true")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)  # â† INCREASED from 1.0
    parser.add_argument("--aux_weight", type=float, default=1.0)
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
                        choices=["transfer", "continue"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3)
    
    args = parser.parse_args()

    # Validate
    if args.ce_only_epochs_after_unfreeze < 0:
        raise ValueError("ce_only_epochs_after_unfreeze must be >= 0")
    if args.freeze_epochs >= args.epochs:
        raise ValueError(f"freeze_epochs must be < total epochs")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"GCNetWithEnhance Training - FIXED VERSION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Image size: {args.img_h}x{args.img_w}")
    print(f"Epochs: {args.epochs}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Gradient clipping: {args.grad_clip}")  # â† SHOW
    print(f"Freeze backbone: {args.freeze_backbone}")
    if args.unfreeze_schedule:
        print(f"Unfreeze schedule: {args.unfreeze_schedule}")
    print(f"Discriminative LR: {args.use_discriminative_lr} (factor={args.backbone_lr_factor})")
    print(f"{'='*70}\n")
    
    # Config
    cfg = ModelConfig.get_config()
    args.loss_config = cfg["loss"]
    
    print(f"ðŸ”§ Model Config:")
    print(f"DWSA alpha: {cfg['backbone']['dwsa_alpha']}")
    print(f"DWSA drop: {cfg['backbone']['dwsa_drop']}")
    print(f"MS alpha: {cfg['backbone']['ms_alpha']}\n")
    
    # Dataloaders
    print(f"Creating dataloaders...")
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=args.use_class_weights,
        dataset_type=args.dataset_type
    )
    print(f"Dataloaders created\n")
    
    # Model
    print(f"{'='*70}")
    print("BUILDING MODEL")
    print(f"{'='*70}\n")
    
    backbone = GCNetWithEnhance(**cfg["backbone"]).to(device)
    
    detected_channels = detect_backbone_channels(backbone, device, (args.img_h, args.img_w))
    
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
    
    model = Segmentor(
        backbone=backbone,
        head=GCNetHead(**head_cfg),
        aux_head=GCNetAuxHead(**aux_head_cfg),
    )
    
    print("\nApplying Optimizations...")
    # print("Converting BN â†’ GN")
    # model = replace_bn_with_gn(model)
    
    print("Kaiming Init")
    model.apply(init_weights)
    
    print("Health Check")
    check_model_health(model)
    print()
    
    # Transfer Learning
    print(f"{'='*70}")
    print("TRANSFER LEARNING SETUP")
    print(f"{'='*70}\n")
    
    if args.pretrained_weights:
        load_pretrained_gcnet_core(model, args.pretrained_weights)
    
    if args.freeze_backbone:
        freeze_backbone(model)
        print()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    count_trainable_params(model)
    
    # Test forward
    model = model.to(device)
    with torch.no_grad():
        sample = torch.randn(1, 3, args.img_h, args.img_w).to(device)
        try:
            outputs = model.forward_train(sample)
            print(f"Forward pass successful!")
            print(f"   Main:  {outputs['main'].shape}")
            if 'aux' in outputs:
                print(f"   Aux:   {outputs['aux'].shape}\n")
        except Exception as e:
            print(f"Forward pass FAILED: {e}\n")
            return
    
    # Optimizer
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
        print(f"Optimizer: AdamW (lr={args.lr})")
    
    # Scheduler
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * args.epochs
        n_groups = len(optimizer.param_groups)
    
        if n_groups == 1:
            max_lrs = args.lr
        elif n_groups == 2:
            max_lrs = [
                args.lr * args.backbone_lr_factor,
                args.lr,
            ]
        else:
            raise ValueError(f"Unexpected param_groups: {n_groups}")
    
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,
            final_div_factor=100000,
        )
        print(f"OneCycleLR (total_steps={total_steps})")
    elif args.scheduler == 'poly':
        print(f"Polynomial LR decay")
        def poly_lr_lambda(epoch):
            return (1 - epoch / args.epochs) ** 0.9
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    else:
        print(f"Cosine Annealing LR")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    print_backbone_structure(model)
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        class_weights=class_weights if args.use_class_weights else None
    )
    
    if args.resume:
        reset_epoch = (args.resume_mode == "transfer")
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=reset_epoch,
            load_optimizer=False,
            reset_best_metric=args.reset_best_metric,
        )
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    unfreeze_epochs = []
    if args.unfreeze_schedule:
        try:
            unfreeze_epochs = sorted(int(e) for e in args.unfreeze_schedule.split(','))
        except Exception:
            raise ValueError("unfreeze_schedule must be comma-separated integers")

        if any(e <= args.freeze_epochs for e in unfreeze_epochs):
            raise ValueError(f"Unfreeze epochs must be > freeze_epochs ({args.freeze_epochs})")
        if any(e >= args.epochs for e in unfreeze_epochs):
            raise ValueError(f"Unfreeze epochs must be < total epochs")

    for epoch in range(trainer.start_epoch, args.epochs):

        # Phase 1: Freeze backbone
        if epoch < args.freeze_epochs:
            freeze_backbone(model)
            trainer.set_loss_phase('full')

        # Phase 2+: Progressive unfreezing
        if epoch in unfreeze_epochs:
            k = len([e for e in unfreeze_epochs if e <= epoch])

            if k == 1:
                targets = ['semantic_branch_layers.2', 'detail_branch_layers.2', 'dwsa6']
            elif k == 2:
                targets = ['semantic_branch_layers.1', 'detail_branch_layers.1', 'dwsa5']
            elif k == 3:
                targets = ['semantic_branch_layers.0', 'detail_branch_layers.0', 'stem']
            else:
                targets = []

            if targets:
                unfreeze_backbone_progressive(model, targets)
                trainer.set_loss_phase('ce_only')
                
                print(f"\n{'='*70}")
                print(f"Learning Rates after unfreezing:")
                print(f"{'='*70}")
                for i, group in enumerate(optimizer.param_groups):
                    name = group.get('name', f'group_{i}')
                    print(f"   {name}: {group['lr']:.2e}")
                print(f"{'='*70}\n")

        # Switch back to full loss
        if unfreeze_epochs:
            past = [e for e in unfreeze_epochs if e <= epoch]
            if past:
                last_unfreeze = max(past)
                if epoch >= last_unfreeze + args.ce_only_epochs_after_unfreeze:
                    trainer.set_loss_phase('full')

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
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
    print("TRAINING COMPLETED!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
