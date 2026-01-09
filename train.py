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

from model.backbone.model import GCNetWithDWSA
from model.head.segmentation_head import (
    GCNetHead,
    GCNetAuxHead,
    EnhancedDecoder,
    GatedFusion,
    DWConvModule,
    ResidualBlock,
)
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# FREEZE/UNFREEZE UTILITIES
# ============================================

def freeze_backbone(model):
    """Freeze toÃ n bá»™ backbone"""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("ðŸ”’ Backbone FROZEN - chá»‰ head Ä‘Æ°á»£c train")


def unfreeze_backbone(model):
    """Unfreeze toÃ n bá»™ backbone"""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("ðŸ”“ Backbone UNFROZEN - táº¥t cáº£ layers trainable")


def unfreeze_backbone_progressive(model, stage_name):
    """
    Unfreeze má»™t stage cá»¥ thá»ƒ cá»§a backbone
    stage_name: 'stem', 'stage1', 'stage2', 'stage3', 'stage4', 'bottleneck', 'ppm'
    """
    unfrozen_count = 0
    for name, module in model.backbone.named_modules():
        if stage_name in name:
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_count += 1
    
    print(f"ðŸ”“ Unfrozen stage: {stage_name} ({unfrozen_count} parameters)")


def get_backbone_stages(model):
    """Láº¥y danh sÃ¡ch cÃ¡c stages trong backbone theo thá»© tá»± tá»« input Ä‘áº¿n output"""
    stages = []
    
    # Thá»© tá»± tá»« tháº¥p Ä‘áº¿n cao (cÃ ng gáº§n input cÃ ng tá»•ng quÃ¡t)
    if hasattr(model.backbone, 'stem'):
        stages.append('stem')
    
    for i in range(1, 6):  # stage1 Ä‘áº¿n stage5
        stage_name = f'stage{i}'
        if hasattr(model.backbone, stage_name):
            stages.append(stage_name)
    
    if hasattr(model.backbone, 'bottleneck'):
        stages.append('bottleneck')
    
    if hasattr(model.backbone, 'ppm'):
        stages.append('ppm')
    
    return stages


def count_trainable_params(model):
    """Äáº¿m vÃ  hiá»ƒn thá»‹ sá»‘ parameters trainable/frozen"""
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
    print("ðŸ“Š PARAMETER STATISTICS")
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


def print_freeze_status(model):
    """Hiá»ƒn thá»‹ tráº¡ng thÃ¡i freeze chi tiáº¿t tá»«ng stage"""
    print(f"\n{'='*70}")
    print("ðŸ” FREEZE STATUS")
    print(f"{'='*70}")
    
    stages = get_backbone_stages(model)
    for stage in stages:
        stage_params = [p for n, p in model.backbone.named_parameters() if stage in n]
        if stage_params:
            trainable = sum(1 for p in stage_params if p.requires_grad)
            total = len(stage_params)
            status = "ðŸŸ¢" if trainable == total else "ðŸ”´" if trainable == 0 else "ðŸŸ¡"
            print(f"{status} {stage:12s}: {trainable:>3}/{total:>3} trainable")
    
    # Heads
    head_trainable = sum(1 for p in model.decode_head.parameters() if p.requires_grad)
    head_total = sum(1 for p in model.decode_head.parameters())
    print(f"ðŸŸ¢ {'head':12s}: {head_trainable:>3}/{head_total:>3} trainable")
    
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        aux_trainable = sum(1 for p in model.aux_head.parameters() if p.requires_grad)
        aux_total = sum(1 for p in model.aux_head.parameters())
        print(f"ðŸŸ¢ {'aux_head':12s}: {aux_trainable:>3}/{aux_total:>3} trainable")
    
    print(f"{'='*70}\n")


def setup_discriminative_lr(model, base_lr, backbone_lr_factor=0.1, weight_decay=1e-4):
    """
    Táº¡o optimizer vá»›i LR khÃ¡c nhau cho backbone vs head
    backbone_lr = base_lr * backbone_lr_factor
    head_lr = base_lr
    """
    backbone_params = [p for n, p in model.named_parameters() 
                      if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() 
                  if 'backbone' not in n and p.requires_grad]
    
    if len(backbone_params) == 0:
        # Backbone fully frozen
        optimizer = torch.optim.AdamW(head_params, lr=base_lr, weight_decay=weight_decay)
        print(f"âš™ï¸  Optimizer: AdamW (lr={base_lr}) - chá»‰ head")
    else:
        backbone_lr = base_lr * backbone_lr_factor
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': base_lr, 'name': 'head'}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        print(f"âš™ï¸  Optimizer: AdamW")
        print(f"   â”œâ”€ Backbone LR: {backbone_lr:.2e} ({len(backbone_params):,} params)")
        print(f"   â””â”€ Head LR:     {base_lr:.2e} ({len(head_params):,} params)")
    
    return optimizer


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
        print("ðŸ” BACKBONE CHANNEL DETECTION")
        print(f"{'='*70}")
        for key in ['c1', 'c2', 'c3', 'c4', 'c5']:
            if key in channels:
                print(f"   {key}: {channels[key]} channels")
        print(f"{'='*70}\n")
        
        return channels


# ============================================
# MODEL CONFIG - ENHANCED BACKBONE WITH UPGRADED HEAD
# ============================================

class ModelConfig:
    """Enhanced Backbone: channels=48 + Upgraded Head with Gated Fusion"""
    
    @staticmethod
    def get_config():
        """Optimized config for best mIoU"""
        return {
            "backbone": {
                "in_channels": 3,
                "channels": 32,
                "ppm_channels": 128,
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
                "dwsa_stages": ['stage3', 'stage4', 'bottleneck'],
                "dwsa_num_heads": 8,
                "use_dcn_in_stage4": True,
                "use_multi_scale_context": True,
                "align_corners": False,
                "deploy": False
            },
            "head": {
                "in_channels": 128,
                "decoder_channels": 128,
                "dropout_ratio": 0.1,
                "align_corners": False,
                "use_gated_fusion": True,
                "norm_cfg": {'type': 'BN', 'requires_grad': True},
                "act_cfg": {'type': 'ReLU', 'inplace': False}
            },
            "aux_head": {
                "in_channels": 256,
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
    """Segmentation model with backbone + upgraded head + auxiliary head"""
    
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
    """Training class with progressive unfreezing, logging vÃ  checkpointing"""
    
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
        
        # ====== PROGRESSIVE UNFREEZING SETUP ======
        self.unfreeze_epochs = [int(x) for x in args.unfreeze_schedule.split(',')] if args.unfreeze_schedule else []
        self.unfreeze_mode = args.unfreeze_mode
        self.current_unfreeze_idx = 0
        
        # Get backbone stages
        self.backbone_stages = get_backbone_stages(model)
        print(f"\nðŸ“‹ Backbone stages: {self.backbone_stages}")
        
        # Setup discriminative LR tracking
        self.base_lr = args.lr
        self.backbone_lr_factor = args.backbone_lr_factor
        
        # Save config
        self.save_config()
        self._print_config(loss_cfg)
    
    def _print_config(self, loss_cfg):
        """Print training configuration"""
        print(f"\n{'='*70}")
        print("âš™ï¸  TRAINER CONFIGURATION")
        print(f"{'='*70}")
        print(f"ðŸ“¦ Batch size: {self.args.batch_size}")
        print(f"ðŸ” Gradient accumulation: {self.args.accumulation_steps}")
        print(f"ðŸ“Š Effective batch size: {self.args.batch_size * self.args.accumulation_steps}")
        print(f"âš¡ Mixed precision: {self.args.use_amp}")
        print(f"âœ‚ï¸  Gradient clipping: {self.args.grad_clip}")
        print(f"ðŸ“‰ Loss: CE({loss_cfg['ce_weight']}) + Dice({loss_cfg['dice_weight']}) + Focal({loss_cfg['focal_weight']})")
        print(f"ðŸ”€ Gated Fusion: ENABLED (upgraded head)")
        print(f"ðŸ’¾ Save dir: {self.args.save_dir}")
        print(f"â„ï¸  Freeze backbone: {self.args.freeze_backbone}")
        print(f"ðŸ“… Unfreeze schedule: {self.unfreeze_epochs}")
        print(f"ðŸ”„ Unfreeze mode: {self.unfreeze_mode}")
        print(f"{'='*70}\n")
    
    def save_config(self):
        """Save training config"""
        config = vars(self.args)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    def _handle_unfreezing(self, epoch):
        """Xá»­ lÃ½ unfreezing táº¡i epoch"""
        print(f"\n{'='*70}")
        print(f"ðŸ”“ UNFREEZING AT EPOCH {epoch}")
        print(f"{'='*70}\n")
        
        if self.unfreeze_mode == 'all_at_once':
            # Unfreeze toÃ n bá»™ backbone ngay láº­p tá»©c
            unfreeze_backbone(self.model)
            self.current_unfreeze_idx = len(self.backbone_stages)
            print("âœ… Unfrozen toÃ n bá»™ backbone!")
            
        else:  # progressive
            # Unfreeze tá»«ng stage má»™t
            if self.current_unfreeze_idx < len(self.backbone_stages):
                stage_to_unfreeze = self.backbone_stages[self.current_unfreeze_idx]
                unfreeze_backbone_progressive(self.model, stage_to_unfreeze)
                self.current_unfreeze_idx += 1
                print(f"âœ… Unfrozen stage: {stage_to_unfreeze}")
            else:
                print("âš ï¸  Táº¥t cáº£ stages Ä‘Ã£ Ä‘Æ°á»£c unfreeze!")
        
        # Update optimizer sau khi unfreeze
        self._update_optimizer_after_unfreeze()
        
        # Print status
        print_freeze_status(self.model)
        count_trainable_params(self.model)
        
        print(f"{'='*70}\n")
    
    def _update_optimizer_after_unfreeze(self):
        """Cáº­p nháº­t optimizer sau khi unfreeze layers má»›i"""
        # Giáº£m LR cho backbone khi unfreeze thÃªm
        new_backbone_lr = self.base_lr * self.backbone_lr_factor * (0.5 ** self.current_unfreeze_idx)
        new_head_lr = self.base_lr * (0.5 ** self.current_unfreeze_idx)
        
        # Táº¡o optimizer má»›i vá»›i params má»›i
        self.optimizer = setup_discriminative_lr(
            self.model,
            base_lr=new_head_lr,
            backbone_lr_factor=self.backbone_lr_factor * (0.5 ** self.current_unfreeze_idx),
            weight_decay=self.args.weight_decay
        )
        
        # Reset scaler Ä‘á»ƒ trÃ¡nh numerical issues
        self.scaler = GradScaler(enabled=self.args.use_amp)
        
        print(f"ðŸ“‰ LR updated: Backbone={new_backbone_lr:.2e}, Head={new_head_lr:.2e}")
    
    def train_epoch(self, loader, epoch):
        """Train one epoch vá»›i progressive unfreezing"""
        
        # ====== PROGRESSIVE UNFREEZING LOGIC ======
        if epoch in self.unfreeze_epochs:
            self._handle_unfreezing(epoch)
        
        # ====== TRAINING CODE ======
        
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
                    # Decay aux weight as training progresses
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
            'global_step': self.global_step,
            'unfreeze_epochs': self.unfreeze_epochs,
            'current_unfreeze_idx': self.current_unfreeze_idx,
            'backbone_stages': self.backbone_stages
        }
        
        torch.save(checkpoint, self.save_dir / "last.pth")
        
        if is_best:
            torch.save(checkpoint, self.save_dir / "best.pth")
            print(f"âœ… Best model saved! mIoU: {metrics['miou']:.4f}")
        
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
            print(f"âœ… Weights loaded from epoch {checkpoint['epoch']}, starting new phase from epoch 0")
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            
            # Restore unfreezing state
            self.unfreeze_epochs = checkpoint.get('unfreeze_epochs', [])
            self.current_unfreeze_idx = checkpoint.get('current_unfreeze_idx', 0)
            
            if self.scheduler and checkpoint.get('scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            print(f"âœ… Checkpoint loaded, resuming from epoch {self.start_epoch}")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="ðŸš€ GCNet Training - Progressive Unfreezing")
    
    # ========== TRANSFER LEARNING ARGUMENTS ==========
    parser.add_argument("--pretrained_weights", type=str, default=None,
                       help="Path to pretrained GCNet weights")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                       help="Freeze toÃ n bá»™ backbone tá»« epoch 0")
    parser.add_argument("--unfreeze_schedule", type=str, default="10,20,30,40",
                       help="CÃ¡c epoch sáº½ unfreeze (VD: '10,20,30,40')")
    parser.add_argument("--unfreeze_mode", type=str, default="progressive",
                       choices=["progressive", "all_at_once"],
                       help="CÃ¡ch unfreeze: progressive (tá»«ng stage) hoáº·c all_at_once")
    parser.add_argument("--use_discriminative_lr", action="store_true", default=True,
                       help="DÃ¹ng LR khÃ¡c nhau cho backbone vs head")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1,
                       help="Backbone LR = head_lr * factor")
    
    # ========== DATASET ARGUMENTS ==========
    parser.add_argument("--train_txt", required=True, help="Path to training list")
    parser.add_argument("--val_txt", required=True, help="Path to validation list")
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    
    # ========== TRAINING ARGUMENTS ==========
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    
    # ========== OPTIMIZATION ARGUMENTS ==========
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4, help="Max LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=1.0, help="Auxiliary head weight (decays over epochs)")
    parser.add_argument("--scheduler", default="onecycle", choices=["onecycle", "poly", "cosine"])
    
    # ========== DATA ARGUMENTS ==========
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    
    # ========== SYSTEM ARGUMENTS ==========
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
    print(f"ðŸš€ GCNet Training - Progressive Unfreezing & Transfer Learning")
    print(f"{'='*70}")
    print(f"ðŸ“± Device: {device}")
    print(f"ðŸ–¼ï¸  Image size: {args.img_h}x{args.img_w}")
    print(f"ðŸ“Š Epochs: {args.epochs}")
    print(f"âš¡ Scheduler: {args.scheduler}")
    print(f"â„ï¸  Freeze backbone: {args.freeze_backbone}")
    print(f"ðŸ“… Unfreeze schedule: {args.unfreeze_schedule}")
    print(f"{'='*70}\n")
    
    # Config
    cfg = ModelConfig.get_config()
    args.loss_config = cfg["loss"]
    
    # Dataloaders
    print(f"ðŸ“‚ Creating dataloaders...")
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=True,
        dataset_type=args.dataset_type
    )
    print(f"âœ… Dataloaders created\n")
    
    # Model
    print(f"{'='*70}")
    print("ðŸ—ï¸  BUILDING MODEL WITH PROGRESSIVE UNFREEZING")
    print(f"{'='*70}\n")
    
    # Build backbone
    backbone = GCNetWithDWSA(**cfg["backbone"]).to(device)
    
    # Auto-detect backbone channels
    detected_channels = detect_backbone_channels(backbone, device, (args.img_h, args.img_w))
    
    # Build head config with detected channels
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
    
    print("\nðŸ”§ Applying Model Optimizations...")
    print("   â”œâ”€ Converting BatchNorm â†’ GroupNorm")
    model = replace_bn_with_gn(model)
    
    print("   â”œâ”€ Applying Kaiming Initialization")
    model.apply(init_weights)
    
    print("   â””â”€ Checking Model Health")
    check_model_health(model)
    print()
    
    # ===================== TRANSFER LEARNING SETUP =====================
    print(f"{'='*70}")
    print("ðŸ”„ TRANSFER LEARNING SETUP")
    print(f"{'='*70}\n")
    
    # Load pre-trained weights
    if args.pretrained_weights:
        print(f"ðŸ“¥ Loading pretrained weights from: {args.pretrained_weights}")
        
        try:
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Remove 'backbone.' prefix if loading from MMSegmentation
                    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load backbone
            backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('decode_head') and not k.startswith('aux_head')}
            
            missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
            if missing:
                print(f"   âš ï¸  Missing keys in backbone: {len(missing)} keys")
                if len(missing) <= 5:
                    print(f"      Keys: {missing}")
            if unexpected:
                print(f"   âš ï¸  Unexpected keys: {len(unexpected)} keys")
            
            print(f"âœ… Weights loaded successfully!\n")
            
        except Exception as e:
            print(f"âŒ Failed to load weights: {e}\n")
            return
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print(f"â„ï¸  Freezing backbone...")
        freeze_backbone(model)
        print()
    
    # Print status
    count_trainable_params(model)
    print_freeze_status(model)
    
    # ===================== END TRANSFER LEARNING SETUP =====================
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ðŸ“Š Total parameters: {total_params:,} ({total_params/1e6:.2f}M)\n")

    # Test forward pass
    model = model.to(device)
    with torch.no_grad():
        sample = torch.randn(1, 3, args.img_h, args.img_w).to(device)
        try:
            outputs = model.forward_train(sample)
            print(f"âœ… Forward pass successful!")
            print(f"   Main head output:  {outputs['main'].shape}")
            if 'aux' in outputs:
                print(f"   Aux head output:   {outputs['aux'].shape}")
        except Exception as e:
            print(f"âŒ Forward pass FAILED: {e}")
            return
    
    print(f"{'='*70}\n")

    # ===================== OPTIMIZER SETUP =====================
    if args.use_discriminative_lr:
        optimizer = setup_discriminative_lr(
            model,
            base_lr=args.lr,
            backbone_lr_factor=args.backbone_lr_factor,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    
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
        print(f"âœ… Using OneCycleLR scheduler (total_steps={total_steps})")
    elif args.scheduler == 'poly':
        print(f"âœ… Using Polynomial LR decay")
        def poly_lr_lambda(epoch):
            return (1 - epoch / args.epochs) ** 0.9
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    else:
        print(f"âœ… Using Cosine Annealing LR")
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
    print("ðŸš€ STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(trainer.start_epoch, args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)
        
        print(f"\n{'='*70}")
        print(f"ðŸ“Š Epoch {epoch+1}/{args.epochs}")
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
    print("âœ… TRAINING COMPLETED!")
    print(f"ðŸ† Best mIoU: {trainer.best_miou:.4f}")
    print(f"ðŸ’¾ Checkpoints saved to: {args.save_dir}")
    print(f"ðŸ“Š Tensorboard logs at: {args.save_dir}/tensorboard")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
