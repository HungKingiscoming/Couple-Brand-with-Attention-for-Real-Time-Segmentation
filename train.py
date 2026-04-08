# ============================================
# train.py — adapted for GCNet v3 + GCNetHead v2
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
from torch.optim.lr_scheduler import LambdaLR
warnings.filterwarnings('ignore')

# ============================================
# IMPORTS — model mới, không còn mmcv/mmseg
# ============================================

from model.backbone.model import GCNetSegmentor        
from model.head.segmentation_head import GCNetHead      # head mới (tích hợp aux)
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# PRETRAINED WEIGHT LOADER
# ============================================

def _remap_stem_key(key: str, N2: int = 4):
    """Remap checkpoint key sang GCNet v3."""
    import re as _re

    for pref in ['backbone.', 'model.', 'module.']:
        if key.startswith(pref):
            key = key[len(pref):]

    m = _re.match(r'stem\.(\d+)\.(.+)$', key)
    if not m:
        return key

    idx  = int(m.group(1))
    rest = m.group(2)

    def _map_convmodule(rest_str, target_prefix):
        if rest_str.startswith('conv.'):
            return f'{target_prefix}.{rest_str[len("conv."):].lstrip(".")}'
        return None

    if idx == 0:
        return _map_convmodule(rest, 'stem_conv1.0')
    elif idx == 1:
        return _map_convmodule(rest, 'stem_conv2.0')
    elif 2 <= idx <= 1 + N2:
        return f'stem_stage2.{idx - 2}.{rest}'
    else:
        return f'stem_stage3.{idx - (2 + N2)}.{rest}'


def load_pretrained_gcnet(model, ckpt_path, strict_match=False):
    """Load pretrained weights into model.backbone (GCNetBackbone v3)."""
    print(f"Loading pretrained weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    model_state = model.backbone.state_dict()
    compatible = {}
    skipped = []

    # Simple matching: try exact name, then remove prefixes
    for ckpt_key, ckpt_val in state.items():
        # Strip common prefixes
        key = ckpt_key
        for pref in ('backbone.', 'model.', 'module.'):
            if key.startswith(pref):
                key = key[len(pref):]
                break
        # If key exists in model and shape matches, load it
        if key in model_state and model_state[key].shape == ckpt_val.shape:
            compatible[key] = ckpt_val
        else:
            skipped.append(ckpt_key)

    loaded = len(compatible)
    total = len(model_state)
    rate = 100 * loaded / total if total > 0 else 0.0

    print(f"\n{'='*70}")
    print("WEIGHT LOADING SUMMARY")
    print(f"{'='*70}")
    print(f"Loaded: {loaded:>5} / {total} ({rate:.1f}%)")
    print(f"Skipped: {len(skipped)} keys (unmatched or shape mismatch)")
    print(f"{'='*70}\n")

    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)
    if missing:
        print(f"Missing keys in loaded state: {len(missing)}")
        for k in missing[:5]:
            print(f"  - {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    return rate


# ============================================
# OPTIMIZER
# ============================================

def build_optimizer(model, args):
    """Use model's get_param_groups for discriminative learning rates."""
    param_groups = model.get_param_groups(
        head_lr=args.lr,
        backbone_lr=args.lr * args.backbone_lr_factor,
        fan_lr=args.lr * args.alpha_lr_factor,      # FoggyAwareNorm alpha
        msc_lr=args.lr * args.dwsa_lr_factor        # MultiScaleContext alpha
    )
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    # Store initial LR for each group
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
        g.setdefault('name', g.get('name', 'unknown'))

    print(f"Optimizer: AdamW (Discriminative LR)")
    for g in optimizer.param_groups:
        print(f"  group '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")
    return optimizer

def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    n_groups = len(optimizer.param_groups)

    if args.scheduler == 'onecycle':
        remaining_epochs = args.epochs - start_epoch
        total_steps = len(train_loader) * remaining_epochs

        if n_groups == 1:
            max_lrs = args.lr
        else:
            max_lrs = [g['initial_lr'] for g in optimizer.param_groups]

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
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9
        )
        print("Polynomial LR decay")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        print("CosineAnnealingLR")
    return scheduler


# ============================================
# LOSS FUNCTIONS
# ============================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255, log_loss=False, class_weights=None):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.log_loss     = log_loss
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else None
        )

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        valid_mask       = (targets != self.ignore_index)
        targets_clamped  = targets.clamp(0, C - 1)
        targets_one_hot  = F.one_hot(targets_clamped, C).permute(0, 3, 1, 2).float()
        targets_one_hot  = targets_one_hot * valid_mask.unsqueeze(1).float()

        probs       = F.softmax(logits, dim=1) * valid_mask.unsqueeze(1).float()
        probs_flat  = probs.reshape(B, C, -1)
        target_flat = targets_one_hot.reshape(B, C, -1)

        intersection = (probs_flat * target_flat).sum(2)
        cardinality  = probs_flat.sum(2) + target_flat.sum(2)
        dice_score   = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        dice_loss = -torch.log(dice_score.clamp(min=self.smooth)) if self.log_loss else 1.0 - dice_score

        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights.unsqueeze(0)

        class_present = target_flat.sum(2) > 0
        dice_loss     = dice_loss * class_present.float()
        n_present     = class_present.float().sum(1).clamp(min=1)
        return (dice_loss.sum(1) / n_present).mean()


class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, keep_ratio=0.3, min_kept=100000, class_weights=None):
        super().__init__()
        self.ignore_index  = ignore_index
        self.keep_ratio    = keep_ratio
        self.min_kept      = min_kept
        self.class_weights = class_weights

    def forward(self, logits, labels):
        weight      = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_pixel  = F.cross_entropy(logits, labels, weight=weight,
                                      ignore_index=self.ignore_index, reduction='none').view(-1)
        valid_mask  = (labels.view(-1) != self.ignore_index)
        valid_losses = loss_pixel[valid_mask]
        n_valid     = valid_losses.numel()
        if n_valid == 0:
            return logits.sum() * 0
        n_keep = max(int(self.keep_ratio * n_valid), min(self.min_kept, n_valid))
        n_keep = min(n_keep, n_valid)
        if n_keep < n_valid:
            threshold   = torch.sort(valid_losses, descending=True)[0][n_keep - 1].detach()
            valid_losses = valid_losses[valid_losses >= threshold]
        return valid_losses.mean()


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


def check_gradients(model, threshold=10.0):
    max_grad = 0.0; max_name = ""; total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            total_sq += g ** 2
            if g > max_grad:
                max_grad = g; max_name = name
    total_norm = total_sq ** 0.5
    if max_grad > threshold:
        print(f"Large gradient: {max_name[:60]}... = {max_grad:.2f}")
    return max_grad, total_norm


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_total = sum(p.numel() for p in model.backbone.parameters())
    bb_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    hd_total = sum(p.numel() for p in model.head.parameters())
    hd_train = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print("PARAMETER STATISTICS")
    print(f"{'='*70}")
    print(f"Total:      {total:>15,} | 100%")
    print(f"Trainable:  {trainable:>15,} | {100*trainable/total:.1f}%")
    print(f"Frozen:     {total-trainable:>15,} | {100*(total-trainable)/total:.1f}%")
    print(f"{'-'*70}")
    print(f"Backbone:   {bb_train:>15,} / {bb_total:,} | {100*bb_train/max(bb_total,1):.1f}%")
    print(f"Head:       {hd_train:>15,} / {hd_total:,} | {100*hd_train/max(hd_total,1):.1f}%")
    print(f"{'='*70}\n")
    return trainable, total - trainable


def freeze_backbone(model):
    """Freeze entire backbone, then selectively unfreeze FoggyAwareNorm and MultiScaleContext."""
    print("Freezing backbone (keeping FoggyAwareNorm + MultiScaleContext trainable)...")
    # Freeze all backbone parameters
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Unfreeze FoggyAwareNorm (stem_conv1, stem_conv2)
    fan_modules = [model.backbone.stem_conv1, model.backbone.stem_conv2]
    fan_params = 0
    for mod in fan_modules:
        for p in mod.parameters():
            p.requires_grad = True
            fan_params += p.numel()
        # Also set BN/IN to train mode
        for m in mod.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.train()
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad = True
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad = True
    print(f"  FoggyAwareNorm trainable: {fan_params:,} params")

    # Unfreeze MultiScaleContext module
    if hasattr(model.backbone, 'ms_context'):
        msc_params = 0
        for p in model.backbone.ms_context.parameters():
            p.requires_grad = True
            msc_params += p.numel()
        print(f"  MultiScaleContext trainable: {msc_params:,} params")

    print("Backbone partially frozen (FAN + MSC trainable)\n")


def unfreeze_backbone_progressive(model, stage_names):
    """Unfreeze specific modules by name (supports dotted names)."""
    if isinstance(stage_names, str):
        stage_names = [stage_names]

    total_unfrozen = 0
    for stage_name in stage_names:
        module = None
        # Try to get attribute
        if hasattr(model.backbone, stage_name):
            module = getattr(model.backbone, stage_name)
        elif '.' in stage_name:
            parts = stage_name.split('.')
            obj = model.backbone
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                module = obj
        if module is None:
            print(f"  [skip] module '{stage_name}' not found in backbone")
            continue

        count = 0
        for p in module.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                count += 1
        # Also set BN/IN to train mode inside module
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.train()
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad = True
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad = True
        total_unfrozen += count
        if count > 0:
            print(f"  Unfrozen: backbone.{stage_name} ({count:,} params)")

    print(f"  Total unfrozen this call: {total_unfrozen:,} params\n")
    return total_unfrozen

def log_msc_alpha(model, writer, epoch):
    """Log alpha value of MultiScaleContext (if exists)."""
    if hasattr(model.backbone, 'ms_context') and hasattr(model.backbone.ms_context, 'alpha'):
        alpha_val = model.backbone.ms_context.alpha.item()
        writer.add_scalar('msc/alpha', alpha_val, epoch)


def print_backbone_structure(model):
    print(f"\n{'='*70}")
    print(" BACKBONE STRUCTURE (GCNet v3 - FAN + MSC)")
    print(f"{'='*70}")
    for name, module in model.backbone.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {type(module).__name__}  ({n_params:,} params)")
    print(f"{'='*70}\n")

# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config():
        C = 32
        return {
            "backbone": {
                "in_channels"          : 3,
                "channels"             : C,
                "ppm_channels"         : 128,
                "num_blocks_per_stage" : [4, 4, [5, 4], [5, 4], [2, 2]],
                "align_corners"        : False,
                "norm_cfg"             : dict(type='BN', requires_grad=True),
                "act_cfg"              : dict(type='ReLU', inplace=True),
                "deploy"               : False,
            },
            "head": {
                "in_channels"     : C * 4,
                "channels"        : 128,
                "align_corners"   : False,
                "dropout_ratio"   : 0.1,
                "loss_weight_aux" : 0.4,
                "norm_cfg"        : dict(type='BN', requires_grad=True),
                "act_cfg"         : dict(type='ReLU', inplace=True),
            },
            "loss": {
                "ce_weight"    : 1.0,
                "dice_weight"  : 0.5,
                "dice_smooth"  : 1e-5,
            },
        }


# ============================================
# SEGMENTOR
# ============================================

class Segmentor(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.decode_head(feat)

    def forward_train(self, x):
        feats = self.backbone(x)
        outputs = self.decode_head(feats)
    
        # outputs = (aux, main)
        if isinstance(outputs, tuple):
            aux_logit, main_logit = outputs
        else:
            aux_logit = None
            main_logit = outputs
    
        return {
            "aux": aux_logit,
            "main": main_logit
        }


# ============================================
# TRAINER
# ============================================

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, args, class_weights=None):
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.args         = args
        self.best_miou    = 0.0
        self.start_epoch  = 0
        self.global_step  = 0

        loss_cfg          = args.loss_config
        self.ce_weight    = loss_cfg['ce_weight']
        self.dice_weight  = loss_cfg['dice_weight']
        self.base_loss_cfg = loss_cfg
        self.loss_phase   = 'full'

        cw_device = class_weights.to(device) if class_weights is not None else None

        self.ohem = OHEMLoss(
            ignore_index=args.ignore_index,
            keep_ratio=0.3,
            min_kept=100000,
            class_weights=class_weights,
        )
        self.dice = DiceLoss(
            smooth=loss_cfg['dice_smooth'],
            ignore_index=args.ignore_index,
        )
        self.ce = nn.CrossEntropyLoss(
            weight=cw_device,
            ignore_index=args.ignore_index,
        )

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=self.save_dir / "tensorboard")

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
        print(f"Loss phase → {phase}  (CE={self.ce_weight}, Dice={self.dice_weight})")

    def _print_config(self, loss_cfg):
        print(f"\n{'='*70}")
        print("TRAINER CONFIGURATION")
        print(f"{'='*70}")
        print(f"Batch size:             {self.args.batch_size}")
        print(f"Gradient accumulation:  {self.args.accumulation_steps}")
        print(f"Effective batch:        {self.args.batch_size * self.args.accumulation_steps}")
        print(f"Mixed precision:        {self.args.use_amp}")
        print(f"Gradient clipping:      {self.args.grad_clip}")
        print(f"Loss: CE({loss_cfg['ce_weight']}) + Dice({loss_cfg['dice_weight']})")
        print(f"{'='*70}\n")

    def save_config(self):
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

    # ---------------------------------------------------------------------- #
    # Training                                                                 #
    # ---------------------------------------------------------------------- #

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = total_ohem = total_dice = 0.0
        max_grad_epoch = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                # Forward pass: model returns (aux_logit, main_logit) in training mode
                aux_logit, main_logit = self.model(imgs)

                target_size = masks.shape[-2:]
                if aux_logit is not None:
                    aux_full = F.interpolate(aux_logit, size=target_size, mode='bilinear', align_corners=False)
                main_full = F.interpolate(main_logit, size=target_size, mode='bilinear', align_corners=False)

                # Main loss (OHEM + optional Dice)
                ohem_loss = self.ohem(main_full, masks)
                dice_loss = self.dice(main_full, masks) if self.dice_weight > 0 else torch.tensor(0.0, device=self.device)
                loss = ohem_loss + self.dice_weight * dice_loss

                # Auxiliary loss (if enabled)
                if self.args.aux_weight > 0 and aux_logit is not None:
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    aux_loss = self.ohem(aux_full, masks)
                    loss = loss + aux_weight * aux_loss

                loss = loss / self.args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nNaN/Inf loss at epoch {epoch}, batch {batch_idx} — skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                max_grad, _ = check_gradients(self.model, threshold=10.0)
                max_grad_epoch = max(max_grad_epoch, max_grad)
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.scheduler and self.args.scheduler == 'onecycle':
                    self.scheduler.step()

            total_loss += loss.item() * self.args.accumulation_steps
            total_ohem += ohem_loss.item()
            total_dice += dice_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ohem': f'{ohem_loss.item():.4f}',
                'dice': f'{dice_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'max_grad': f'{max_grad:.2f}',
            })

            if batch_idx % 50 == 0:
                clear_gpu_memory()

            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ohem', ohem_loss.item(), self.global_step)
                self.writer.add_scalar('train/dice', dice_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/max_grad', max_grad, self.global_step)

        n = len(loader)
        print(f"\nEpoch {epoch+1} — Max gradient: {max_grad_epoch:.2f}")
        torch.cuda.empty_cache()
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()
        # Log FoggyAwareNorm alpha (first stem)
        if hasattr(self.model.backbone, 'stem_conv1') and hasattr(self.model.backbone.stem_conv1[1], 'alpha'):
            fan_alpha = self.model.backbone.stem_conv1[1].alpha.mean().item()
            self.writer.add_scalar('train/fan_alpha', fan_alpha, epoch)
        return {
            'loss': total_loss / n,
            'ohem': total_ohem / n,
            'dice': total_dice / n,
        }


    # ---------------------------------------------------------------------- #
    # Validation                                                               #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss = 0.0
        num_classes = self.args.num_classes
        ignore_idx = getattr(self.args, "ignore_index", 255)

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        pbar = tqdm(loader, desc=f"Validation Epoch {epoch}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                # In eval mode, model returns only main logit
                logits = self.model(imgs)
                if isinstance(logits, tuple):
                    logits = logits[0]  # safety
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss_ce = self.ohem(logits, masks)
                if self.dice_weight > 0:
                    loss_dice = self.dice(logits, masks)
                else:
                    loss_dice = torch.tensor(0.0, device=self.device)
                loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

            total_loss += loss.item()

            pred = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            valid = (target != ignore_idx) & (target < num_classes)
            label = num_classes * target[valid].astype(int) + pred[valid]
            count = np.bincount(label, minlength=num_classes ** 2)
            conf_matrix += count.reshape(num_classes, num_classes)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        intersection = np.diag(conf_matrix)
        union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        acc = intersection.sum() / (conf_matrix.sum() + 1e-10)

        return {
            'loss': total_loss / len(loader),
            'miou': miou,
            'accuracy': acc,
            'per_class_iou': iou,
        }
    # ---------------------------------------------------------------------- #
    # Checkpoint                                                               #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, epoch, metrics, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'best_miou': self.best_miou,
            'metrics': metrics,
            'global_step': self.global_step,
        }
        torch.save(ckpt, self.save_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pth")
            print(f"Best model saved! mIoU: {metrics['miou']:.4f}")
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(ckpt, self.save_dir / f"epoch_{epoch+1}.pth")


    def load_checkpoint(self, path, reset_epoch=True, load_optimizer=True, reset_best_metric=False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        if load_optimizer and ckpt.get('optimizer'):
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError as e:
                print(f"Optimizer state not loaded: {e}")
        if load_optimizer and 'scaler' in ckpt and ckpt['scaler']:
            try:
                self.scaler.load_state_dict(ckpt['scaler'])
            except Exception as e:
                print(f"Scaler state not loaded: {e}")
        if reset_epoch:
            self.start_epoch = 0
            self.global_step = 0
            self.best_miou = 0.0 if reset_best_metric else ckpt.get('best_miou', 0.0)
            print(f"Weights loaded (epoch {ckpt['epoch']}), starting from epoch 0")
        else:
            self.start_epoch = ckpt['epoch'] + 1
            self.best_miou = ckpt.get('best_miou', 0.0)
            self.global_step = ckpt.get('global_step', 0)
            if self.scheduler and ckpt.get('scheduler') and load_optimizer:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                except Exception as e:
                    print(f"Scheduler state not loaded: {e}")
            print(f"Checkpoint loaded, resuming from epoch {self.start_epoch}")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 Training (FAN + MSC)")

    # Transfer learning
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--unfreeze_schedule", type=str, default="",
                        help="Comma-separated epochs to progressively unfreeze backbone")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--dwsa_lr_factor", type=float, default=0.5,
                        help="LR factor for MultiScaleContext alpha (msc_lr)")
    parser.add_argument("--alpha_lr_factor", type=float, default=0.1,
                        help="LR factor for FoggyAwareNorm alpha (fan_lr)")
    parser.add_argument("--use_class_weights", action="store_true")

    # Dataset
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--dataset_type", default="foggy", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--aux_weight", type=float, default=0.4)
    parser.add_argument("--scheduler", default="onecycle",
                        choices=["onecycle", "poly", "cosine"])
    parser.add_argument("--freeze_epochs", type=int, default=0)
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3)

    # Image
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
    parser.add_argument("--reset_best_metric", action="store_true")

    args = parser.parse_args()
    if args.freeze_epochs >= args.epochs:
        raise ValueError("freeze_epochs must be < total epochs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"GCNet v3 Training  |  FoggyAwareNorm + MultiScaleContext (no DWSA)")
    print(f"{'='*70}")
    print(f"Device:     {device}")
    print(f"Image size: {args.img_h}x{args.img_w}")
    print(f"Epochs:     {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip:  {args.grad_clip}  |  AMP: {args.use_amp}")
    print(f"LR MSC:     {args.lr * args.dwsa_lr_factor:.2e}  (factor={args.dwsa_lr_factor})")
    print(f"LR FAN:     {args.lr * args.alpha_lr_factor:.2e}  (factor={args.alpha_lr_factor})")
    print(f"{'='*70}\n")

    cfg = ModelConfig.get_config()
    args.loss_config = cfg["loss"]

    print("Creating dataloaders...")
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=args.use_class_weights,
        dataset_type=args.dataset_type,
    )
    print("Dataloaders ready\n")

    print(f"{'='*70}")
    print("BUILDING MODEL")
    print(f"{'='*70}\n")

    model = GCNetSegmentor(
        in_channels=3,
        channels=cfg["channels"],
        ppm_channels=cfg["ppm_channels"],
        num_blocks_per_stage=cfg["num_blocks_per_stage"],
        num_classes=args.num_classes,
        decoder_channels=cfg["decoder_channels"],
        dropout_ratio=cfg["dropout_ratio"],
        align_corners=cfg["align_corners"],
        norm_cfg=cfg["norm_cfg"],
        act_cfg=cfg["act_cfg"],
        fan_eps=cfg["fan_eps"],
        fan_momentum=cfg["fan_momentum"],
        ms_scales=cfg["ms_scales"],
        ms_branch_ratio=cfg["ms_branch_ratio"],
        ms_alpha=cfg["ms_alpha"],
        deploy=False,
    ).to(device)


    print("Applying init_weights...")
    model.apply(init_weights)
    check_model_health(model)
    print()

    print(f"{'='*70}")
    print("TRANSFER LEARNING SETUP")
    print(f"{'='*70}\n")

    if args.pretrained_weights:
        load_pretrained_gcnet(model, args.pretrained_weights)

    if args.freeze_backbone:
        freeze_backbone(model)

    count_trainable_params(model)
    print_backbone_structure(model)

    with torch.no_grad():
        sample = torch.randn(2, 3, args.img_h, args.img_w).to(device)
        try:
            out = model.forward_train(sample)
            c4_logit = aux_logit
            c6_logit = main_logit
            print(f"Forward pass OK:")
            print(f"  c4_logit: {c4_logit.shape}")
            print(f"  c6_logit: {c6_logit.shape}\n")
        except Exception as e:
            print(f"Forward pass FAILED: {e}")
            return

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=0)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        class_weights=class_weights if args.use_class_weights else None,
    )

    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=(args.resume_mode == "transfer"),
            load_optimizer=(args.resume_mode == "continue"),
            reset_best_metric=args.reset_best_metric,
        )

    unfreeze_epochs = []
    if args.unfreeze_schedule:
        try:
            unfreeze_epochs = sorted(int(e) for e in args.unfreeze_schedule.split(','))
        except Exception:
            raise ValueError("unfreeze_schedule phải là chuỗi số nguyên cách nhau bởi dấu phẩy")

    UNFREEZE_STAGES = [
        ['stem_conv1', 'stem_conv2'],                     # FoggyAwareNorm
        ['stem_stage2', 'stem_stage3'],                   # early stages
        ['semantic_branch.0', 'detail_branch.0', 'compression_1', 'down_1'],
        ['semantic_branch.1', 'detail_branch.1', 'compression_2', 'down_2'],
        ['semantic_branch.2', 'detail_branch.2', 'spp', 'ms_context', 'final_proj'],
    ]
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(trainer.start_epoch, args.epochs):
        # Progressive unfreezing
        past = [e for e in unfreeze_epochs if e <= epoch]
        k = len(past)
        targets = []
        for i in range(min(k, len(UNFREEZE_STAGES))):
            targets += UNFREEZE_STAGES[i]
        if targets:
            unfreeze_backbone_progressive(model, targets)

        # Rebuild optimizer when unfreezing
        if epoch in unfreeze_epochs:
            trainer.set_loss_phase('full')
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=epoch)
            trainer.optimizer = optimizer
            trainer.scheduler = scheduler
            print("Learning rates after unfreezing:")
            for g in optimizer.param_groups:
                print(f"  {g.get('name','?')}: {g['lr']:.2e}")
            print()

        # Loss phase adjustment
        if unfreeze_epochs:
            last_unfreeze = max((e for e in unfreeze_epochs if e <= epoch), default=None)
            if last_unfreeze is not None:
                if epoch >= last_unfreeze + args.ce_only_epochs_after_unfreeze:
                    trainer.set_loss_phase('full')
                elif epoch == last_unfreeze:
                    trainer.set_loss_phase('ce_only')

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader, epoch)

        # Log MSC alpha if present
        log_msc_alpha(model, trainer.writer, epoch)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | OHEM: {train_metrics['ohem']:.4f} | Dice: {train_metrics['dice']:.4f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | mIoU: {val_metrics['miou']:.4f}  | Acc: {val_metrics['accuracy']:.4f}")
        # Log FAN alpha
        if hasattr(model.backbone, 'stem_conv1') and hasattr(model.backbone.stem_conv1[1], 'alpha'):
            fan_alpha = model.backbone.stem_conv1[1].alpha.mean().item()
            print(f"  FAN alpha (stem_conv1): {fan_alpha:.4f}")
        if hasattr(model.backbone, 'ms_context') and hasattr(model.backbone.ms_context, 'alpha'):
            msc_alpha = model.backbone.ms_context.alpha.item()
            print(f"  MSC alpha: {msc_alpha:.4f}")
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
