# ============================================
# train_v3.py — GCNetBackbone (FAN+MSC) + GCNetHeadLite
# Decoder dừng tại /2, không có activation tại full res
# Pretrained loader tương thích checkpoint GCNet gốc
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
warnings.filterwarnings('ignore')

# ============================================
# IMPORTS
# ============================================

from model.backbone.model import GCNet        
from model.head.segmentation_head import GCNetHeadLite    
from data.custom import create_dataloaders
from model.model_utils import init_weights, check_model_health


# ============================================
# WRAPPER
# ============================================

class GCNetModel(nn.Module):
    """Wrapper: GCNetBackbone (FAN + MSC) + GCNetHeadLite.

    Train : forward(x) → (aux_logit/2, main_logit/2)
    Eval  : forward(x) → main_logit/2
    """

    def __init__(self, backbone_cfg: dict, head_cfg: dict):
        super().__init__()
        self.backbone = GCNetBackbone(**backbone_cfg)
        self.head     = GCNetHeadLite(**head_cfg)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)   # train→(c4,c6,c1,c2) | eval→(c6,c1,c2)
        # GCNetHeadLite chỉ cần (c4, c6) khi train, c6 khi eval
        # Tách đúng theo mode
        if self.training:
            c4_feat, c6_feat, c1, c2 = feats
            return self.head((c4_feat, c6_feat))
        else:
            c6_feat, c1, c2 = feats
            return self.head(c6_feat)

    def switch_to_deploy(self):
        self.backbone.switch_to_deploy()

    def get_param_groups(self,
                         head_lr: float = 5e-4,
                         backbone_lr: float = 5e-5,
                         fan_lr: float = 5e-5,
                         msc_lr: float = 1e-4) -> list:
        fan_params = (list(self.backbone.stem_conv1.parameters())
                      + list(self.backbone.stem_conv2.parameters()))
        fan_ids = {id(p) for p in fan_params}

        msc_params = (list(self.backbone.ms_context.parameters())
                      + list(self.backbone.final_proj.parameters()))
        msc_ids = {id(p) for p in msc_params}

        head_params = list(self.head.parameters())
        head_ids    = {id(p) for p in head_params}

        bb_params = [p for p in self.backbone.parameters()
                     if id(p) not in fan_ids and id(p) not in msc_ids]

        return [
            {'params': head_params, 'lr': head_lr,     'name': 'head'},
            {'params': bb_params,   'lr': backbone_lr, 'name': 'backbone'},
            {'params': fan_params,  'lr': fan_lr,      'name': 'fan'},
            {'params': msc_params,  'lr': msc_lr,      'name': 'msc'},
        ]


# ============================================
# PRETRAINED WEIGHT LOADER — tương thích checkpoint GCNet gốc
# ============================================

# Mapping: key_prefix_trong_ckpt → key_prefix_trong_backbone_mới
#
# Checkpoint gốc dùng:
#   backbone.stem.0/1           → ConvModule (có .conv + .bn)
#   backbone.stem.2..9          → GCBlock stage2 (4 blocks) + stage3 (1+3 blocks)
#   backbone.semantic_branch_layers.N  → semantic_branch[N]
#   backbone.detail_branch_layers.N    → detail_branch[N]
#   backbone.compression_1/2, down_1/2, spp → giữ nguyên tên
#
# Backbone mới dùng:
#   stem_conv1 (Conv2d + FAN + ReLU) — KHÔNG có .conv/.bn (naked Conv2d)
#   stem_conv2 (Conv2d + FAN + ReLU) — KHÔNG có .conv/.bn
#   stem_stage2 (GCBlock x4)
#   stem_stage3 (GCBlock x4: 1 stride=2 + 3 stride=1)
#   semantic_branch[N]
#   detail_branch[N]

def _remap_ckpt_key(key: str) -> str | None:
    """Chuyển key của checkpoint gốc sang key của backbone mới.

    Trả về None nếu key không thể map (ví dụ: decode_head, stem conv/bn).
    """
    # Bỏ decode_head hoàn toàn
    if key.startswith('decode_head.'):
        return None

    # Bỏ prefix 'backbone.' để xử lý phần còn lại
    if not key.startswith('backbone.'):
        return None
    rest = key[len('backbone.'):]  # phần sau 'backbone.'

    # --- stem.0 / stem.1 : ConvModule → KHÔNG map được vào FAN ---
    # stem.0.conv.weight → stem_conv1[0].weight  (Conv2d naked, key = '0.weight')
    # stem.0.bn.*        → SKIP (FAN thay BN, không tương thích shape-wise)
    # stem.1.*           → tương tự cho stem_conv2
    if rest.startswith('stem.0.'):
        sub = rest[len('stem.0.'):]       # e.g. 'conv.weight', 'bn.weight'
        if sub == 'conv.weight':
            return 'stem_conv1.0.weight'  # Conv2d(in,C,3,stride=2,...).weight
        # bn.* → không map vào FAN (khác cấu trúc)
        return None

    if rest.startswith('stem.1.'):
        sub = rest[len('stem.1.'):]
        if sub == 'conv.weight':
            return 'stem_conv2.0.weight'
        return None

    # --- stem.2 ~ stem.5 : GCBlock stage2 (index 0..3 trong stem_stage2) ---
    # stem.2 → stem_stage2.0, stem.3 → stem_stage2.1, ...
    for stem_idx in range(2, 6):
        prefix = f'stem.{stem_idx}.'
        if rest.startswith(prefix):
            stage2_idx = stem_idx - 2   # 0..3
            return f'stem_stage2.{stage2_idx}.' + rest[len(prefix):]

    # --- stem.6 : GCBlock stage3[0] (stride=2, C→C*2) ---
    if rest.startswith('stem.6.'):
        return 'stem_stage3.0.' + rest[len('stem.6.'):]

    # --- stem.7 ~ stem.9 : GCBlock stage3[1..3] (stride=1) ---
    for stem_idx in range(7, 10):
        prefix = f'stem.{stem_idx}.'
        if rest.startswith(prefix):
            stage3_idx = stem_idx - 6   # 1..3
            return f'stem_stage3.{stage3_idx}.' + rest[len(prefix):]

    # --- semantic_branch_layers → semantic_branch ---
    if rest.startswith('semantic_branch_layers.'):
        return 'semantic_branch.' + rest[len('semantic_branch_layers.'):]

    # --- detail_branch_layers → detail_branch ---
    if rest.startswith('detail_branch_layers.'):
        return 'detail_branch.' + rest[len('detail_branch_layers.'):]

    # --- compression_1/2, down_1/2, spp : giữ nguyên ---
    for name in ('compression_1.', 'compression_2.',
                 'down_1.', 'down_2.', 'spp.'):
        if rest.startswith(name):
            return rest   # key trong backbone mới không có prefix 'backbone.'

    return None


def load_pretrained_gcnet(model: GCNetModel, ckpt_path: str) -> float:
    """Load pretrained weights từ checkpoint GCNet gốc vào GCNetModel mới.

    Mapping được thực hiện qua _remap_ckpt_key():
      - stem conv weights (không có BN/FAN) được load vào Conv2d đầu tiên của stem_conv1/2
      - GCBlock stage2/3 được map sang stem_stage2/3
      - semantic/detail branch được map sang tên mới
      - ms_context, final_proj, FAN alpha, head: khởi tạo random (không có trong ckpt cũ)
      - decode_head: bỏ hoàn toàn

    Returns:
        float: tỉ lệ phần trăm keys được load thành công
    """
    print(f"\n{'='*70}")
    print("TRANSFER LEARNING — GCNet gốc → GCNetBackbone (FAN + MSC)")
    print(f"{'='*70}")
    print(f"Checkpoint: {ckpt_path}\n")

    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))

    backbone_state = model.backbone.state_dict()

    compatible: dict = {}
    skipped_decode  = 0
    skipped_no_map  = 0
    skipped_shape   = []
    loaded_keys     = []

    for ckpt_key, ckpt_val in state.items():
        # Thử remap
        new_key = _remap_ckpt_key(ckpt_key)

        if new_key is None:
            if ckpt_key.startswith('decode_head.'):
                skipped_decode += 1
            else:
                skipped_no_map += 1
            continue

        if new_key not in backbone_state:
            skipped_no_map += 1
            continue

        if backbone_state[new_key].shape != ckpt_val.shape:
            skipped_shape.append(
                f"  {ckpt_key} → {new_key}: "
                f"ckpt {tuple(ckpt_val.shape)} vs model {tuple(backbone_state[new_key].shape)}"
            )
            continue

        compatible[new_key] = ckpt_val
        loaded_keys.append(new_key)

    total   = len(backbone_state)
    loaded  = len(compatible)
    rate    = 100.0 * loaded / total if total > 0 else 0.0

    print(f"Checkpoint keys total : {len(state)}")
    print(f"  decode_head skipped : {skipped_decode}")
    print(f"  no mapping / not in model: {skipped_no_map}")
    print(f"  shape mismatch      : {len(skipped_shape)}")
    print(f"  loaded successfully : {loaded}")
    print(f"\nBackbone keys total   : {total}")
    print(f"  Loaded              : {loaded} ({rate:.1f}%)")
    print(f"  Random init         : {total - loaded} (FAN alpha, MSC, final_proj, ...)")

    if skipped_shape:
        print(f"\nShape mismatches:")
        for s in skipped_shape[:10]:
            print(s)

    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)

    # Phân loại missing keys
    fan_missing  = [k for k in missing if 'stem_conv' in k]
    msc_missing  = [k for k in missing if 'ms_context' in k or 'final_proj' in k]
    gcb_missing  = [k for k in missing if k not in fan_missing and k not in msc_missing]

    print(f"\nMissing keys breakdown:")
    print(f"  FAN (alpha, IN, BN new params) : {len(fan_missing)}")
    print(f"  MSC + final_proj (new modules) : {len(msc_missing)}")
    print(f"  GCBlock / branch (unmatched)   : {len(gcb_missing)}")
    if gcb_missing:
        print(f"  First 5 unmatched GCBlock keys:")
        for k in gcb_missing[:5]:
            print(f"    {k}")

    print(f"\n{'='*70}")
    print(f"Load rate: {rate:.1f}%  |  New modules will train from scratch")
    print(f"{'='*70}\n")

    return rate


# ============================================
# FREEZE / UNFREEZE
# ============================================

def freeze_backbone(model: GCNetModel):
    """Freeze backbone, giữ FAN + MSC trainable."""
    print("Freezing backbone (keeping FAN + MSC trainable)...")
    for p in model.backbone.parameters():
        p.requires_grad = False

    # FAN: stem_conv1, stem_conv2
    fan_params = 0
    for mod in (model.backbone.stem_conv1, model.backbone.stem_conv2):
        for p in mod.parameters():
            p.requires_grad = True
            fan_params += p.numel()
        for m in mod.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.train()
    print(f"  FAN trainable: {fan_params:,} params")

    # MSC + final_proj
    msc_params = 0
    for mod in (model.backbone.ms_context, model.backbone.final_proj):
        for p in mod.parameters():
            p.requires_grad = True
            msc_params += p.numel()
    print(f"  MSC + final_proj trainable: {msc_params:,} params\n")


def unfreeze_backbone_progressive(model: GCNetModel, stage_names):
    if isinstance(stage_names, str):
        stage_names = [stage_names]
    total = 0
    for name in stage_names:
        module = None
        for part in name.split('.'):
            obj = model.backbone if module is None else module
            module = getattr(obj, part, None)
            if module is None:
                break
        if module is None:
            print(f"  [skip] '{name}' not found in backbone")
            continue
        count = sum(1 for p in module.parameters() if not p.requires_grad)
        for p in module.parameters():
            p.requires_grad = True
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.train()
        total += count
        if count:
            print(f"  Unfrozen: backbone.{name} ({count:,} params)")
    print(f"  Total unfrozen: {total:,} params\n")
    return total


# ============================================
# OPTIMIZER / SCHEDULER
# ============================================

def build_optimizer(model: GCNetModel, args) -> optim.AdamW:
    param_groups = model.get_param_groups(
        head_lr=args.lr,
        backbone_lr=args.lr * args.backbone_lr_factor,
        fan_lr=args.lr * args.alpha_lr_factor,
        msc_lr=args.lr * args.msc_lr_factor,
    )
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])

    print("Optimizer: AdamW (Discriminative LR)")
    for g in optimizer.param_groups:
        print(f"  group '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")
    return optimizer


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * (args.epochs - start_epoch)
        max_lrs = [g['initial_lr'] for g in optimizer.param_groups]
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_steps,
            pct_start=0.05, anneal_strategy='cos',
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
            div_factor=25, final_div_factor=100000,
        )
        print(f"OneCycleLR (total_steps={total_steps})")
    elif args.scheduler == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: (1 - e / args.epochs) ** 0.9)
        print("Polynomial LR")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
        print("CosineAnnealingLR")
    return scheduler


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
        valid = (targets != self.ignore_index)
        t_clamp = targets.clamp(0, C - 1)
        t_oh = F.one_hot(t_clamp, C).permute(0, 3, 1, 2).float()
        t_oh = t_oh * valid.unsqueeze(1).float()
        probs = F.softmax(logits, dim=1) * valid.unsqueeze(1).float()
        pf = probs.reshape(B, C, -1)
        tf = t_oh.reshape(B, C, -1)
        inter = (pf * tf).sum(2)
        card  = pf.sum(2) + tf.sum(2)
        dice  = 1.0 - (2.0 * inter + self.smooth) / (card + self.smooth)
        present = tf.sum(2) > 0
        dice = dice * present.float()
        n = present.float().sum(1).clamp(min=1)
        return (dice.sum(1) / n).mean()


class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, keep_ratio=0.3, min_kept=100000,
                 class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.keep_ratio   = keep_ratio
        self.min_kept     = min_kept
        self.class_weights = class_weights

    def forward(self, logits, labels):
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        pixel_loss = F.cross_entropy(
            logits, labels, weight=w,
            ignore_index=self.ignore_index, reduction='none').view(-1)
        valid   = labels.view(-1) != self.ignore_index
        losses  = pixel_loss[valid]
        n = losses.numel()
        if n == 0:
            return logits.sum() * 0
        n_keep = max(int(self.keep_ratio * n), min(self.min_kept, n))
        n_keep = min(n_keep, n)
        if n_keep < n:
            thr    = torch.sort(losses, descending=True)[0][n_keep - 1].detach()
            losses = losses[losses >= thr]
        return losses.mean()


# ============================================
# UTILITIES
# ============================================

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


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
    if max_grad > threshold:
        print(f"Large gradient: {max_name[:60]}... = {max_grad:.2f}")
    return max_grad, total_sq ** 0.5


def count_trainable_params(model: GCNetModel):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_total  = sum(p.numel() for p in model.backbone.parameters())
    bb_train  = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    hd_total  = sum(p.numel() for p in model.head.parameters())
    hd_train  = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
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


# ============================================
# CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config(channels: int = 32):
        C = channels
        return {
            "backbone": dict(
                in_channels=3,
                channels=C,
                ppm_channels=128,
                num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
                align_corners=False,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                fan_eps=1e-5,
                fan_momentum=0.1,
                ms_scales=(1, 2),
                ms_branch_ratio=8,
                ms_alpha=0.1,
                deploy=False,
            ),
            "head": dict(
                in_channels=C * 4,       # c6_feat = channels*4
                channels=C,
                num_classes=19,
                decoder_channels=96,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                align_corners=False,
                ignore_index=255,
                loss_weight_aux=0.4,
                dropout_ratio=0.1,
            ),
            "loss": dict(
                ce_weight=1.0,
                dice_weight=0.5,
                dice_smooth=1e-5,
            ),
        }


# ============================================
# TRAINER
# ============================================

class Trainer:
    def __init__(self, model: GCNetModel, optimizer, scheduler, device, args,
                 class_weights=None):
        self.model       = model.to(device)
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.args        = args
        self.best_miou   = 0.0
        self.start_epoch = 0
        self.global_step = 0

        loss_cfg = args.loss_config
        self.ce_weight        = loss_cfg['ce_weight']
        self.dice_weight      = loss_cfg['dice_weight']
        self.base_dice_weight = loss_cfg['dice_weight']
        self.loss_phase       = 'full'

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

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=self.save_dir / "tensorboard")

        self._save_config()
        self._print_config(loss_cfg)

    def set_loss_phase(self, phase: str):
        if phase == self.loss_phase:
            return
        self.dice_weight = 0.0 if phase == 'ce_only' else self.base_dice_weight
        self.loss_phase  = phase
        print(f"Loss phase → {phase}  (CE={self.ce_weight}, Dice={self.dice_weight})")

    def _save_config(self):
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

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
        print(f"Decoder stops at /2, resize in loss loop")
        print(f"{'='*70}\n")

    # ---------------------------------------------------------------------- #
    # Train epoch                                                              #
    # ---------------------------------------------------------------------- #

    def train_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        total_loss = total_ohem = total_dice = 0.0
        max_grad_epoch = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            target_size = masks.shape[-2:]

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                aux_logit, main_logit = self.model(imgs)   # both at /2

                # Resize /2 → full res NGOÀI backward graph decoder
                main_full = F.interpolate(main_logit, size=target_size,
                                          mode='bilinear', align_corners=False)
                aux_full  = F.interpolate(aux_logit,  size=target_size,
                                          mode='bilinear', align_corners=False)

                ohem_loss = self.ohem(main_full, masks)
                dice_loss = (self.dice(main_full, masks)
                             if self.dice_weight > 0
                             else torch.tensor(0.0, device=self.device))
                loss = self.ce_weight * ohem_loss + self.dice_weight * dice_loss

                if self.args.aux_weight > 0:
                    aux_w    = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    aux_loss = self.ohem(aux_full, masks)
                    loss     = loss + aux_w * aux_loss

                loss = loss / self.args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nNaN/Inf at epoch {epoch} batch {batch_idx} — skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                max_grad, _ = check_gradients(self.model)
                max_grad_epoch = max(max_grad_epoch, max_grad)
                if self.args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip)
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
                'lr':   f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            })

            if batch_idx % 50 == 0:
                clear_gpu_memory()

            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss',
                    loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ohem', ohem_loss.item(), self.global_step)
                self.writer.add_scalar('train/dice', dice_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr',
                    self.optimizer.param_groups[0]['lr'], self.global_step)

        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        # Log FAN alpha
        try:
            fan_alpha = self.model.backbone.stem_conv1[1].alpha.mean().item()
            self.writer.add_scalar('monitor/fan_alpha', fan_alpha, epoch)
            print(f"  FAN alpha (stem_conv1): {fan_alpha:.4f}")
        except Exception:
            pass

        # Log MSC alpha
        try:
            msc_alpha = self.model.backbone.ms_context.alpha.item()
            self.writer.add_scalar('monitor/msc_alpha', msc_alpha, epoch)
            print(f"  MSC alpha:              {msc_alpha:.4f}")
        except Exception:
            pass

        n = len(loader)
        print(f"Epoch {epoch+1} — max_grad: {max_grad_epoch:.2f}")
        clear_gpu_memory()
        return dict(loss=total_loss/n, ohem=total_ohem/n, dice=total_dice/n)

    # ---------------------------------------------------------------------- #
    # Validate                                                                 #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> dict:
        self.model.eval()
        total_loss  = 0.0
        num_classes = self.args.num_classes
        ignore_idx  = self.args.ignore_index
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        pbar = tqdm(loader, desc=f"Val {epoch+1}")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits = self.model(imgs)
                if isinstance(logits, tuple):
                    logits = logits[-1]
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode='bilinear', align_corners=False)
                loss = self.ohem(logits, masks)

            total_loss += loss.item()

            pred   = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            valid  = (target != ignore_idx) & (target < num_classes)
            label  = num_classes * target[valid].astype(int) + pred[valid]
            count  = np.bincount(label, minlength=num_classes ** 2)
            conf_matrix += count.reshape(num_classes, num_classes)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        inter = np.diag(conf_matrix)
        union = conf_matrix.sum(1) + conf_matrix.sum(0) - inter
        iou   = inter / (union + 1e-10)
        miou  = float(np.nanmean(iou))
        acc   = float(inter.sum() / (conf_matrix.sum() + 1e-10))

        return dict(loss=total_loss/len(loader), miou=miou,
                    accuracy=acc, per_class_iou=iou)

    # ---------------------------------------------------------------------- #
    # Checkpoint                                                               #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, epoch: int, metrics: dict, is_best=False):
        ckpt = dict(
            epoch=epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
            scaler=self.scaler.state_dict(),
            best_miou=self.best_miou,
            metrics=metrics,
            global_step=self.global_step,
        )
        torch.save(ckpt, self.save_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pth")
            print(f"Best model saved! mIoU: {metrics['miou']:.4f}")
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(ckpt, self.save_dir / f"epoch_{epoch+1}.pth")

    def load_checkpoint(self, path: str, reset_epoch=True,
                        load_optimizer=True, reset_best_metric=False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        if load_optimizer and ckpt.get('optimizer'):
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError as e:
                print(f"Optimizer not loaded: {e}")
        if load_optimizer and 'scaler' in ckpt and ckpt['scaler']:
            try:
                self.scaler.load_state_dict(ckpt['scaler'])
            except Exception as e:
                print(f"Scaler not loaded: {e}")
        if reset_epoch:
            self.start_epoch = 0
            self.global_step = 0
            self.best_miou   = 0.0 if reset_best_metric else ckpt.get('best_miou', 0.0)
        else:
            self.start_epoch = ckpt['epoch'] + 1
            self.best_miou   = ckpt.get('best_miou', 0.0)
            self.global_step = ckpt.get('global_step', 0)
            if self.scheduler and ckpt.get('scheduler') and load_optimizer:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                except Exception as e:
                    print(f"Scheduler not loaded: {e}")
        mode = "transfer" if reset_epoch else "resume"
        print(f"Checkpoint loaded ({mode}), starting epoch {self.start_epoch}")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="GCNet Training — FAN + MSC + GCNetHeadLite")

    # Pretrained / Transfer
    parser.add_argument("--pretrained_weights",  type=str, default=None)
    parser.add_argument("--freeze_backbone",     action="store_true", default=False)
    parser.add_argument("--unfreeze_schedule",   type=str, default="",
                        help="Comma-separated epochs, e.g. '5,10,15,20'")
    parser.add_argument("--backbone_lr_factor",  type=float, default=0.1)
    parser.add_argument("--msc_lr_factor",       type=float, default=0.2)
    parser.add_argument("--alpha_lr_factor",     type=float, default=0.1)
    parser.add_argument("--use_class_weights",   action="store_true")

    # Dataset
    parser.add_argument("--train_txt",    required=True)
    parser.add_argument("--val_txt",      required=True)
    parser.add_argument("--dataset_type", default="foggy",
                        choices=["normal", "foggy"])
    parser.add_argument("--num_classes",  type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)

    # Training
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--accumulation_steps",type=int,   default=5)
    parser.add_argument("--lr",                type=float, default=5e-4)
    parser.add_argument("--weight_decay",      type=float, default=1e-4)
    parser.add_argument("--grad_clip",         type=float, default=5.0)
    parser.add_argument("--aux_weight",        type=float, default=0.4)
    parser.add_argument("--scheduler",         default="onecycle",
                        choices=["onecycle", "poly", "cosine"])
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3)

    # Image
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)

    # System
    parser.add_argument("--use_amp",       action="store_true", default=True)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--save_dir",      default="./checkpoints")
    parser.add_argument("--resume",        type=str, default=None)
    parser.add_argument("--resume_mode",   type=str, default="transfer",
                        choices=["transfer", "continue"])
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--log_interval",  type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--reset_best_metric", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print("GCNet Training v3  |  FAN + MSC + GCNetHeadLite (decoder /2)")
    print(f"{'='*70}")
    print(f"Device: {device}  |  Image: {args.img_h}x{args.img_w}")
    print(f"Batch: {args.batch_size} × accum {args.accumulation_steps} "
          f"= effective {args.batch_size*args.accumulation_steps}")
    print(f"{'='*70}\n")

    # Config
    cfg = ModelConfig.get_config(channels=32)
    cfg['head']['num_classes']  = args.num_classes
    cfg['head']['ignore_index'] = args.ignore_index
    args.loss_config = cfg['loss']

    # Data
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

    # Model
    print(f"{'='*70}")
    print("BUILDING MODEL — GCNetBackbone (FAN+MSC) + GCNetHeadLite")
    print(f"{'='*70}\n")

    model = GCNetModel(
        backbone_cfg=cfg['backbone'],
        head_cfg=cfg['head'],
    ).to(device)

    model.apply(init_weights)
    check_model_health(model)

    # Transfer learning từ checkpoint GCNet gốc
    if args.pretrained_weights:
        load_pretrained_gcnet(model, args.pretrained_weights)

    if args.freeze_backbone:
        freeze_backbone(model)

    count_trainable_params(model)

    # Sanity check forward pass
    with torch.no_grad():
        sample = torch.randn(2, 3, args.img_h, args.img_w).to(device)
        model.train()
        try:
            aux_logit, main_logit = model(sample)
            h2, w2 = args.img_h // 2, args.img_w // 2
            print(f"Forward pass (train) OK:")
            print(f"  aux_logit:  {tuple(aux_logit.shape)}  (expected {h2}x{w2})")
            print(f"  main_logit: {tuple(main_logit.shape)} (expected {h2}x{w2})\n")
        except Exception as e:
            print(f"Forward pass FAILED: {e}")
            raise
        model.eval()
        try:
            logit = model(sample)
            if isinstance(logit, tuple):
                logit = logit[-1]
            print(f"Forward pass (eval) OK: {tuple(logit.shape)}\n")
        except Exception as e:
            print(f"Eval forward FAILED: {e}")
            raise

    # Optimizer / scheduler
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

    # Progressive unfreeze schedule
    unfreeze_epochs = []
    if args.unfreeze_schedule:
        try:
            unfreeze_epochs = sorted(
                int(e) for e in args.unfreeze_schedule.split(','))
        except Exception:
            raise ValueError("unfreeze_schedule: chuỗi số nguyên, vd '5,10,15'")

    # Tên modules trong GCNetBackbone mới
    UNFREEZE_STAGES = [
        ['stem_conv1', 'stem_conv2'],           # stage 1 (FAN)
        ['stem_stage2', 'stem_stage3'],          # stage 2-3
        ['semantic_branch.0', 'detail_branch.0',
         'compression_1', 'down_1'],             # stage 4
        ['semantic_branch.1', 'detail_branch.1',
         'compression_2', 'down_2'],             # stage 5
        ['semantic_branch.2', 'detail_branch.2',
         'spp', 'ms_context', 'final_proj'],     # stage 6 + MSC
    ]

    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(trainer.start_epoch, args.epochs):

        # Progressive unfreezing — tích lũy tất cả stages đến epoch hiện tại
        past    = [e for e in unfreeze_epochs if e <= epoch]
        targets = []
        for i in range(min(len(past), len(UNFREEZE_STAGES))):
            targets += UNFREEZE_STAGES[i]
        if targets:
            unfreeze_backbone_progressive(model, targets)

        # Rebuild optimizer khi unfreeze
        if epoch in unfreeze_epochs:
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(
                optimizer, args, train_loader, start_epoch=epoch)
            trainer.optimizer = optimizer
            trainer.scheduler = scheduler

        # Loss phase: ce_only ngay sau unfreeze, full sau vài epoch
        if unfreeze_epochs:
            last_uf = max((e for e in unfreeze_epochs if e <= epoch), default=None)
            if last_uf is not None:
                if epoch == last_uf:
                    trainer.set_loss_phase('ce_only')
                elif epoch >= last_uf + args.ce_only_epochs_after_unfreeze:
                    trainer.set_loss_phase('full')

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics   = trainer.validate(val_loader, epoch)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | "
              f"OHEM: {train_metrics['ohem']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | "
              f"mIoU: {val_metrics['miou']:.4f}  | "
              f"Acc:  {val_metrics['accuracy']:.4f}")
        print(f"{'='*70}\n")

        trainer.writer.add_scalar('val/loss',     val_metrics['loss'],     epoch)
        trainer.writer.add_scalar('val/miou',     val_metrics['miou'],     epoch)
        trainer.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)

        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)

    trainer.writer.close()
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED!  Best mIoU: {trainer.best_miou:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
