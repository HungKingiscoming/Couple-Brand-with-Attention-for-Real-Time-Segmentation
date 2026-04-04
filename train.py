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

from model.backbone.model import GCNet          # backbone mới
from model.head.segmentation_head import GCNetHead      # head mới (tích hợp aux)
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# PRETRAINED WEIGHT LOADER
# ============================================

def _remap_stem_key(key: str, N2: int = 4):
    """Remap checkpoint key sang GCNet v3.

    Xử lý hai vấn đề:
    1. Strip prefix backbone./model./module. trước khi remap.
    2. Stem gốc = 1 Sequential liên tục (ConvModule),
       v3 tách thành stem_conv1/conv2 (nn.Conv2d + FoggyAwareNorm):
         stem.0.conv.weight → stem_conv1.0.weight  (bỏ tiền tố .conv.)
         stem.0.bn.*        → None  (BN gốc thay bằng FoggyAwareNorm)
         stem.1.conv.weight → stem_conv2.0.weight
         stem.2..{N2+1}.*   → stem_stage2.{0..N2-1}.*
         stem.{N2+2}...*    → stem_stage3.{0..}.*

    Returns:
        str  — key mới, hoặc
        None — key nên bị bỏ qua (BN gốc không dùng nữa)
    """
    import re as _re

    # 1. Strip prefix
    for pref in ['backbone.', 'model.', 'module.']:
        if key.startswith(pref):
            key = key[len(pref):]

    # 2. Chỉ xử lý key bắt đầu bằng 'stem.'
    m = _re.match(r'stem\.(\d+)\.(.+)$', key)
    if not m:
        return key

    idx  = int(m.group(1))
    rest = m.group(2)

    # stem.0 và stem.1 là ConvModule → conv.weight / bn.* / activate.*
    # v3 dùng nn.Conv2d trực tiếp → chỉ lấy conv.weight, bỏ bn
    def _map_convmodule(rest_str, target_prefix):
        if rest_str.startswith('conv.'):
            return f'{target_prefix}.{rest_str[len("conv."):].lstrip(".")}'
        # bn.*, activate.* → drop (FoggyAwareNorm thay thế BN, ReLU không có weight)
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
    """Load pretrained weights vào model.backbone (GCNet v3).

    Xử lý:
    1. Strip prefix backbone./model./module.
    2. Remap stem key: checkpoint gốc (stem.N.*) → v3 (stem_conv1/2/stage2/3.*)
    3. Module mới (DWSA, FoggyAwareNorm) không có trong checkpoint → init random.
    """
    print(f"Loading pretrained weights from: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    model_state = model.backbone.state_dict()
    compatible  = {}
    skipped     = []

    model_key_map = {}
    for mk in model_state.keys():
        norm = mk
        for pref in ['backbone.', 'model.', 'module.']:
            if norm.startswith(pref):
                norm = norm[len(pref):]
        model_key_map[norm] = mk

    # Tập hợp các key BN cũ bị bỏ do FoggyAwareNorm (return None từ remap)
    bn_dropped = []
    # Các prefix không thuộc backbone — checkpoint có thể chứa head/aux
    non_backbone_prefixes = ('decode_head.', 'aux_head.', 'head.')

    for ckpt_key, ckpt_val in state.items():
        # Bỏ qua key không thuộc backbone (decode_head, aux_head, ...)
        # Strip prefix backbone./model./module. trước khi kiểm tra
        stripped = ckpt_key
        for pref in ('backbone.', 'model.', 'module.'):
            if stripped.startswith(pref):
                stripped = stripped[len(pref):]
                break
        if any(stripped.startswith(p) for p in non_backbone_prefixes):
            continue  # key thuộc head — không load vào backbone

        # _remap_stem_key tự strip prefix VÀ remap
        norm_ckpt = _remap_stem_key(ckpt_key)

        # None = BN gốc của stem.0/1 đã được thay bằng FoggyAwareNorm → skip đúng ý
        if norm_ckpt is None:
            bn_dropped.append(ckpt_key)
            continue

        matched = False
        if norm_ckpt in model_key_map:
            mk = model_key_map[norm_ckpt]
            if model_state[mk].shape == ckpt_val.shape:
                compatible[mk] = ckpt_val
                matched = True

        if not matched and not strict_match:
            for norm_model, mk in model_key_map.items():
                if (norm_model.endswith(norm_ckpt) or norm_ckpt.endswith(norm_model)):
                    if model_state[mk].shape == ckpt_val.shape:
                        compatible[mk] = ckpt_val
                        matched = True
                        break

        if not matched:
            skipped.append(ckpt_key)

    loaded = len(compatible)
    total  = len(model_state)
    rate   = 100 * loaded / total if total > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Phân loại skip                                                       #
    #                                                                      #
    # "Expected skip" gồm 3 loại:                                         #
    # 1. DWSA / FoggyAwareNorm — module hoàn toàn mới                     #
    # 2. DAPPM (spp) BN shape mismatch — checkpoint dùng conv_cfg         #
    #    order=(norm,act,conv) → BN(in_channels), v3 dùng                 #
    #    order=(conv,norm,act) → BN(out_channels). Key tên giống nhau     #
    #    nhưng shape khác → không load được, init random là đúng.         #
    # 3. stem_conv1/2 BN — FoggyAwareNorm.bn là module mới, không có      #
    #    trong checkpoint gốc → expected missing.                          #
    # ------------------------------------------------------------------ #
    expected_skip_markers = (
        'dwsa_stage', 'foggy', 'alpha', 'in_.',   # module mới
        '.spp.',                                    # DAPPM BN shape mismatch
        'backbone.spp.',                            # với prefix
    )
    truly_expected = [k for k in skipped if any(s in k for s in expected_skip_markers)]
    truly_unmatched = [k for k in skipped if k not in truly_expected]

    sep = '=' * 70
    print(f"\n{sep}")
    print("WEIGHT LOADING SUMMARY")
    print(sep)
    print(f"Loaded:                  {loaded:>5} / {total} ({rate:.1f}%)")
    print(f"BN dropped (expected):   {len(bn_dropped):>5}  (stem BN → FoggyAwareNorm ✓)")
    print(f"Skipped total:           {len(skipped):>5}")
    print(f"  Expected (shape/new):  {len(truly_expected):>5}  (DWSA / FoggyNorm / DAPPM BN order ✓)")
    if truly_unmatched:
        print(f"  Unmatched:             {len(truly_unmatched):>5}  ← cần kiểm tra")
        for k in truly_unmatched[:5]:
            print(f"      {k}")
    print(sep + "\n")

    if truly_unmatched:
        print(f"WARNING: {len(truly_unmatched)} key không match — kiểm tra checkpoint format\n")

    missing, _ = model.backbone.load_state_dict(compatible, strict=False)

    # Expected missing = module mới + FoggyAwareNorm.bn (không có trong ckpt)
    # + DAPPM BN (shape mismatch → missing trong model vì không được load)
    expected_missing_markers = (
        'dwsa',    # DWSA modules
        'alpha',   # FoggyAwareNorm.alpha
        'in_.',    # FoggyAwareNorm.in_
        'foggy',   # FoggyAwareNorm class name
        '.1.bn.',  # FoggyAwareNorm.bn trong stem_conv1/2 (index [1])
        'spp.',    # DAPPM BN shape mismatch
    )
    expected_missing   = [k for k in missing if any(s in k for s in expected_missing_markers)]
    unexpected_missing = [k for k in missing if k not in expected_missing]

    if unexpected_missing:
        print(f"Unexpected missing ({len(unexpected_missing)}) — cần kiểm tra:")
        for k in unexpected_missing[:10]:
            print(f"  - {k}")
        print()
    print(f"Expected missing: {len(expected_missing)} keys (new modules + DAPPM BN order) → OK\n")
    return rate


# ============================================
# OPTIMIZER
# ============================================

def build_optimizer(model, args):
    """Phân tách params thành 3 nhóm với LR khác nhau:
      - alpha (FoggyAwareNorm gate + DWSA gamma): LR rất nhỏ
      - backbone: LR * backbone_lr_factor
      - head: LR đầy đủ
    """
    alpha_params    = []
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'gamma' in name:
            alpha_params.append(param)
        elif 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    head_lr = args.lr * getattr(args, 'head_lr_factor', 1.0)
    if head_params:
        groups.append({'params': head_params,    'lr': head_lr,   'name': 'head'})
    if backbone_params:
        groups.append({'params': backbone_params,'lr': args.lr * args.backbone_lr_factor,    'name': 'backbone'})
    if alpha_params:
        groups.append({'params': alpha_params,   'lr': args.lr * args.alpha_lr_factor,       'name': 'alpha'})

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])

    print(f"Optimizer: AdamW (Discriminative LR)")
    for g in optimizer.param_groups:
        print(f"  group '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")

    return optimizer


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    n_groups = len(optimizer.param_groups)

    if args.scheduler == 'onecycle':
        remaining_epochs = args.epochs - start_epoch
        total_steps      = len(train_loader) * remaining_epochs

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
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_total  = sum(p.numel() for p in model.backbone.parameters())
    bb_train  = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    hd_total  = sum(p.numel() for p in model.decode_head.parameters())
    hd_train  = sum(p.numel() for p in model.decode_head.parameters() if p.requires_grad)

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
    """Freeze toàn bộ backbone trừ DWSA và FoggyAwareNorm.

    DWSA (gamma) và FoggyAwareNorm (alpha) là module MỚI không có trong
    checkpoint → phải trainable ngay từ đầu để học được. Freeze chúng
    sẽ khiến DWSA hoạt động như identity (gamma=0 cố định) và
    FoggyAwareNorm không học được gate tối ưu cho foggy.
    """
    print("Freezing backbone (keeping DWSA + FoggyAwareNorm trainable)...")

    # Freeze toàn bộ trước
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Lock BN running stats
    bn_count = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None: m.weight.requires_grad = False
            if m.bias   is not None: m.bias.requires_grad   = False
            bn_count += 1
    print(f"  {bn_count} BN layers locked")

    # Luôn giữ DWSA trainable — gamma=0 lúc init, cần học từ epoch 0
    dwsa_params = 0
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        module = getattr(model.backbone, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
                dwsa_params += p.numel()
    print(f"  DWSA kept trainable: {dwsa_params:,} params")

    # Luôn giữ FoggyAwareNorm trainable — alpha cần học từ đầu
    fan_params = 0
    for name in ['stem_conv1', 'stem_conv2']:
        module = getattr(model.backbone, name, None)
        if module is not None:
            # stem_conv1 = [Conv2d, FoggyAwareNorm, ReLU] → index [1]
            if len(module) > 1 and hasattr(module[1], 'alpha'):
                for p in module[1].parameters():
                    p.requires_grad = True
                    fan_params += p.numel()
    print(f"  FoggyAwareNorm kept trainable: {fan_params:,} params")
    print("Backbone frozen\n")


def unfreeze_backbone_progressive(model, stage_names):
    """Unfreeze từng stage theo tên — hỗ trợ cả dotted names.

    Module names trong GCNet v3 (model.backbone.*):
      stem_conv1, stem_conv2, stem_stage2, stem_stage3
      semantic_branch_layers.0 / .1 / .2
      detail_branch_layers.0   / .1 / .2
      dwsa_stage4, dwsa_stage5, dwsa_stage6
      compression_1, compression_2, down_1, down_2
      spp
    """
    if isinstance(stage_names, str):
        stage_names = [stage_names]

    total_unfrozen = 0
    for stage_name in stage_names:
        module = None

        # Direct attr trên backbone
        if hasattr(model.backbone, stage_name):
            module = getattr(model.backbone, stage_name)

        # Dotted: 'semantic_branch_layers.0'
        elif '.' in stage_name:
            parts    = stage_name.split('.', 1)
            base_mod = getattr(model.backbone, parts[0], None)
            if base_mod is not None and parts[1].isdigit():
                try:
                    module = base_mod[int(parts[1])]
                except (IndexError, TypeError):
                    pass

        if module is None:
            print(f"  [skip] module '{stage_name}' not found in backbone")
            continue

        count = 0; bn_count = 0
        for p in module.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                count += 1
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if m.weight is not None: m.weight.requires_grad = True
                if m.bias   is not None: m.bias.requires_grad   = True
                bn_count += 1
        total_unfrozen += count
        if count > 0:
            print(f"  Unfrozen: backbone.{stage_name} ({count:,} params, {bn_count} BN)")

    print(f"  Total unfrozen this call: {total_unfrozen:,} params\n")
    return total_unfrozen


def print_backbone_structure(model):
    print(f"\n{'='*70}")
    print(" BACKBONE STRUCTURE (GCNet v3)")
    print(f"{'='*70}")
    for name, module in model.backbone.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        if isinstance(module, nn.ModuleList):
            print(f"  {name}: ModuleList[{len(module)}]  ({n_params:,} params)")
            for i, sub in enumerate(module):
                sp = sum(p.numel() for p in sub.parameters())
                print(f"    [{i}]: {type(sub).__name__}  ({sp:,} params)")
        else:
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
            # ---- backbone (GCNet v3) ------------------------------------ #
            "backbone": {
                "in_channels"          : 3,
                "channels"             : C,
                "ppm_channels"         : 128,
                "num_blocks_per_stage" : [4, 4, [5, 4], [5, 4], [2, 2]],
                "align_corners"        : False,
                "norm_cfg"             : dict(type='BN', requires_grad=True),
                "act_cfg"              : dict(type='ReLU', inplace=True),
                "dwsa_reduction"       : 8,   # DWSA channel reduction ratio
                "deploy"               : False,
            },
            # ---- head (GCNetHead v2) ------------------------------------ #
            # c6_feat output của backbone = channels*4 = 128
            # c4_feat (aux)              = channels*2 = 64
            # head nhận in_channels=128, aux tự xử lý với in_channels//2=64
            "head": {
                "in_channels"     : C * 4,   # 128 — c6 (backbone output)
                "channels"        : 128,      # hidden channels trong head
                "align_corners"   : False,
                "dropout_ratio"   : 0.1,
                "loss_weight_aux" : 0.4,
                "norm_cfg"        : dict(type='BN', requires_grad=True),
                "act_cfg"         : dict(type='ReLU', inplace=True),
            },
            # ---- loss -------------------------------------------------- #
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
    """Wrapper ghép backbone + decode_head.

    Backbone (GCNet v3):
      training  → (c4_feat, c6_feat)
      inference → c6_feat

    Head (GCNetHead v2):
      training  → forward((c4_feat, c6_feat)) → (c4_logit, c6_logit)
      inference → forward(c6_feat)            → c6_logit
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        """Inference path: trả về c6_logit (B, C, H/8, W/8)."""
        feat = self.backbone(x)             # c6_feat
        return self.decode_head(feat)       # c6_logit

    def forward_train(self, x):
        """Training path: trả về dict với 'main' = (c4_logit, c6_logit)."""
        feats  = self.backbone(x)           # (c4_feat, c6_feat)
        logits = self.decode_head(feats)    # (c4_logit, c6_logit)
        return {"main": logits}


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
        # CE với class weights (dùng cho validate và backup)
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

        # Warmup: scale LR từ 1/10 lên full trong warmup_epochs đầu
        warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for g in self.optimizer.param_groups:
                g['lr'] = g['initial_lr'] * warmup_factor

        total_loss = total_ohem = total_dice = 0.0
        max_grad_epoch = 0.0
        max_grad = 0.0   # khởi tạo trước vòng lặp — tránh UnboundLocalError
                         # khi accumulation_steps > 1 và batch đầu chưa trigger update
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)

                # outputs["main"] = (c4_logit, c6_logit) từ GCNetHead.forward()
                c4_logit, c6_logit = outputs["main"]

                # Upsample cả hai về full resolution để tính loss
                target_size = masks.shape[-2:]
                c4_full = F.interpolate(c4_logit, size=target_size,
                                        mode='bilinear', align_corners=False)
                c6_full = F.interpolate(c6_logit, size=target_size,
                                        mode='bilinear', align_corners=False)

                # Main loss trên c6 (OHEM + Dice)
                ohem_loss = self.ohem(c6_full, masks)

                if self.dice_weight > 0:
                    # Dice tính ở resolution thấp (c6_logit), downsample mask
                    masks_small = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=c6_logit.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                    dice_loss = self.dice(c6_logit, masks_small)
                else:
                    dice_loss = torch.tensor(0.0, device=self.device)

                loss = self.ce_weight * ohem_loss + self.dice_weight * dice_loss

                # Auxiliary loss trên c4 — weight tuỳ epoch
                if self.args.aux_weight > 0:
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    aux_loss   = self.ohem(c4_full, masks)
                    loss       = loss + aux_weight * aux_loss

                loss = loss / self.args.accumulation_steps

            # NaN guard
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
                'loss'    : f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ohem'    : f'{ohem_loss.item():.4f}',
                'dice'    : f'{dice_loss.item():.4f}',
                'lr'      : f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'max_grad': f'{max_grad:.2f}',
            })

            if batch_idx % 50 == 0:
                clear_gpu_memory()

            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss',     loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ohem',     ohem_loss.item(), self.global_step)
                self.writer.add_scalar('train/dice',     dice_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr',       self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/max_grad', max_grad, self.global_step)

        n = len(loader)
        print(f"\nEpoch {epoch+1} — Max gradient: {max_grad_epoch:.2f}")
        torch.cuda.empty_cache()
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

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
        total_loss   = 0.0
        num_classes  = self.args.num_classes
        conf_matrix  = np.zeros((num_classes, num_classes), dtype=np.int64)
        pbar         = tqdm(loader, desc="Validation")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                # Inference: backbone trả c6_feat, head trả c6_logit
                logits = self.model(imgs)   # (B, C, H/8, W/8)

                logits_full = F.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                ce_loss = self.ce(logits_full, masks)

                if self.dice_weight > 0:
                    masks_small = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=logits.shape[-2:], mode='nearest'
                    ).squeeze(1).long()
                    dice_loss = self.dice(logits, masks_small)
                else:
                    dice_loss = torch.tensor(0.0, device=self.device)

                loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

            total_loss += loss.item()

            pred   = logits_full.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            valid  = (target >= 0) & (target < num_classes)
            label  = num_classes * target[valid].astype(int) + pred[valid]
            count  = np.bincount(label, minlength=num_classes ** 2)
            conf_matrix += count.reshape(num_classes, num_classes)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        intersection = np.diag(conf_matrix)
        union        = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
        iou          = intersection / (union + 1e-10)
        miou         = np.nanmean(iou)
        acc          = intersection.sum() / (conf_matrix.sum() + 1e-10)

        return {
            'loss'         : total_loss / len(loader),
            'miou'         : miou,
            'accuracy'     : acc,
            'per_class_iou': iou,
        }

    # ---------------------------------------------------------------------- #
    # Checkpoint                                                               #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, epoch, metrics, is_best=False):
        ckpt = {
            'epoch'      : epoch,
            'model'      : self.model.state_dict(),
            'optimizer'  : self.optimizer.state_dict(),
            'scheduler'  : self.scheduler.state_dict() if self.scheduler else None,
            'scaler'     : self.scaler.state_dict(),
            'best_miou'  : self.best_miou,
            'metrics'    : metrics,
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
            self.best_miou   = 0.0 if reset_best_metric else ckpt.get('best_miou', 0.0)
            print(f"Weights loaded (epoch {ckpt['epoch']}), starting from epoch 0")
        else:
            self.start_epoch = ckpt['epoch'] + 1
            self.best_miou   = ckpt.get('best_miou', 0.0)
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
    parser = argparse.ArgumentParser(description="GCNet v3 Training")

    # Transfer learning
    parser.add_argument("--pretrained_weights",    type=str,   default=None)
    parser.add_argument("--freeze_backbone",        action="store_true", default=False)
    parser.add_argument("--unfreeze_schedule",      type=str,   default="",
                        help="Comma-separated epochs to progressively unfreeze backbone")
    parser.add_argument("--backbone_lr_factor",     type=float, default=0.1)
    parser.add_argument("--head_lr_factor",         type=float, default=1.0,
                        help="LR factor cho head. Giảm xuống 0.1-0.5 khi head đã pretrained")
    parser.add_argument("--alpha_lr_factor",        type=float, default=0.01,
                        help="LR factor for FoggyAwareNorm.alpha and DWSA.gamma")
    parser.add_argument("--warmup_epochs",          type=int,   default=0,
                        help="Số epoch warmup LR từ lr/10 lên lr đầy đủ")
    parser.add_argument("--use_class_weights",      action="store_true")

    # Dataset
    parser.add_argument("--train_txt",   required=True)
    parser.add_argument("--val_txt",     required=True)
    parser.add_argument("--dataset_type", default="foggy", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)

    # Training
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--accumulation_steps",type=int,   default=2)
    parser.add_argument("--lr",                type=float, default=5e-4)
    parser.add_argument("--weight_decay",      type=float, default=1e-4)
    parser.add_argument("--grad_clip",         type=float, default=5.0)
    parser.add_argument("--aux_weight",        type=float, default=0.4,
                        help="Weight của auxiliary c4 loss. 0 = tắt aux loss.")
    parser.add_argument("--scheduler",         default="onecycle",
                        choices=["onecycle", "poly", "cosine"])
    parser.add_argument("--freeze_epochs",     type=int,   default=0)
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

    # Validate
    if args.freeze_epochs >= args.epochs:
        raise ValueError("freeze_epochs must be < total epochs")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"GCNet v3 Training  |  FoggyAwareNorm + DWSA stage 4/5/6")
    print(f"{'='*70}")
    print(f"Device:     {device}")
    print(f"Image size: {args.img_h}x{args.img_w}")
    print(f"Epochs:     {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip:  {args.grad_clip}  |  AMP: {args.use_amp}")
    print(f"{'='*70}\n")

    cfg          = ModelConfig.get_config()
    args.loss_config = cfg["loss"]

    # ---- Dataloaders ---- #
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

    # ---- Build model ---- #
    print(f"{'='*70}")
    print("BUILDING MODEL")
    print(f"{'='*70}\n")

    backbone = GCNet(**cfg["backbone"])
    head     = GCNetHead(
        **cfg["head"],
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )
    model = Segmentor(backbone=backbone, head=head).to(device)

    print("Applying init_weights...")
    model.apply(init_weights)
    check_model_health(model)
    print()

    # ---- Transfer learning ---- #
    print(f"{'='*70}")
    print("TRANSFER LEARNING SETUP")
    print(f"{'='*70}\n")

    if args.pretrained_weights:
        load_pretrained_gcnet(model, args.pretrained_weights)

    if args.freeze_backbone:
        freeze_backbone(model)

    count_trainable_params(model)
    print_backbone_structure(model)

    # ---- Sanity forward ---- #
    with torch.no_grad():
        sample = torch.randn(2, 3, args.img_h, args.img_w).to(device)
        try:
            out = model.forward_train(sample)
            c4_logit, c6_logit = out["main"]
            print(f"Forward pass OK:")
            print(f"  c4_logit: {c4_logit.shape}")
            print(f"  c6_logit: {c6_logit.shape}\n")
        except Exception as e:
            print(f"Forward pass FAILED: {e}")
            return

    # ---- Optimizer + Scheduler ---- #
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=0)

    # ---- Trainer ---- #
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

    # ---- Unfreeze schedule ---- #
    unfreeze_epochs = []
    if args.unfreeze_schedule:
        try:
            unfreeze_epochs = sorted(int(e) for e in args.unfreeze_schedule.split(','))
        except Exception:
            raise ValueError("unfreeze_schedule phải là chuỗi số nguyên cách nhau bởi dấu phẩy")

    # Module names của GCNet v3 để unfreeze dần
    # Stage 6 → 5 → 4 → stem (từ sâu nhất ra ngoài)
    UNFREEZE_STAGES = [
        # k=1: unfreeze stage 6
        ['semantic_branch_layers.2', 'detail_branch_layers.2', 'dwsa_stage6', 'spp'],
        # k=2: unfreeze stage 5
        ['semantic_branch_layers.1', 'detail_branch_layers.1', 'dwsa_stage5',
         'compression_2', 'down_2'],
        # k=3: unfreeze stage 4
        ['semantic_branch_layers.0', 'detail_branch_layers.0', 'dwsa_stage4',
         'compression_1', 'down_1'],
        # k=4: unfreeze stem
        ['stem_conv1', 'stem_conv2', 'stem_stage2', 'stem_stage3'],
    ]

    # ---- Training loop ---- #
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(trainer.start_epoch, args.epochs):

        # Cumulative unfreeze
        past = [e for e in unfreeze_epochs if e <= epoch]
        k    = len(past)
        targets = []
        for i in range(min(k, len(UNFREEZE_STAGES))):
            targets += UNFREEZE_STAGES[i]
        if targets:
            unfreeze_backbone_progressive(model, targets)

        # Rebuild optimizer + scheduler khi đúng epoch unfreeze
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

        # Switch back to full loss sau ce_only phase
        if unfreeze_epochs:
            last_unfreeze = max((e for e in unfreeze_epochs if e <= epoch), default=None)
            if last_unfreeze is not None:
                if epoch >= last_unfreeze + args.ce_only_epochs_after_unfreeze:
                    trainer.set_loss_phase('full')
                elif epoch == last_unfreeze:
                    trainer.set_loss_phase('ce_only')

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics   = trainer.validate(val_loader, epoch)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | OHEM: {train_metrics['ohem']:.4f} | Dice: {train_metrics['dice']:.4f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | mIoU: {val_metrics['miou']:.4f}  | Acc: {val_metrics['accuracy']:.4f}")
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
    print("TRAINING COMPLETED!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
