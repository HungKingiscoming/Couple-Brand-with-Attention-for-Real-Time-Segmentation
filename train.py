# ============================================
# train.py — GCNet v3 + GCNetHead v2
# FIXES:
#   1. Progressive unfreeze: chỉ gọi đúng epoch, không cumulative mỗi epoch
#   2. Scheduler: CosineAnnealingLR khi freeze_backbone (OneCycleLR không
#      tương thích rebuild giữa chừng); OneCycleLR vẫn dùng được khi không freeze
#   3. compression_1/down_1 unfreeze cùng stem (stage 1), không phải stage 2
#   4. DWSA gamma init comment sửa từ "should be 0.0" → "should be 0.1"
#   5. Optimizer rebuild sau unfreeze đảm bảo new params có LR đúng
#   6. Knowledge Distillation: thêm Trainer.set_teacher() + loss_kd với ramp-up
#      Feature distillation (PKD-style, normalized MSE) tại fused output
#      Logit distillation (KL, T=4) tại head output
#      Cả hai đều weight=0 lúc đầu, ramp up qua kd_warmup_epochs
#   7. Segmentor.forward_train expose fused_feat để distill
#   8. Trainer.train_epoch nhận epoch param để tính kd_weight đúng
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

from model.backbone.model import GCNet
from model.head.segmentation_head import GCNetHead
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# PRETRAINED WEIGHT LOADER
# ============================================

def _remap_stem_key(key: str, N2: int = 4):
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
    print(f"Loading pretrained weights from: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    HEAD_KEY_MAP = {}
    for k in state.keys():
        if not k.startswith('decode_head.'):
            continue
        suffix = k[len('decode_head.'):]
        if suffix.startswith('conv_seg.'):
            dst = 'cls_seg.' + suffix[len('conv_seg.'):]
        else:
            dst = suffix
        HEAD_KEY_MAP[k] = dst

    model_state   = model.backbone.state_dict()
    compatible    = {}
    skipped       = []
    model_key_map = {}
    for mk in model_state.keys():
        norm = mk
        for pref in ['backbone.', 'model.', 'module.']:
            if norm.startswith(pref):
                norm = norm[len(pref):]
        model_key_map[norm] = mk

    bn_dropped = []

    for ckpt_key, ckpt_val in state.items():
        if ckpt_key.startswith('decode_head.'):
            continue

        stripped = ckpt_key
        for pref in ('backbone.', 'model.', 'module.'):
            if stripped.startswith(pref):
                stripped = stripped[len(pref):]
                break

        norm_ckpt = _remap_stem_key(ckpt_key)
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

    head_state           = model.decode_head.state_dict()
    head_loaded          = {}
    head_skipped_shape   = []
    head_skipped_missing = []

    for ckpt_key, dst_suffix in HEAD_KEY_MAP.items():
        ckpt_val = state[ckpt_key]
        if dst_suffix not in head_state:
            head_skipped_missing.append(f"{ckpt_key} → {dst_suffix}")
            continue
        if head_state[dst_suffix].shape != ckpt_val.shape:
            head_skipped_shape.append(
                f"{ckpt_key}: ckpt{list(ckpt_val.shape)} vs model{list(head_state[dst_suffix].shape)}"
            )
            continue
        head_loaded[dst_suffix] = ckpt_val

    loaded_bb = len(compatible)
    loaded_hd = len(head_loaded)
    total_bb  = len(model_state)
    total_hd  = len(head_state)

    expected_skip_markers = ('dwsa_stage', 'foggy', 'alpha', 'in_.', '.spp.', 'backbone.spp.')
    truly_unmatched = [k for k in skipped if not any(s in k for s in expected_skip_markers)]

    sep = '=' * 70
    print(f"\n{sep}")
    print("WEIGHT LOADING SUMMARY")
    print(sep)
    print(f"Backbone:  {loaded_bb:>5} / {total_bb}  ({100*loaded_bb/max(total_bb,1):.1f}%)")
    print(f"Head:      {loaded_hd:>5} / {total_hd}  ({100*loaded_hd/max(total_hd,1):.1f}%)")
    print(f"BN dropped (expected): {len(bn_dropped):>3}  (stem BN → FoggyAwareNorm)")
    if head_skipped_shape:
        print(f"Head shape mismatch:   {len(head_skipped_shape)}")
        for s in head_skipped_shape:
            print(f"    SHAPE: {s}")
    if head_skipped_missing:
        print(f"Head key missing:      {len(head_skipped_missing)}")
        for s in head_skipped_missing:
            print(f"    MISSING: {s}")
    if truly_unmatched:
        print(f"Backbone unmatched:    {len(truly_unmatched)}  ← cần kiểm tra")
        for k in truly_unmatched[:5]:
            print(f"    {k}")
    print(sep + "\n")

    missing_bb, _ = model.backbone.load_state_dict(compatible, strict=False)
    missing_hd, _ = model.decode_head.load_state_dict(head_loaded, strict=False)

    # stem_conv1.1 / stem_conv2.1 = FoggyAwareNorm — module mới, expected missing
    expected_missing_markers = ('dwsa', 'alpha', 'in_.', 'foggy', '.1.bn.', 'spp.',
                                'loss_', 'fog_consistency',
                                'stem_conv1.1.', 'stem_conv2.1.')
    unexpected_bb = [k for k in missing_bb
                     if not any(s in k for s in expected_missing_markers)]
    unexpected_hd = [k for k in missing_hd
                     if not any(s in k for s in expected_missing_markers)]

    if unexpected_bb:
        print(f"Unexpected backbone missing ({len(unexpected_bb)}):")
        for k in unexpected_bb[:5]:
            print(f"  - {k}")
    if unexpected_hd:
        print(f"Unexpected head missing ({len(unexpected_hd)}):")
        for k in unexpected_hd[:5]:
            print(f"  - {k}")

    n_expected_missing = (len(missing_bb) - len(unexpected_bb)
                          + len(missing_hd) - len(unexpected_hd))
    print(f"Expected missing: {n_expected_missing} keys (DWSA/FoggyNorm/loss buffers) → OK\n")

    return 100 * (loaded_bb + loaded_hd) / max(total_bb + total_hd, 1)


# ============================================
# OPTIMIZER
# ============================================

def build_optimizer(model, args):
    """4 param groups: head | backbone | dwsa | alpha."""
    dwsa_params     = []
    alpha_params    = []
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'dwsa' in name:
            dwsa_params.append(param)
        elif 'alpha' in name:
            alpha_params.append(param)
        elif 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if head_params:
        groups.append({'params': head_params,     'lr': args.lr,                           'name': 'head'})
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': args.lr * args.backbone_lr_factor, 'name': 'backbone'})
    if dwsa_params:
        groups.append({'params': dwsa_params,     'lr': args.lr * args.dwsa_lr_factor,     'name': 'dwsa'})
    if alpha_params:
        groups.append({'params': alpha_params,    'lr': args.lr * args.alpha_lr_factor,    'name': 'alpha'})

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])

    print("Optimizer: AdamW (Discriminative LR)")
    for g in optimizer.param_groups:
        print(f"  group '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")

    return optimizer


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    """
    FIX: khi dùng freeze_backbone + progressive unfreeze, dùng CosineAnnealingLR.
    OneCycleLR không tương thích với rebuild optimizer giữa chừng vì:
      - total_steps được tính cố định lúc khởi tạo
      - Mỗi lần rebuild, scheduler reset về step 0 nhưng epoch đã đi xa
      - Kết quả: LR spike đột ngột sau mỗi unfreeze stage
    CosineAnnealingLR an toàn hơn: decay từ LR hiện tại về eta_min,
    không phụ thuộc vào step count tuyệt đối.
    """
    n_groups = len(optimizer.param_groups)

    # Nếu đang freeze backbone → dùng cosine bất kể args.scheduler
    use_cosine = args.freeze_backbone and args.unfreeze_schedule

    if use_cosine or args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - start_epoch,
            eta_min=1e-6
        )
        if use_cosine and args.scheduler != 'cosine':
            print("CosineAnnealingLR (auto-selected: freeze_backbone + unfreeze_schedule)")
        else:
            print("CosineAnnealingLR")

    elif args.scheduler == 'onecycle':
        remaining_epochs = args.epochs - start_epoch
        total_steps      = len(train_loader) * remaining_epochs
        max_lrs = args.lr if n_groups == 1 else [g['initial_lr'] for g in optimizer.param_groups]
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

    else:  # poly
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9
        )
        print("Polynomial LR decay")

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
        probs            = F.softmax(logits, dim=1) * valid_mask.unsqueeze(1).float()
        probs_flat       = probs.reshape(B, C, -1)
        target_flat      = targets_one_hot.reshape(B, C, -1)
        intersection     = (probs_flat * target_flat).sum(2)
        cardinality      = probs_flat.sum(2) + target_flat.sum(2)
        dice_score       = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = (-torch.log(dice_score.clamp(min=self.smooth))
                     if self.log_loss else 1.0 - dice_score)
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
        weight       = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_pixel   = F.cross_entropy(logits, labels, weight=weight,
                                       ignore_index=self.ignore_index, reduction='none').view(-1)
        valid_mask   = (labels.view(-1) != self.ignore_index)
        valid_losses = loss_pixel[valid_mask]
        n_valid      = valid_losses.numel()
        if n_valid == 0:
            return logits.sum() * 0
        n_keep = max(int(self.keep_ratio * n_valid), min(self.min_kept, n_valid))
        n_keep = min(n_keep, n_valid)
        if n_keep < n_valid:
            threshold    = torch.sort(valid_losses, descending=True)[0][n_keep - 1].detach()
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
    """Freeze toàn bộ backbone trừ DWSA và FoggyAwareNorm."""
    print("Freezing backbone (keeping DWSA + FoggyAwareNorm trainable)...")

    for p in model.backbone.parameters():
        p.requires_grad = False

    bn_count = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None: m.weight.requires_grad = False
            if m.bias   is not None: m.bias.requires_grad   = False
            bn_count += 1
    print(f"  {bn_count} BN layers locked")

    dwsa_params = 0; dwsa_bn_count = 0
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        module = getattr(model.backbone, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
                dwsa_params += p.numel()
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    if m.weight is not None: m.weight.requires_grad = True
                    if m.bias   is not None: m.bias.requires_grad   = True
                    dwsa_bn_count += 1
    print(f"  DWSA kept trainable: {dwsa_params:,} params, {dwsa_bn_count} BN unfrozen")

    fan_params = 0
    for name in ['stem_conv1', 'stem_conv2']:
        module = getattr(model.backbone, name, None)
        if module is not None:
            if len(module) > 1 and hasattr(module[1], 'alpha'):
                for p in module[1].parameters():
                    p.requires_grad = True
                    fan_params += p.numel()
                fan_bn = module[1].bn
                fan_bn.train()
                if fan_bn.weight is not None: fan_bn.weight.requires_grad = True
                if fan_bn.bias   is not None: fan_bn.bias.requires_grad   = True
    print(f"  FoggyAwareNorm kept trainable: {fan_params:,} params")
    print("Backbone frozen\n")


def unfreeze_backbone_progressive(model, stage_names):
    """Unfreeze một danh sách stage cụ thể — chỉ gọi đúng epoch unfreeze."""
    if isinstance(stage_names, str):
        stage_names = [stage_names]

    total_unfrozen = 0
    for stage_name in stage_names:
        module = None

        if hasattr(model.backbone, stage_name):
            module = getattr(model.backbone, stage_name)
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


def log_dwsa_gamma(model, writer, epoch):
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        module = getattr(model.backbone, name, None)
        if module is not None and hasattr(module, 'gamma'):
            writer.add_scalar(f'dwsa/{name}_gamma', module.gamma.item(), epoch)


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
            "backbone": {
                "in_channels"          : 3,
                "channels"             : C,
                "ppm_channels"         : 128,
                "num_blocks_per_stage" : [4, 4, [5, 4], [5, 4], [2, 2]],
                "align_corners"        : False,
                "norm_cfg"             : dict(type='BN', requires_grad=True),
                "act_cfg"              : dict(type='ReLU', inplace=True),
                "dwsa_reduction"       : 8,
                "deploy"               : False,
            },
            "head": {
                "in_channels"     : C * 4,
                "channels"        : 64,
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
        """
        Returns:
            dict với keys:
              'main'       : (c4_logit, c6_logit)
              'fused_feat' : fused backbone feature (B, ch*4, H/8, W/8)
                             dùng cho feature-level distillation

        FIX: dùng return_aux=True explicit thay vì phụ thuộc self.training.
        Teacher model bị set eval() nên self.training=False → backbone trả
        về Tensor đơn thay vì tuple → ValueError khi unpack.
        return_aux=True buộc backbone luôn trả về (c4_feat, fused) tuple.
        """
        # return_aux=True: luôn trả về (c4_feat, fused) dù training hay eval mode
        feats              = self.backbone(x, return_aux=True)
        c4_feat, fused_feat = feats
        # Head cần training mode để nhận tuple input và trả về (c4_logit, c6_logit)
        # Tạm thời set head sang train mode nếu đang ở eval (teacher case)
        head_was_training = self.decode_head.training
        self.decode_head.train()
        logits = self.decode_head(feats)
        if not head_was_training:
            self.decode_head.eval()
        return {
            "main"       : logits,
            "fused_feat" : fused_feat,
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

        # ------------------------------------------------------------------ #
        # Distillation state
        # ------------------------------------------------------------------ #
        self.teacher            = None   # set via set_teacher()
        self.kd_feat_weight     = getattr(args, 'kd_feat_weight',  0.5)
        self.kd_logit_weight    = getattr(args, 'kd_logit_weight', 0.5)
        self.kd_temperature     = getattr(args, 'kd_temperature',  4.0)
        self.kd_warmup_epochs   = getattr(args, 'kd_warmup_epochs', 10)
        # Adapter: nếu teacher channel != student channel, cần 1x1 conv
        self._kd_feat_adapter   = None
        self._kd_active         = True   # False khi ce_only phase
        self._teacher_cache     = None   # cache teacher output để tránh forward mỗi batch
        self._teacher_cache_idx = -1     # batch_idx của cache hiện tại

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=self.save_dir / "tensorboard")

        self.save_config()
        self._print_config(loss_cfg)

    # ---------------------------------------------------------------------- #
    # Distillation setup                                                       #
    # ---------------------------------------------------------------------- #

    def set_teacher(self, teacher_model, teacher_feat_channels=None):
        """Load teacher model cho knowledge distillation.

        Args:
            teacher_model: nn.Module đã load weights, sẽ bị freeze hoàn toàn.
            teacher_feat_channels: số channels của fused_feat trong teacher.
                Nếu None → giả sử bằng student (args.model_channels * 4).
                Nếu khác student → tạo 1×1 conv adapter để align.
        """
        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        cfg            = ModelConfig.get_config()
        student_ch     = cfg['backbone']['channels'] * 4
        teacher_ch     = teacher_feat_channels if teacher_feat_channels else student_ch

        if teacher_ch != student_ch:
            # Adapter: project student → teacher space để MSE có nghĩa hơn
            self._kd_feat_adapter = nn.Conv2d(
                student_ch, teacher_ch, kernel_size=1, bias=False
            ).to(self.device)
            nn.init.kaiming_normal_(self._kd_feat_adapter.weight,
                                    mode='fan_out', nonlinearity='relu')
            # Thêm adapter vào optimizer
            self.optimizer.add_param_group({
                'params'     : list(self._kd_feat_adapter.parameters()),
                'lr'         : self.args.lr,
                'name'       : 'kd_adapter',
                'initial_lr' : self.args.lr,
            })
            print(f"KD feat adapter: {student_ch} → {teacher_ch} channels")
        else:
            self._kd_feat_adapter = None

        print(f"Teacher model loaded and frozen.")
        print(f"  KD feat weight:   {self.kd_feat_weight}")
        print(f"  KD logit weight:  {self.kd_logit_weight}")
        print(f"  KD temperature:   {self.kd_temperature}")
        print(f"  KD warmup epochs: {self.kd_warmup_epochs}\n")

    def _get_kd_weight(self, epoch):
        """Ramp-up KD weight từ 0 → target trong kd_warmup_epochs.

        Lý do ramp-up: ở epoch đầu teacher logits với foggy input cũng không
        ổn định, KD loss lớn ngay từ đầu sẽ lấn át OHEM loss.
        """
        if self.teacher is None or self.kd_warmup_epochs == 0:
            return 1.0
        return min(1.0, epoch / self.kd_warmup_epochs)

    def _compute_kd_loss(self, student_outputs, imgs, epoch, batch_idx=0):
        """Tính KD loss (feature + logit).

        Feature distillation (PKD-style):
          - Normalize cả student và teacher feature trước MSE
          - Giúp tránh scale mismatch khi teacher/student có channels khác nhau
          - Detach teacher hoàn toàn

        Logit distillation (KL với temperature):
          - KL(student || teacher) * T²
          - T=4 làm mềm distribution, focus vào dark knowledge
          - Detach teacher logit

        Returns:
            loss_feat  (Tensor): scalar hoặc 0.0
            loss_logit (Tensor): scalar hoặc 0.0
        """
        if self.teacher is None or not self._kd_active:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        kd_w = self._get_kd_weight(epoch)
        if kd_w == 0.0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        # FIX memory: cache teacher output trong cùng accumulation window
        # Tránh teacher forward 2 lần khi accumulation_steps=2 — tiết kiệm ~0.5GB
        accum = getattr(self.args, 'accumulation_steps', 1)
        accum_group = batch_idx // accum  # cùng group = cùng accumulation window
        if self._teacher_cache is not None and self._teacher_cache_idx == accum_group:
            t_feat, t_logit = self._teacher_cache
        else:

            # Teacher forward bên ngoài autocast: frozen model không cần AMP,
            # BN trong teacher cần float32 để tránh numerical issues.
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                imgs_f32 = imgs.float()
                t_feat   = self.teacher.backbone(imgs_f32, return_aux=False).detach()
                t_logit  = self.teacher.decode_head(t_feat).detach()
            self._teacher_cache     = (t_feat, t_logit)
            self._teacher_cache_idx = accum_group

        s_feat  = student_outputs['fused_feat']
        s_c4, s_c6 = student_outputs['main']
        s_logit = s_c6

        # ---- Feature distillation ---------------------------------------- #
        if self._kd_feat_adapter is not None:
            s_feat_proj = self._kd_feat_adapter(s_feat)
        else:
            s_feat_proj = s_feat

        # Upsample nếu spatial size khác (teacher có thể channels=64)
        if s_feat_proj.shape[-2:] != t_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=s_feat_proj.shape[-2:],
                                   mode='bilinear', align_corners=False)

        # PKD: normalize trên dim=1 trước MSE — dùng float32 để tránh NaN
        s_norm = F.normalize(s_feat_proj.float(), dim=1)
        t_norm = F.normalize(t_feat.float(),      dim=1)
        loss_feat = F.mse_loss(s_norm, t_norm) * kd_w * self.kd_feat_weight
        if torch.isnan(loss_feat) or torch.isinf(loss_feat):
            loss_feat = torch.tensor(0.0, device=self.device)

        # ---- Logit distillation ------------------------------------------ #
        T = self.kd_temperature
        if s_logit.shape[-2:] != t_logit.shape[-2:]:
            t_logit = F.interpolate(t_logit, size=s_logit.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # Clamp logits trước softmax để tránh overflow/NaN
        # Teacher logit có thể có range lớn sau load partial weights
        s_logit_c = s_logit.float().clamp(-50, 50)
        t_logit_c = t_logit.float().clamp(-50, 50)
        log_p    = F.log_softmax(s_logit_c / T, dim=1)
        q        = F.softmax(t_logit_c  / T, dim=1).clamp(min=1e-7)
        loss_kd  = F.kl_div(log_p, q, reduction='batchmean') * (T ** 2)
        # Guard NaN — có thể xảy ra ở early training khi logits unstable
        if torch.isnan(loss_kd) or torch.isinf(loss_kd):
            loss_kd = torch.tensor(0.0, device=self.device)
        loss_logit = loss_kd * kd_w * self.kd_logit_weight

        return loss_feat, loss_logit

    # ---------------------------------------------------------------------- #
    # Loss phase                                                               #
    # ---------------------------------------------------------------------- #

    def set_loss_phase(self, phase: str):
        if phase == self.loss_phase:
            return
        if phase == 'ce_only':
            self.dice_weight = 0.0
            # FIX: tắt KD trong ce_only để tránh gradient shock sau unfreeze
            self._kd_active = False
        elif phase == 'full':
            self.dice_weight = self.base_loss_cfg['dice_weight']
            self._kd_active = True
        self.loss_phase = phase
        print(f"Loss phase → {phase}  (CE={self.ce_weight}, Dice={self.dice_weight}, KD={self._kd_active})")

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
        kd_on = getattr(self.args, 'teacher_weights', None) is not None
        if kd_on:
            print(f"Distillation: feat_w={self.kd_feat_weight}, "
                  f"logit_w={self.kd_logit_weight}, T={self.kd_temperature}, "
                  f"warmup={self.kd_warmup_epochs}")
        print(f"{'='*70}\n")

    def save_config(self):
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

    # ---------------------------------------------------------------------- #
    # Training                                                                 #
    # ---------------------------------------------------------------------- #

    def train_epoch(self, loader, epoch):
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
            self._teacher_cache     = None  # clear cache ở đầu epoch
            self._teacher_cache_idx = -1

        total_loss   = total_ohem = total_dice = 0.0
        total_kd_feat = total_kd_logit = 0.0
        max_grad_epoch = 0.0
        max_grad       = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs = self.model.forward_train(imgs)

                c4_logit, c6_logit = outputs["main"]
                target_size        = masks.shape[-2:]

                c4_full = F.interpolate(c4_logit, size=target_size,
                                        mode='bilinear', align_corners=False)
                c6_full = F.interpolate(c6_logit, size=target_size,
                                        mode='bilinear', align_corners=False)

                ohem_loss = self.ohem(c6_full, masks)

                if self.dice_weight > 0:
                    masks_small = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=c6_logit.shape[-2:], mode='nearest'
                    ).squeeze(1).long()
                    dice_loss = self.dice(c6_logit, masks_small)
                else:
                    dice_loss = torch.tensor(0.0, device=self.device)

                loss = self.ce_weight * ohem_loss + self.dice_weight * dice_loss

                if self.args.aux_weight > 0:
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                    aux_loss   = self.ohem(c4_full, masks)
                    loss       = loss + aux_weight * aux_loss

                # ---- Knowledge Distillation ------------------------------ #
                loss_feat, loss_logit = self._compute_kd_loss(outputs, imgs, epoch, batch_idx)
                loss = loss + loss_feat + loss_logit

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

            total_loss    += loss.item() * self.args.accumulation_steps
            total_ohem    += ohem_loss.item()
            total_dice    += dice_loss.item()
            total_kd_feat  += loss_feat.item() if isinstance(loss_feat, torch.Tensor) else 0.0
            total_kd_logit += loss_logit.item() if isinstance(loss_logit, torch.Tensor) else 0.0

            postfix = {
                'loss'    : f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ohem'    : f'{ohem_loss.item():.4f}',
                'dice'    : f'{dice_loss.item():.4f}',
                'lr'      : f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'max_g'   : f'{max_grad:.2f}',
            }
            if self.teacher is not None:
                postfix['kd_f'] = f'{loss_feat.item():.4f}'
                postfix['kd_l'] = f'{loss_logit.item():.4f}'
            pbar.set_postfix(postfix)

            if batch_idx % 50 == 0:
                clear_gpu_memory()

            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss',     loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ohem',     ohem_loss.item(),  self.global_step)
                self.writer.add_scalar('train/dice',     dice_loss.item(),  self.global_step)
                self.writer.add_scalar('train/lr',       self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/max_grad', max_grad,          self.global_step)
                if self.teacher is not None:
                    self.writer.add_scalar('train/kd_feat',  loss_feat.item(),  self.global_step)
                    self.writer.add_scalar('train/kd_logit', loss_logit.item(), self.global_step)
                    self.writer.add_scalar('train/kd_weight', self._get_kd_weight(epoch), self.global_step)

        n = len(loader)
        print(f"\nEpoch {epoch+1} — Max gradient: {max_grad_epoch:.2f}")
        torch.cuda.empty_cache()

        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        return {
            'loss'     : total_loss   / n,
            'ohem'     : total_ohem   / n,
            'dice'     : total_dice   / n,
            'kd_feat'  : total_kd_feat  / n,
            'kd_logit' : total_kd_logit / n,
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
                logits = self.model(imgs)
                logits_full = F.interpolate(logits, size=masks.shape[-2:],
                                            mode='bilinear', align_corners=False)
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
# UNFREEZE SCHEDULE
# ============================================

# FIX: compression_1 + down_1 dipindahkan ke stage 1 (bersama stem).
# Alasan: bilateral fusion layers menghubungkan stem output ke stage 4.
# Jika stem di-unfreeze tapi compression_1/down_1 masih frozen,
# gradient dari stage 4 tidak bisa mengalir ke stem → manfaat unfreeze stem
# tidak optimal karena jalur gradient terputus.
UNFREEZE_STAGES = [
    # Stage 1 — stem + bilateral fusion layers yang terkait langsung
    ['stem_conv1', 'stem_conv2', 'stem_stage2', 'stem_stage3',
     'compression_1', 'down_1'],
    # Stage 2 — stage 4 (semantic + detail branch) + DWSA stage 4
    ['semantic_branch_layers.0', 'detail_branch_layers.0', 'dwsa_stage4'],
    # Stage 3 — stage 5 + DWSA stage 5 + compression_2/down_2
    ['semantic_branch_layers.1', 'detail_branch_layers.1', 'dwsa_stage5',
     'compression_2', 'down_2'],
    # Stage 4 — stage 6 + DWSA stage 6 + DAPPM
    ['semantic_branch_layers.2', 'detail_branch_layers.2', 'dwsa_stage6', 'spp'],
]


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 Training")

    # Transfer learning
    parser.add_argument("--pretrained_weights",    type=str,   default=None)
    parser.add_argument("--freeze_backbone",        action="store_true", default=False)
    parser.add_argument("--unfreeze_schedule",      type=str,   default="",
                        help="Comma-separated epochs to progressively unfreeze, e.g. '10,20,30,40'")
    parser.add_argument("--backbone_lr_factor",     type=float, default=0.1)
    parser.add_argument("--dwsa_lr_factor",         type=float, default=0.5,
                        help="LR factor cho DWSA. >= 0.3 để gamma thoát 0.")
    parser.add_argument("--alpha_lr_factor",        type=float, default=0.1,
                        help="LR factor cho FoggyAwareNorm.alpha")
    parser.add_argument("--use_class_weights",      action="store_true")

    # Knowledge Distillation
    parser.add_argument("--teacher_weights",        type=str,   default=None,
                        help="Path tới checkpoint của teacher model (GCNet-L hoặc tương tự). "
                             "Nếu None → không dùng distillation.")
    parser.add_argument("--teacher_channels",       type=int,   default=64,
                        help="'channels' param của teacher backbone. "
                             "Student=32 → teacher=64 thì teacher_feat_ch=256.")
    parser.add_argument("--kd_feat_weight",         type=float, default=0.5,
                        help="Weight cho feature distillation loss (PKD-style MSE).")
    parser.add_argument("--kd_logit_weight",        type=float, default=0.5,
                        help="Weight cho logit distillation loss (KL divergence).")
    parser.add_argument("--kd_temperature",         type=float, default=4.0,
                        help="Temperature T cho softmax trong logit KD. T=4 là default.")
    parser.add_argument("--kd_warmup_epochs",       type=int,   default=10,
                        help="Số epoch ramp-up KD weight từ 0 → 1. "
                             "0 = bật ngay từ epoch đầu (không khuyến nghị).")
    parser.add_argument("--teacher_head_channels",  type=int,   default=None,
                        help="channels của teacher head. Mặc định = teacher_channels * 2.")

    # Dataset
    parser.add_argument("--train_txt",    required=True)
    parser.add_argument("--val_txt",      required=True)
    parser.add_argument("--dataset_type", default="foggy", choices=["normal", "foggy"])
    parser.add_argument("--num_classes",  type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)

    # Training
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--accumulation_steps", type=int,   default=2)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--weight_decay",       type=float, default=1e-4)
    parser.add_argument("--grad_clip",          type=float, default=5.0)
    parser.add_argument("--aux_weight",         type=float, default=0.4)
    parser.add_argument("--scheduler",          default="onecycle",
                        choices=["onecycle", "poly", "cosine"])
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3,
                        help="Số epoch chỉ dùng CE loss ngay sau mỗi lần unfreeze "
                             "(Dice + KD tắt tạm để tránh gradient shock).")

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

    if args.freeze_backbone and args.unfreeze_schedule:
        unfreeze_list = sorted(int(e) for e in args.unfreeze_schedule.split(','))
        if max(unfreeze_list) >= args.epochs:
            raise ValueError("unfreeze_schedule contains epoch >= total epochs")
    else:
        unfreeze_list = []

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Scheduler auto-override: cosine khi dùng freeze + unfreeze
    effective_scheduler = args.scheduler
    if args.freeze_backbone and unfreeze_list and args.scheduler == 'onecycle':
        effective_scheduler = 'cosine'
        print(f"[INFO] scheduler auto-switched: onecycle → cosine "
              f"(freeze_backbone + unfreeze_schedule không tương thích với OneCycleLR)")
    args.scheduler = effective_scheduler

    print(f"\n{'='*70}")
    print(f"GCNet v3 Training  |  FoggyAwareNorm + DWSA stage 4/5/6")
    print(f"{'='*70}")
    print(f"Device:     {device}")
    print(f"Image size: {args.img_h}x{args.img_w}")
    print(f"Epochs:     {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip:  {args.grad_clip}  |  AMP: {args.use_amp}")
    print(f"LR DWSA:    {args.lr * args.dwsa_lr_factor:.2e}  (factor={args.dwsa_lr_factor})")
    print(f"LR alpha:   {args.lr * args.alpha_lr_factor:.2e}  (factor={args.alpha_lr_factor})")
    if args.teacher_weights:
        print(f"Distillation: teacher={args.teacher_weights}")
        print(f"  feat_w={args.kd_feat_weight}, logit_w={args.kd_logit_weight}, "
              f"T={args.kd_temperature}, warmup={args.kd_warmup_epochs}")
    if unfreeze_list:
        print(f"Unfreeze schedule: epochs {unfreeze_list}")
    print(f"{'='*70}\n")

    cfg              = ModelConfig.get_config()
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

    print(f"{'='*70}")
    print("TRANSFER LEARNING SETUP")
    print(f"{'='*70}\n")

    if args.pretrained_weights:
        load_pretrained_gcnet(model, args.pretrained_weights)

    if args.freeze_backbone:
        freeze_backbone(model)

    count_trainable_params(model)
    print_backbone_structure(model)

    # FIX: gamma khởi tạo = 0.1 (bukan 0) — comment lama sai
    print("DWSA gamma values at init:")
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        m = getattr(model.backbone, name, None)
        if m is not None and hasattr(m, 'gamma'):
            print(f"  {name}.gamma = {m.gamma.item():.6f}  (expected ~0.1)")
    print()

    with torch.no_grad():
        sample = torch.randn(2, 3, args.img_h, args.img_w).to(device)
        try:
            out = model.forward_train(sample)
            c4_logit, c6_logit = out["main"]
            fused = out["fused_feat"]
            print(f"Forward pass OK:")
            print(f"  c4_logit:    {c4_logit.shape}")
            print(f"  c6_logit:    {c6_logit.shape}")
            print(f"  fused_feat:  {fused.shape}\n")
        except Exception as e:
            print(f"Forward pass FAILED: {e}")
            raise

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

    # ------------------------------------------------------------------ #
    # Load teacher nếu có
    # ------------------------------------------------------------------ #
    if args.teacher_weights:
        teacher_cfg = ModelConfig.get_config()
        teacher_cfg["backbone"]["channels"] = args.teacher_channels
        teacher_cfg["head"]["in_channels"]  = args.teacher_channels * 4
        # FIX: teacher_head_channels tự động = teacher_channels * 2 nếu không chỉ định
        t_head_ch = (args.teacher_head_channels
                     if args.teacher_head_channels is not None
                     else args.teacher_channels * 2)
        teacher_cfg["head"]["channels"]     = t_head_ch

        t_backbone = GCNet(**teacher_cfg["backbone"])
        t_head     = GCNetHead(
            **teacher_cfg["head"],
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        teacher_model = Segmentor(backbone=t_backbone, head=t_head)
        load_pretrained_gcnet(teacher_model, args.teacher_weights)

        teacher_feat_channels = args.teacher_channels * 4
        trainer.set_teacher(teacher_model, teacher_feat_channels=teacher_feat_channels)

    # ------------------------------------------------------------------ #
    # Resume checkpoint
    # ------------------------------------------------------------------ #
    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=(args.resume_mode == "transfer"),
            load_optimizer=(args.resume_mode == "continue"),
            reset_best_metric=args.reset_best_metric,
        )

    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    # Track which unfreeze stages have been applied (để không gọi lại)
    applied_unfreeze_stages = set()

    for epoch in range(trainer.start_epoch, args.epochs):

        # ------------------------------------------------------------------ #
        # FIX: Progressive unfreeze — chỉ gọi ĐÚNG epoch unfreeze, không
        # cumulative mỗi epoch. Bản cũ gọi unfreeze_backbone_progressive với
        # toàn bộ targets tích lũy mỗi epoch → lãng phí + confuse log.
        # ------------------------------------------------------------------ #
        if epoch in unfreeze_list and epoch not in applied_unfreeze_stages:
            stage_idx = unfreeze_list.index(epoch)
            if stage_idx < len(UNFREEZE_STAGES):
                print(f"[Epoch {epoch+1}] Progressive unfreeze — stage {stage_idx + 1}/{len(UNFREEZE_STAGES)}")
                unfreeze_backbone_progressive(model, UNFREEZE_STAGES[stage_idx])
                applied_unfreeze_stages.add(epoch)

                # Rebuild optimizer + scheduler sau unfreeze để new params có LR đúng
                # FIX: bản cũ rebuild scheduler với start_epoch=epoch nhưng
                # CosineAnnealingLR tính T_max = epochs - start_epoch → OK
                optimizer = build_optimizer(model, args)
                scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=epoch)
                trainer.optimizer = optimizer
                trainer.scheduler = scheduler

                # FIX: re-add kd_feat_adapter vào optimizer mới nếu tồn tại
                # Adapter không thuộc model.parameters() nên bị mất sau rebuild
                if trainer._kd_feat_adapter is not None:
                    optimizer.add_param_group({
                        'params'     : list(trainer._kd_feat_adapter.parameters()),
                        'lr'         : args.lr,
                        'name'       : 'kd_adapter',
                        'initial_lr' : args.lr,
                    })
                    print("  kd_feat_adapter re-added to optimizer")

                # Tắt Dice + KD tạm thời sau unfreeze để ổn định gradient
                trainer.set_loss_phase('ce_only')
                print(f"  Loss → ce_only for {args.ce_only_epochs_after_unfreeze} epochs "
                      f"(gradient stabilization)\n")

                print("Learning rates after unfreezing:")
                for g in optimizer.param_groups:
                    print(f"  {g.get('name','?')}: {g['lr']:.2e}")
                print()

        # Switch back to full loss sau ce_only phase
        if unfreeze_list and trainer.loss_phase == 'ce_only':
            last_unfreeze = max((e for e in unfreeze_list if e in applied_unfreeze_stages
                                 and e <= epoch), default=None)
            if (last_unfreeze is not None
                    and epoch >= last_unfreeze + args.ce_only_epochs_after_unfreeze):
                trainer.set_loss_phase('full')

        # ------------------------------------------------------------------ #
        # Train + Validate
        # ------------------------------------------------------------------ #
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics   = trainer.validate(val_loader, epoch)

        log_dwsa_gamma(model, trainer.writer, epoch)

        # ------------------------------------------------------------------ #
        # Logging
        # ------------------------------------------------------------------ #
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | "
              f"OHEM: {train_metrics['ohem']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}", end="")
        if trainer.teacher is not None:
            print(f" | KD_feat: {train_metrics['kd_feat']:.4f} | "
                  f"KD_logit: {train_metrics['kd_logit']:.4f}", end="")
        print()
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | "
              f"mIoU: {val_metrics['miou']:.4f}  | "
              f"Acc: {val_metrics['accuracy']:.4f}")

        for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
            m = getattr(model.backbone, name, None)
            if m is not None and hasattr(m, 'gamma'):
                print(f"  {name}.gamma = {m.gamma.item():.6f}")
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
