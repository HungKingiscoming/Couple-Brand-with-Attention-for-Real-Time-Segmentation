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
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False

class _DummyWriter:
    """Fallback khi tensorboard/tensorflow bị conflict (thường xảy ra trên Kaggle).
    Ghi metrics ra CSV thay vì tensorboard.
    """
    def __init__(self, log_dir):
        import pathlib, csv
        self._log_dir = pathlib.Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._log_dir / "metrics.csv"
        self._file = open(self._csv_path, 'w', newline='')
        self._csv = csv.writer(self._file)
        self._csv.writerow(['tag', 'step', 'value'])
        self._file.flush()
        print(f"TensorBoard unavailable — logging metrics to {self._csv_path}")

    def add_scalar(self, tag, value, step):
        self._csv.writerow([tag, step, f"{value:.6f}"])
        self._file.flush()

    def close(self):
        self._file.close()

def _make_writer(log_dir):
    if _TENSORBOARD_AVAILABLE:
        try:
            return SummaryWriter(log_dir=str(log_dir))
        except Exception:
            pass
    return _DummyWriter(log_dir)
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Dynamic import — được chọn theo --model_variant
# fan_dwsa: model.py (FoggyAwareNorm + DWSA) — default
# fan_only: model_fan_only.py (FoggyAwareNorm, no DWSA)
# dwsa_only: model_dwsa_only.py (DWSA, no FoggyAwareNorm, BN stem)
from model.head.segmentation_head import GCNetHead
# GCNet được import dynamically trong main() theo args.model_variant
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


def load_pretrained_gcnet(model, ckpt_path, strict_match=False, variant="fan_dwsa"):
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

    # Skip markers: keys trong checkpoint không có trong model (expected mismatch)
    _v = variant
    _skip_base = ['.spp.', 'backbone.spp.']
    if 'dwsa' not in _v:  _skip_base.append('dwsa_stage')  # model không có DWSA
    if 'fan'  not in _v:  _skip_base += ['foggy', 'alpha', 'in_.']  # không có FAN
    expected_skip_markers = tuple(_skip_base)
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
    # Missing markers: keys trong model không có trong checkpoint (expected new)
    _miss = ['.1.bn.', 'spp.', 'loss_', 'fog_consistency']
    if 'dwsa' in _v: _miss.append('dwsa')
    if 'fan'  in _v: _miss += ['alpha', 'in_.', 'foggy', 'stem_conv1.1.', 'stem_conv2.1.']
    expected_missing_markers = tuple(_miss)
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
    _comp = []
    if 'dwsa' in _v: _comp.append('DWSA')
    if 'fan'  in _v: _comp.append('FoggyNorm')
    _comp.append('loss buffers')
    print(f"Expected missing: {n_expected_missing} keys ({'/ '.join(_comp)}) → OK\n")

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

    opt_type = getattr(args, 'optimizer', 'adamw').lower()
    if opt_type == 'sgd':
        momentum = getattr(args, 'sgd_momentum', 0.9)
        optimizer = torch.optim.SGD(
            groups, momentum=momentum,
            weight_decay=args.weight_decay, nesterov=True)
        print(f"Optimizer: SGD (momentum={momentum}, nesterov=True)")
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        print("Optimizer: AdamW (Discriminative LR)")

    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
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

    elif args.scheduler == 'cosine_wr':
        T_0 = getattr(args, 'cosine_wr_t0', 10)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=1e-7)
        print(f"CosineAnnealingWarmRestarts (T_0={T_0})")

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
    def __init__(self, ignore_index=255, keep_ratio=0.3, min_kept=100000,
                 thresh=None, class_weights=None):
        """OHEM Cross Entropy.
        thresh    : threshold-based OHEM như config gốc GCNet-S (thres=0.9).
                    Giữ pixel có max_prob < thresh (hard pixels).
        keep_ratio: ratio-based OHEM nếu thresh=None.
        """
        super().__init__()
        self.ignore_index  = ignore_index
        self.keep_ratio    = keep_ratio
        self.min_kept      = min_kept
        self.thresh        = thresh
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

        if self.thresh is not None:
            # Threshold-based: giữ pixel có confidence thấp (hard pixels)
            with torch.no_grad():
                max_probs = torch.softmax(logits.detach(), dim=1).max(1)[0].view(-1)[valid_mask]
                hard_mask = max_probs < self.thresh
                if hard_mask.sum() < self.min_kept:
                    _, idx    = torch.topk(max_probs, min(self.min_kept, n_valid), largest=False)
                    hard_mask = torch.zeros(n_valid, dtype=torch.bool, device=logits.device)
                    hard_mask[idx] = True
            valid_losses = valid_losses[hard_mask]
        else:
            # Ratio-based: giữ top-K% pixels có loss cao nhất
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


def freeze_backbone(model, variant='fan_dwsa'):
    """Freeze toàn bộ backbone.
    Tự động detect variant để giữ đúng components trainable:
      fan_dwsa : giữ DWSA + FoggyAwareNorm trainable
      fan_only : giữ FoggyAwareNorm trainable
      dwsa_only: giữ DWSA trainable
    """
    has_dwsa = hasattr(model.backbone, 'dwsa_stage4')
    has_fan  = (hasattr(model.backbone, 'stem_conv1') and
                len(model.backbone.stem_conv1) > 1 and
                hasattr(model.backbone.stem_conv1[1], 'alpha'))
    keep = []
    if has_dwsa: keep.append('DWSA')
    if has_fan:  keep.append('FoggyAwareNorm')
    print(f"Freezing backbone (keeping {' + '.join(keep) if keep else 'nothing'} trainable)...")

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
    if has_dwsa:
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
    if has_fan:
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
    def get_config(variant='fan_dwsa'):
        """variant: 'fan_dwsa' | 'fan_only' | 'dwsa_only'"""
        C = 32
        backbone_base = {
            "in_channels"          : 3,
            "channels"             : C,
            "ppm_channels"         : 128,
            "num_blocks_per_stage" : [4, 4, [5, 4], [5, 4], [2, 2]],
            "align_corners"        : False,
            "norm_cfg"             : dict(type='BN', requires_grad=True),
            "act_cfg"              : dict(type='ReLU', inplace=True),
            "deploy"               : False,
        }
        # fan_dwsa và fan_only cần dwsa_reduction; dwsa_only không có FAN
        if variant == 'fan_dwsa':
            backbone_base["dwsa_reduction"] = 8
        elif variant == 'dwsa_only':
            backbone_base["dwsa_reduction"] = 8
        # fan_only: không có dwsa_reduction param
        return {
            "backbone": backbone_base,
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
                "dice_weight"  : 1.0,
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
        feats  = self.backbone(x)      # training: (c4_feat, fused)
        logits = self.decode_head(feats)
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

        _ohem_kr  = getattr(args, 'ohem_keep_ratio', 0.3)
        _ohem_mk  = getattr(args, 'ohem_min_kept',   100000)
        _ohem_thr = getattr(args, 'ohem_thresh',     None)
        self.ohem = OHEMLoss(
            ignore_index=args.ignore_index,
            keep_ratio=_ohem_kr,
            min_kept=_ohem_mk,
            thresh=_ohem_thr,
            class_weights=class_weights,
        )
        if _ohem_thr:
            print(f"OHEM: threshold-based (thres={_ohem_thr}, min_kept={_ohem_mk})")
        else:
            print(f"OHEM: ratio-based (keep_ratio={_ohem_kr}, min_kept={_ohem_mk})")
        self.dice = DiceLoss(
            smooth=loss_cfg['dice_smooth'],
            ignore_index=args.ignore_index,
            class_weights=class_weights,   # FIX: truyền class_weights vào Dice
        )
        _ls = getattr(args, 'label_smoothing', 0.0)
        self.ce = nn.CrossEntropyLoss(
            weight=cw_device,
            ignore_index=args.ignore_index,
            label_smoothing=_ls,
        )
        if _ls > 0: print(f"Label smoothing: {_ls}")

        # ------------------------------------------------------------------ #
        # Distillation state
        # ------------------------------------------------------------------ #

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = _make_writer(self.save_dir / "tensorboard")

        self.save_config()
        self._print_config(loss_cfg)

    # ---------------------------------------------------------------------- #
    # Loss phase    # ---------------------------------------------------------------------- #
    # Loss phase                                                               #
    # ---------------------------------------------------------------------- #

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

        # FIX: SPP BN eval mode để tránh gradient inf ở spp.scales.N.bn
        if getattr(self.args, "freeze_spp_bn", False):
            spp = getattr(self.model.backbone, "spp", None)
            if spp is not None:
                for m in spp.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

        total_loss   = total_ohem = total_dice = 0.0
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
                    aux_exp    = getattr(self.args, 'aux_decay_exp', 0.9)
                    aux_weight = self.args.aux_weight * (1 - epoch / self.args.epochs) ** aux_exp
                    aux_loss   = self.ohem(c4_full, masks)
                    loss       = loss + aux_weight * aux_loss

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

            postfix = {
                'loss'    : f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ohem'    : f'{ohem_loss.item():.4f}',
                'dice'    : f'{dice_loss.item():.4f}',
                'lr'      : f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'max_g'   : f'{max_grad:.2f}',
            }
            pbar.set_postfix(postfix)

            if batch_idx % 50 == 0:
                clear_gpu_memory()

            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss',     loss.item() * self.args.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/ohem',     ohem_loss.item(),  self.global_step)
                self.writer.add_scalar('train/dice',     dice_loss.item(),  self.global_step)
                self.writer.add_scalar('train/lr',       self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/max_grad', max_grad,          self.global_step)

        n = len(loader)
        print(f"\nEpoch {epoch+1} — Max gradient: {max_grad_epoch:.2f}")
        torch.cuda.empty_cache()

        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        return {
            'loss'     : total_loss   / n,
            'ohem'     : total_ohem   / n,
            'dice'     : total_dice   / n,
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

    # Model variant
    parser.add_argument("--model_variant",         type=str,   default="fan_dwsa",
                        choices=["fan_dwsa", "fan_only", "dwsa_only"],
                        help="fan_dwsa: FoggyAwareNorm+DWSA | "
                             "fan_only: FAN only | "
                             "dwsa_only: DWSA only, BN stem")

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
    parser.add_argument("--use_class_weights",      action="store_true",
                        help="Compute class weights từ training data (chậm, chạy 1 lần).")
    parser.add_argument("--class_weights_file",     type=str, default=None,
                        help="Load precomputed weights từ file .pt (nhanh hơn compute lại). "
                             "Tạo bằng: python analyze_distribution.py --save_weights w.pt")
    parser.add_argument("--class_weights_method",   type=str, default="median_freq",
                        choices=["inverse_freq", "sqrt_inverse", "median_freq"],
                        help="Phương pháp tính weights. median_freq khuyến nghị cho Cityscapes.")


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
    parser.add_argument("--optimizer",          type=str,   default="adamw",
                        choices=["adamw", "sgd"],
                        help="adamw (default) hoặc sgd (như config gốc GCNet-S).")
    parser.add_argument("--sgd_momentum",       type=float, default=0.9,
                        help="Momentum cho SGD.")
    parser.add_argument("--grad_clip",          type=float, default=5.0)
    parser.add_argument("--aux_weight",         type=float, default=0.4)
    parser.add_argument("--aux_decay_exp",      type=float, default=0.9,
                        help="Exponent cho aux decay. 0.5=chậm hơn, giữ signal lâu hơn.")
    parser.add_argument("--dice_weight",        type=float, default=None,
                        help="Override dice loss weight. None=dùng config (1.0). "
                             "Tăng lên 1.5-2.0 để boost class nhỏ.")
    parser.add_argument("--label_smoothing",    type=float, default=0.0,
                        help="Label smoothing cho CE (0.05-0.1).")
    parser.add_argument("--ohem_keep_ratio",    type=float, default=0.3,
                        help="OHEM keep ratio (0.5 giữ nhiều hard pixels hơn).")
    parser.add_argument("--ohem_min_kept",      type=int,   default=100000)
    parser.add_argument("--ohem_thresh",        type=float, default=None,
                        help="Threshold-based OHEM như GCNet-S config (thres=0.9). "
                             "Nếu set, override keep_ratio.")
    parser.add_argument("--ce_weight",          type=float, default=None,
                        help="Override CE loss weight. None=dùng config (1.0).")
    parser.add_argument("--scheduler",          default="cosine",
                        choices=["onecycle", "poly", "cosine", "cosine_wr"])
    parser.add_argument("--cosine_wr_t0",       type=int, default=10,
                        help="Restart period cho cosine_wr (epochs).")
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3,
                        help="Số epoch chỉ dùng CE loss ngay sau mỗi lần unfreeze "
                             "(Dice + KD tắt tạm để tránh gradient shock).")

    # Image
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--high_res_h", type=int, default=640,
                        help="Resolution cao cho phase đầu (head-only training).")
    parser.add_argument("--high_res_w", type=int, default=1280)
    parser.add_argument("--high_res_epochs", type=int, default=0,
                        help="Số epochs train ở high_res trước khi switch về img_h/img_w. "
                             "0 = không dùng high resolution phase.")

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
    parser.add_argument("--freeze_stem_conv",  action="store_true", default=False,
                        help="Freeze stem_conv1/2 conv weights (giữ FoggyAwareNorm trainable). "
                             "Dùng khi resume để chống gradient inf ở stem.")
    parser.add_argument("--freeze_spp_bn",     action="store_true", default=False,
                        help="Set SPP BN sang eval mode khi train. "
                             "Loại bỏ gradient inf ở spp.scales.N.bn.")

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
    _variant_label = {
        'fan_dwsa' : 'FoggyAwareNorm + DWSA',
        'fan_only' : 'FoggyAwareNorm only',
        'dwsa_only': 'DWSA only (BN stem)',
    }.get(getattr(args, 'model_variant', 'fan_dwsa'), 'Unknown')
    print(f"GCNet v3 Training  |  {_variant_label}")
    print(f"{'='*70}")
    print(f"Device:     {device}")
    print(f"Image size: {args.img_h}x{args.img_w}")
    print(f"Epochs:     {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip:  {args.grad_clip}  |  AMP: {args.use_amp}")
    if getattr(args, "high_res_epochs", 0) > 0:
        print(f"High-res phase:  {args.high_res_h}x{args.high_res_w} for {args.high_res_epochs} epochs, then {args.img_h}x{args.img_w}")
    _v = getattr(args, 'model_variant', 'fan_dwsa')
    if 'dwsa' in _v:
        print(f"LR DWSA:    {args.lr * args.dwsa_lr_factor:.2e}  (factor={args.dwsa_lr_factor})")
    if 'fan' in _v:
        print(f"LR alpha:   {args.lr * args.alpha_lr_factor:.2e}  (factor={args.alpha_lr_factor})")
    if unfreeze_list:
        print(f"Unfreeze schedule: epochs {unfreeze_list}")
    print(f"{'='*70}\n")

    # Dynamic import GCNet theo variant
    variant = getattr(args, 'model_variant', 'fan_dwsa')
    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet
    elif variant == 'fan_only':
        from model.backbone.fan import GCNet
    elif variant == 'dwsa_only':
        from model.backbone.dwsa import GCNet
    else:
        raise ValueError(f"Unknown model_variant: {variant}")
    print(f"Model variant: {variant}")

    cfg = ModelConfig.get_config(variant=variant)
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

    # High-resolution dataloader cho phase đầu (nếu được yêu cầu)
    # Batch size nhỏ hơn vì resolution lớn hơn tốn memory
    high_res_train_loader = None
    if getattr(args, "high_res_epochs", 0) > 0:
        hi_h, hi_w = args.high_res_h, args.high_res_w
        # Tự động tính batch size phù hợp: scale theo diện tích
        area_ratio = (args.img_h * args.img_w) / (hi_h * hi_w)
        hi_batch   = max(4, int(args.batch_size * area_ratio))
        print(f"High-res dataloader: {hi_h}x{hi_w}, "
              f"batch_size={hi_batch} (auto from area ratio {area_ratio:.2f})")
        high_res_train_loader, _, _ = create_dataloaders(
            train_txt=args.train_txt,
            val_txt=args.val_txt,
            batch_size=hi_batch,
            num_workers=args.num_workers,
            img_size=(hi_h, hi_w),
            pin_memory=True,
            compute_class_weights=False,
            dataset_type=args.dataset_type,
        )
        print(f"  {len(high_res_train_loader)} batches/epoch at high res")
    print("Dataloaders ready\n")

    # ------------------------------------------------------------------ #
    # Class weights — load từ file hoặc dùng kết quả compute từ dataloader
    # ------------------------------------------------------------------ #
    if getattr(args, "class_weights_file", None):
        import pathlib
        cw_path = pathlib.Path(args.class_weights_file)
        if cw_path.exists():
            class_weights = torch.load(cw_path, map_location="cpu")
            print(f"Class weights loaded from: {cw_path}")
            print(f"  min={class_weights.min():.3f}  max={class_weights.max():.3f}  "
                  f"mean={class_weights.mean():.3f}")
        else:
            print(f"WARNING: {cw_path} not found — class_weights_file ignored")
            class_weights = None

    if class_weights is not None:
        print("\nClass weights summary:")
        cnames = ["road","sidewalk","building","wall","fence","pole",
                  "t.light","t.sign","vegetation","terrain","sky",
                  "person","rider","car","truck","bus","train","moto","bicycle"]
        for i, (name, w) in enumerate(zip(cnames, class_weights)):
            bar = "█" * int(w.item() * 4)
            print(f"  {name:<12} {w.item():5.2f}  {bar}")
        print()

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
        load_pretrained_gcnet(model, args.pretrained_weights,
                          variant=getattr(args, 'model_variant', 'fan_dwsa'))

    if args.freeze_backbone:
        freeze_backbone(model, variant=getattr(args, 'model_variant', 'fan_dwsa'))

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
            print(f"Forward pass OK:")
            print(f"  c4_logit: {c4_logit.shape}")
            print(f"  c6_logit: {c6_logit.shape}\n")
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

    # Override loss weights từ command line nếu được chỉ định
    if getattr(args, "dice_weight", None) is not None:
        trainer.dice_weight = args.dice_weight
        trainer.base_loss_cfg["dice_weight"] = args.dice_weight
        print(f"Dice weight overridden: {args.dice_weight}")
    if getattr(args, "ce_weight", None) is not None:
        trainer.ce_weight = args.ce_weight
        print(f"CE weight overridden: {args.ce_weight}")

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

    # Freeze stem conv weights nếu được yêu cầu
    # Giữ FoggyAwareNorm (alpha, bn, in_) trainable — chỉ freeze conv weight
    # Lý do: stem_conv1.0.weight nhận gradient từ toàn network → spike khi batch khó
    # Sau nhiều epochs fine-tune, conv stem đã converge, không cần train thêm
    if getattr(args, "freeze_stem_conv", False):
        frozen_stem = 0
        for stem_name in ["stem_conv1", "stem_conv2"]:
            module = getattr(model.backbone, stem_name, None)
            if module is None:
                continue
            for pname, param in module.named_parameters():
                # Chỉ freeze conv weight (index 0) — giữ FAN params
                is_fan = any(k in pname for k in ("alpha", "bn.", "in_."))
                if not is_fan:
                    param.requires_grad = False
                    frozen_stem += param.numel()
        print(f"Stem conv frozen: {frozen_stem:,} params "
              f"(FoggyAwareNorm still trainable)")
        # Rebuild optimizer để loại frozen params khỏi groups
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args, train_loader,
                                    start_epoch=trainer.start_epoch)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        print("Optimizer rebuilt after stem freeze")
        for g in optimizer.param_groups:
            print(f"  {g["name"]}: lr={g["lr"]:.2e}, params={len(g["params"])}")


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
        # Resolution switching: dùng high_res trong high_res_epochs đầu
        # ------------------------------------------------------------------ #
        hi_epochs = getattr(args, "high_res_epochs", 0)
        if hi_epochs > 0 and high_res_train_loader is not None and epoch < hi_epochs:
            active_loader = high_res_train_loader
            if epoch == 0:
                print(f"[Resolution] Using HIGH RES "
                      f"({args.high_res_h}x{args.high_res_w}) for epochs 1-{hi_epochs}")
        else:
            active_loader = train_loader
            if hi_epochs > 0 and epoch == hi_epochs:
                print(f"[Resolution] Switching to NORMAL RES "
                      f"({args.img_h}x{args.img_w}) from epoch {epoch+1}")
                # Reset scheduler để cosine decay tính lại từ epoch này
                scheduler = build_scheduler(
                    trainer.optimizer, args, train_loader,
                    start_epoch=epoch)
                trainer.scheduler = scheduler
                print("  Scheduler reset for remaining epochs")

        # ------------------------------------------------------------------ #
        # Train + Validate
        # ------------------------------------------------------------------ #
        train_metrics = trainer.train_epoch(active_loader, epoch)
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
        # Log per-class IoU — cực hữu ích để debug class nào đang kéo mIoU xuống
        CLASS_NAMES = ['road','sidewalk','building','wall','fence','pole',
                       'traffic_light','traffic_sign','vegetation','terrain',
                       'sky','person','rider','car','truck','bus',
                       'train','motorcycle','bicycle']
        for ci, (cname, ciou) in enumerate(
                zip(CLASS_NAMES, val_metrics['per_class_iou'])):
            trainer.writer.add_scalar(f'val/iou_{cname}', float(ciou), epoch)

        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
            # In per-class IoU khi đạt best model để dễ debug
            print("  Per-class IoU (best model):")
            cnames = ['road','sidewalk','building','wall','fence','pole',
                      'traffic_light','traffic_sign','vegetation','terrain',
                      'sky','person','rider','car','truck','bus',
                      'train','motorcycle','bicycle']
            low_classes = []
            for cn, ciou in zip(cnames, val_metrics['per_class_iou']):
                mark = ' ←LOW' if ciou < 0.4 else ''
                print(f"    {cn:<16} {ciou:.4f}{mark}")
                if ciou < 0.4: low_classes.append(cn)
            if low_classes:
                print(f"  Classes below 0.40: {low_classes}")
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)

    trainer.writer.close()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
