# ============================================
# train.py — GCNet v3 + GCNetHead v2
# KNOWLEDGE RETENTION ADDITIONS (phase 3):
#   K1. BN Statistics Reset + Warmup — reset running stats, momentum cao 3-5 epoch đầu
#   K2. Layer-wise Learning Rate Decay (LLRD) — LR decay theo depth
#   K3. Knowledge Distillation — teacher=checkpoint 0.7191, student=model đang train
#   K4. Elastic Weight Consolidation (EWC) — bảo vệ weight quan trọng
#   K5. Progressive Resolution Transition — giảm dần resolution thay vì jump thẳng
# LOGGING ADDITIONS (trên bản gốc):
#   L1. DiagnosticLogger
#   L2. DWSA health check
#   L3. FoggyAwareNorm alpha monitor
#   L4. Per-class IoU every epoch
#   L5. Gradient flow map
#   L6. SPP BN health check
#   L7. Loss decomposition trend
#   L8. Learning rate tracker
#   L9. Batch-level hard pixel ratio
#   L10. Epoch summary table
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import math

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False


class _DummyWriter:
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

# Module-level separator constant used across all print functions
SEP = "=" * 70


from model.head.segmentation_head import GCNetHead
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn, init_weights, check_model_health


# ============================================
# L1. DIAGNOSTIC LOGGER
# ============================================

class DiagnosticLogger:
    def __init__(self, save_dir: Path, class_names: list):
        self.save_dir    = Path(save_dir)
        self.class_names = class_names
        self.history     = defaultdict(list)
        self._csv_path   = self.save_dir / "diagnostics.csv"
        self._csv_file   = open(self._csv_path, 'w', newline='')
        import csv
        self._csv        = csv.writer(self._csv_file)
        self._csv.writerow(['epoch', 'key', 'value'])
        self._csv_file.flush()

    def log(self, epoch: int, key: str, value: float):
        self.history[key].append((epoch, value))
        self._csv.writerow([epoch, key, f"{value:.6f}"])
        self._csv_file.flush()

    def log_dict(self, epoch: int, d: dict, prefix: str = ''):
        for k, v in d.items():
            self.log(epoch, f"{prefix}{k}" if prefix else k, float(v))

    def print_epoch_summary(self, epoch: int):
        print(f"\n{'─'*80}")
        print(f"  EPOCH {epoch+1:>3} SUMMARY")
        print(f"{'─'*80}")
        print(f"  {'Metric':<28}  {'Value':>10}  {'Trend':>12}")
        print(f"  {'─'*28}  {'─'*10}  {'─'*12}")

        metrics = [
            ('val/miou',          'Val mIoU',        '.4f'),
            ('val/accuracy',      'Val Accuracy',    '.4f'),
            ('val/loss',          'Val Loss',        '.4f'),
            ('train/ohem',        'Train OHEM',      '.4f'),
            ('train/dice',        'Train Dice',      '.4f'),
            ('train/max_grad',    'Max Gradient',    '.3f'),
            ('dwsa/gamma4',       'DWSA gamma4',     '.4f'),
            ('dwsa/gamma5',       'DWSA gamma5',     '.4f'),
            ('dwsa/gamma6',       'DWSA gamma6',     '.4f'),
            ('fan/alpha1_mean',   'FAN alpha1 mean', '.4f'),
            ('fan/alpha2_mean',   'FAN alpha2 mean', '.4f'),
            ('train/hard_ratio',  'OHEM hard ratio', '.3f'),
        ]
        for key, label, fmt in metrics:
            h = self.history.get(key, [])
            if not h:
                continue
            val = h[-1][1]
            if len(h) >= 3:
                delta = h[-1][1] - h[-3][1]
                arrow = '↑' if delta > 1e-4 else ('↓' if delta < -1e-4 else '→')
                trend = f"{arrow} {abs(delta):.4f}"
            else:
                trend = '(new)'
            print(f"  {label:<28}  {val:>10{fmt}}  {trend:>12}")
        print(f"{'─'*80}\n")

    def print_full_history(self):
        print(f"\n{'='*80}")
        print("  FULL TRAINING HISTORY")
        print(f"{'='*80}")
        key_miou = self.history.get('val/miou', [])
        if key_miou:
            best_ep, best_val = max(key_miou, key=lambda x: x[1])
            print(f"  Best mIoU: {best_val:.4f} at epoch {best_ep+1}")
            print(f"  Final mIoU: {key_miou[-1][1]:.4f}")
            if len(key_miou) >= 10:
                last10 = [v for _, v in key_miou[-10:]]
                spread = max(last10) - min(last10)
                print(f"  Last-10 mIoU spread: {spread:.4f} "
                      f"{'← PLATEAU' if spread < 0.003 else ''}")
        print(f"\n  Epoch │ mIoU   │ OHEM   │ Dice   │ gamma4 │ gamma5 │ hard%")
        print(f"  {'─'*65}")
        n_epochs = max((len(v) for v in self.history.values()), default=0)
        for i in range(n_epochs):
            def _g(key):
                h = self.history.get(key, [])
                return h[i][1] if i < len(h) else float('nan')
            miou   = _g('val/miou')
            ohem   = _g('train/ohem')
            dice   = _g('train/dice')
            g4     = _g('dwsa/gamma4')
            g5     = _g('dwsa/gamma5')
            hratio = _g('train/hard_ratio')
            mark   = ' ← BEST' if not math.isnan(miou) and miou == max(
                v for _, v in self.history.get('val/miou', [(0,0)])) else ''
            print(f"  {i+1:>5} │ {miou:.4f} │ {ohem:.4f} │ {dice:.4f} │ "
                  f"{g4:.4f} │ {g5:.4f} │ {hratio:.3f}{mark}")
        print(f"{'='*80}\n")
        print(f"  Diagnostics CSV saved → {self._csv_path}")

    def close(self):
        self._csv_file.close()


# ============================================
# K1. BN STATISTICS RESET + WARMUP
# Giải quyết vấn đề BN stats mismatch khi chuyển resolution
# ============================================

def reset_bn_stats_with_warmup(model, reset_momentum: float = 0.3):
    """
    Reset BN running stats và set momentum cao để BN adapt nhanh
    về distribution mới trong vài epoch đầu.

    Bình thường momentum=0.1 (update 10% mỗi batch).
    reset_momentum=0.3 → BN adapt nhanh 3x trong warmup period.

    Gọi SAU khi load checkpoint, TRƯỚC training loop.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = reset_momentum
            count += 1
    print(f"  🔄 K1: Reset {count} BN layers running stats, "
          f"momentum={reset_momentum} (warmup mode)")
    return count


def restore_bn_momentum(model, momentum: float = 0.1):
    """
    Sau bn_warmup_epochs, restore momentum về giá trị bình thường.
    Gọi tại đầu epoch sau khi warmup kết thúc.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            count += 1
    print(f"  ✅ K1: BN momentum restored to {momentum} ({count} layers)")


# ============================================
# K2. LAYER-WISE LEARNING RATE DECAY (LLRD)
# Layer sâu hơn (gần output) → LR cao hơn
# Layer nông hơn (gần input) → LR thấp hơn → bảo vệ knowledge đã học
# ============================================

def build_llrd_optimizer(model, base_lr: float = 5e-6,
                         decay_factor: float = 0.7,
                         weight_decay: float = 0.01,
                         dwsa_lr_multiplier: float = 2.0,
                         alpha_lr_multiplier: float = 0.5):
    """
    Layer-wise LR Decay:
      head            → base_lr * 1.0   (adapt nhanh nhất)
      semantic_branch[2] → base_lr * decay^1
      semantic_branch[1] → base_lr * decay^2
      semantic_branch[0] → base_lr * decay^3
      spp/compression → base_lr * decay^4
      detail_branch   → base_lr * decay^4
      stem            → base_lr * decay^5  (bảo vệ nhất)
      dwsa            → base_lr * dwsa_lr_multiplier  (cần học nhiều)
      alpha (FAN)     → base_lr * alpha_lr_multiplier
    """
    # Collect params theo layer group
    assigned = set()

    def get_params_by_name(name_fragments, exclude_assigned=True):
        params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_assigned and id(p) in assigned:
                continue
            if any(frag in n for frag in name_fragments):
                params.append(p)
                assigned.add(id(p))
        return params

    groups = []

    # DWSA — LR cao nhất để học attention pattern
    dwsa_p = get_params_by_name(['dwsa_stage'])
    if dwsa_p:
        groups.append({
            'params': dwsa_p,
            'lr': base_lr * dwsa_lr_multiplier,
            'name': 'dwsa'
        })

    # FAN alpha — LR thấp để không phá foggy adaptation
    alpha_p = get_params_by_name(['alpha'])
    if alpha_p:
        groups.append({
            'params': alpha_p,
            'lr': base_lr * alpha_lr_multiplier,
            'name': 'alpha'
        })

    # Head — adapt nhanh
    head_p = get_params_by_name(['decode_head'])
    if head_p:
        groups.append({
            'params': head_p,
            'lr': base_lr,
            'name': 'head'
        })

    # Semantic branch theo depth (layer sâu → LR cao hơn)
    for depth_idx, decay_pow in [(2, 1), (1, 2), (0, 3)]:
        sem_p = get_params_by_name([f'semantic_branch_layers.{depth_idx}'])
        if sem_p:
            groups.append({
                'params': sem_p,
                'lr': base_lr * (decay_factor ** decay_pow),
                'name': f'semantic_{depth_idx}'
            })

    # SPP + compression layers
    spp_p = get_params_by_name(['backbone.spp', 'compression_', 'down_'])
    if spp_p:
        groups.append({
            'params': spp_p,
            'lr': base_lr * (decay_factor ** 4),
            'name': 'spp_compress'
        })

    # Detail branch
    detail_p = get_params_by_name(['detail_branch_layers'])
    if detail_p:
        groups.append({
            'params': detail_p,
            'lr': base_lr * (decay_factor ** 4),
            'name': 'detail'
        })

    # Stem — bảo vệ nhất (đã converge tốt từ nhiều run trước)
    stem_p = get_params_by_name(['stem_conv1', 'stem_conv2', 'stem_stage2', 'stem_stage3'])
    if stem_p:
        groups.append({
            'params': stem_p,
            'lr': base_lr * (decay_factor ** 5),
            'name': 'stem'
        })

    # Remaining params (nếu có)
    remaining = [p for n, p in model.named_parameters()
                 if p.requires_grad and id(p) not in assigned]
    if remaining:
        groups.append({
            'params': remaining,
            'lr': base_lr * (decay_factor ** 3),
            'name': 'other'
        })

    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)

    print(f"\n  K2: LLRD Optimizer (base_lr={base_lr:.2e}, decay={decay_factor}):")
    for g in groups:
        print(f"    {g['name']:<16} lr={g['lr']:.2e}  ({len(g['params'])} tensors)")

    # Set initial_lr cho scheduler
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])

    return optimizer


# ============================================
# K3. KNOWLEDGE DISTILLATION
# Teacher = checkpoint 0.7191 (frozen, chạy ở resolution cao hơn nếu cần)
# Student = model đang train
# Loss = (1-alpha) * task_loss + alpha * KL(student || teacher)
# ============================================

class KnowledgeDistillationLoss(nn.Module):
    """
    Self-distillation: dùng checkpoint tốt nhất làm teacher để
    guide model trong quá trình adapt sang resolution mới.

    temperature > 1 làm softmax distribution mềm hơn → student học
    được uncertainty của teacher, không chỉ học hard labels.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.kl    = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, task_loss):
        """
        Args:
            student_logits: (B, C, H, W) — output của model đang train
            teacher_logits: (B, C, H', W') — output của teacher (có thể khác size)
            task_loss: scalar — CE+Dice loss đã tính trước
        """
        # Resize teacher về cùng size với student nếu khác nhau
        if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
            teacher_logits = F.interpolate(
                teacher_logits.float(),
                size=student_logits.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # KL divergence với temperature scaling
        student_soft  = F.log_softmax(student_logits.float() / self.T, dim=1)
        teacher_soft  = F.softmax(teacher_logits.float() / self.T, dim=1)
        kl_loss       = self.kl(student_soft, teacher_soft) * (self.T ** 2)

        total_loss = (1.0 - self.alpha) * task_loss + self.alpha * kl_loss
        return total_loss, kl_loss.item()


def load_teacher_model(ckpt_path: str, model_class, cfg: dict,
                       head_class, head_cfg: dict,
                       num_classes: int, ignore_index: int,
                       device: str):
    """
    Load teacher model từ checkpoint — frozen hoàn toàn.
    Teacher chạy ở eval mode, không có gradient.
    """
    from model.backbone.model import GCNet as GCNetFanDwsa

    backbone = model_class(**cfg)
    head     = head_class(**head_cfg, num_classes=num_classes,
                          ignore_index=ignore_index)

    class _Seg(nn.Module):
        def __init__(self, bb, hd):
            super().__init__()
            self.backbone    = bb
            self.decode_head = hd
        def forward(self, x):
            return self.decode_head(self.backbone(x))

    teacher = _Seg(backbone, head).to(device)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (ckpt.get('model') or ckpt.get('model_state_dict') or
             ckpt.get('state_dict') or ckpt)
    teacher.load_state_dict(state, strict=False)

    # Freeze hoàn toàn
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"  K3: Teacher loaded from {ckpt_path}")
    print(f"      {n_params:,} params, fully frozen, eval mode")
    return teacher


# ============================================
# K4. ELASTIC WEIGHT CONSOLIDATION (EWC)
# Tính Fisher Information Matrix để xác định weight quan trọng
# Penalize thay đổi lớn ở những weight quan trọng
# ============================================

class EWCRegularizer:
    """
    EWC bảo vệ các weight quan trọng với task cũ (640×1280).
    
    Fisher Information ≈ gradient² → weight nào có gradient lớn
    trong task cũ → quan trọng → penalize nếu thay đổi.
    
    Penalty = lambda * Σ F_i * (θ_i - θ*_i)²
    """
    def __init__(self, model, dataloader, device,
                 n_samples: int = 200, ewc_lambda: float = 500.0):
        self.ewc_lambda = ewc_lambda
        self.device     = device

        print(f"\n  K4: Computing Fisher Information Matrix "
              f"({n_samples} samples)...")

        # Lưu tham số tham chiếu (θ*)
        self.params_ref = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # Tính Fisher
        self.fisher = self._compute_fisher(model, dataloader, n_samples)

        total_fisher = sum(f.sum().item() for f in self.fisher.values())
        print(f"  K4: Fisher computed. Total Fisher mass: {total_fisher:.4f}")
        print(f"  K4: EWC lambda: {ewc_lambda}")

    def _compute_fisher(self, model, dataloader, n_samples: int):
        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.eval()
        count = 0
        for imgs, masks in dataloader:
            if count >= n_samples:
                break
            imgs = imgs.to(self.device)
            with torch.no_grad():
                logits = model(imgs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[-1]
                logits_full = F.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False)
                probs = F.softmax(logits_full.float(), dim=1)

            # Sample từ model distribution (không dùng ground truth)
            B, C, H, W = probs.shape
            probs_flat  = probs.permute(0,2,3,1).reshape(-1, C)
            sampled     = torch.multinomial(probs_flat, 1).reshape(B, H, W)

            model.zero_grad()
            with torch.enable_grad():
                logits2 = model(imgs)
                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[-1]
                logits2_full = F.interpolate(
                    logits2, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False)
                loss = F.cross_entropy(
                    logits2_full, sampled.to(self.device),
                    ignore_index=255)
                loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

            count += imgs.size(0)
            if count % 50 == 0:
                print(f"    Fisher progress: {count}/{n_samples}")

        for n in fisher:
            fisher[n] /= max(count, 1)

        model.zero_grad()
        model.train()
        return fisher

    def penalty(self, model) -> torch.Tensor:
        """Tính EWC penalty — thêm vào total loss."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.params_ref:
                loss = loss + (
                    self.fisher[n] * (p - self.params_ref[n]).pow(2)
                ).sum()
        return self.ewc_lambda * loss


# ============================================
# K5. PROGRESSIVE RESOLUTION TRANSITION
# Giảm dần resolution thay vì jump thẳng 640→512
# Mỗi bước giảm → BN adapt từng phần → ít disruption hơn
# ============================================

class ProgressiveResolutionScheduler:
    """
    Lịch giảm resolution theo epoch:
      epoch 0–2:   608×1216  (bước 1)
      epoch 3–5:   576×1152  (bước 2)
      epoch 6–8:   544×1088  (bước 3)
      epoch 9+:    512×1024  (target)

    Có thể customize qua resolution_schedule dict.
    """
    def __init__(self, resolution_schedule: dict,
                 train_txt: str, val_txt: str,
                 batch_size: int, num_workers: int,
                 dataset_type: str, use_class_weights: bool = False):
        self.schedule      = sorted(resolution_schedule.items())
        self.train_txt     = train_txt
        self.val_txt       = val_txt
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.dataset_type  = dataset_type
        self.use_cw        = use_class_weights
        self.current_res   = None
        self.current_loader = None
        self._last_change_epoch = -999

        print(f"\n  K5: Progressive Resolution Schedule:")
        prev_ep = 0
        for start_ep, (h, w) in self.schedule:
            print(f"    epoch {prev_ep:>2}–{start_ep-1 if start_ep > 0 else '?':>2}: "
                  f"{h}×{w}")
            prev_ep = start_ep
        print(f"    epoch {prev_ep:>2}+    : target resolution")

    def get_loader(self, epoch: int, val_loader):
        """
        Trả về train_loader phù hợp với epoch hiện tại.
        Rebuild nếu resolution thay đổi.
        """
        target_res = self.schedule[0][1]  # default: resolution đầu tiên
        for start_ep, res in self.schedule:
            if epoch >= start_ep:
                target_res = res

        if target_res != self.current_res:
            h, w = target_res
            print(f"\n  K5: Resolution transition → {h}×{w} (epoch {epoch+1})")
            train_loader, _, _ = create_dataloaders(
                train_txt=self.train_txt,
                val_txt=self.val_txt,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                img_size=(h, w),
                pin_memory=True,
                compute_class_weights=False,
                dataset_type=self.dataset_type
            )
            self.current_res    = target_res
            self.current_loader = train_loader
            self._last_change_epoch = epoch
            return train_loader, True  # True = resolution changed

        return self.current_loader, False

    def should_boost_bn_momentum(self, epoch: int,
                                 boost_duration: int = 2) -> bool:
        """Sau mỗi resolution change, boost BN momentum trong boost_duration epoch."""
        return (epoch - self._last_change_epoch) < boost_duration


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
                f"{ckpt_key}: ckpt{list(ckpt_val.shape)} vs model{list(head_state[dst_suffix].shape)}")
            continue
        head_loaded[dst_suffix] = ckpt_val

    loaded_bb = len(compatible);  loaded_hd = len(head_loaded)
    total_bb  = len(model_state); total_hd  = len(head_state)
    _v = variant
    _skip_base = ['.spp.', 'backbone.spp.']
    if 'dwsa' not in _v: _skip_base.append('dwsa_stage')
    if 'fan'  not in _v: _skip_base += ['foggy', 'alpha', 'in_.']
    expected_skip_markers = tuple(_skip_base)
    truly_unmatched = [k for k in skipped
                       if not any(s in k for s in expected_skip_markers)]

    print(f"\n{SEP}\nWEIGHT LOADING SUMMARY\n{SEP}")
    print(f"Backbone:  {loaded_bb:>5} / {total_bb}  ({100*loaded_bb/max(total_bb,1):.1f}%)")
    print(f"Head:      {loaded_hd:>5} / {total_hd}  ({100*loaded_hd/max(total_hd,1):.1f}%)")
    print(f"BN dropped (expected): {len(bn_dropped):>3}")
    if head_skipped_shape:
        print(f"Head shape mismatch:   {len(head_skipped_shape)}")
        for s in head_skipped_shape: print(f"    SHAPE: {s}")
    if head_skipped_missing:
        print(f"Head key missing:      {len(head_skipped_missing)}")
        for s in head_skipped_missing: print(f"    MISSING: {s}")
    if truly_unmatched:
        print(f"Backbone unmatched:    {len(truly_unmatched)}  ← cần kiểm tra")
        for k in truly_unmatched[:5]: print(f"    {k}")
    print(sep + "\n")

    missing_bb, _ = model.backbone.load_state_dict(compatible, strict=False)
    missing_hd, _ = model.decode_head.load_state_dict(head_loaded, strict=False)

    _miss = ['.1.bn.', 'spp.', 'loss_', 'fog_consistency']
    if 'dwsa' in _v: _miss.append('dwsa')
    if 'fan'  in _v: _miss += ['alpha', 'in_.', 'foggy',
                                'stem_conv1.1.', 'stem_conv2.1.']
    expected_missing_markers = tuple(_miss)
    unexpected_bb = [k for k in missing_bb
                     if not any(s in k for s in expected_missing_markers)]
    unexpected_hd = [k for k in missing_hd
                     if not any(s in k for s in expected_missing_markers)]
    if unexpected_bb:
        print(f"Unexpected backbone missing ({len(unexpected_bb)}):")
        for k in unexpected_bb[:5]: print(f"  - {k}")
    if unexpected_hd:
        print(f"Unexpected head missing ({len(unexpected_hd)}):")
        for k in unexpected_hd[:5]: print(f"  - {k}")

    return 100 * (loaded_bb + loaded_hd) / max(total_bb + total_hd, 1)


# ============================================
# OPTIMIZER (standard + LLRD variant)
# ============================================

def build_optimizer(model, args):
    STEM_MODULES = {'stem_conv1', 'stem_conv2', 'stem_stage2', 'stem_stage3'}

    # Nếu dùng LLRD, delegate sang build_llrd_optimizer
    if getattr(args, 'use_llrd', False):
        return build_llrd_optimizer(
            model,
            base_lr=args.lr,
            decay_factor=getattr(args, 'llrd_decay', 0.7),
            weight_decay=args.weight_decay,
            dwsa_lr_multiplier=getattr(args, 'dwsa_lr_factor', 2.0),
            alpha_lr_multiplier=getattr(args, 'alpha_lr_factor', 0.5),
        )

    dwsa_params     = []
    alpha_params    = []
    stem_params     = []
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
            parts       = name.split('.')
            module_name = parts[1] if len(parts) > 1 else ''
            if module_name in STEM_MODULES:
                stem_params.append(param)
            else:
                backbone_params.append(param)
        else:
            head_params.append(param)

    stem_lr_factor = getattr(args, 'stem_lr_factor', 0.001)

    groups = []
    if head_params:
        groups.append({'params': head_params,
                       'lr': args.lr, 'name': 'head'})
    if backbone_params:
        groups.append({'params': backbone_params,
                       'lr': args.lr * args.backbone_lr_factor,
                       'name': 'backbone'})
    if stem_params:
        groups.append({'params': stem_params,
                       'lr': args.lr * stem_lr_factor,
                       'name': 'stem'})
    if dwsa_params:
        groups.append({'params': dwsa_params,
                       'lr': args.lr * args.dwsa_lr_factor,
                       'name': 'dwsa'})
    if alpha_params:
        groups.append({'params': alpha_params,
                       'lr': args.lr * args.alpha_lr_factor,
                       'name': 'alpha'})

    opt_type = getattr(args, 'optimizer', 'adamw').lower()
    if opt_type == 'sgd':
        momentum  = getattr(args, 'sgd_momentum', 0.9)
        optimizer = torch.optim.SGD(
            groups, momentum=momentum,
            weight_decay=args.weight_decay, nesterov=True)
        print(f"Optimizer: SGD (momentum={momentum}, nesterov=True)")
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        print("Optimizer: AdamW (Discriminative LR)")

    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
        print(f"  group '{g['name']}': lr={g['lr']:.2e}, "
              f"params={len(g['params'])}")

    return optimizer


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    n_groups   = len(optimizer.param_groups)
    use_cosine = args.freeze_backbone and args.unfreeze_schedule

    if use_cosine or args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - start_epoch, eta_min=1e-6)
        print("CosineAnnealingLR")
    elif args.scheduler == 'onecycle':
        remaining_epochs = args.epochs - start_epoch
        total_steps      = len(train_loader) * remaining_epochs
        max_lrs = (args.lr if n_groups == 1
                   else [g['initial_lr'] for g in optimizer.param_groups])
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_steps,
            pct_start=0.05, anneal_strategy='cos',
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
            div_factor=25, final_div_factor=100000)
        print(f"OneCycleLR (total_steps={total_steps})")
    elif args.scheduler == 'cosine_wr':
        T_0 = getattr(args, 'cosine_wr_t0', 10)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=1e-7)
        print(f"CosineAnnealingWarmRestarts (T_0={T_0})")
    else:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9)
        print("Polynomial LR decay")
    return scheduler


# ============================================
# LOSS FUNCTIONS
# ============================================

class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, keep_ratio=0.3,
                 min_kept=100000, thresh=None, class_weights=None):
        super().__init__()
        self.ignore_index    = ignore_index
        self.keep_ratio      = keep_ratio
        self.min_kept        = min_kept
        self.thresh          = thresh
        self.class_weights   = class_weights
        self.last_hard_ratio = 0.0

    def forward(self, logits, labels):
        weight = (self.class_weights.to(logits.device)
                  if self.class_weights is not None else None)

        loss_pixel = F.cross_entropy(
            logits.float(), labels,
            weight=weight.float() if weight is not None else None,
            ignore_index=self.ignore_index,
            reduction='none'
        ).view(-1)

        valid_mask   = (labels.view(-1) != self.ignore_index)
        valid_losses = loss_pixel[valid_mask]
        n_valid      = valid_losses.numel()
        if n_valid == 0:
            self.last_hard_ratio = 0.0
            return logits.sum() * 0

        if self.thresh is not None:
            with torch.no_grad():
                max_probs = torch.softmax(
                    logits.detach().float(), dim=1
                ).max(1)[0].view(-1)[valid_mask]
                hard_mask = max_probs < self.thresh
                if hard_mask.sum() < self.min_kept:
                    _, idx    = torch.topk(max_probs,
                                           min(self.min_kept, n_valid),
                                           largest=False)
                    hard_mask = torch.zeros(n_valid, dtype=torch.bool,
                                            device=logits.device)
                    hard_mask[idx] = True
            self.last_hard_ratio = hard_mask.float().mean().item()
            valid_losses = valid_losses[hard_mask]
        else:
            n_keep = max(int(self.keep_ratio * n_valid),
                         min(self.min_kept, n_valid))
            n_keep = min(n_keep, n_valid)
            self.last_hard_ratio = n_keep / n_valid
            if n_keep < n_valid:
                threshold    = torch.sort(
                    valid_losses, descending=True)[0][n_keep - 1].detach()
                valid_losses = valid_losses[valid_losses >= threshold]

        return valid_losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255,
                 log_loss=False, class_weights=None):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.log_loss     = log_loss
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None else None)

    def forward(self, logits, targets):
        logits = logits.float()
        B, C, H, W  = logits.shape
        valid_mask  = (targets != self.ignore_index)
        tgt_clamped = targets.clamp(0, C - 1)
        tgt_one_hot = (F.one_hot(tgt_clamped, C)
                       .permute(0, 3, 1, 2).float())
        tgt_one_hot = tgt_one_hot * valid_mask.unsqueeze(1).float()
        probs       = (F.softmax(logits, dim=1)
                       * valid_mask.unsqueeze(1).float())
        probs_flat  = probs.reshape(B, C, -1)
        target_flat = tgt_one_hot.reshape(B, C, -1)
        intersection = (probs_flat * target_flat).sum(2)
        cardinality  = probs_flat.sum(2) + target_flat.sum(2)
        dice_score   = ((2.0 * intersection + self.smooth)
                        / (cardinality + self.smooth))
        dice_loss = (
            -torch.log(dice_score.clamp(min=self.smooth))
            if self.log_loss else 1.0 - dice_score)
        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights.float().unsqueeze(0)
        class_present = target_flat.sum(2) > 0
        dice_loss     = dice_loss * class_present.float()
        n_present     = class_present.float().sum(1).clamp(min=1)
        return (dice_loss.sum(1) / n_present).mean()


# ============================================
# UTILITIES
# ============================================

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_memory_efficient_training():
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def check_gradients(model, threshold=10.0):
    # Chỉ track max_grad — bỏ grad_map để tránh iterate 480+ params mỗi step
    max_grad = 0.0; max_name = ""
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            if g > max_grad:
                max_grad = g; max_name = name
    if max_grad > threshold:
        print(f"Large gradient: {max_name[:60]}... = {max_grad:.2f}")
    return max_grad


def debug_inf_gradients(model, batch_idx: int, loss_components: dict):
    """
    Debug chi tiết khi phát hiện inf/nan gradient.
    In ra:
    - Loss components (task, kl, ewc) tại batch này
    - Tất cả layers có inf/nan gradient (không chỉ top-1)
    - Phân loại theo module (spp, detail, semantic, head, dwsa)
    Gọi ngay sau unscale_, trước clip_grad_norm_.
    """
    inf_layers   = []
    nan_layers   = []
    large_layers = []  # > 1000 nhưng không phải inf/nan

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.norm().item()
        if not math.isfinite(g):
            if math.isnan(g):
                nan_layers.append(name)
            else:
                inf_layers.append((name, g))
        elif g > 1000:
            large_layers.append((name, g))

    if not inf_layers and not nan_layers:
        return  # không có vấn đề, không in gì

    print(f"\n  {SEP}")
    print(f"  🔴 INF/NAN GRADIENT DEBUG — batch {batch_idx}")
    print(f"  {SEP}")

    # Loss components
    print(f"  Loss components:")
    for k, v in loss_components.items():
        v_str = f"{v:.4f}" if math.isfinite(v) else f"⚠️ {v}"
        print(f"    {k:<20} {v_str}")

    # Phân loại inf layers theo module
    module_groups = {
        'spp':          [],
        'detail_branch': [],
        'semantic_branch': [],
        'dwsa':         [],
        'decode_head':  [],
        'stem':         [],
        'other':        [],
    }
    for name, g in inf_layers:
        if 'spp' in name:
            module_groups['spp'].append(name)
        elif 'detail_branch' in name:
            module_groups['detail_branch'].append(name)
        elif 'semantic_branch' in name:
            module_groups['semantic_branch'].append(name)
        elif 'dwsa' in name:
            module_groups['dwsa'].append(name)
        elif 'decode_head' in name:
            module_groups['decode_head'].append(name)
        elif 'stem' in name:
            module_groups['stem'].append(name)
        else:
            module_groups['other'].append(name)

    print(f"  INF gradient layers ({len(inf_layers)} total):")
    for group, layers in module_groups.items():
        if layers:
            print(f"    [{group}] {len(layers)} layers:")
            for l in layers[:3]:
                print(f"      - {l}")
            if len(layers) > 3:
                print(f"      ... and {len(layers)-3} more")

    if nan_layers:
        print(f"  NAN gradient layers ({len(nan_layers)}):")
        for l in nan_layers[:5]:
            print(f"    - {l}")

    if large_layers:
        top3 = sorted(large_layers, key=lambda x: x[1], reverse=True)[:3]
        print(f"  LARGE (>1000) gradient layers:")
        for name, g in top3:
            print(f"    {name[-55:]}: {g:.1f}")

    print(f"  {SEP}\n")



def log_gradient_flow(grad_map: dict, writer, epoch: int, top_k: int = 5):
    if not grad_map:
        return
    sorted_grads = sorted(grad_map.items(), key=lambda x: x[1], reverse=True)
    inf_layers   = [(n, g) for n, g in sorted_grads if not math.isfinite(g)]
    top_finite   = [(n, g) for n, g in sorted_grads if math.isfinite(g)][:top_k]

    if inf_layers:
        print(f"  ⚠️  INF gradient layers ({len(inf_layers)}):")
        for n, g in inf_layers[:3]:
            print(f"      {n[:70]}")

    print(f"  Top-{top_k} gradient layers (epoch {epoch+1}):")
    for i, (name, g) in enumerate(top_finite):
        bar = '█' * min(int(g * 5), 30)
        print(f"    {i+1}. {name[-55:]:<55} {g:7.4f}  {bar}")

    if top_finite:
        writer.add_scalar('grad/top1_norm', top_finite[0][1], epoch)
    bottom5 = [(n, g) for n, g in reversed(sorted_grads)
               if math.isfinite(g) and g > 0][:5]
    if bottom5:
        min_grad = bottom5[-1][1]
        writer.add_scalar('grad/min_norm', min_grad, epoch)
        if min_grad < 1e-7:
            print(f"  ⚠️  VANISHING gradient: "
                  f"{bottom5[-1][0][-55:]} = {min_grad:.2e}")


def log_dwsa_health(model, writer, epoch: int, diag: DiagnosticLogger):
    print(f"\n  DWSA Health (epoch {epoch+1}):")
    print(f"  {'Stage':<12} {'gamma':>8}  {'Δgamma':>8}  {'Status'}")
    print(f"  {'─'*50}")

    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        module = getattr(model.backbone, name, None)
        if module is None:
            continue
        gamma_val = module.gamma.item()
        diag.log(epoch, f'dwsa/{"gamma4" if "4" in name else "gamma5" if "5" in name else "gamma6"}',
                 gamma_val)

        if gamma_val < 0.11:
            status = '⚠️  NOT LEARNING'
        elif gamma_val < 0.2:
            status = '📈 Warming up'
        elif gamma_val < 0.4:
            status = '✅ Active'
        else:
            status = '🔥 Highly active'

        key = f'dwsa/{"gamma4" if "4" in name else "gamma5" if "5" in name else "gamma6"}'
        h   = diag.history.get(key, [])
        delta_str = (f"{gamma_val - h[-2][1]:+.5f}"
                     if len(h) >= 2 else '(first)')
        print(f"  {name:<12} {gamma_val:>8.5f}  {delta_str:>8}  {status}")
    print()


def log_fan_health(model, writer, epoch: int, diag: DiagnosticLogger):
    fan_info = []
    for stem_name in ['stem_conv1', 'stem_conv2']:
        module = getattr(model.backbone, stem_name, None)
        if module is None or len(module) < 2:
            continue
        fan = module[1]
        if not hasattr(fan, 'alpha'):
            continue
        alpha_raw  = fan.alpha.data
        alpha_sig  = torch.sigmoid(alpha_raw)
        alpha_mean = alpha_sig.mean().item()
        alpha_std  = alpha_sig.std().item()
        alpha_min  = alpha_sig.min().item()
        alpha_max  = alpha_sig.max().item()
        fan_info.append((stem_name, alpha_mean, alpha_std, alpha_min, alpha_max))
        tag = '1' if '1' in stem_name else '2'
        diag.log(epoch, f'fan/alpha{tag}_mean', alpha_mean)

    if fan_info:
        print(f"  FoggyAwareNorm alpha (epoch {epoch+1}):")
        print(f"  {'Layer':<12} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}  Blend")
        print(f"  {'─'*55}")
        for stem_name, mean, std, mn, mx in fan_info:
            blend_bar = '█' * int(mean * 20)
            bias = ("→ IN" if mean > 0.6
                    else ("→ BN" if mean < 0.4 else "balanced"))
            print(f"  {stem_name:<12} {mean:>7.4f} {std:>7.4f} "
                  f"{mn:>7.4f} {mx:>7.4f}  {bias} {blend_bar}")
        print()


def check_spp_bn_health(model, epoch: int):
    spp = getattr(model.backbone, 'spp', None)
    if spp is None:
        return
    issues = []
    for name, m in spp.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.running_var is not None:
                rv = m.running_var
                if rv.min().item() < 1e-6:
                    issues.append(f"spp.{name}: var_min={rv.min().item():.2e}")
                if rv.max().item() > 1e6:
                    issues.append(f"spp.{name}: var_max={rv.max().item():.2e}")
                if torch.isnan(rv).any():
                    issues.append(f"spp.{name}: NaN")
                if torch.isinf(rv).any():
                    issues.append(f"spp.{name}: INF")
    if issues:
        print(f"\n  ⚠️  SPP BN HEALTH WARNING (epoch {epoch+1}):")
        for iss in issues:
            print(f"    {iss}")
        print(f"    → Auto-resetting corrupted BN running stats...")
        for name, m in spp.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if (m.running_var is not None and
                        (torch.isnan(m.running_var).any() or
                         torch.isinf(m.running_var).any() or
                         m.running_var.min().item() < 1e-6)):
                    m.running_mean.zero_()
                    m.running_var.fill_(1.0)
                    print(f"    Reset: spp.{name}")
    # Không print OK — chỉ cảnh báo khi có vấn đề


def count_trainable_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_total  = sum(p.numel() for p in model.backbone.parameters())
    bb_train  = sum(p.numel() for p in model.backbone.parameters()
                    if p.requires_grad)
    hd_total  = sum(p.numel() for p in model.decode_head.parameters())
    hd_train  = sum(p.numel() for p in model.decode_head.parameters()
                    if p.requires_grad)
    print(f"\n{SEP}\nPARAMETER STATISTICS\n{SEP}")
    print(f"Total:      {total:>15,} | 100%")
    print(f"Trainable:  {trainable:>15,} | {100*trainable/total:.1f}%")
    print(f"Frozen:     {total-trainable:>15,} | "
          f"{100*(total-trainable)/total:.1f}%")
    print(f"{'-'*70}")
    print(f"Backbone:   {bb_train:>15,} / {bb_total:,} | "
          f"{100*bb_train/max(bb_total,1):.1f}%")
    print(f"Head:       {hd_train:>15,} / {hd_total:,} | "
          f"{100*hd_train/max(hd_total,1):.1f}%")
    print(f"{SEP}\n")
    return trainable, total - trainable


def freeze_backbone(model, variant='fan_dwsa'):
    has_dwsa = hasattr(model.backbone, 'dwsa_stage4')
    has_fan  = (hasattr(model.backbone, 'stem_conv1') and
                len(model.backbone.stem_conv1) > 1 and
                hasattr(model.backbone.stem_conv1[1], 'alpha'))
    keep = []
    if has_dwsa: keep.append('DWSA')
    if has_fan:  keep.append('FoggyAwareNorm')
    print(f"Freezing backbone "
          f"(keeping {' + '.join(keep) if keep else 'nothing'} trainable)...")
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
                    p.requires_grad = True; dwsa_params += p.numel()
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
                        if m.weight is not None: m.weight.requires_grad = True
                        if m.bias   is not None: m.bias.requires_grad   = True
                        dwsa_bn_count += 1
        print(f"  DWSA kept trainable: {dwsa_params:,} params, "
              f"{dwsa_bn_count} BN unfrozen")
    fan_params = 0
    if has_fan:
        for name in ['stem_conv1', 'stem_conv2']:
            module = getattr(model.backbone, name, None)
            if (module is not None and len(module) > 1 and
                    hasattr(module[1], 'alpha')):
                for p in module[1].parameters():
                    p.requires_grad = True; fan_params += p.numel()
                fan_bn = module[1].bn; fan_bn.train()
                if fan_bn.weight is not None: fan_bn.weight.requires_grad = True
                if fan_bn.bias   is not None: fan_bn.bias.requires_grad   = True
        print(f"  FoggyAwareNorm kept trainable: {fan_params:,} params")
    print("Backbone frozen\n")


def unfreeze_backbone_progressive(model, stage_names):
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
                try: module = base_mod[int(parts[1])]
                except (IndexError, TypeError): pass
        if module is None:
            print(f"  [skip] module '{stage_name}' not found"); continue
        count = 0; bn_count = 0
        for p in module.parameters():
            if not p.requires_grad:
                p.requires_grad = True; count += 1
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if m.weight is not None: m.weight.requires_grad = True
                if m.bias   is not None: m.bias.requires_grad   = True
                bn_count += 1
        total_unfrozen += count
        if count > 0:
            print(f"  Unfrozen: backbone.{stage_name} "
                  f"({count:,} params, {bn_count} BN)")
    print(f"  Total unfrozen: {total_unfrozen:,} params\n")
    return total_unfrozen


def print_backbone_structure(model):
    print(f"\n{SEP}\n BACKBONE STRUCTURE (GCNet v3)\n{SEP}")
    for name, module in model.backbone.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        if isinstance(module, nn.ModuleList):
            print(f"  {name}: ModuleList[{len(module)}]  ({n_params:,} params)")
            for i, sub in enumerate(module):
                sp = sum(p.numel() for p in sub.parameters())
                print(f"    [{i}]: {type(sub).__name__}  ({sp:,} params)")
        else:
            print(f"  {name}: {type(module).__name__}  ({n_params:,} params)")
    print(f"{SEP}\n")


# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config(variant='fan_dwsa'):
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
        if variant in ('fan_dwsa', 'dwsa_only'):
            backbone_base["dwsa_reduction"] = 8
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
        feats  = self.backbone(x)
        logits = self.decode_head(feats)
        return {"main": logits}


# ============================================
# TRAINER
# ============================================

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, args,
                 class_weights=None, diag: DiagnosticLogger = None):
        self.model       = model.to(device)
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.args        = args
        self.best_miou   = 0.0
        self.start_epoch = 0
        self.global_step = 0
        self.diag        = diag

        loss_cfg          = args.loss_config
        self.ce_weight    = loss_cfg['ce_weight']
        self.dice_weight  = loss_cfg['dice_weight']
        self.base_loss_cfg = loss_cfg
        self.loss_phase   = 'full'

        # K3: Distillation loss (init nếu dùng)
        self.distill_loss_fn = None
        self.teacher_model   = None
        if getattr(args, 'use_distillation', False):
            self.distill_loss_fn = KnowledgeDistillationLoss(
                temperature=getattr(args, 'distill_temperature', 4.0),
                alpha=getattr(args, 'distill_alpha', 0.3)
            )
            print(f"  K3: Distillation enabled "
                  f"(T={args.distill_temperature}, alpha={args.distill_alpha})")

        # K4: EWC regularizer (init nếu dùng)
        self.ewc = None  # Set sau khi dataloader sẵn sàng

        cw_device = (class_weights.to(device)
                     if class_weights is not None else None)
        _ohem_kr  = getattr(args, 'ohem_keep_ratio', 0.3)
        _ohem_mk  = getattr(args, 'ohem_min_kept',   100000)
        _ohem_thr = getattr(args, 'ohem_thresh',     None)
        self.ohem = OHEMLoss(
            ignore_index=args.ignore_index, keep_ratio=_ohem_kr,
            min_kept=_ohem_mk, thresh=_ohem_thr,
            class_weights=class_weights)
        if _ohem_thr:
            print(f"OHEM: threshold-based (thres={_ohem_thr})")
        else:
            print(f"OHEM: ratio-based (keep_ratio={_ohem_kr})")

        self.dice = DiceLoss(
            smooth=loss_cfg['dice_smooth'],
            ignore_index=args.ignore_index,
            class_weights=class_weights)
        _ls = getattr(args, 'label_smoothing', 0.0)
        self.ce = nn.CrossEntropyLoss(
            weight=cw_device,
            ignore_index=args.ignore_index,
            label_smoothing=_ls)
        if _ls > 0: print(f"Label smoothing: {_ls}")

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = _make_writer(self.save_dir / "tensorboard")
        self.save_config()
        self._print_config(loss_cfg)

    def set_teacher(self, teacher_model):
        """K3: Gán teacher model sau khi init."""
        self.teacher_model = teacher_model
        print(f"  K3: Teacher model set, "
              f"{sum(p.numel() for p in teacher_model.parameters()):,} params")

    def set_ewc(self, ewc_regularizer):
        """K4: Gán EWC regularizer sau khi compute Fisher."""
        self.ewc = ewc_regularizer
        print(f"  K4: EWC regularizer set (lambda={ewc_regularizer.ewc_lambda})")

    def set_loss_phase(self, phase: str):
        if phase == self.loss_phase: return
        if phase == 'ce_only':    self.dice_weight = 0.0
        elif phase == 'full':     self.dice_weight = self.base_loss_cfg['dice_weight']
        self.loss_phase = phase
        print(f"Loss phase → {phase}  "
              f"(CE={self.ce_weight}, Dice={self.dice_weight})")

    def _print_config(self, loss_cfg):
        print(f"\n{SEP}\nTRAINER CONFIGURATION\n{SEP}")
        print(f"Batch size:             {self.args.batch_size}")
        print(f"Gradient accumulation:  {self.args.accumulation_steps}")
        print(f"Effective batch:        "
              f"{self.args.batch_size * self.args.accumulation_steps}")
        print(f"Mixed precision:        {self.args.use_amp}")
        print(f"Gradient clipping:      {self.args.grad_clip}")
        print(f"Loss: CE({loss_cfg['ce_weight']}) + "
              f"Dice({loss_cfg['dice_weight']})")
        if getattr(self.args, 'use_distillation', False):
            print(f"Distillation: T={self.args.distill_temperature}, "
                  f"alpha={self.args.distill_alpha}")
        if getattr(self.args, 'use_ewc', False):
            print(f"EWC: lambda={self.args.ewc_lambda}")
        print(f"{SEP}\n")

    def save_config(self):
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

    def train_epoch(self, loader, epoch):
        self.model.train()

        if getattr(self.args, "freeze_spp_bn", False):
            spp = getattr(self.model.backbone, "spp", None)
            if spp is not None:
                # Freeze TẤT CẢ params trong SPP mỗi epoch (conv + BN)
                # BN eval() đã không đủ — conv weight vẫn nhận inf gradient
                # khi distillation loss scale lớn ở epoch đầu
                for p in spp.parameters():
                    p.requires_grad = False
                for m in spp.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        if getattr(self.args, "freeze_stem_conv", False):
            # Re-lock stem mỗi epoch sau model.train() — train() không restore requires_grad
            for stem_name in ["stem_conv1", "stem_conv2"]:
                module = getattr(self.model.backbone, stem_name, None)
                if module is None: continue
                for pname, param in module.named_parameters():
                    is_fan = any(k in pname for k in ("alpha", "bn.", "in_."))
                    if not is_fan:
                        param.requires_grad = False
            for stem_name in ["stem_stage2", "stem_stage3"]:
                module = getattr(self.model.backbone, stem_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
                    for m in module.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()

        total_loss = total_ohem = total_dice = 0.0
        total_kl   = 0.0
        max_grad_epoch = 0.0
        max_grad       = 0.0
        hard_ratio_acc = 0.0
        pbar = tqdm(loader,
                    desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            # K3: Teacher inference (no grad, bên ngoài autocast để tránh OOM)
            teacher_logits_c6 = None
            if (self.teacher_model is not None and
                    self.distill_loss_fn is not None):
                with torch.no_grad():
                    # Teacher có thể chạy ở resolution khác nếu cần
                    teacher_size = getattr(self.args, 'teacher_img_size', None)
                    if teacher_size is not None:
                        imgs_teacher = F.interpolate(
                            imgs, size=teacher_size,
                            mode='bilinear', align_corners=False)
                    else:
                        imgs_teacher = imgs
                    t_out = self.teacher_model(imgs_teacher)
                    if isinstance(t_out, (tuple, list)):
                        t_raw = t_out[-1].detach()
                    else:
                        t_raw = t_out.detach()
                    # Resize teacher logit về H/8×W/8 để match c6_logit
                    # Tránh tính KL trên full resolution (512×1024) → OOM + inf grad
                    _th = self.args.img_h // 8
                    _tw = self.args.img_w // 8
                    teacher_logits_c6 = F.interpolate(
                        t_raw.float(), size=(_th, _tw),
                        mode='bilinear', align_corners=False)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                outputs  = self.model.forward_train(imgs)
                c4_logit, c6_logit = outputs["main"]
                target_size = masks.shape[-2:]
                c4_full = F.interpolate(c4_logit, size=target_size,
                                        mode='bilinear', align_corners=False)
                c6_full = F.interpolate(c6_logit, size=target_size,
                                        mode='bilinear', align_corners=False)

                ohem_loss = self.ohem(c6_full, masks)

                if self.dice_weight > 0:
                    masks_small = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=c6_logit.shape[-2:],
                        mode='nearest').squeeze(1).long()
                    dice_loss = self.dice(c6_logit, masks_small)
                else:
                    dice_loss = torch.tensor(0.0, device=self.device)

                task_loss = (self.ce_weight * ohem_loss +
                             self.dice_weight * dice_loss)

                if self.args.aux_weight > 0:
                    aux_exp    = getattr(self.args, 'aux_decay_exp', 0.9)
                    aux_weight = (self.args.aux_weight *
                                  (1 - epoch / self.args.epochs) ** aux_exp)
                    aux_loss   = self.ohem(c4_full, masks)
                    task_loss  = task_loss + aux_weight * aux_loss

                # K3: Combine với distillation loss
                # Dùng c6_logit (nhỏ, H/8×W/8) thay vì c6_full (H×W)
                # c6_full = 512×1024 → log_softmax tốn 836MB → OOM + inf gradient
                # c6_logit = 64×128 → nhỏ hơn 64x, gradient ổn định hơn
                kl_val = 0.0
                if (teacher_logits_c6 is not None and
                        self.distill_loss_fn is not None):
                    loss, kl_val = self.distill_loss_fn(
                        c6_logit, teacher_logits_c6, task_loss)
                else:
                    loss = task_loss

                # K4: Thêm EWC penalty
                if self.ewc is not None:
                    ewc_penalty = self.ewc.penalty(self.model)
                    loss = loss + ewc_penalty

                loss = loss / self.args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  NaN/Inf loss at epoch {epoch+1}, "
                      f"batch {batch_idx} — skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                max_grad = check_gradients(self.model, threshold=10.0)
                max_grad_epoch = max(max_grad_epoch, max_grad)
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if (self.scheduler and
                        self.args.scheduler == 'onecycle'):
                    self.scheduler.step()

            total_loss += loss.item() * self.args.accumulation_steps
            total_ohem += ohem_loss.item()
            total_dice += dice_loss.item()
            total_kl   += kl_val
            hard_ratio_acc += self.ohem.last_hard_ratio

            postfix = {
                'loss' : f'{loss.item() * self.args.accumulation_steps:.4f}',
                'ohem' : f'{ohem_loss.item():.4f}',
                'dice' : f'{dice_loss.item():.4f}',
                'lr'   : f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'hard%': f'{self.ohem.last_hard_ratio:.2f}',
                'max_g': f'{max_grad:.2f}',
            }
            if kl_val > 0:
                postfix['kl'] = f'{kl_val:.4f}'
            pbar.set_postfix(postfix)

            # clear_gpu_memory chỉ gọi mỗi 200 batch thay vì 50
            if batch_idx % 200 == 0:
                torch.cuda.empty_cache()

        n = len(loader)
        avg_hard_ratio = hard_ratio_acc / n

        print(f"\nEpoch {epoch+1} — Max gradient: {max_grad_epoch:.2f}  |  "
              f"Avg hard pixel ratio: {avg_hard_ratio:.3f}")
        if total_kl > 0:
            print(f"  Avg KL distill loss: {total_kl/n:.4f}")

        # Chỉ in LR của group đầu (head) — đủ để monitor scheduler
        print(f"  LR head={self.optimizer.param_groups[0]['lr']:.2e}")

        torch.cuda.empty_cache()
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        result = {
            'loss'      : total_loss / n,
            'ohem'      : total_ohem / n,
            'dice'      : total_dice / n,
            'hard_ratio': avg_hard_ratio,
        }

        if self.diag:
            self.diag.log_dict(epoch, result, prefix='train/')
            self.diag.log(epoch, 'train/max_grad', max_grad_epoch)

        return result

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss  = 0.0
        num_classes = self.args.num_classes
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        pbar        = tqdm(loader, desc="Validation")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits      = self.model(imgs)
                logits_full = F.interpolate(
                    logits, size=masks.shape[-2:],
                    mode='bilinear', align_corners=False)
                ce_loss     = self.ce(logits_full, masks)
                if self.dice_weight > 0:
                    masks_small = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=logits.shape[-2:],
                        mode='nearest').squeeze(1).long()
                    dice_loss = self.dice(logits, masks_small)
                else:
                    dice_loss = torch.tensor(0.0, device=self.device)
                loss = (self.ce_weight * ce_loss +
                        self.dice_weight * dice_loss)

            total_loss += loss.item()
            pred   = logits_full.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            valid  = (target >= 0) & (target < num_classes)
            label  = (num_classes * target[valid].astype(int) +
                      pred[valid])
            count  = np.bincount(label, minlength=num_classes ** 2)
            conf_matrix += count.reshape(num_classes, num_classes)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            if batch_idx % 20 == 0:
                clear_gpu_memory()

        intersection = np.diag(conf_matrix)
        union        = (conf_matrix.sum(1) + conf_matrix.sum(0)
                        - intersection)
        iou          = intersection / (union + 1e-10)
        miou         = np.nanmean(iou)
        acc          = intersection.sum() / (conf_matrix.sum() + 1e-10)

        result = {
            'loss'         : total_loss / len(loader),
            'miou'         : miou,
            'accuracy'     : acc,
            'per_class_iou': iou,
        }

        if self.diag:
            self.diag.log(epoch, 'val/miou', miou)

        return result

    def save_checkpoint(self, epoch, metrics, is_best=False):
        ckpt = {
            'epoch'      : epoch,
            'model'      : self.model.state_dict(),
            'optimizer'  : self.optimizer.state_dict(),
            'scheduler'  : (self.scheduler.state_dict()
                            if self.scheduler else None),
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

    def load_checkpoint(self, path, reset_epoch=True,
                        load_optimizer=True, reset_best_metric=False):
        ckpt  = torch.load(path, map_location=self.device,
                           weights_only=False)
        state = (ckpt.get('model') or ckpt.get('model_state_dict') or
                 ckpt.get('state_dict') or ckpt)
        self.model.load_state_dict(state, strict=False)

        if load_optimizer and not reset_epoch:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError) as e:
                print(f"Optimizer state not loaded: {e}")
            if self.scheduler and ckpt.get('scheduler'):
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                except Exception as e:
                    print(f"Scheduler state not loaded: {e}")

        if (load_optimizer and 'scaler' in ckpt and
                ckpt['scaler'] and not reset_epoch):
            try:
                self.scaler.load_state_dict(ckpt['scaler'])
            except Exception as e:
                print(f"Scaler state not loaded: {e}")

        if reset_epoch:
            self.start_epoch = 0
            self.global_step = 0
            self.best_miou   = (0.0 if reset_best_metric
                                else ckpt.get('best_miou', 0.0))
            print(f"Weights loaded (epoch {ckpt['epoch']}), "
                  f"starting from epoch 0")
            print(f"Optimizer state NOT loaded (transfer mode)")
        else:
            self.start_epoch = ckpt['epoch'] + 1
            self.best_miou   = ckpt.get('best_miou', 0.0)
            self.global_step = ckpt.get('global_step', 0)
            print(f"Checkpoint loaded, resuming from epoch "
                  f"{self.start_epoch}")


# ============================================
# CONSTANTS
# ============================================

UNFREEZE_STAGES = [
    ['stem_conv1', 'stem_conv2', 'stem_stage2', 'stem_stage3',
     'compression_1', 'down_1'],
    ['semantic_branch_layers.0', 'detail_branch_layers.0', 'dwsa_stage4'],
    ['semantic_branch_layers.1', 'detail_branch_layers.1', 'dwsa_stage5',
     'compression_2', 'down_2'],
    ['semantic_branch_layers.2', 'detail_branch_layers.2', 'dwsa_stage6',
     'spp'],
]

CLASS_NAMES = ['road','sidewalk','building','wall','fence','pole',
               'traffic_light','traffic_sign','vegetation','terrain',
               'sky','person','rider','car','truck','bus',
               'train','motorcycle','bicycle']


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 Training")
    parser.add_argument("--model_variant",         type=str, default="fan_dwsa",
                        choices=["fan_dwsa", "fan_only", "dwsa_only"])
    parser.add_argument("--pretrained_weights",    type=str, default=None)
    parser.add_argument("--freeze_backbone",       action="store_true", default=False)
    parser.add_argument("--unfreeze_schedule",     type=str, default="")
    parser.add_argument("--backbone_lr_factor",    type=float, default=0.1)
    parser.add_argument("--dwsa_lr_factor",        type=float, default=0.5)
    parser.add_argument("--alpha_lr_factor",       type=float, default=0.1)
    parser.add_argument("--use_class_weights",     action="store_true")
    parser.add_argument("--class_weights_file",    type=str, default=None)
    parser.add_argument("--class_weights_method",  type=str, default="median_freq",
                        choices=["inverse_freq", "sqrt_inverse", "median_freq"])
    parser.add_argument("--train_txt",    required=True)
    parser.add_argument("--val_txt",      required=True)
    parser.add_argument("--dataset_type", default="foggy",
                        choices=["normal", "foggy"])
    parser.add_argument("--num_classes",  type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--accumulation_steps", type=int,   default=2)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--weight_decay",       type=float, default=1e-4)
    parser.add_argument("--optimizer",          type=str,   default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--stem_lr_factor",     type=float, default=0.01)
    parser.add_argument("--sgd_momentum",       type=float, default=0.9)
    parser.add_argument("--grad_clip",          type=float, default=5.0)
    parser.add_argument("--aux_weight",         type=float, default=0.4)
    parser.add_argument("--aux_decay_exp",      type=float, default=0.9)
    parser.add_argument("--dice_weight",        type=float, default=None)
    parser.add_argument("--label_smoothing",    type=float, default=0.0)
    parser.add_argument("--ohem_keep_ratio",    type=float, default=0.3)
    parser.add_argument("--ohem_min_kept",      type=int,   default=100000)
    parser.add_argument("--ohem_thresh",        type=float, default=None)
    parser.add_argument("--ce_weight",          type=float, default=None)
    parser.add_argument("--scheduler",          default="cosine",
                        choices=["onecycle", "poly", "cosine", "cosine_wr"])
    parser.add_argument("--cosine_wr_t0",       type=int,   default=10)
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3)
    parser.add_argument("--img_h",              type=int,   default=512)
    parser.add_argument("--img_w",              type=int,   default=1024)
    parser.add_argument("--high_res_h",         type=int,   default=640)
    parser.add_argument("--high_res_w",         type=int,   default=1280)
    parser.add_argument("--high_res_epochs",    type=int,   default=0)
    parser.add_argument("--use_amp",            action="store_true", default=True)
    parser.add_argument("--num_workers",        type=int,   default=4)
    parser.add_argument("--save_dir",           default="./checkpoints")
    parser.add_argument("--resume",             type=str,   default=None)
    parser.add_argument("--resume_mode",        type=str,   default="transfer",
                        choices=["transfer", "continue"])
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--log_interval",       type=int,   default=50)
    parser.add_argument("--save_interval",      type=int,   default=10)
    parser.add_argument("--reset_best_metric",  action="store_true")
    parser.add_argument("--freeze_stem_conv",   action="store_true", default=False)
    parser.add_argument("--freeze_spp_bn",      action="store_true", default=False)
    parser.add_argument("--diag_interval",      type=int,   default=1)

    # ── K1: BN Reset + Warmup ──────────────────────────────────────────────
    parser.add_argument("--reset_bn_stats",     action="store_true", default=False,
                        help="K1: Reset BN running stats sau load checkpoint "
                             "(quan trọng khi chuyển resolution)")
    parser.add_argument("--bn_warmup_epochs",   type=int,   default=3,
                        help="K1: Số epoch giữ BN momentum cao sau reset "
                             "(BN adapt nhanh hơn về distribution mới)")
    parser.add_argument("--bn_warmup_momentum", type=float, default=0.3,
                        help="K1: BN momentum trong warmup period (default 0.3 = 3x nhanh)")

    # ── K2: LLRD ──────────────────────────────────────────────────────────
    parser.add_argument("--use_llrd",           action="store_true", default=False,
                        help="K2: Dùng Layer-wise LR Decay thay vì discriminative LR thông thường")
    parser.add_argument("--llrd_decay",         type=float, default=0.7,
                        help="K2: LR decay factor mỗi layer (0.7 = mỗi layer sâu hơn LR giảm 30%%)")

    # ── K3: Knowledge Distillation ─────────────────────────────────────────
    parser.add_argument("--use_distillation",   action="store_true", default=False,
                        help="K3: Dùng knowledge distillation từ teacher checkpoint")
    parser.add_argument("--teacher_ckpt",       type=str,   default=None,
                        help="K3: Path tới teacher checkpoint (thường là best.pth 0.7191)")
    parser.add_argument("--distill_alpha",      type=float, default=0.3,
                        help="K3: Weight của distillation loss (0.3 = 30%% KL + 70%% task)")
    parser.add_argument("--distill_temperature",type=float, default=4.0,
                        help="K3: Temperature cho softmax (>1 làm distribution mềm hơn)")
    parser.add_argument("--teacher_img_size",   type=int,   nargs=2,
                        default=None, metavar=('H', 'W'),
                        help="K3: Teacher inference size (default = same as student). "
                             "Ví dụ: --teacher_img_size 640 1280")

    # ── K4: EWC ───────────────────────────────────────────────────────────
    parser.add_argument("--use_ewc",            action="store_true", default=False,
                        help="K4: Dùng Elastic Weight Consolidation để bảo vệ weight quan trọng")
    parser.add_argument("--ewc_lambda",         type=float, default=500.0,
                        help="K4: EWC penalty strength (cao hơn = bảo vệ weight nhiều hơn)")
    parser.add_argument("--ewc_samples",        type=int,   default=200,
                        help="K4: Số samples để tính Fisher Information Matrix")

    # ── K5: Progressive Resolution ─────────────────────────────────────────
    parser.add_argument("--use_progressive_res",action="store_true", default=False,
                        help="K5: Giảm dần resolution thay vì jump thẳng về target")
    parser.add_argument("--prog_res_schedule",  type=str,
                        default="0:608x1216,3:576x1152,6:544x1088,9:512x1024",
                        help="K5: Schedule dạng 'epoch:HxW,...' "
                             "(default: 4 bước từ 608 xuống 512)")

    args = parser.parse_args()

    if args.freeze_backbone and args.unfreeze_schedule:
        unfreeze_list = sorted(int(e)
                               for e in args.unfreeze_schedule.split(','))
        if max(unfreeze_list) >= args.epochs:
            raise ValueError("unfreeze_schedule epoch >= total epochs")
    else:
        unfreeze_list = []

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_memory_efficient_training()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if (args.freeze_backbone and unfreeze_list and
            args.scheduler == 'onecycle'):
        args.scheduler = 'cosine'
        print("[INFO] scheduler auto-switched: onecycle → cosine")

    print(f"\n{SEP}")
    print(f"GCNet v3 Training  |  {args.model_variant}")
    print(f"{SEP}")
    print(f"Device: {device}  |  Image: {args.img_h}x{args.img_w}")
    print(f"Epochs: {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip: {args.grad_clip}  |  AMP: {args.use_amp}")
    print(f"Diag interval: every {args.diag_interval} epoch(s)")
    # Print enabled knowledge retention features
    kr_features = []
    if args.reset_bn_stats:       kr_features.append("K1:BN-Reset")
    if args.use_llrd:             kr_features.append("K2:LLRD")
    if args.use_distillation:     kr_features.append("K3:Distill")
    if args.use_ewc:              kr_features.append("K4:EWC")
    if args.use_progressive_res:  kr_features.append("K5:ProgRes")
    if kr_features:
        print(f"Knowledge Retention: {' | '.join(kr_features)}")
    print(f"{SEP}\n")

    variant = getattr(args, 'model_variant', 'fan_dwsa')
    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet
    elif variant == 'fan_only':
        from model.backbone.fan import GCNet
    elif variant == 'dwsa_only':
        from model.backbone.dwsa import GCNet
    else:
        raise ValueError(f"Unknown model_variant: {variant}")

    cfg          = ModelConfig.get_config(variant=variant)
    args.loss_config = cfg["loss"]

    # ── K5: Parse progressive resolution schedule ───────────────────────
    prog_res_scheduler = None
    if args.use_progressive_res:
        prog_res_dict = {}
        for entry in args.prog_res_schedule.split(','):
            ep_str, res_str = entry.strip().split(':')
            h_str, w_str   = res_str.split('x')
            prog_res_dict[int(ep_str)] = (int(h_str), int(w_str))
        # Override img_h/img_w với resolution đầu tiên trong schedule
        first_res    = prog_res_dict[min(prog_res_dict.keys())]
        args.img_h   = first_res[0]
        args.img_w   = first_res[1]
        print(f"  K5: Starting resolution: {args.img_h}×{args.img_w}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt, val_txt=args.val_txt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w), pin_memory=True,
        compute_class_weights=args.use_class_weights,
        dataset_type=args.dataset_type)

    high_res_train_loader = None
    if getattr(args, "high_res_epochs", 0) > 0:
        hi_h, hi_w = args.high_res_h, args.high_res_w
        area_ratio = (args.img_h * args.img_w) / (hi_h * hi_w)
        hi_batch   = max(4, int(args.batch_size * area_ratio))
        high_res_train_loader, _, _ = create_dataloaders(
            train_txt=args.train_txt, val_txt=args.val_txt,
            batch_size=hi_batch, num_workers=args.num_workers,
            img_size=(hi_h, hi_w), pin_memory=True,
            compute_class_weights=False,
            dataset_type=args.dataset_type)

    if getattr(args, "class_weights_file", None):
        import pathlib
        cw_path = pathlib.Path(args.class_weights_file)
        if cw_path.exists():
            class_weights = torch.load(cw_path, map_location="cpu")
            print(f"Class weights loaded from: {cw_path}  "
                  f"(min={class_weights.min():.3f}, "
                  f"max={class_weights.max():.3f}, "
                  f"ratio={class_weights.max()/class_weights.min():.1f}x)")
        else:
            print(f"WARNING: {cw_path} not found — ignored")
            class_weights = None

    # ── Model ──────────────────────────────────────────────────────────────
    backbone = GCNet(**cfg["backbone"])
    head     = GCNetHead(**cfg["head"], num_classes=args.num_classes,
                         ignore_index=args.ignore_index)
    model    = Segmentor(backbone=backbone, head=head).to(device)
    model.apply(init_weights)
    check_model_health(model)

    if args.pretrained_weights:
        load_pretrained_gcnet(model, args.pretrained_weights,
                              variant=variant)
    if args.freeze_backbone:
        freeze_backbone(model, variant=variant)

    count_trainable_params(model)
    print_backbone_structure(model)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader,
                                start_epoch=0)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    diag = DiagnosticLogger(save_dir=save_path, class_names=CLASS_NAMES)

    trainer = Trainer(model=model, optimizer=optimizer,
                      scheduler=scheduler, device=device, args=args,
                      class_weights=(class_weights
                                     if args.use_class_weights else None),
                      diag=diag)

    if getattr(args, "dice_weight", None) is not None:
        trainer.dice_weight = args.dice_weight
        trainer.base_loss_cfg["dice_weight"] = args.dice_weight
    if getattr(args, "ce_weight", None) is not None:
        trainer.ce_weight = args.ce_weight

    # ── Load checkpoint ────────────────────────────────────────────────────
    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=(args.resume_mode == "transfer"),
            load_optimizer=(args.resume_mode == "continue"),
            reset_best_metric=args.reset_best_metric,
        )

    # ── K1: BN Reset sau load checkpoint ──────────────────────────────────
    if args.reset_bn_stats:
        reset_bn_stats_with_warmup(model,
                                   reset_momentum=args.bn_warmup_momentum)
        print(f"  K1: BN warmup active for {args.bn_warmup_epochs} epochs")

    # ── Freeze stem (sau load để không ảnh hưởng weight loading) ──────────
    if getattr(args, "freeze_stem_conv", False):
        frozen_stem = 0
        for stem_name in ["stem_conv1", "stem_conv2"]:
            module = getattr(model.backbone, stem_name, None)
            if module is None: continue
            for pname, param in module.named_parameters():
                is_fan = any(k in pname for k in ("alpha", "bn.", "in_."))
                if not is_fan:
                    param.requires_grad = False
                    frozen_stem += param.numel()
        for stem_name in ["stem_stage2", "stem_stage3"]:
            module = getattr(model.backbone, stem_name, None)
            if module is None: continue
            for param in module.parameters():
                param.requires_grad = False
                frozen_stem += param.numel()
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        print(f"Stem frozen: {frozen_stem:,} params "
              f"(stem_conv1/2/stage2/3, FAN still trainable)")
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args, train_loader,
                                    start_epoch=trainer.start_epoch)
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler

    # ── Freeze SPP BN — disable cả gradient để tránh inf backward ────────
    if getattr(args, "freeze_spp_bn", False):
        spp = getattr(model.backbone, "spp", None)
        if spp is not None:
            frozen_spp_params = 0
            # Freeze TẤT CẢ params trong SPP: conv + BN weight/bias
            # Chỉ freeze BN không đủ — SPP conv weight vẫn nhận inf gradient
            # qua distillation loss hoặc khi batch_size nhỏ
            for p in spp.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen_spp_params += p.numel()
            for m in spp.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            print(f"SPP fully frozen: {frozen_spp_params:,} params "
                  f"(all conv + BN — prevents inf gradient from distillation)")
            # Rebuild optimizer để loại bỏ SPP params khỏi param groups
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args, train_loader,
                                        start_epoch=trainer.start_epoch)
            trainer.optimizer = optimizer
            trainer.scheduler = scheduler

    # ── K3: Load teacher model ─────────────────────────────────────────────
    if args.use_distillation and args.teacher_ckpt:
        teacher = load_teacher_model(
            ckpt_path=args.teacher_ckpt,
            model_class=GCNet,
            cfg=cfg["backbone"],
            head_class=GCNetHead,
            head_cfg=cfg["head"],
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            device=device
        )
        trainer.set_teacher(teacher)
        if args.teacher_img_size is not None:
            args.teacher_img_size = tuple(args.teacher_img_size)
            print(f"  K3: Teacher inference size: "
                  f"{args.teacher_img_size[0]}×{args.teacher_img_size[1]}")
    elif args.use_distillation and not args.teacher_ckpt:
        print("  ⚠️  K3: --use_distillation set but --teacher_ckpt not "
              "provided. Distillation disabled.")
        args.use_distillation = False

    # ── K4: Compute Fisher Information ────────────────────────────────────
    if args.use_ewc:
        print(f"\n  K4: Initializing EWC "
              f"(lambda={args.ewc_lambda}, "
              f"samples={args.ewc_samples})...")
        ewc = EWCRegularizer(
            model=model,
            dataloader=train_loader,
            device=device,
            n_samples=args.ewc_samples,
            ewc_lambda=args.ewc_lambda
        )
        trainer.set_ewc(ewc)

    # ── K5: Init progressive resolution scheduler ─────────────────────────
    if args.use_progressive_res:
        prog_res_scheduler = ProgressiveResolutionScheduler(
            resolution_schedule=prog_res_dict,
            train_txt=args.train_txt,
            val_txt=args.val_txt,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataset_type=args.dataset_type,
            use_class_weights=False
        )

    print(f"\n{SEP}\nSTARTING TRAINING\n{SEP}\n")

    applied_unfreeze_stages = set()

    for epoch in range(trainer.start_epoch, args.epochs):

        # ── K1: Restore BN momentum sau warmup period ──────────────────────
        if (args.reset_bn_stats and
                epoch == trainer.start_epoch + args.bn_warmup_epochs):
            restore_bn_momentum(model, momentum=0.1)

        # ── K5: Get correct loader for this epoch ─────────────────────────
        if args.use_progressive_res and prog_res_scheduler is not None:
            active_loader, res_changed = prog_res_scheduler.get_loader(
                epoch, val_loader)
            # Nếu resolution thay đổi, boost BN momentum tạm thời
            if res_changed and not args.reset_bn_stats:
                # Chỉ boost nếu không đang trong BN warmup từ K1
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.momentum = 0.3
                print(f"  K5+K1: BN momentum boosted to 0.3 "
                      f"after resolution change")
            elif (prog_res_scheduler.should_boost_bn_momentum(epoch) and
                  not res_changed):
                # Sau 2 epoch với resolution mới, restore momentum
                restore_bn_momentum(model, momentum=0.1)
        else:
            # Progressive unfreeze
            if (epoch in unfreeze_list and
                    epoch not in applied_unfreeze_stages):
                stage_idx = unfreeze_list.index(epoch)
                if stage_idx < len(UNFREEZE_STAGES):
                    print(f"[Epoch {epoch+1}] Progressive unfreeze — "
                          f"stage {stage_idx+1}/{len(UNFREEZE_STAGES)}")
                    unfreeze_backbone_progressive(
                        model, UNFREEZE_STAGES[stage_idx])
                    applied_unfreeze_stages.add(epoch)
                    optimizer = build_optimizer(model, args)
                    scheduler = build_scheduler(
                        optimizer, args, train_loader, start_epoch=epoch)
                    trainer.optimizer = optimizer
                    trainer.scheduler = scheduler
                    trainer.set_loss_phase('ce_only')

            if unfreeze_list and trainer.loss_phase == 'ce_only':
                last_unfreeze = max(
                    (e for e in unfreeze_list
                     if e in applied_unfreeze_stages and e <= epoch),
                    default=None)
                if (last_unfreeze is not None and
                        epoch >= last_unfreeze +
                        args.ce_only_epochs_after_unfreeze):
                    trainer.set_loss_phase('full')

            # Resolution switch (non-progressive)
            hi_epochs = getattr(args, "high_res_epochs", 0)
            if (hi_epochs > 0 and
                    high_res_train_loader is not None and
                    epoch < hi_epochs):
                active_loader = high_res_train_loader
            else:
                active_loader = train_loader
                if hi_epochs > 0 and epoch == hi_epochs:
                    scheduler = build_scheduler(
                        trainer.optimizer, args, train_loader,
                        start_epoch=epoch)
                    trainer.scheduler = scheduler

        # ── L6: SPP BN health check ─────────────────────────────────────
        check_spp_bn_health(model, epoch)

        # ── Train + Validate ────────────────────────────────────────────
        train_metrics = trainer.train_epoch(active_loader, epoch)
        val_metrics   = trainer.validate(val_loader, epoch)

        # ── L2 + L3: DWSA & FAN health check ────────────────────────────
        if epoch % args.diag_interval == 0:
            log_dwsa_health(model, trainer.writer, epoch, diag)
            log_fan_health(model,  trainer.writer, epoch, diag)

        # ── L4: Per-class IoU ────────────────────────────────────────────
        iou_arr    = val_metrics['per_class_iou']
        low_thresh = 0.4
        low_cls    = [(n, v) for n, v in zip(CLASS_NAMES, iou_arr)
                      if v < low_thresh]

        print(f"\n  Per-class IoU (epoch {epoch+1}):")
        print(f"  {'Class':<16} {'IoU':>6}  Bar")
        print(f"  {'─'*45}")
        for cname, ciou in zip(CLASS_NAMES, iou_arr):
            bar  = '█' * int(ciou * 20)
            mark = (' ⚠️' if ciou < low_thresh
                    else (' ★' if ciou > 0.75 else ''))
            print(f"  {cname:<16} {ciou:>6.4f}  {bar}{mark}")
        if low_cls:
            print(f"\n  ⚠️  LOW classes (<{low_thresh}): "
                  f"{[n for n,_ in low_cls]}")

        # Bỏ per-class tensorboard (19 write/epoch) — chỉ log best/worst class vào diag
        best_cls  = max(zip(CLASS_NAMES, iou_arr), key=lambda x: x[1])
        worst_cls = min(zip(CLASS_NAMES, iou_arr), key=lambda x: x[1])
        diag.log(epoch, 'iou/best',  best_cls[1])
        diag.log(epoch, 'iou/worst', worst_cls[1])

        # ── Standard logging ─────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{SEP}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | "
              f"OHEM: {train_metrics['ohem']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | "
              f"Hard%: {train_metrics['hard_ratio']:.3f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | "
              f"mIoU: {val_metrics['miou']:.4f}  | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"{SEP}\n")

        # Tensorboard val metrics đã bỏ — dùng diagnostics.csv để track

        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
            print(f"  ★ NEW BEST mIoU: {trainer.best_miou:.4f}")
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)

    diag.print_full_history()
    diag.close()
    trainer.writer.close()

    print(f"\n{SEP}")
    print(f"TRAINING COMPLETED!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
