import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from collections import defaultdict
import math, gc, json, time, warnings
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except Exception:
    _TB = False

SEP = "=" * 70

from model.head.segmentation_head import GCNetHead
from data.custom import create_dataloaders
from model.model_utils import init_weights, check_model_health
from arch_config import apply_arch_config, load_arch_json


# ============================================================
# LOGGING
# ============================================================

class _DummyWriter:
    def __init__(self, log_dir):
        import csv, pathlib
        p = pathlib.Path(log_dir); p.mkdir(parents=True, exist_ok=True)
        self._f   = open(p / "metrics.csv", 'w', newline='')
        self._csv = csv.writer(self._f)
        self._csv.writerow(['tag', 'step', 'value'])

    def add_scalar(self, tag, value, step):
        self._csv.writerow([tag, step, f"{value:.6f}"]); self._f.flush()

    def close(self): self._f.close()


class _NullWriter:
    """No-op writer for short proxy runs where per-candidate I/O is wasteful."""
    def add_scalar(self, tag, value, step):
        pass

    def close(self):
        pass


def _make_writer(log_dir):
    if _TB:
        try: return SummaryWriter(log_dir=str(log_dir))
        except Exception: pass
    return _DummyWriter(log_dir)


class DiagnosticLogger:
    def __init__(self, save_dir, class_names):
        import csv
        self.save_dir    = Path(save_dir)
        self.class_names = class_names
        self.history     = defaultdict(list)
        self._f   = open(self.save_dir / "diagnostics.csv", 'w', newline='')
        self._csv = csv.writer(self._f)
        self._csv.writerow(['epoch', 'key', 'value'])

    def log(self, epoch, key, value):
        self.history[key].append((epoch, float(value)))
        self._csv.writerow([epoch, key, f"{float(value):.6f}"])
        self._f.flush()

    def log_dict(self, epoch, d, prefix=''):
        for k, v in d.items():
            self.log(epoch, f"{prefix}{k}" if prefix else k, v)

    def print_epoch_summary(self, epoch):
        print(f"\n{'─'*70}\n  EPOCH {epoch+1:>3} SUMMARY\n{'─'*70}")
        metrics = [
            ('val/miou',         'Val mIoU',        '.4f'),
            ('val/accuracy',     'Val Accuracy',    '.4f'),
            ('val/loss',         'Val Loss',        '.4f'),
            ('train/ohem',       'Train OHEM',      '.4f'),
            ('train/dice',       'Train Dice',      '.4f'),
            ('train/max_grad',   'Max Gradient',    '.3f'),
            ('dwsa/gamma4',      'DWSA gamma4',     '.4f'),
            ('dwsa/gamma5',      'DWSA gamma5',     '.4f'),
            ('dwsa/gamma6',      'DWSA gamma6',     '.4f'),
            ('fan/alpha1_mean',  'FAN alpha1',      '.4f'),
            ('fan/alpha2_mean',  'FAN alpha2',      '.4f'),
            ('train/hard_ratio', 'OHEM hard ratio', '.3f'),
        ]
        print(f"  {'Metric':<24}  {'Value':>10}  {'Trend':>12}")
        print(f"  {'─'*24}  {'─'*10}  {'─'*12}")
        for key, label, fmt in metrics:
            h = self.history.get(key, [])
            if not h: continue
            val = h[-1][1]
            if len(h) < 3:
                trend = '(new)'
            else:
                delta = h[-1][1] - h[-3][1]
                arrow = '↑' if delta > 1e-4 else ('↓' if delta < -1e-4 else '→')
                trend = f"{arrow} {abs(delta):.4f}"
            print(f"  {label:<24}  {val:>10{fmt}}  {trend:>12}")
        print(f"{'─'*70}\n")

    def print_full_history(self):
        print(f"\n{'='*70}\n  FULL TRAINING HISTORY\n{'='*70}")
        miou_hist = self.history.get('val/miou', [])
        if miou_hist:
            best_ep, best_val = max(miou_hist, key=lambda x: x[1])
            print(f"  Best mIoU: {best_val:.4f} at epoch {best_ep+1}")
            print(f"  Final mIoU: {miou_hist[-1][1]:.4f}")
            if len(miou_hist) >= 10:
                last10 = [v for _, v in miou_hist[-10:]]
                spread = max(last10) - min(last10)
                print(f"  Last-10 spread: {spread:.4f} {'← PLATEAU' if spread<0.003 else ''}")
        print(f"\n  Epoch │ mIoU   │ OHEM   │ Dice   │ gamma4 │ gamma5 │ hard%")
        print(f"  {'─'*65}")
        n = max((len(v) for v in self.history.values()), default=0)
        best_miou = max((v for _, v in self.history.get('val/miou', [(0,0)])), default=0)
        for i in range(n):
            def _g(k):
                h = self.history.get(k, [])
                return h[i][1] if i < len(h) else float('nan')
            miou = _g('val/miou')
            mark = ' ← BEST' if not math.isnan(miou) and miou == best_miou else ''
            print(f"  {i+1:>5} │ {miou:.4f} │ {_g('train/ohem'):.4f} │ "
                  f"{_g('train/dice'):.4f} │ {_g('dwsa/gamma4'):.4f} │ "
                  f"{_g('dwsa/gamma5'):.4f} │ {_g('train/hard_ratio'):.3f}{mark}")
        print(f"{'='*70}\n")

    def close(self): self._f.close()


# ============================================================
# BN RESET (K1)
# ============================================================

def reset_bn_stats(model, momentum=0.3):
    n = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = momentum
    print(f"  K1: Reset {n} BN layers, momentum={momentum}")

def restore_bn_momentum(model, momentum=0.1):
    n = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
    print(f"  K1: BN momentum restored to {momentum} ({n} layers)")


# ============================================================
# WEIGHT LOADING
# ============================================================

def _remap_stem_key(key, N2=4):
    import re
    for pref in ['backbone.', 'model.', 'module.']:
        if key.startswith(pref): key = key[len(pref):]
    m = re.match(r'stem\.(\d+)\.(.+)$', key)
    if not m: return key
    idx, rest = int(m.group(1)), m.group(2)
    def _cm(rest, pref):
        return f'{pref}.{rest[len("conv."):].lstrip(".")}' if rest.startswith('conv.') else None
    if idx == 0:   return _cm(rest, 'stem_conv1.0')
    elif idx == 1: return _cm(rest, 'stem_conv2.0')
    elif 2 <= idx <= 1+N2: return f'stem_stage2.{idx-2}.{rest}'
    else: return f'stem_stage3.{idx-(2+N2)}.{rest}'


def load_pretrained_gcnet(model, ckpt_path, strict_match=False, variant="fan_dwsa"):
    """Load compatible GCNet weights from a path or an in-memory checkpoint.

    Architecture search builds many models from the same checkpoint. Accepting
    the already-loaded object avoids repeatedly reading and deserializing the
    same file for every candidate while preserving the original path API.
    """
    if isinstance(ckpt_path, (str, os.PathLike)):
        print(f"Loading pretrained weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    else:
        print("Loading pretrained weights from cached checkpoint")
        ckpt = ckpt_path
    # FIX: Trainer.save_checkpoint() saves the model state dict under the
    # 'model' key (see save_checkpoint below), not 'state_dict' -- the old
    # `ckpt.get('state_dict', ckpt)` silently fell back to treating the
    # WHOLE checkpoint dict (with keys like 'epoch', 'model', 'optimizer')
    # as if it were the state dict itself whenever loading one of this
    # project's own checkpoints, matching 0% of parameters with no error.
    # Same fallback chain as Trainer.load_checkpoint() below.
    state = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt

    # Head key remap
    HEAD_MAP = {}
    for k in state:
        if not k.startswith('decode_head.'): continue
        s = k[len('decode_head.'):]
        HEAD_MAP[k] = ('cls_seg.' + s[len('conv_seg.'):] if s.startswith('conv_seg.') else s)

    # Backbone
    model_state  = model.backbone.state_dict()
    model_keymap = {}
    for mk in model_state:
        norm = mk
        for p in ['backbone.','model.','module.']: norm = norm[len(p):] if norm.startswith(p) else norm
        model_keymap[norm] = mk

    compatible, skipped, bn_dropped = {}, [], []
    for ck, cv in state.items():
        if ck.startswith('decode_head.'): continue
        norm = _remap_stem_key(ck)
        if norm is None: bn_dropped.append(ck); continue
        matched = False
        if norm in model_keymap:
            mk = model_keymap[norm]
            if model_state[mk].shape == cv.shape:
                compatible[mk] = cv; matched = True
        if not matched and not strict_match:
            for nm, mk in model_keymap.items():
                if (nm.endswith(norm) or norm.endswith(nm)) and model_state[mk].shape == cv.shape:
                    compatible[mk] = cv; matched = True; break
        if not matched: skipped.append(ck)

    # Head
    head_state  = model.decode_head.state_dict()
    head_loaded = {}
    for ck, dst in HEAD_MAP.items():
        cv = state[ck]
        if dst in head_state and head_state[dst].shape == cv.shape:
            head_loaded[dst] = cv

    lbb, lhd = len(compatible), len(head_loaded)
    tbb, thd = len(model_state), len(head_state)
    print(f"\n{SEP}\nWEIGHT LOADING SUMMARY\n{SEP}")
    print(f"Backbone:  {lbb:>5} / {tbb}  ({100*lbb/max(tbb,1):.1f}%)")
    print(f"Head:      {lhd:>5} / {thd}  ({100*lhd/max(thd,1):.1f}%)")
    print(f"BN dropped: {len(bn_dropped)}\n{SEP}\n")

    model.backbone.load_state_dict(compatible, strict=False)
    model.decode_head.load_state_dict(head_loaded, strict=False)
    return 100 * (lbb + lhd) / max(tbb + thd, 1)


# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

def build_optimizer(model, args):
    STEM = {'stem_conv1','stem_conv2','stem_stage2','stem_stage3'}
    dwsa, alpha, stem, backbone, head = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'dwsa' in name:
            dwsa.append(p)
        elif 'alpha' in name:
            alpha.append(p)
        elif 'backbone' in name:
            part = name.split('.')[1] if len(name.split('.')) > 1 else ''
            if part in STEM:
                stem.append(p)
            else:
                backbone.append(p)
        else:
            head.append(p)

    slr = getattr(args, 'stem_lr_factor', 0.01)
    groups = []
    if head:     groups.append({'params': head,     'lr': args.lr,                          'name': 'head'})
    if backbone: groups.append({'params': backbone, 'lr': args.lr * args.backbone_lr_factor,'name': 'backbone'})
    if stem:     groups.append({'params': stem,     'lr': args.lr * slr,                    'name': 'stem'})
    if dwsa:     groups.append({'params': dwsa,     'lr': args.lr * args.dwsa_lr_factor,    'name': 'dwsa'})
    if alpha:    groups.append({'params': alpha,    'lr': args.lr * args.alpha_lr_factor,   'name': 'alpha'})

    for g in groups: g.setdefault('initial_lr', g['lr'])

    if getattr(args, 'optimizer', 'adamw').lower() == 'sgd':
        opt = torch.optim.SGD(groups, momentum=getattr(args,'sgd_momentum',0.9),
                              weight_decay=args.weight_decay, nesterov=True)
        print(f"Optimizer: SGD (momentum={getattr(args,'sgd_momentum',0.9)})")
    else:
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        print("Optimizer: AdamW")

    for g in groups:
        print(f"  '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")
    return opt


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    use_cosine = (args.freeze_backbone and args.unfreeze_schedule) or args.scheduler == 'cosine'
    if use_cosine:
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs-start_epoch, eta_min=1e-6)
        print("CosineAnnealingLR")
    elif args.scheduler == 'onecycle':
        steps   = len(train_loader) * (args.epochs - start_epoch)
        max_lrs = [g['initial_lr'] for g in optimizer.param_groups]
        sch = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=steps,
            pct_start=0.05, anneal_strategy='cos',
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
            div_factor=25, final_div_factor=100000)
        print(f"OneCycleLR (steps={steps})")
    elif args.scheduler == 'cosine_wr':
        T0 = getattr(args, 'cosine_wr_t0', 10)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=1, eta_min=1e-7)
        print(f"CosineAnnealingWarmRestarts (T_0={T0})")
    else:
        sch = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: (1 - e/args.epochs)**0.9)
        print("Polynomial LR")
    return sch


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, keep_ratio=0.3,
                 min_kept=100000, thresh=None, class_weights=None):
        super().__init__()
        self.ignore_index  = ignore_index
        self.keep_ratio    = keep_ratio
        self.min_kept      = min_kept
        self.thresh        = thresh
        self.class_weights = class_weights
        self.last_hard_ratio = 0.0

    def forward(self, logits, labels):
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_px = F.cross_entropy(logits.float(), labels,
                                  weight=w.float() if w is not None else None,
                                  ignore_index=self.ignore_index,
                                  reduction='none').view(-1)
        valid = labels.view(-1) != self.ignore_index
        loss_px = loss_px[valid]
        n = loss_px.numel()
        if n == 0:
            self.last_hard_ratio = 0.0
            return logits.sum() * 0

        if self.thresh is not None:
            with torch.no_grad():
                probs     = torch.softmax(logits.detach().float(), dim=1).max(1)[0].view(-1)[valid]
                hard_mask = probs < self.thresh
                if hard_mask.sum() < self.min_kept:
                    _, idx = torch.topk(probs, min(self.min_kept, n), largest=False)
                    hard_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
                    hard_mask[idx] = True
            self.last_hard_ratio = hard_mask.float().mean().item()
            loss_px = loss_px[hard_mask]
        else:
            n_keep = min(max(int(self.keep_ratio * n), min(self.min_kept, n)), n)
            self.last_hard_ratio = n_keep / n
            if n_keep < n:
                # kthvalue is linear-time and preserves the old threshold/tie
                # semantics without fully sorting every valid pixel.
                thr = torch.kthvalue(loss_px, n - n_keep + 1).values.detach()
                loss_px = loss_px[loss_px >= thr]
        return loss_px.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255, class_weights=None):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        logits = logits.float()
        B, C, H, W = logits.shape
        valid   = targets != self.ignore_index
        tgt_oh  = F.one_hot(targets.clamp(0,C-1), C).permute(0,3,1,2).float() * valid.unsqueeze(1)
        probs   = F.softmax(logits, dim=1) * valid.unsqueeze(1)
        pf, tf  = probs.reshape(B,C,-1), tgt_oh.reshape(B,C,-1)
        inter   = (pf * tf).sum(2)
        dice    = (2*inter + self.smooth) / (pf.sum(2) + tf.sum(2) + self.smooth)
        loss    = 1.0 - dice
        if self.class_weights is not None:
            loss = loss * self.class_weights.float().unsqueeze(0)
        present = tf.sum(2) > 0
        return (loss * present.float()).sum(1).div(present.float().sum(1).clamp(1)).mean()


# ============================================================
# UTILITIES
# ============================================================

def check_gradients(model, threshold=10.0):
    max_g, max_n = 0.0, ""
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            if g > max_g: max_g, max_n = g, name
    if max_g > threshold:
        print(f"Large gradient: {max_n[:60]}... = {max_g:.2f}")
    return max_g


def check_spp_bn_health(model, epoch):
    spp = getattr(model.backbone, 'spp', None)
    if spp is None: return
    for name, m in spp.named_modules():
        if not isinstance(m, nn.BatchNorm2d): continue
        rv = m.running_var
        if rv is None: continue
        bad = torch.isnan(rv).any() or torch.isinf(rv).any() or rv.min() < 1e-6
        if bad:
            print(f"  ⚠️ SPP BN bad: spp.{name} — resetting")
            m.running_mean.zero_(); m.running_var.fill_(1.0)


def log_dwsa_health(model, epoch, diag):
    print(f"\n  DWSA Health (epoch {epoch+1}):")
    print(f"  {'Stage':<12} {'gamma':>8}  {'Δgamma':>8}  Status")
    print(f"  {'─'*48}")
    for name, tag in [('dwsa_stage4','gamma4'),('dwsa_stage5','gamma5'),('dwsa_stage6','gamma6')]:
        mod = getattr(model.backbone, name, None)
        if mod is None: continue
        g = mod.gamma.item()
        diag.log(epoch, f'dwsa/{tag}', g)
        h = diag.history.get(f'dwsa/{tag}', [])
        delta = f"{g-h[-2][1]:+.5f}" if len(h) >= 2 else '(first)'
        status = ('⚠️  NOT LEARNING' if g < 0.11 else '📈 Warming up' if g < 0.2 else '✅ Active' if g < 0.4 else '🔥 Highly active')
        print(f"  {name:<12} {g:>8.5f}  {delta:>8}  {status}")
    print()


def log_fan_health(model, epoch, diag):
    info = []
    for stem, tag in [('stem_conv1','1'),('stem_conv2','2')]:
        mod = getattr(model.backbone, stem, None)
        if mod is None or len(mod) < 2 or not hasattr(mod[1], 'alpha'): continue
        a   = torch.sigmoid(mod[1].alpha.data)
        info.append((stem, a.mean().item(), a.std().item(), a.min().item(), a.max().item()))
        diag.log(epoch, f'fan/alpha{tag}_mean', a.mean().item())
    if not info: return
    print(f"  FoggyAwareNorm alpha (epoch {epoch+1}):")
    print(f"  {'Layer':<12} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}  Blend")
    print(f"  {'─'*53}")
    for stem, mean, std, mn, mx in info:
        bias = '→ IN' if mean > 0.6 else ('→ BN' if mean < 0.4 else 'balanced')
        print(f"  {stem:<12} {mean:>7.4f} {std:>7.4f} {mn:>7.4f} {mx:>7.4f}  {bias} {'█'*int(mean*20)}")
    print()


def count_trainable_params(model):
    tot  = sum(p.numel() for p in model.parameters())
    tr   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bbt  = sum(p.numel() for p in model.backbone.parameters())
    bbtr = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    hdt  = sum(p.numel() for p in model.decode_head.parameters())
    hdtr = sum(p.numel() for p in model.decode_head.parameters() if p.requires_grad)
    print(f"\n{SEP}\nPARAMETER STATISTICS\n{SEP}")
    print(f"Total:      {tot:>15,} | 100%")
    print(f"Trainable:  {tr:>15,} | {100*tr/tot:.1f}%")
    print(f"Frozen:     {tot-tr:>15,} | {100*(tot-tr)/tot:.1f}%")
    print(f"{'─'*70}")
    print(f"Backbone:   {bbtr:>15,} / {bbt:,} | {100*bbtr/max(bbt,1):.1f}%")
    print(f"Head:       {hdtr:>15,} / {hdt:,} | {100*hdtr/max(hdt,1):.1f}%")
    print(f"{SEP}\n")


def freeze_backbone(model, variant='fan_dwsa'):
    has_dwsa = hasattr(model.backbone, 'dwsa_stage4')
    has_fan  = (hasattr(model.backbone, 'stem_conv1') and
                len(model.backbone.stem_conv1) > 1 and
                hasattr(model.backbone.stem_conv1[1], 'alpha'))
    print(f"Freezing backbone...")
    for p in model.backbone.parameters(): p.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None: m.weight.requires_grad = False
            if m.bias   is not None: m.bias.requires_grad   = False

    if has_dwsa:
        for name in ['dwsa_stage4','dwsa_stage5','dwsa_stage6']:
            mod = getattr(model.backbone, name, None)
            if mod is None: continue
            for p in mod.parameters(): p.requires_grad = True
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    if m.weight is not None: m.weight.requires_grad = True
                    if m.bias   is not None: m.bias.requires_grad   = True

    if has_fan:
        for name in ['stem_conv1','stem_conv2']:
            mod = getattr(model.backbone, name, None)
            if mod is None or len(mod) < 2 or not hasattr(mod[1], 'alpha'): continue
            for p in mod[1].parameters(): p.requires_grad = True
            mod[1].bn.train()
            if mod[1].bn.weight is not None: mod[1].bn.weight.requires_grad = True
            if mod[1].bn.bias   is not None: mod[1].bn.bias.requires_grad   = True
    print("Backbone frozen\n")


def unfreeze_backbone_progressive(model, stage_names):
    if isinstance(stage_names, str): stage_names = [stage_names]
    total = 0
    for name in stage_names:
        mod = getattr(model.backbone, name, None)
        if mod is None and '.' in name:
            parts = name.split('.', 1)
            base  = getattr(model.backbone, parts[0], None)
            if base is not None and parts[1].isdigit():
                try: mod = base[int(parts[1])]
                except: pass
        if mod is None: print(f"  [skip] '{name}' not found"); continue
        cnt = 0
        for p in mod.parameters():
            if not p.requires_grad: p.requires_grad = True; cnt += 1
        for m in mod.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if m.weight is not None: m.weight.requires_grad = True
                if m.bias   is not None: m.bias.requires_grad   = True
        total += cnt
        if cnt: print(f"  Unfrozen: backbone.{name} ({cnt:,} params)")
    print(f"  Total unfrozen: {total:,} params\n")


def print_backbone_structure(model):
    print(f"\n{SEP}\n BACKBONE STRUCTURE\n{SEP}")
    for name, mod in model.backbone.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.ModuleList):
            print(f"  {name}: ModuleList[{len(mod)}]  ({n:,} params)")
            for i, sub in enumerate(mod):
                sp = sum(p.numel() for p in sub.parameters())
                print(f"    [{i}]: {type(sub).__name__}  ({sp:,} params)")
        else:
            print(f"  {name}: {type(mod).__name__}  ({n:,} params)")
    print(f"{SEP}\n")


# ============================================================
# MODEL CONFIG
# ============================================================

class ModelConfig:
    @staticmethod
    def get_config(variant='fan_dwsa'):
        C = 32
        bb = {
            "in_channels": 3, "channels": C, "ppm_channels": 128,
            "num_blocks_per_stage": [4, 4, [5,4], [5,4], [2,2]],
            "align_corners": False, "deploy": False,
            "norm_cfg": dict(type='BN', requires_grad=True),
            "act_cfg":  dict(type='ReLU', inplace=True),
        }
        if variant in ('fan_dwsa', 'dwsa_only'):
            bb["dwsa_reduction"] = 8
        return {
            "backbone": bb,
            "head": {
                "in_channels": C*4, "channels": 64,
                "align_corners": False, "dropout_ratio": 0.1,
                "norm_cfg": dict(type='BN', requires_grad=True),
                "act_cfg":  dict(type='ReLU', inplace=True),
            },
            "loss": {"ce_weight": 1.0, "dice_weight": 0.5, "dice_smooth": 1e-5},
        }


# ============================================================
# SEGMENTOR
# ============================================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        return self.decode_head(self.backbone(x))

    def forward_train(self, x, return_aux=True):
        features = self.backbone(x, return_aux=return_aux)
        return {"main": self.decode_head(features)}


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, args,
                 class_weights=None, diag=None):
        self.model       = model.to(device)
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.args        = args
        self.best_miou   = 0.0
        self.start_epoch = 0
        self.global_step = 0
        self.diag        = diag
        self.raindrop_multipliers = {}

        lcfg = args.loss_config
        self.ce_weight    = lcfg['ce_weight']
        self.dice_weight  = lcfg['dice_weight']
        self.base_loss_cfg = lcfg
        self.loss_phase   = 'full'

        cw = class_weights.to(device) if class_weights is not None else None
        self.ohem = OHEMLoss(
            ignore_index=args.ignore_index,
            keep_ratio=getattr(args,'ohem_keep_ratio',0.3),
            min_kept=getattr(args,'ohem_min_kept',100000),
            thresh=getattr(args,'ohem_thresh',None),
            class_weights=cw)

        self.dice = DiceLoss(smooth=lcfg['dice_smooth'],
                             ignore_index=args.ignore_index,
                             class_weights=cw)
        _ls = getattr(args, 'label_smoothing', 0.0)
        self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=args.ignore_index,
                                      label_smoothing=_ls)

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = (_make_writer(self.save_dir / "tensorboard")
                       if getattr(args, "enable_tensorboard", True)
                       else _NullWriter())
        self._save_config()
        self._print_config()

        ohem_mode = (f"threshold-based (thresh={getattr(args,'ohem_thresh',None)})"
                     if getattr(args,'ohem_thresh',None)
                     else f"ratio-based (keep_ratio={getattr(args,'ohem_keep_ratio',0.3)})")
        print(f"OHEM: {ohem_mode}")
        if _ls > 0: print(f"Label smoothing: {_ls}")

    def _save_config(self):
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

    def _print_config(self):
        print(f"\n{SEP}\nTRAINER CONFIGURATION\n{SEP}")
        print(f"Batch size:            {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.accumulation_steps}")
        print(f"Effective batch:       {self.args.batch_size * self.args.accumulation_steps}")
        print(f"Mixed precision:       {self.args.use_amp}")
        print(f"Gradient clipping:     {self.args.grad_clip}")
        print(f"Loss: CE({self.ce_weight}) + Dice({self.dice_weight})")
        print(f"{SEP}\n")

    def set_loss_phase(self, phase):
        if phase == self.loss_phase: return
        self.dice_weight = 0.0 if phase == 'ce_only' else self.base_loss_cfg['dice_weight']
        self.loss_phase  = phase
        print(f"Loss phase → {phase}  (CE={self.ce_weight}, Dice={self.dice_weight})")

    def set_raindrop_multipliers(self, multipliers):
        """Set per-group LR multipliers without changing scheduler state."""
        self.raindrop_multipliers = dict(multipliers or {})

    def train_epoch(self, loader, epoch):
        self.model.train()

        # .train() recursively reactivates BN even after a backbone was frozen.
        # Eval mode fixes running_mean/var while affine parameters may still train.
        if getattr(self.args, "lock_bn_stats", False):
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        # Re-apply freezes each epoch (model.train() doesn't restore requires_grad)
        if getattr(self.args, "freeze_spp_bn", False):
            spp = getattr(self.model.backbone, "spp", None)
            if spp:
                for p in spp.parameters(): p.requires_grad = False
                for m in spp.modules():
                    if isinstance(m, nn.BatchNorm2d): m.eval()

        if getattr(self.args, "freeze_stem_conv", False):
            for sn in ["stem_conv1","stem_conv2"]:
                mod = getattr(self.model.backbone, sn, None)
                if mod is None: continue
                for pn, p in mod.named_parameters():
                    if not any(k in pn for k in ("alpha","bn.","in_.")): p.requires_grad = False
            for sn in ["stem_stage2","stem_stage3"]:
                mod = getattr(self.model.backbone, sn, None)
                if mod is None: continue
                for p in mod.parameters(): p.requires_grad = False
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm2d): m.eval()

        total_loss = torch.zeros((), device=self.device)
        total_ohem = torch.zeros((), device=self.device)
        total_dice = torch.zeros((), device=self.device)
        max_grad_epoch = hard_ratio_acc = 0.0
        mg = 0.0
        grad_check_interval = getattr(self.args, "gradient_check_interval", 100)
        progress_interval = max(1, getattr(self.args, "progress_interval", 20))
        empty_cache_interval = getattr(self.args, "empty_cache_interval", 0)
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4: masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                use_aux = self.args.aux_weight > 0
                outputs = self.model.forward_train(imgs, return_aux=use_aux)["main"]
                if use_aux:
                    c4_logit, c6_logit = outputs
                else:
                    c4_logit, c6_logit = None, outputs
                target_size = masks.shape[-2:]
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

                task_loss = self.ce_weight * ohem_loss + self.dice_weight * dice_loss

                if use_aux:
                    c4_full = F.interpolate(c4_logit, size=target_size,
                                            mode='bilinear', align_corners=False)
                    aux_decay = getattr(self.args, 'aux_decay_exp', 0.9)
                    aux_w     = self.args.aux_weight * (1 - epoch / self.args.epochs) ** aux_decay
                    task_loss = task_loss + aux_w * self.ohem(c4_full, masks)

                loss = task_loss / self.args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf loss at batch {batch_idx} — skipping")
                self.optimizer.zero_grad(set_to_none=True); continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                if (grad_check_interval > 0 and
                        (self.global_step + 1) % grad_check_interval == 0):
                    mg = check_gradients(self.model, threshold=10.0)
                    max_grad_epoch = max(max_grad_epoch, mg)
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                base_lrs = [group["lr"] for group in self.optimizer.param_groups]
                for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                    factor = self.raindrop_multipliers.get(group.get("name"), 1.0)
                    group["lr"] = base_lr * factor
                try:
                    self.scaler.step(self.optimizer)
                finally:
                    for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                        group["lr"] = base_lr
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.scheduler and self.args.scheduler == 'onecycle':
                    self.scheduler.step()

            total_loss += loss.detach() * self.args.accumulation_steps
            total_ohem += ohem_loss.detach()
            total_dice += dice_loss.detach()
            hard_ratio_acc += self.ohem.last_hard_ratio

            if batch_idx % progress_interval == 0 or batch_idx + 1 == len(loader):
                pbar.set_postfix({
                    'loss': f'{loss.detach().item()*self.args.accumulation_steps:.4f}',
                    'ohem': f'{ohem_loss.detach().item():.4f}',
                    'dice': f'{dice_loss.detach().item():.4f}',
                    'lr':   f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'hard%':f'{self.ohem.last_hard_ratio:.2f}',
                    'mg':   f'{mg:.2f}',
                })
            if empty_cache_interval > 0 and (batch_idx + 1) % empty_cache_interval == 0:
                torch.cuda.empty_cache()

        n = len(loader)
        avg_hr = hard_ratio_acc / n
        print(f"\nEpoch {epoch+1} — Max grad: {max_grad_epoch:.2f}  |  Hard%: {avg_hr:.3f}")
        print(f"  LR head={self.optimizer.param_groups[0]['lr']:.2e}")

        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        result = {'loss': total_loss.item()/n, 'ohem': total_ohem.item()/n,
                  'dice': total_dice.item()/n, 'hard_ratio': avg_hr}
        if self.diag:
            self.diag.log_dict(epoch, result, prefix='train/')
            self.diag.log(epoch, 'train/max_grad', max_grad_epoch)
        return result

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        C  = self.args.num_classes
        cm = torch.zeros((C, C), dtype=torch.int64, device=self.device)
        progress_interval = max(1, getattr(self.args, "progress_interval", 20))
        pbar = tqdm(loader, desc="Validation")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits = self.model(imgs)
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode='bilinear', align_corners=False)
                loss   = self.ce(logits, masks)

            total_loss += loss.detach()
            pred = logits.argmax(1)
            target = masks
            valid  = (target >= 0) & (target < C)
            lbl = C * target[valid].long() + pred[valid].long()
            cm += torch.bincount(lbl, minlength=C * C).reshape(C, C)
            if batch_idx % progress_interval == 0 or batch_idx + 1 == len(loader):
                pbar.set_postfix({'loss': f'{loss.detach().item():.4f}'})

        cm = cm.cpu().numpy()
        inter = np.diag(cm)
        union = cm.sum(1) + cm.sum(0) - inter
        iou   = inter / (union + 1e-10)
        result = {
            'loss'         : total_loss.item() / len(loader),
            'miou'         : float(np.nanmean(iou)),
            'accuracy'     : float(inter.sum() / (cm.sum() + 1e-10)),
            'per_class_iou': iou,
        }
        if self.diag:
            self.diag.log(epoch, 'val/miou',     result['miou'])
            self.diag.log(epoch, 'val/loss',     result['loss'])
            self.diag.log(epoch, 'val/accuracy', result['accuracy'])
        return result

    def save_checkpoint(self, epoch, metrics, is_best=False):
        ckpt = {
            'epoch': epoch, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'best_miou': self.best_miou, 'metrics': metrics,
            'global_step': self.global_step,
            'model_config': getattr(self.args, 'model_config', None),
            'model_variant': getattr(self.args, 'model_variant', None),
        }
        torch.save(ckpt, self.save_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pth")
            print(f"Best model saved! mIoU: {metrics['miou']:.4f}")
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(ckpt, self.save_dir / f"epoch_{epoch+1}.pth")

    def load_checkpoint(self, path, reset_epoch=True, load_optimizer=True, reset_best_metric=False):
        ckpt  = torch.load(path, map_location=self.device, weights_only=False)
        if not reset_epoch and ckpt.get('model_config') is not None:
            if ckpt['model_config'] != getattr(self.args, 'model_config', None):
                raise ValueError("Resume checkpoint architecture differs from the current "
                                 "model. Pass the same --arch_json used for the original "
                                 "run, or use --resume_mode transfer intentionally.")
        state = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
        self.model.load_state_dict(state, strict=False)
        if load_optimizer and not reset_epoch:
            try: self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError) as e: print(f"Optimizer not loaded: {e}")
            if self.scheduler and ckpt.get('scheduler'):
                try: self.scheduler.load_state_dict(ckpt['scheduler'])
                except Exception as e: print(f"Scheduler not loaded: {e}")
            if 'scaler' in ckpt and ckpt['scaler']:
                try: self.scaler.load_state_dict(ckpt['scaler'])
                except Exception as e: print(f"Scaler not loaded: {e}")
        if reset_epoch:
            self.start_epoch = 0; self.global_step = 0
            self.best_miou   = 0.0 if reset_best_metric else ckpt.get('best_miou', 0.0)
            print(f"Weights loaded (epoch {ckpt.get('epoch','?')}), starting from 0")
        else:
            self.start_epoch = ckpt['epoch'] + 1
            self.best_miou   = ckpt.get('best_miou', 0.0)
            self.global_step = ckpt.get('global_step', 0)
            print(f"Resuming from epoch {self.start_epoch}")


# ============================================================
# CONSTANTS
# ============================================================

UNFREEZE_STAGES = [
    ['stem_conv1','stem_conv2','stem_stage2','stem_stage3','compression_1','down_1'],
    ['semantic_branch_layers.0','detail_branch_layers.0','dwsa_stage4'],
    ['semantic_branch_layers.1','detail_branch_layers.1','dwsa_stage5','compression_2','down_2'],
    ['semantic_branch_layers.2','detail_branch_layers.2','dwsa_stage6','spp'],
]

CLASS_NAMES = ['road','sidewalk','building','wall','fence','pole',
               'traffic_light','traffic_sign','vegetation','terrain',
               'sky','person','rider','car','truck','bus',
               'train','motorcycle','bicycle']


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 Training")
    # Model
    parser.add_argument("--model_variant",      type=str, default="fan_dwsa",
                        choices=["fan_dwsa","fan_only","dwsa_only"])
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--arch_json", type=str, default=None,
                        help="Search output JSON containing best_config, or a bare "
                             "architecture JSON. Applies the selected architecture "
                             "before building the model.")
    # Backbone freeze/unfreeze
    parser.add_argument("--freeze_backbone",    action="store_true")
    parser.add_argument("--freeze_all_backbone", action="store_true",
                        help="Head-only fine-tuning: freeze every backbone parameter")
    parser.add_argument("--lock_bn_stats", action="store_true",
                        help="Keep BatchNorm running stats fixed during every train epoch")
    parser.add_argument("--unfreeze_schedule",  type=str, default="")
    parser.add_argument("--freeze_stem_conv",   action="store_true")
    parser.add_argument("--freeze_spp_bn",      action="store_true")
    # LR factors
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--dwsa_lr_factor",     type=float, default=0.5)
    parser.add_argument("--alpha_lr_factor",    type=float, default=0.1)
    parser.add_argument("--stem_lr_factor",     type=float, default=0.01)
    # Data
    parser.add_argument("--train_txt",          required=True)
    parser.add_argument("--val_txt",            required=True)
    parser.add_argument("--dataset_type",       default="foggy", choices=["normal","foggy"])
    parser.add_argument("--num_classes",        type=int, default=19)
    parser.add_argument("--ignore_index",       type=int, default=255)
    parser.add_argument("--use_class_weights",  action="store_true")
    parser.add_argument("--class_weights_file", type=str, default=None)
    # Training
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--accumulation_steps", type=int,   default=2)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--weight_decay",       type=float, default=1e-4)
    parser.add_argument("--optimizer",          type=str,   default="adamw",
                        choices=["adamw","sgd"])
    parser.add_argument("--sgd_momentum",       type=float, default=0.9)
    parser.add_argument("--grad_clip",          type=float, default=5.0)
    parser.add_argument("--scheduler",          default="cosine",
                        choices=["onecycle","poly","cosine","cosine_wr"])
    parser.add_argument("--cosine_wr_t0",       type=int,   default=10)
    # Loss
    parser.add_argument("--aux_weight",         type=float, default=0.4)
    parser.add_argument("--aux_decay_exp",      type=float, default=0.9)
    parser.add_argument("--dice_weight",        type=float, default=None)
    parser.add_argument("--ce_weight",          type=float, default=None)
    parser.add_argument("--label_smoothing",    type=float, default=0.0)
    parser.add_argument("--ohem_keep_ratio",    type=float, default=0.3)
    parser.add_argument("--ohem_min_kept",      type=int,   default=100000)
    parser.add_argument("--ohem_thresh",        type=float, default=None)
    # Resolution
    parser.add_argument("--img_h",              type=int,   default=512)
    parser.add_argument("--img_w",              type=int,   default=1024)
    # BN warmup (K1)
    parser.add_argument("--reset_bn_stats",     action="store_true")
    parser.add_argument("--bn_warmup_epochs",   type=int,   default=3)
    parser.add_argument("--bn_warmup_momentum", type=float, default=0.3)
    # Misc
    parser.add_argument("--use_amp",            action="store_true", default=True)
    parser.add_argument("--num_workers",        type=int,   default=4)
    parser.add_argument("--persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--save_dir",           default="./checkpoints")
    parser.add_argument("--target_miou", type=float, default=None,
                        help="Stop after the first full-validation mIoU strictly above this value")
    parser.add_argument("--hpo_summary_json", type=str, default=None,
                        help="Write time-to-target and best mIoU for HPO orchestration")
    parser.add_argument("--max_trial_epochs", type=int, default=None,
                        help="Limit this trial's epochs without changing the scheduler horizon")
    parser.add_argument("--skip_checkpoint", action="store_true",
                        help="Do not save model weights for disposable proxy trials")
    parser.add_argument("--resume",             type=str,   default=None)
    parser.add_argument("--resume_mode",        type=str,   default="transfer",
                        choices=["transfer","continue"])
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--save_interval",      type=int,   default=10)
    parser.add_argument("--reset_best_metric",  action="store_true")
    parser.add_argument("--diag_interval",      type=int,   default=1)
    parser.add_argument("--gradient_check_interval", type=int, default=100,
                        help="Check all parameter gradients every N optimizer steps; 0 disables")
    parser.add_argument("--progress_interval", type=int, default=20,
                        help="Refresh tqdm batch metrics every N batches")
    parser.add_argument("--empty_cache_interval", type=int, default=0,
                        help="Call CUDA empty_cache every N batches; 0 keeps the allocator warm")
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3)
    # Training-only outer loop that selects AdamW group step multipliers.
    parser.add_argument("--raindrop_guided", action="store_true")
    parser.add_argument("--raindrop_guide_samples", type=int, default=192)
    parser.add_argument("--raindrop_guide_gate_fraction", type=float, default=1/3)
    parser.add_argument("--raindrop_gradient_batches", type=int, default=4)
    parser.add_argument("--raindrop_proxy_batch_size", type=int, default=4)
    parser.add_argument("--raindrop_pop_size", type=int, default=4)
    parser.add_argument("--raindrop_max_iter", type=int, default=1)
    parser.add_argument("--raindrop_factor_min", type=float, default=0.5)
    parser.add_argument("--raindrop_factor_max", type=float, default=1.5)
    parser.add_argument("--raindrop_min_ce_gain", type=float, default=1e-4)
    args = parser.parse_args()

    if args.target_miou is not None and (not math.isfinite(args.target_miou)
                                         or not 0 <= args.target_miou < 1):
        raise ValueError("--target_miou must be finite and in [0, 1)")
    if args.target_miou is not None and not args.pretrained_weights:
        raise ValueError("Time-to-target HPO requires --pretrained_weights")
    if args.max_trial_epochs is not None and not 1 <= args.max_trial_epochs <= args.epochs:
        raise ValueError("--max_trial_epochs must be between 1 and --epochs")
    if args.skip_checkpoint and not args.hpo_summary_json:
        raise ValueError("--skip_checkpoint requires --hpo_summary_json")
    if args.freeze_all_backbone and args.freeze_backbone:
        raise ValueError("Use only one backbone-freezing mode")
    if args.freeze_all_backbone and args.unfreeze_schedule:
        raise ValueError("Head-only fine-tuning does not support unfreeze_schedule")
    if args.lock_bn_stats and args.reset_bn_stats:
        raise ValueError("Cannot reset and lock BatchNorm running stats together")
    if args.raindrop_guided and args.optimizer != "adamw":
        raise ValueError("--raindrop_guided currently requires --optimizer adamw")
    if args.raindrop_guided and args.unfreeze_schedule:
        raise ValueError("Raindrop-guided mode does not support optimizer rebuilds")

    # Validate unfreeze schedule
    unfreeze_list = []
    if args.freeze_backbone and args.unfreeze_schedule:
        unfreeze_list = sorted(int(e) for e in args.unfreeze_schedule.split(','))
        if max(unfreeze_list) >= args.epochs:
            raise ValueError("unfreeze_schedule epoch >= total epochs")
        if args.scheduler == 'onecycle':
            args.scheduler = 'cosine'
            print("[INFO] scheduler auto-switched: onecycle → cosine")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{SEP}\nGCNet v3  |  {args.model_variant}\n{SEP}")
    print(f"Device: {device}  |  Image: {args.img_h}x{args.img_w}")
    print(f"Epochs: {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip: {args.grad_clip}  |  AMP: {args.use_amp}")
    if args.reset_bn_stats:
        print(f"K1: BN Reset (warmup={args.bn_warmup_epochs} ep, mom={args.bn_warmup_momentum})")
    print(f"{SEP}\n")

    # Import backbone
    if args.model_variant == 'fan_dwsa':
        from model.backbone.model import GCNet
    elif args.model_variant == 'fan_only':
        from model.backbone.fan import GCNet
    else:
        from model.backbone.dwsa import GCNet

    cfg = ModelConfig.get_config(variant=args.model_variant)
    if args.arch_json:
        arch = load_arch_json(args.arch_json)
        cfg = apply_arch_config(cfg, arch)
        print(f"Selected architecture from {args.arch_json}: {arch}")
    elif args.resume and args.resume_mode == "continue":
        # A new Kaggle session can continue a selected architecture without
        # needing the original search JSON, because it is embedded in ckpt.
        resume_meta = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_meta.get("model_config") is not None:
            saved_variant = resume_meta.get("model_variant")
            if saved_variant and saved_variant != args.model_variant:
                raise ValueError(f"Resume checkpoint uses {saved_variant!r}; pass "
                                 f"--model_variant {saved_variant}")
            cfg = resume_meta["model_config"]
            print(f"Restored architecture from resume checkpoint: {args.resume}")
    args.model_config = cfg
    args.loss_config = cfg["loss"]

    # DataLoaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt, val_txt=args.val_txt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w), pin_memory=True,
        compute_class_weights=args.use_class_weights,
        dataset_type=args.dataset_type,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed if (args.target_miou is not None or args.hpo_summary_json) else None)

    if getattr(args, "class_weights_file", None):
        cw_path = Path(args.class_weights_file)
        if cw_path.exists():
            class_weights = torch.load(cw_path, map_location="cpu")
            print(f"Class weights: {cw_path}  "
                  f"(min={class_weights.min():.3f}, max={class_weights.max():.3f})")
        else:
            print(f"WARNING: {cw_path} not found"); class_weights = None

    # Build model
    model = Segmentor(GCNet(**cfg["backbone"]),
                      GCNetHead(**cfg["head"], num_classes=args.num_classes)).to(device)
    model.apply(init_weights)
    check_model_health(model)

    if args.pretrained_weights:
        loaded_pct = load_pretrained_gcnet(
            model, args.pretrained_weights, variant=args.model_variant)
        # The 26 head keys alone are about 1% of a full model; require a
        # meaningful backbone match rather than accepting a head-only load.
        if loaded_pct < 20.0:
            raise RuntimeError(f"Only {loaded_pct:.2f}% pretrained weights matched. "
                               "Check --pretrained_weights before training.")
        if (args.target_miou is not None or args.hpo_summary_json) and loaded_pct < 99.0:
            raise RuntimeError(
                f"HPO needs the same starting model in every trial, but only "
                f"{loaded_pct:.2f}% of checkpoint weights matched")
    if args.freeze_all_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone fully frozen; training decode head only")
    elif args.freeze_backbone:
        freeze_backbone(model, variant=args.model_variant)

    count_trainable_params(model)
    print_backbone_structure(model)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader)

    save_path = Path(args.save_dir); save_path.mkdir(parents=True, exist_ok=True)
    diag    = DiagnosticLogger(save_dir=save_path, class_names=CLASS_NAMES)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler,
                      device=device, args=args,
                      class_weights=class_weights if args.use_class_weights else None,
                      diag=diag)

    if args.dice_weight is not None:
        trainer.dice_weight = args.dice_weight
        trainer.base_loss_cfg["dice_weight"] = args.dice_weight
    if args.ce_weight is not None:
        trainer.ce_weight = args.ce_weight

    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=(args.resume_mode == "transfer"),
            load_optimizer=(args.resume_mode == "continue"),
            reset_best_metric=args.reset_best_metric)

    if args.reset_bn_stats:
        reset_bn_stats(model, momentum=args.bn_warmup_momentum)

    # Freeze stem after checkpoint load
    if args.freeze_stem_conv:
        frozen = 0
        for sn in ["stem_conv1","stem_conv2"]:
            mod = getattr(model.backbone, sn, None)
            if mod is None: continue
            for pn, p in mod.named_parameters():
                if not any(k in pn for k in ("alpha","bn.","in_.")):
                    p.requires_grad = False; frozen += p.numel()
        for sn in ["stem_stage2","stem_stage3"]:
            mod = getattr(model.backbone, sn, None)
            if mod is None: continue
            for p in mod.parameters(): p.requires_grad = False; frozen += p.numel()
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
        print(f"Stem frozen: {frozen:,} params (FAN still trainable)")
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=trainer.start_epoch)
        trainer.optimizer = optimizer; trainer.scheduler = scheduler

    # Freeze SPP
    if args.freeze_spp_bn:
        spp = getattr(model.backbone, "spp", None)
        if spp:
            frozen = sum(p.numel() for p in spp.parameters() if p.requires_grad)
            for p in spp.parameters(): p.requires_grad = False
            for m in spp.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
            print(f"SPP frozen: {frozen:,} params")
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=trainer.start_epoch)
            trainer.optimizer = optimizer; trainer.scheduler = scheduler

    raindrop_guide = None
    if args.raindrop_guided:
        from torch.utils.data import DataLoader, Subset
        from data.custom import CityscapesDataset, get_val_transforms
        from raindrop_guided import RaindropAdamWGuide, split_guide_indices

        guide_dataset = CityscapesDataset(
            txt_file=args.train_txt,
            transforms=get_val_transforms(img_size=(args.img_h, args.img_w)),
            img_size=(args.img_h, args.img_w), label_mapping="train_id",
            dataset_type=args.dataset_type)
        gradient_indices, gate_indices = split_guide_indices(
            len(guide_dataset), args.raindrop_guide_samples,
            args.raindrop_guide_gate_fraction, args.seed + 10000)
        guide_loader_options = {
            "batch_size": args.raindrop_proxy_batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": True,
        }
        gradient_loader = DataLoader(
            Subset(guide_dataset, gradient_indices.tolist()),
            **guide_loader_options)
        guide_gate_loader = DataLoader(
            Subset(guide_dataset, gate_indices.tolist()),
            **guide_loader_options)
        save_path.joinpath("raindrop_guide_indices.json").write_text(
            json.dumps({"gradient": gradient_indices.tolist(),
                        "gate": gate_indices.tolist()}, indent=2),
            encoding="utf-8")
        raindrop_guide = RaindropAdamWGuide(
            model, trainer.optimizer, gradient_loader, guide_gate_loader,
            torch.device(device), pop_size=args.raindrop_pop_size,
            max_iter=args.raindrop_max_iter,
            factor_min=args.raindrop_factor_min,
            factor_max=args.raindrop_factor_max,
            gradient_batches=args.raindrop_gradient_batches,
            min_ce_gain=args.raindrop_min_ce_gain, seed=args.seed,
            log_path=save_path / "raindrop_guidance.jsonl")
        print(f"Raindrop-guided AdamW enabled: groups={raindrop_guide.group_names}, "
              f"gradient_samples={len(gradient_indices)}, "
              f"gate_samples={len(gate_indices)}")

    print(f"\n{SEP}\nSTARTING TRAINING\n{SEP}\n")
    applied_unfreeze = set()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_started = time.perf_counter()
    epochs_completed = 0
    target_epoch = None
    time_to_target_sec = None

    trial_end_epoch = (min(args.epochs, args.max_trial_epochs)
                       if args.max_trial_epochs is not None else args.epochs)
    for epoch in range(trainer.start_epoch, trial_end_epoch):

        # K1: Restore BN momentum after warmup
        if args.reset_bn_stats and epoch == trainer.start_epoch + args.bn_warmup_epochs:
            restore_bn_momentum(model)

        # Progressive unfreeze
        if epoch in unfreeze_list and epoch not in applied_unfreeze:
            idx = unfreeze_list.index(epoch)
            if idx < len(UNFREEZE_STAGES):
                print(f"[Epoch {epoch+1}] Unfreeze stage {idx+1}/{len(UNFREEZE_STAGES)}")
                unfreeze_backbone_progressive(model, UNFREEZE_STAGES[idx])
                applied_unfreeze.add(epoch)
                optimizer = build_optimizer(model, args)
                scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=epoch)
                trainer.optimizer = optimizer; trainer.scheduler = scheduler
                trainer.set_loss_phase('ce_only')

        if unfreeze_list and trainer.loss_phase == 'ce_only':
            last_un = max((e for e in unfreeze_list if e in applied_unfreeze and e <= epoch), default=None)
            if last_un is not None and epoch >= last_un + args.ce_only_epochs_after_unfreeze:
                trainer.set_loss_phase('full')

        check_spp_bn_health(model, epoch)

        if raindrop_guide is not None:
            multipliers, _ = raindrop_guide.select(epoch)
            trainer.set_raindrop_multipliers(multipliers)

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics   = trainer.validate(val_loader, epoch)
        epochs_completed += 1
        if (args.target_miou is not None
                and val_metrics['miou'] > args.target_miou):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            target_epoch = epoch + 1
            time_to_target_sec = time.perf_counter() - train_started

        if epoch % args.diag_interval == 0:
            log_dwsa_health(model, epoch, diag)
            log_fan_health(model,  epoch, diag)

        # Per-class IoU
        iou_arr = val_metrics['per_class_iou']
        print(f"\n  Per-class IoU (epoch {epoch+1}):")
        print(f"  {'Class':<16} {'IoU':>6}  Bar")
        print(f"  {'─'*43}")
        for cname, ciou in zip(CLASS_NAMES, iou_arr):
            mark = ' ⚠️' if ciou < 0.4 else (' ★' if ciou > 0.75 else '')
            print(f"  {cname:<16} {ciou:>6.4f}  {'█'*int(ciou*20)}{mark}")
        low = [n for n, v in zip(CLASS_NAMES, iou_arr) if v < 0.4]
        if low: print(f"\n  ⚠️  LOW (<0.4): {low}")

        diag.log(epoch, 'iou/best',  float(max(iou_arr)))
        diag.log(epoch, 'iou/worst', float(min(iou_arr)))

        print(f"\n{SEP}\nEpoch {epoch+1}/{args.epochs}\n{SEP}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | "
              f"OHEM: {train_metrics['ohem']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | "
              f"Hard%: {train_metrics['hard_ratio']:.3f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | "
              f"mIoU: {val_metrics['miou']:.4f}  | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"{SEP}\n")

        diag.print_epoch_summary(epoch)

        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
            print(f"  ★ NEW BEST mIoU: {trainer.best_miou:.4f}")
        if not args.skip_checkpoint:
            trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)
        if target_epoch is not None:
            print(f"Target mIoU > {args.target_miou:.6f} reached at epoch "
                  f"{target_epoch} in {time_to_target_sec/60:.1f} min")
            break

    if args.hpo_summary_json:
        summary_path = Path(args.hpo_summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            'target_miou': args.target_miou,
            'target_reached': target_epoch is not None,
            'target_epoch': target_epoch,
            'time_to_target_sec': time_to_target_sec,
            'epochs_completed': epochs_completed,
            'best_miou': float(trainer.best_miou),
            'best_checkpoint': (None if args.skip_checkpoint else
                                str(Path(args.save_dir) / 'best.pth')),
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    diag.print_full_history()
    diag.close()
    trainer.writer.close()

    print(f"\n{SEP}\nTRAINING COMPLETED!\nBest mIoU: {trainer.best_miou:.4f}\n{SEP}\n")


if __name__ == "__main__":
    main()
