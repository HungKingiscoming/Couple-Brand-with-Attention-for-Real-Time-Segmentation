"""
train.py — GCNet v3 + GCNetHead, tối ưu cho Foggy Cityscapes
=============================================================

Những thay đổi so với bản gốc:
  1. Scheduler: PolyLR tính trên step (không bị phá vỡ khi add param group)
  2. Unfreeze: add_param_group() thay vì rebuild optimizer → giữ state
  3. DWSA gamma init = 0.1 (không phải 0) → thoát zero gradient ngay
  4. Bỏ ce_only phase → warmup LR 1 epoch sau mỗi unfreeze thay thế
  5. Label smoothing trong CE loss → regularize class imbalance
  6. Multi-scale validation (x0.75, x1.0, x1.25) → +1-2% mIoU miễn phí
  7. drop_last=False trong val_loader → metric chính xác
  8. EMA (Exponential Moving Average) weights → ổn định cuối training
"""

import os
import math
import gc
import json
import time
import warnings
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings('ignore')

from model.backbone.model import GCNet
from model.head.segmentation_head import GCNetHead
from data.custom import create_dataloaders
from model.model_utils import init_weights, check_model_health


# =============================================================================
# EMA — Exponential Moving Average weights
# =============================================================================

class ModelEMA:
    """EMA của model weights để ổn định dự đoán cuối training.

    decay ~ 0.9999 → weights EMA thay đổi rất chậm, loại bỏ oscillation.
    Chỉ dùng EMA để validate/save best — training vẫn dùng model gốc.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema    = deepcopy(model).eval()
        self.decay  = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# =============================================================================
# Poly LR (step-based) — không bị phá vỡ khi add param group
# =============================================================================

class PolyLRScheduler:
    """LR = base_lr * (1 - step / total_steps) ^ power

    Tính theo global step thay vì epoch → không quan tâm đến
    việc optimizer bị thay đổi param group sau mỗi unfreeze.
    Mỗi param group nhớ base_lr riêng.
    """

    def __init__(self, optimizer: optim.Optimizer,
                 total_steps: int, power: float = 0.9,
                 min_lr: float = 1e-6):
        self.optimizer   = optimizer
        self.total_steps = total_steps
        self.power       = power
        self.min_lr      = min_lr
        self._step       = 0
        # Snapshot base_lr của từng group lúc init
        for g in optimizer.param_groups:
            g.setdefault('base_lr', g['lr'])

    def step(self):
        self._step = min(self._step + 1, self.total_steps)
        factor = (1.0 - self._step / self.total_steps) ** self.power
        for g in self.optimizer.param_groups:
            g['lr'] = max(g['base_lr'] * factor, self.min_lr)

    def add_param_group(self, group: dict):
        """Gọi sau khi optimizer.add_param_group() để sync base_lr."""
        group.setdefault('base_lr', group['lr'])

    def state_dict(self):
        return {'_step': self._step}

    def load_state_dict(self, sd: dict):
        self._step = sd['_step']

    def get_last_lr(self) -> list:
        return [g['lr'] for g in self.optimizer.param_groups]


# =============================================================================
# Loss functions
# =============================================================================

class OHEMLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, keep_ratio: float = 0.3,
                 min_kept: int = 100_000, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index   = ignore_index
        self.keep_ratio     = keep_ratio
        self.min_kept       = min_kept
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_pixel = F.cross_entropy(
            logits, labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none',
        ).view(-1)

        valid       = labels.view(-1) != self.ignore_index
        valid_loss  = loss_pixel[valid]
        n_valid     = valid_loss.numel()
        if n_valid == 0:
            return logits.sum() * 0.0

        n_keep = max(int(self.keep_ratio * n_valid),
                     min(self.min_kept, n_valid))
        n_keep = min(n_keep, n_valid)
        if n_keep < n_valid:
            threshold  = torch.sort(valid_loss, descending=True)[0][n_keep - 1].detach()
            valid_loss = valid_loss[valid_loss >= threshold]
        return valid_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        valid       = (targets != self.ignore_index)
        t_clamp     = targets.clamp(0, C - 1)
        one_hot     = F.one_hot(t_clamp, C).permute(0, 3, 1, 2).float()
        one_hot    *= valid.unsqueeze(1).float()

        probs      = F.softmax(logits, dim=1) * valid.unsqueeze(1).float()
        p_flat     = probs.reshape(B, C, -1)
        t_flat     = one_hot.reshape(B, C, -1)

        inter      = (p_flat * t_flat).sum(2)
        card       = p_flat.sum(2) + t_flat.sum(2)
        dice       = (2.0 * inter + self.smooth) / (card + self.smooth)

        present    = t_flat.sum(2) > 0
        dice_loss  = (1.0 - dice) * present.float()
        n_present  = present.float().sum(1).clamp(min=1)
        return (dice_loss.sum(1) / n_present).mean()


# =============================================================================
# Model config
# =============================================================================

class ModelConfig:
    @staticmethod
    def get_config(num_classes: int = 19) -> dict:
        C = 32
        return {
            'backbone': {
                'in_channels'          : 3,
                'channels'             : C,
                'ppm_channels'         : 128,
                'num_blocks_per_stage' : [4, 4, [5, 4], [5, 4], [2, 2]],
                'align_corners'        : False,
                'norm_cfg'             : dict(type='BN', requires_grad=True),
                'act_cfg'              : dict(type='ReLU', inplace=True),
                'dwsa_reduction'       : 8,
                'deploy'               : False,
            },
            'head': {
                'in_channels'     : C * 4,
                'channels'        : 128,
                'num_classes'     : num_classes,
                'align_corners'   : False,
                'dropout_ratio'   : 0.1,
                'loss_weight_aux' : 0.4,
                'norm_cfg'        : dict(type='BN', requires_grad=True),
                'act_cfg'         : dict(type='ReLU', inplace=True),
            },
        }


# =============================================================================
# Segmentor wrapper
# =============================================================================

class Segmentor(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.backbone(x))

    def forward_train(self, x: torch.Tensor) -> dict:
        feats  = self.backbone(x)
        logits = self.decode_head(feats)    # (c4_logit, c6_logit)
        return {'main': logits}


# =============================================================================
# Optimizer helpers
# =============================================================================

def _build_param_groups(model: nn.Module, lr: float,
                         backbone_lr_factor: float,
                         dwsa_lr_factor: float,
                         alpha_lr_factor: float) -> list:
    """Phân nhóm params có requires_grad=True thành 4 nhóm LR."""
    head_p, backbone_p, dwsa_p, alpha_p = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'dwsa' in name:
            dwsa_p.append(param)
        elif 'alpha' in name:
            alpha_p.append(param)
        elif 'backbone' in name:
            backbone_p.append(param)
        else:
            head_p.append(param)

    groups = []
    if head_p:
        groups.append({'params': head_p,     'lr': lr,
                       'base_lr': lr,        'name': 'head'})
    if backbone_p:
        _lr = lr * backbone_lr_factor
        groups.append({'params': backbone_p, 'lr': _lr,
                       'base_lr': _lr,       'name': 'backbone'})
    if dwsa_p:
        _lr = lr * dwsa_lr_factor
        groups.append({'params': dwsa_p,     'lr': _lr,
                       'base_lr': _lr,       'name': 'dwsa'})
    if alpha_p:
        _lr = lr * alpha_lr_factor
        groups.append({'params': alpha_p,    'lr': _lr,
                       'base_lr': _lr,       'name': 'alpha'})
    return groups


def build_optimizer(model: nn.Module, args) -> optim.AdamW:
    groups = _build_param_groups(
        model, args.lr,
        args.backbone_lr_factor,
        args.dwsa_lr_factor,
        args.alpha_lr_factor,
    )
    opt = optim.AdamW(groups, weight_decay=args.weight_decay)
    print("Optimizer: AdamW (Discriminative LR)")
    for g in opt.param_groups:
        print(f"  [{g['name']}] lr={g['lr']:.2e}  params={len(g['params'])}")
    return opt


# =============================================================================
# Backbone freeze / unfreeze
# =============================================================================

def freeze_backbone(model: nn.Module):
    """Freeze toàn bộ backbone, giữ DWSA + FoggyAwareNorm trainable."""
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    # Lock tất cả BN
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None: m.weight.requires_grad_(False)
            if m.bias   is not None: m.bias.requires_grad_(False)

    # Unfreeze DWSA (BN bên trong cũng cần train mode)
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        mod = getattr(model.backbone, name, None)
        if mod is None: continue
        for p in mod.parameters(): p.requires_grad_(True)
        for m in mod.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if m.weight is not None: m.weight.requires_grad_(True)
                if m.bias   is not None: m.bias.requires_grad_(True)

    # Unfreeze FoggyAwareNorm (alpha gate + BN)
    for name in ['stem_conv1', 'stem_conv2']:
        mod = getattr(model.backbone, name, None)
        if mod is None or len(mod) < 2: continue
        fan = mod[1]
        if hasattr(fan, 'alpha'):
            for p in fan.parameters(): p.requires_grad_(True)
            fan.bn.train()
            if fan.bn.weight is not None: fan.bn.weight.requires_grad_(True)
            if fan.bn.bias   is not None: fan.bn.bias.requires_grad_(True)

    print("Backbone frozen (DWSA + FoggyAwareNorm remain trainable)")


# DWSA gamma init fix: gọi sau load_pretrained để đặt về 0.1 thay vì 0
def fix_dwsa_gamma(model: nn.Module, init_val: float = 0.1):
    """Đặt DWSA gamma = init_val (không phải 0) để tránh dead attention."""
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        mod = getattr(model.backbone, name, None)
        if mod is not None and hasattr(mod, 'gamma'):
            with torch.no_grad():
                mod.gamma.fill_(init_val)
    print(f"DWSA gamma initialized to {init_val}")


UNFREEZE_STAGES = [
    # epoch k=1: stem (domain shift bắt đầu ở low-level)
    ['stem_conv1', 'stem_conv2', 'stem_stage2', 'stem_stage3'],
    # epoch k=2: stage 4
    ['semantic_branch_layers.0', 'detail_branch_layers.0',
     'compression_1', 'down_1'],
    # epoch k=3: stage 5
    ['semantic_branch_layers.1', 'detail_branch_layers.1',
     'compression_2', 'down_2'],
    # epoch k=4: stage 6 + DAPPM
    ['semantic_branch_layers.2', 'detail_branch_layers.2', 'spp'],
]


def _get_submodule(backbone: nn.Module, name: str) -> Optional[nn.Module]:
    if hasattr(backbone, name):
        return getattr(backbone, name)
    if '.' in name:
        base, idx = name.rsplit('.', 1)
        parent = getattr(backbone, base, None)
        if parent is not None and idx.isdigit():
            try: return parent[int(idx)]
            except: pass
    return None


def unfreeze_stages(model: nn.Module, optimizer: optim.Optimizer,
                    scheduler: PolyLRScheduler,
                    stage_names: list, args,
                    warmup_factor: float = 0.2):
    """Unfreeze params và add vào optimizer hiện tại (giữ state).

    Dùng add_param_group thay vì rebuild optimizer → Adam moment không mất.
    LR khởi đầu nhỏ (warmup_factor * base_lr) để ổn định 1 epoch đầu.
    """
    existing_params = set()
    for g in optimizer.param_groups:
        existing_params.update(id(p) for p in g['params'])

    new_backbone_params = []
    for sname in stage_names:
        mod = _get_submodule(model.backbone, sname)
        if mod is None:
            print(f"  [skip] {sname} not found")
            continue
        count = 0
        for p in mod.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)
                count += 1
        for m in mod.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if m.weight is not None: m.weight.requires_grad_(True)
                if m.bias   is not None: m.bias.requires_grad_(True)
        new_backbone_params += [p for p in mod.parameters()
                                 if id(p) not in existing_params]
        existing_params.update(id(p) for p in mod.parameters())
        if count:
            print(f"  Unfrozen: backbone.{sname} ({count} params)")

    if new_backbone_params:
        warmup_lr = args.lr * args.backbone_lr_factor * warmup_factor
        group = {'params': new_backbone_params, 'lr': warmup_lr,
                 'base_lr': args.lr * args.backbone_lr_factor,
                 'name': 'backbone', 'weight_decay': args.weight_decay}
        optimizer.add_param_group(group)
        scheduler.add_param_group(group)
        print(f"  Added {len(new_backbone_params)} params to optimizer "
              f"(warmup lr={warmup_lr:.2e})")


def restore_warmup_lr(optimizer: optim.Optimizer):
    """Sau 1 epoch warmup: khôi phục base_lr cho các group vừa unfreeze."""
    for g in optimizer.param_groups:
        if g['lr'] < g['base_lr']:
            g['lr'] = g['base_lr']
            print(f"  [{g['name']}] LR restored to base_lr={g['base_lr']:.2e}")


# =============================================================================
# Pretrained weight loader
# =============================================================================

def load_pretrained(model: nn.Module, ckpt_path: str):
    import re
    print(f"Loading pretrained: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    def _remap(key):
        for pref in ('backbone.', 'model.', 'module.'):
            if key.startswith(pref): key = key[len(pref):]
        m = re.match(r'stem\.(\d+)\.(.+)$', key)
        if not m: return key
        idx, rest = int(m.group(1)), m.group(2)
        if idx == 0: return f'stem_conv1.0.{rest[5:]}' if rest.startswith('conv.') else None
        if idx == 1: return f'stem_conv2.0.{rest[5:]}' if rest.startswith('conv.') else None
        N2 = 4
        if 2 <= idx <= 1 + N2: return f'stem_stage2.{idx-2}.{rest}'
        return f'stem_stage3.{idx-(2+N2)}.{rest}'

    model_state = model.backbone.state_dict()
    compatible  = {}
    for ck, cv in state.items():
        if any(ck.startswith(p) for p in ('decode_head.', 'aux_head.', 'head.')): continue
        nk = _remap(ck)
        if nk and nk in model_state and model_state[nk].shape == cv.shape:
            compatible[nk] = cv

    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)
    loaded_pct = 100 * len(compatible) / max(len(model_state), 1)
    print(f"  Loaded: {len(compatible)}/{len(model_state)} ({loaded_pct:.1f}%)")
    exp_miss = [k for k in missing if any(s in k for s in
                ('dwsa', 'alpha', 'in_.', 'spp.'))]
    unexp    = [k for k in missing if k not in exp_miss]
    if unexp:
        print(f"  WARNING: {len(unexp)} unexpected missing keys")
        for k in unexp[:5]: print(f"    {k}")
    print(f"  Expected missing (new modules): {len(exp_miss)}\n")


# =============================================================================
# Multi-scale validation
# =============================================================================

@torch.no_grad()
def validate_multiscale(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    scales: list = (0.75, 1.0, 1.25),
    use_amp: bool = True,
) -> dict:
    """Validation với multi-scale inference + horizontal flip.

    Average logits trước khi argmax → thường +1-2% mIoU so với single-scale.
    """
    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss  = 0.0
    ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    for imgs, masks in tqdm(loader, desc="Val (MS)", leave=False):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4: masks = masks.squeeze(1)

        H, W = masks.shape[-2:]
        accum = torch.zeros(imgs.size(0), num_classes, H, W, device=device)

        for scale in scales:
            sH = int(math.ceil(H * scale / 8) * 8)
            sW = int(math.ceil(W * scale / 8) * 8)
            imgs_s = F.interpolate(imgs, size=(sH, sW), mode='bilinear',
                                   align_corners=False)

            for flip in [False, True]:
                if flip: imgs_s = imgs_s.flip(-1)

                with autocast(device_type='cuda', enabled=use_amp):
                    logit = model(imgs_s)
                    logit = F.interpolate(logit, size=(H, W), mode='bilinear',
                                          align_corners=False)

                if flip: logit = logit.flip(-1)
                accum += logit

        # Loss trên single-scale (1.0) để so sánh với train loss
        with autocast(device_type='cuda', enabled=use_amp):
            logit_1x = model(imgs)
            logit_1x = F.interpolate(logit_1x, size=(H, W), mode='bilinear',
                                      align_corners=False)
            total_loss += ce_fn(logit_1x, masks).item()

        pred   = accum.argmax(1).cpu().numpy()
        target = masks.cpu().numpy()
        valid  = (target >= 0) & (target < num_classes)
        lbl    = num_classes * target[valid].astype(np.int64) + pred[valid]
        cnt    = np.bincount(lbl, minlength=num_classes ** 2)
        conf_matrix += cnt.reshape(num_classes, num_classes)

    inter = np.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - inter
    iou   = inter / (union + 1e-10)
    return {
        'loss'         : total_loss / max(len(loader), 1),
        'miou'         : float(np.nanmean(iou)),
        'accuracy'     : float(inter.sum() / (conf_matrix.sum() + 1e-10)),
        'per_class_iou': iou,
    }


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 scheduler: PolyLRScheduler, device, args,
                 class_weights=None):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.args      = args

        self.best_miou   = 0.0
        self.start_epoch = 0
        self.global_step = 0

        # EMA
        self.ema = ModelEMA(model, decay=0.9999)

        cw = class_weights.to(device) if class_weights is not None else None

        self.ohem = OHEMLoss(
            ignore_index=args.ignore_index,
            keep_ratio=0.3,
            min_kept=100_000,
            label_smoothing=args.label_smoothing,
        )
        self.dice = DiceLoss(ignore_index=args.ignore_index)
        self.ce   = nn.CrossEntropyLoss(weight=cw, ignore_index=args.ignore_index,
                                         label_smoothing=args.label_smoothing)

        self.ce_weight   = args.ce_weight
        self.dice_weight = args.dice_weight
        self.aux_weight  = args.aux_weight

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=str(self.save_dir / 'tensorboard'))

        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    # ---------------------------------------------------------------------- #
    # Train epoch                                                              #
    # ---------------------------------------------------------------------- #

    def train_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        total_loss = total_ohem = total_dice = 0.0
        max_grad_ep = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4: masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                out = self.model.forward_train(imgs)
                c4_logit, c6_logit = out['main']

                tgt_size = masks.shape[-2:]
                c6_full  = F.interpolate(c6_logit, size=tgt_size,
                                          mode='bilinear', align_corners=False)
                c4_full  = F.interpolate(c4_logit, size=tgt_size,
                                          mode='bilinear', align_corners=False)

                ohem_loss = self.ohem(c6_full, masks)

                if self.dice_weight > 0:
                    masks_s   = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=c6_logit.shape[-2:], mode='nearest',
                    ).squeeze(1).long()
                    dice_loss = self.dice(c6_logit, masks_s)
                else:
                    dice_loss = c6_logit.sum() * 0.0

                # Auxiliary loss với poly decay
                aux_w = self.aux_weight * (1 - epoch / self.args.epochs) ** 0.9
                aux_loss = self.ohem(c4_full, masks)

                loss = (self.ce_weight * ohem_loss
                        + self.dice_weight * dice_loss
                        + aux_w * aux_loss)
                loss = loss / self.args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nNaN/Inf loss epoch={epoch} batch={batch_idx}, skip")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                max_grad = self._check_grad()
                max_grad_ep = max(max_grad_ep, max_grad)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1

            total_loss += loss.item() * self.args.accumulation_steps
            total_ohem += ohem_loss.item()
            total_dice += dice_loss.item()

            if batch_idx % self.args.log_interval == 0:
                lr_head = next((g['lr'] for g in self.optimizer.param_groups
                                if g.get('name') == 'head'), 0)
                self.writer.add_scalar('train/loss', total_loss / (batch_idx+1),
                                        self.global_step)
                self.writer.add_scalar('train/lr', lr_head, self.global_step)
                pbar.set_postfix(loss=f'{loss.item()*self.args.accumulation_steps:.4f}',
                                  ohem=f'{ohem_loss.item():.4f}',
                                  dice=f'{dice_loss.item():.4f}',
                                  max_g=f'{max_grad:.1f}')

            if batch_idx % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()

        n = len(loader)
        print(f"  Max gradient: {max_grad_ep:.2f}")
        torch.cuda.empty_cache()
        return {'loss': total_loss/n, 'ohem': total_ohem/n, 'dice': total_dice/n}

    def _check_grad(self, threshold: float = 10.0) -> float:
        max_g = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                g = p.grad.norm().item()
                if g > max_g: max_g = g
                if g > threshold:
                    print(f"  Large grad: {name[:60]} = {g:.2f}")
        return max_g

    # ---------------------------------------------------------------------- #
    # Checkpoint                                                               #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        ckpt = {
            'epoch'      : epoch,
            'model'      : self.model.state_dict(),
            'ema'        : self.ema.state_dict(),
            'optimizer'  : self.optimizer.state_dict(),
            'scheduler'  : self.scheduler.state_dict(),
            'scaler'     : self.scaler.state_dict(),
            'best_miou'  : self.best_miou,
            'metrics'    : metrics,
            'global_step': self.global_step,
        }
        torch.save(ckpt, self.save_dir / 'last.pth')
        if is_best:
            torch.save(ckpt, self.save_dir / 'best.pth')
            print(f"  Best model saved! mIoU: {metrics['miou']:.4f}")
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(ckpt, self.save_dir / f'epoch_{epoch+1}.pth')

    def load_checkpoint(self, path: str, reset_epoch: bool = True,
                         reset_best: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            self.ema.ema.load_state_dict(ckpt['ema'])
        if not reset_epoch:
            self.start_epoch = ckpt['epoch'] + 1
            self.global_step = ckpt.get('global_step', 0)
            try: self.optimizer.load_state_dict(ckpt['optimizer'])
            except: pass
            try: self.scheduler.load_state_dict(ckpt['scheduler'])
            except: pass
            try: self.scaler.load_state_dict(ckpt['scaler'])
            except: pass
        self.best_miou = 0.0 if reset_best else ckpt.get('best_miou', 0.0)
        print(f"Checkpoint loaded from {path} "
              f"(epoch={ckpt['epoch']}, best_miou={self.best_miou:.4f})")


# =============================================================================
# Utilities
# =============================================================================

def setup_env():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def log_dwsa(model: nn.Module, writer, epoch: int):
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        mod = getattr(model.backbone, name, None)
        if mod is not None and hasattr(mod, 'gamma'):
            g = mod.gamma.item()
            writer.add_scalar(f'dwsa/{name}_gamma', g, epoch)
            print(f"  {name}.gamma = {g:.6f}")


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total | {train:,} trainable ({100*train/total:.1f}%)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser('GCNet v3 — Foggy Cityscapes')

    # Pretrained / freeze
    parser.add_argument('--pretrained_weights',  type=str, default=None)
    parser.add_argument('--freeze_backbone',      action='store_true')
    parser.add_argument('--unfreeze_schedule',    type=str, default='',
                        help='Comma-separated epochs, e.g. "5,12,20,28"')
    parser.add_argument('--backbone_lr_factor',   type=float, default=0.1)
    parser.add_argument('--dwsa_lr_factor',       type=float, default=0.5)
    parser.add_argument('--alpha_lr_factor',      type=float, default=0.1)
    parser.add_argument('--dwsa_gamma_init',      type=float, default=0.1,
                        help='DWSA gamma init value (0=dead, 0.1=recommended)')

    # Dataset
    parser.add_argument('--train_txt',   required=True)
    parser.add_argument('--val_txt',     required=True)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--ignore_index',type=int, default=255)
    parser.add_argument('--use_class_weights', action='store_true')

    # Training
    parser.add_argument('--epochs',            type=int,   default=80)
    parser.add_argument('--batch_size',        type=int,   default=4)
    parser.add_argument('--accumulation_steps',type=int,   default=2)
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--grad_clip',         type=float, default=5.0)
    parser.add_argument('--ce_weight',         type=float, default=1.0)
    parser.add_argument('--dice_weight',       type=float, default=0.5)
    parser.add_argument('--aux_weight',        type=float, default=0.4)
    parser.add_argument('--label_smoothing',   type=float, default=0.05)
    parser.add_argument('--poly_power',        type=float, default=0.9)

    # Val
    parser.add_argument('--val_scales',   type=str, default='0.75,1.0,1.25',
                        help='Multi-scale inference, comma-separated')
    parser.add_argument('--val_flip',     action='store_true', default=True)

    # Image
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=1024)

    # System
    parser.add_argument('--use_amp',       action='store_true', default=True)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--save_dir',      default='./checkpoints')
    parser.add_argument('--resume',        type=str, default=None)
    parser.add_argument('--resume_mode',   choices=['transfer','continue'],
                        default='transfer')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--log_interval',  type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setup_env()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"GCNet v3  |  Foggy Cityscapes  |  device={device}")
    print(f"{'='*70}")

    # --- Data ---
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=args.use_class_weights,
        dataset_type='foggy',
    )

    # --- Model ---
    cfg      = ModelConfig.get_config(args.num_classes)
    backbone = GCNet(**cfg['backbone'])
    head     = GCNetHead(**cfg['head'])
    model    = Segmentor(backbone, head).to(device)
    model.apply(init_weights)
    check_model_health(model)

    if args.pretrained_weights:
        load_pretrained(model, args.pretrained_weights)

    # FIX: set DWSA gamma = 0.1, không phải 0
    fix_dwsa_gamma(model, init_val=args.dwsa_gamma_init)

    if args.freeze_backbone:
        freeze_backbone(model)

    count_params(model)

    # --- Optimizer + Scheduler (PolyLR step-based) ---
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    optimizer   = build_optimizer(model, args)
    scheduler   = PolyLRScheduler(optimizer, total_steps=total_steps,
                                   power=args.poly_power, min_lr=1e-6)

    # --- Unfreeze schedule ---
    unfreeze_epochs = []
    if args.unfreeze_schedule:
        unfreeze_epochs = sorted(int(e) for e in args.unfreeze_schedule.split(','))
    warmup_active_until = {}  # epoch → unfreeze epoch trigger

    # --- Val scales ---
    val_scales = [float(s) for s in args.val_scales.split(',')]

    # --- Trainer ---
    trainer = Trainer(model, optimizer, scheduler, device, args,
                       class_weights=class_weights)

    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=(args.resume_mode == 'transfer'),
            reset_best=(args.resume_mode == 'transfer'),
        )

    print(f"\n{'='*70}")
    print(f"Training  {args.epochs} epochs  |  steps={total_steps}")
    print(f"Unfreeze @ epochs: {unfreeze_epochs}")
    print(f"Val scales: {val_scales}")
    print(f"{'='*70}\n")

    for epoch in range(trainer.start_epoch, args.epochs):

        # --- Progressive unfreeze ---
        past = [e for e in unfreeze_epochs if e <= epoch]
        k    = len(past)
        for i in range(min(k, len(UNFREEZE_STAGES))):
            stage_names = UNFREEZE_STAGES[i]
            # Chỉ unfreeze lần đầu khi đúng epoch
            if i == k - 1 and epoch == past[-1]:
                print(f"\nUnfreezing stage {i+1} at epoch {epoch+1}:")
                unfreeze_stages(model, optimizer, scheduler,
                                stage_names, args, warmup_factor=0.2)
                warmup_active_until[epoch] = epoch + 1  # warmup 1 epoch

        # Khôi phục LR sau warmup 1 epoch
        if epoch in warmup_active_until.values():
            restore_warmup_lr(optimizer)

        # --- Train ---
        train_m = trainer.train_epoch(train_loader, epoch)

        # --- Validate với EMA weights + multi-scale ---
        val_m = validate_multiscale(
            trainer.ema.ema, val_loader, device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            scales=val_scales,
            use_amp=args.use_amp,
        )

        # --- Log ---
        log_dwsa(model, trainer.writer, epoch)
        trainer.writer.add_scalar('val/miou',     val_m['miou'],     epoch)
        trainer.writer.add_scalar('val/accuracy', val_m['accuracy'], epoch)
        trainer.writer.add_scalar('val/loss',     val_m['loss'],     epoch)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        print(f"Train — Loss: {train_m['loss']:.4f} | OHEM: {train_m['ohem']:.4f} | Dice: {train_m['dice']:.4f}")
        print(f"Val   — Loss: {val_m['loss']:.4f}  | mIoU: {val_m['miou']:.4f}  | Acc: {val_m['accuracy']:.4f}")

        is_best = val_m['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_m['miou']
        trainer.save_checkpoint(epoch, val_m, is_best=is_best)

    trainer.writer.close()
    print(f"\nDone! Best mIoU: {trainer.best_miou:.4f}")


if __name__ == '__main__':
    main()
