"""
finetune.py — Finetune để cải thiện mDice trên small objects

Mục tiêu: thắng GCNet gốc ở cả 3 độ đo
  Current:  mIoU=0.6783 (+)  mAcc=0.7753 (+)  mDice=0.7973 (-)
  GCNet:    mIoU=0.6751      mAcc=0.7554      mDice=0.7989

Gap cần lấp: mDice +0.002 — do over-segmentation ở rider, motorcycle, person

3 thay đổi so với training gốc:
  1. BoundaryLoss — penalize FP/FN tại boundary pixels → cải thiện Dice trực tiếp
  2. Class weights tăng cho small objects (rider×1.4, motorcycle×1.5, person×1.2)
  3. Frozen backbone, chỉ train head + DWSA — tránh catastrophic forgetting

Chạy:
    python finetune.py \
        --resume ./checkpoints/best.pth \
        --train_txt /kaggle/working/train.txt \
        --val_txt   /kaggle/working/val.txt \
        --epochs 15 \
        --lr 5e-5
"""
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
NUM_CLASSES  = 19
IGNORE_INDEX = 255


# ============================================================
# BOUNDARY LOSS
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware Cross Entropy Loss.

    Tại sao giúp Dice:
      Dice = 2TP / (2TP + FP + FN)
      FP và FN tập trung nhiều nhất tại boundary pixels.
      Tăng weight cho boundary pixels → model học boundary sắc nét hơn
      → giảm FP/FN → Dice tăng.

    Implementation:
      1. Detect boundary pixels bằng max-pooling trick:
         pixel là boundary nếu neighborhood có label khác nhau
      2. Assign weight cao hơn (boundary_weight) cho boundary pixels
      3. CE loss weighted bởi boundary mask

    Hiệu quả đặc biệt với thin/small objects (rider, pole, motorcycle)
    vì chúng có tỷ lệ boundary/area cao nhất.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 boundary_weight: float = 5.0,
                 kernel_size: int = 3,
                 loss_weight: float = 0.5):
        super().__init__()
        self.ignore_index    = ignore_index
        self.boundary_weight = boundary_weight
        self.kernel_size     = kernel_size
        self.loss_weight     = loss_weight

    def _get_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        Detect boundary pixels bằng max-pool trick.
        Pixel là boundary nếu max(neighborhood) != min(neighborhood).

        target: (B, H, W) long, ignore pixels = IGNORE_INDEX
        return: (B, H, W) float, 1.0 = boundary, 0.0 = interior
        """
        # Thay ignore pixels bằng -1 để không tạo false boundary
        t = target.float().clone()
        t[target == self.ignore_index] = -1.0
        t = t.unsqueeze(1)   # (B, 1, H, W)

        pad  = self.kernel_size // 2
        tmax = F.max_pool2d(t, self.kernel_size, stride=1, padding=pad)
        tmin = -F.max_pool2d(-t, self.kernel_size, stride=1, padding=pad)

        boundary = (tmax != tmin).squeeze(1).float()   # (B, H, W)
        # Bỏ ignore pixels ra khỏi boundary mask
        boundary[target == self.ignore_index] = 0.0
        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, C, H, W) logits
        target: (B, H, W) long
        """
        boundary = self._get_boundary_mask(target)   # (B, H, W)

        # Pixel weight: boundary_weight cho boundary, 1.0 cho interior
        weight = torch.ones_like(boundary)
        weight[boundary > 0] = self.boundary_weight

        # Per-pixel CE loss
        loss_per_pixel = F.cross_entropy(
            pred, target,
            ignore_index=self.ignore_index,
            reduction='none',
        )   # (B, H, W)

        # Apply boundary weight
        valid = (target != self.ignore_index)
        loss  = (loss_per_pixel * weight * valid.float()).sum() / \
                (valid.float().sum().clamp(min=1))

        return self.loss_weight * loss


# ============================================================
# OHEM LOSS (giữ từ training gốc)
# ============================================================

class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.9, min_kept=100_000,
                 class_weights=None):
        super().__init__()
        self.ignore_index  = ignore_index
        self.thresh        = thresh
        self.min_kept      = min_kept
        self.class_weights = class_weights

    def forward(self, pred, target):
        weight = self.class_weights.to(pred.device) \
                 if self.class_weights is not None else None

        loss_px = F.cross_entropy(
            pred.float(), target,
            weight=weight.float() if weight is not None else None,
            ignore_index=self.ignore_index,
            reduction='none',
        ).view(-1)

        valid = target.view(-1) != self.ignore_index
        loss_px = loss_px[valid]
        if loss_px.numel() == 0:
            return pred.sum() * 0

        with torch.no_grad():
            probs    = torch.softmax(pred.detach().float(), dim=1).max(1)[0].view(-1)[valid]
            hard     = probs < self.thresh
            if hard.sum() < self.min_kept:
                _, idx = torch.topk(probs, min(self.min_kept, probs.numel()), largest=False)
                hard   = torch.zeros_like(hard)
                hard[idx] = True

        return loss_px[hard].mean()


# ============================================================
# DICE LOSS — thêm vào finetune để optimize Dice trực tiếp
# ============================================================

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss tính trực tiếp trên soft probabilities.
    Optimize Dice metric trực tiếp (surrogate).

    Khác với DiceLoss trong train.py (dùng hard prediction):
      - Soft: dùng softmax prob → gradient smooth, dễ optimize
      - Hard: dùng argmax → không differentiable

    loss_weight nhỏ (0.3) để không lấn át OHEM và boundary loss.
    """

    def __init__(self, ignore_index=255, smooth=1.0, loss_weight=0.3):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth       = smooth
        self.loss_weight  = loss_weight

    def forward(self, pred, target):
        pred   = pred.float()
        B, C, H, W = pred.shape

        valid     = (target != self.ignore_index)
        tgt_clamp = target.clamp(0, C-1)
        one_hot   = F.one_hot(tgt_clamp, C).permute(0, 3, 1, 2).float()
        one_hot   = one_hot * valid.unsqueeze(1).float()

        probs = F.softmax(pred, dim=1) * valid.unsqueeze(1).float()

        # Per-class Dice
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice  = (2 * inter + self.smooth) / (total + self.smooth)

        # Chỉ tính class có GT pixels
        present = one_hot.sum(dim=(0, 2, 3)) > 0
        loss    = 1.0 - dice[present].mean()

        return self.loss_weight * loss


# ============================================================
# METRICS (GCNet official)
# ============================================================

def update_accum(pred, target, ti, tu, tp, tl):
    mask = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
    p    = pred[mask].astype(np.int64)
    t    = target[mask].astype(np.int64)
    inter = p[p == t]
    ai = np.bincount(inter, minlength=NUM_CLASSES)
    ap = np.bincount(p,     minlength=NUM_CLASSES)
    al = np.bincount(t,     minlength=NUM_CLASSES)
    ti += ai;  tu += ap + al - ai;  tp += ap;  tl += al


def compute_metrics(ti, tu, tp, tl):
    present = tl > 0
    iou     = ti / (tu + 1e-10)
    acc     = ti / (tl + 1e-10)
    dice    = 2 * ti / (tp + tl + 1e-10)
    return {
        'aacc' : float(ti[present].sum() / (tl[present].sum() + 1e-10)),
        'miou' : float(np.nanmean(iou[present])),
        'macc' : float(np.nanmean(acc[present])),
        'mdice': float(np.nanmean(dice[present])),
        'per_class_iou' : iou,
        'per_class_dice': dice,
        'present'       : present,
    }


# ============================================================
# BUILD MODEL
# ============================================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.decode_head(feat)

    def forward_train(self, x):
        # Backbone ở training mode trả về (c4_feat, fused)
        feats  = self.backbone(x)           # (c4_feat, c6_feat) tuple
        logits = self.decode_head(feats)    # (c4_logit, c6_logit) tuple
        return logits


def build_model(variant, ckpt_path, device):
    from model.head.segmentation_head import GCNetHead
    C   = 32
    cfg = dict(
        in_channels=3, channels=C, ppm_channels=128,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
        align_corners=False, deploy=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
    )
    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet
        cfg['dwsa_reduction'] = 8
    elif variant == 'fan_only':
        from model.backbone.fan import GCNet
    else:
        from model.backbone.dwsa import GCNet
        cfg['dwsa_reduction'] = 8

    backbone = GCNet(**cfg)
    head     = GCNetHead(
        in_channels=C*4, channels=64, num_classes=NUM_CLASSES,
        align_corners=False, dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
    )
    model = Segmentor(backbone, head).to(device)

    ck    = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (ck.get('model') or ck.get('model_state_dict') or
             ck.get('state_dict') or ck)
    missing, _ = model.load_state_dict(state, strict=False)
    skip = ('dwsa', 'alpha', 'in_.', 'loss_', 'fog_')
    bad  = [k for k in missing if not any(s in k for s in skip)]
    print(f"  Loaded from epoch {ck.get('epoch','?')} | "
          f"recorded mIoU: {ck.get('best_miou','?')}")
    if bad:
        print(f"  ⚠️  Missing: {bad[:3]}")
    return model, ck.get('best_miou', 0.0)


# ============================================================
# FREEZE STRATEGY
# ============================================================

def freeze_for_finetune(model, variant='fan_dwsa'):
    """
    Freeze backbone hoàn toàn trừ DWSA và FAN alpha.
    Chỉ train: head (full) + DWSA gammas + FAN alpha.

    Lý do:
      - Backbone đã converge tốt (mIoU 0.678) → không cần thay đổi features
      - Head là nơi quyết định boundary → train full head
      - DWSA có thể fine-adjust global context → giữ trainable
      - Tránh catastrophic forgetting với LR thấp
    """
    # Freeze toàn bộ
    for p in model.backbone.parameters():
        p.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # Unfreeze DWSA
    dwsa_params = 0
    for name in ['dwsa_stage4', 'dwsa_stage5', 'dwsa_stage6']:
        mod = getattr(model.backbone, name, None)
        if mod:
            for p in mod.parameters():
                p.requires_grad = True
                dwsa_params += p.numel()

    # Unfreeze FAN alpha
    fan_params = 0
    for name in ['stem_conv1', 'stem_conv2']:
        mod = getattr(model.backbone, name, None)
        if mod and len(mod) > 1 and hasattr(mod[1], 'alpha'):
            for p in mod[1].parameters():
                p.requires_grad = True
                fan_params += p.numel()

    # Head: full trainable
    head_params = sum(p.numel() for p in model.decode_head.parameters())
    for p in model.decode_head.parameters():
        p.requires_grad = True
    for m in model.decode_head.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {total_train:,} params")
    print(f"    Head:  {head_params:,}")
    print(f"    DWSA:  {dwsa_params:,}")
    print(f"    FAN α: {fan_params:,}")


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    ti = np.zeros(NUM_CLASSES, dtype=np.int64)
    tu = np.zeros(NUM_CLASSES, dtype=np.int64)
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    tl = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_loss = 0.0
    ce_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for imgs, masks in tqdm(loader, desc="Val", ncols=80):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(imgs)
            if isinstance(logits, (tuple, list)):
                logits = logits[1]
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
            loss = ce_fn(logits, masks)
        total_loss += loss.item()
        update_accum(logits.argmax(1).cpu().numpy(),
                     masks.cpu().numpy(), ti, tu, tp, tl)

    m = compute_metrics(ti, tu, tp, tl)
    m['loss'] = total_loss / len(loader)
    return m


def print_val(m, epoch, best_miou, save_dir):
    iou  = m['per_class_iou']
    dice = m['per_class_dice']
    pres = m['present']

    # Summary
    print(f"\n{'='*65}")
    print(f"  EPOCH {epoch+1} RESULTS")
    print(f"{'='*65}")
    print(f"  aAcc:   {m['aacc']:.4f}")
    print(f"  mIoU:   {m['miou']:.4f}  {'★ BEST' if m['miou'] > best_miou else ''}")
    print(f"  mAcc:   {m['macc']:.4f}")
    print(f"  mDice:  {m['mdice']:.4f}")
    print(f"  Loss:   {m['loss']:.4f}")

    # vs GCNet target
    print(f"\n  vs GCNet gốc:")
    print(f"  {'Metric':<10} {'Ours':>8} {'GCNet':>8} {'Delta':>8}")
    print(f"  {'─'*38}")
    targets = [('mIoU', 0.6751), ('mAcc', 0.7554), ('mDice', 0.7989)]
    for name, gcnet_val in targets:
        ours  = m[name.lower()]
        delta = ours - gcnet_val
        mark  = '✅' if delta > 0 else '❌'
        print(f"  {mark} {name:<8} {ours:>8.4f} {gcnet_val:>8.4f} {delta:>+8.4f}")

    # Per-class Dice (vì đây là metric cần cải thiện)
    print(f"\n  Per-class Dice (target >GCNet):")
    print(f"  {'Class':<16} {'IoU':>6}  {'Dice':>6}  Bar")
    print(f"  {'─'*50}")
    gcnet_dice = {
        'road':0.986,'sidewalk':0.888,'building':0.874,'wall':0.675,
        'fence':0.691,'pole':0.691,'traffic_light':0.711,'traffic_sign':0.807,
        'vegetation':0.870,'terrain':0.703,'sky':0.801,'person':0.855,
        'rider':0.720,'car':0.955,'truck':0.812,'bus':0.840,
        'train':0.795,'motorcycle':0.681,'bicycle':0.824,
    }
    for name, i, d, p in zip(CLASS_NAMES, iou, dice, pres):
        if not p:
            continue
        bar   = '█' * int(i * 16)
        gd    = gcnet_dice.get(name, 0)
        delta = d - gd
        mark  = '✅' if delta > 0 else '  '
        print(f"  {mark}{name:<14} {i:>6.4f}  {d:>6.4f}  {bar}")

    print(f"{'='*65}\n")


# ============================================================
# TRAIN EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, scheduler,
                ohem_loss, boundary_loss, dice_loss,
                aux_weight, device, use_amp, scaler,
                accumulation_steps, epoch, args_epochs):
    model.train()
    # Giữ backbone BN ở eval (đã freeze)
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    total_ohem = total_bnd = total_dice = 0.0
    pbar = tqdm(loader, desc=f"Train E{epoch+1}", ncols=90)

    for batch_idx, (imgs, masks) in enumerate(pbar):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)

        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model.forward_train(imgs)
            if isinstance(outputs, (tuple, list)):
                c4_logit, c6_logit = outputs
            else:
                c4_logit, c6_logit = None, outputs

            c6_full = F.interpolate(c6_logit, size=masks.shape[-2:],
                                    mode='bilinear', align_corners=False)

            l_ohem = ohem_loss(c6_full, masks)
            l_bnd  = boundary_loss(c6_full, masks)

            # Dice trên c6_logit (1/8 resolution) thay vì full resolution
            # Tránh OOM: one_hot (B,19,H,W) tốn ~836MB ở 512×1024
            masks_small = F.interpolate(
                masks.unsqueeze(1).float(),
                size=c6_logit.shape[-2:],
                mode='nearest').squeeze(1).long()
            l_dice = dice_loss(c6_logit, masks_small)

            loss = l_ohem + l_bnd + l_dice

            if c4_logit is not None and aux_weight > 0:
                c4_full = F.interpolate(c4_logit, size=masks.shape[-2:],
                                        mode='bilinear', align_corners=False)
                aux_decay = aux_weight * (1 - epoch / max(args_epochs, 1)) ** 0.9
                loss = loss + aux_decay * ohem_loss(c4_full, masks)

            loss = loss / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_ohem += l_ohem.item()
        total_bnd  += l_bnd.item()
        total_dice += l_dice.item()

        pbar.set_postfix({
            'ohem': f'{l_ohem.item():.3f}',
            'bnd' : f'{l_bnd.item():.3f}',
            'dice': f'{l_dice.item():.3f}',
        })

    if scheduler:
        scheduler.step()

    n = len(loader)
    return {
        'ohem': total_ohem / n,
        'bnd' : total_bnd  / n,
        'dice': total_dice / n,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 Finetune — Boundary+Dice")
    parser.add_argument('--resume',        required=True,
                        help='Path to best.pth từ training gốc')
    parser.add_argument('--train_txt',     required=True)
    parser.add_argument('--val_txt',       required=True)
    parser.add_argument('--model_variant', default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--dataset_type',  default='foggy')
    parser.add_argument('--epochs',        type=int,   default=15)
    parser.add_argument('--batch_size',    type=int,   default=8,
                        help='Nhỏ hơn training gốc vì thêm boundary loss')
    parser.add_argument('--accumulation',  type=int,   default=4)
    parser.add_argument('--lr',            type=float, default=5e-5,
                        help='LR thấp — chỉ finetune head+DWSA')
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--img_h',         type=int,   default=512)
    parser.add_argument('--img_w',         type=int,   default=1024)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--use_amp',       action='store_true', default=True)
    parser.add_argument('--save_dir',      default='./checkpoints_finetune')
    parser.add_argument('--aux_weight',    type=float, default=0.2)
    # Loss weights
    parser.add_argument('--boundary_weight', type=float, default=5.0,
                        help='Weight của boundary pixels dalam boundary loss')
    parser.add_argument('--boundary_loss_w', type=float, default=0.5,
                        help='Scalar weight của toàn bộ boundary loss')
    parser.add_argument('--dice_loss_w',     type=float, default=0.3,
                        help='Scalar weight của soft dice loss')
    parser.add_argument('--ohem_thresh',     type=float, default=0.9)
    parser.add_argument('--class_weights_file', type=str, default=None,
                        help='Path to .pt file chứa class weights')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.use_amp and device.type == 'cuda'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  GCNet v3 Finetune — Boundary + SoftDice")
    print(f"{'='*65}")
    print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    print(f"  LR:        {args.lr}  |  Epochs: {args.epochs}")
    print(f"  Batch:     {args.batch_size} × acc{args.accumulation} = "
          f"{args.batch_size * args.accumulation} effective")
    print(f"  BndWeight: {args.boundary_weight}×  |  "
          f"BndLossW: {args.boundary_loss_w}  |  DiceLossW: {args.dice_loss_w}")
    print(f"{'='*65}\n")

    # ---- Model ----
    model, recorded_miou = build_model(args.model_variant, args.resume, device)
    best_miou = float(recorded_miou) if recorded_miou != '?' else 0.0

    print("\nFreezing backbone (keep head + DWSA + FAN trainable):")
    freeze_for_finetune(model, args.model_variant)

    # ---- Class weights ----
    class_weights = None
    if args.class_weights_file:
        class_weights = torch.load(args.class_weights_file, map_location='cpu')
        print(f"\n  Class weights: {args.class_weights_file}")
        print(f"  ratio={class_weights.max()/class_weights.min():.2f}x")
    else:
        # Default: boost small objects thêm so với official
        # rider×1.4, motorcycle×1.5, person×1.2, pole×1.3
        class_weights = torch.tensor([
            0.837, 0.918, 0.866, 1.050, 1.150, 1.300,   # road→pole
            0.975, 1.049, 0.879, 1.002, 0.954, 1.200,   # tl→person (person ×1.2)
            1.500, 0.904, 1.087, 1.096, 1.087, 1.600,   # rider(×1.5)→motorcycle(×1.6)
            1.051,
        ], dtype=torch.float32)
        class_weights = class_weights / class_weights.mean()
        print(f"\n  Class weights: built-in finetune preset")
        print(f"  ratio={class_weights.max()/class_weights.min():.2f}x")

    # ---- Losses ----
    ohem_loss     = OHEMLoss(
        ignore_index=IGNORE_INDEX, thresh=args.ohem_thresh,
        min_kept=100_000, class_weights=class_weights)
    boundary_loss = BoundaryLoss(
        ignore_index=IGNORE_INDEX,
        boundary_weight=args.boundary_weight,
        loss_weight=args.boundary_loss_w)
    dice_loss     = SoftDiceLoss(
        ignore_index=IGNORE_INDEX,
        loss_weight=args.dice_loss_w)

    # ---- Optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler(enabled=use_amp)

    print(f"\n  Optimizer: AdamW lr={args.lr} | "
          f"Scheduler: CosineAnnealing T={args.epochs}")

    # ---- DataLoader ----
    from data.custom import CityscapesDataset, get_train_transforms, get_val_transforms

    train_ds = CityscapesDataset(
        txt_file=args.train_txt,
        transforms=get_train_transforms(
            img_size=(args.img_h, args.img_w),
            dataset_type=args.dataset_type),
        img_size=(args.img_h, args.img_w),
        label_mapping='train_id',
        dataset_type=args.dataset_type,
    )
    val_ds = CityscapesDataset(
        txt_file=args.val_txt,
        transforms=get_val_transforms(img_size=(args.img_h, args.img_w)),
        img_size=(args.img_h, args.img_w),
        label_mapping='train_id',
        dataset_type=args.dataset_type,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,  # same batch, avoid OOM
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print(f"\n  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"\n{'='*65}")
    print(f"  STARTING FINETUNE")
    print(f"{'='*65}\n")

    # ---- Training loop ----
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            ohem_loss, boundary_loss, dice_loss,
            args.aux_weight, device, use_amp, scaler,
            args.accumulation, epoch, args.epochs)

        print(f"\n  Train — OHEM: {train_metrics['ohem']:.4f} | "
              f"Boundary: {train_metrics['bnd']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}")

        val_metrics = validate(model, val_loader, device, use_amp)
        print_val(val_metrics, epoch, best_miou, save_dir)

        is_best = val_metrics['miou'] > best_miou
        if is_best:
            best_miou = val_metrics['miou']

        ckpt = {
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_miou': best_miou,
            'metrics'  : val_metrics,
        }
        torch.save(ckpt, save_dir / 'last.pth')
        if is_best:
            torch.save(ckpt, save_dir / 'best.pth')
            print(f"  ★ NEW BEST saved: mIoU={best_miou:.4f}")

    print(f"\n{'='*65}")
    print(f"  FINETUNE COMPLETE")
    print(f"  Best mIoU: {best_miou:.4f}")
    print(f"  Checkpoints: {save_dir}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
