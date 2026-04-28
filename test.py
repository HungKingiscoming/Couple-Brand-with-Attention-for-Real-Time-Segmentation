"""
validate.py
───────────
Standalone validation script — replicate chính xác logic validate()
trong train.py để ra đúng mIoU đã ghi nhận (e.g. 0.6783).

Chạy:
    python validate.py \
        --checkpoint ./checkpoints/best.pth \
        --val_txt    /kaggle/working/val.txt \
        --model_variant fan_dwsa

Kết quả sẽ khớp với log training nếu:
  1. Dùng đúng checkpoint (best.pth)
  2. Dùng đúng val.txt
  3. img_h/img_w khớp với lúc train (default 512x1024)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
NUM_CLASSES  = 19
IGNORE_INDEX = 255


# ============================================================
# VALIDATE — replicate chính xác từ Trainer.validate()
# ============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp=True):
    model.eval()

    total_loss  = 0.0
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    ce_loss_fn  = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    pbar = tqdm(loader, desc="Validating", ncols=90)

    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)

        with autocast(device_type='cuda', enabled=use_amp):
            logits      = model(imgs)                           # (B, C, h, w)
            logits_full = F.interpolate(
                logits, size=masks.shape[-2:],
                mode='bilinear', align_corners=False)           # (B, C, H, W)
            loss = ce_loss_fn(logits_full, masks)

        total_loss += loss.item()

        pred   = logits_full.argmax(1).cpu().numpy()            # (B, H, W)
        target = masks.cpu().numpy()                            # (B, H, W)

        # confusion matrix — exact same logic as train.py
        valid = (target >= 0) & (target < NUM_CLASSES)
        label = NUM_CLASSES * target[valid].astype(int) + pred[valid]
        count = np.bincount(label, minlength=NUM_CLASSES ** 2)
        conf_matrix += count.reshape(NUM_CLASSES, NUM_CLASSES)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # metrics
    intersection = np.diag(conf_matrix)
    union        = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou          = intersection / (union + 1e-10)
    miou         = float(np.nanmean(iou))
    acc          = float(intersection.sum() / (conf_matrix.sum() + 1e-10))
    avg_loss     = total_loss / len(loader)

    return {
        'miou'         : miou,
        'accuracy'     : acc,
        'loss'         : avg_loss,
        'per_class_iou': iou,
        'conf_matrix'  : conf_matrix,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(metrics):
    iou_arr = metrics['per_class_iou']
    miou    = metrics['miou']
    acc     = metrics['accuracy']
    loss    = metrics['loss']

    print(f"\n{'='*70}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  mIoU:     {miou:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Val Loss: {loss:.4f}")
    print(f"{'='*70}")

    print(f"\n  Per-class IoU:")
    print(f"  {'Class':<16} {'IoU':>6}  Bar")
    print(f"  {'─'*45}")
    for cname, ciou in zip(CLASS_NAMES, iou_arr):
        bar  = '█' * int(ciou * 20)
        mark = ' ⚠️ ' if ciou < 0.40 else (' ★' if ciou > 0.75 else '')
        print(f"  {cname:<16} {ciou:>6.4f}  {bar}{mark}")

    low_cls = [(n, v) for n, v in zip(CLASS_NAMES, iou_arr) if v < 0.40]
    if low_cls:
        print(f"\n  ⚠️  LOW classes (<0.40): {[n for n, _ in low_cls]}")
    print(f"{'='*70}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone validation for GCNet v3")
    parser.add_argument('--checkpoint',    type=str, required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--val_txt',       type=str, required=True,
                        help='Path to val.txt')
    parser.add_argument('--model_variant', type=str, default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',         type=int, default=512)
    parser.add_argument('--img_w',         type=int, default=1024)
    parser.add_argument('--batch_size',    type=int, default=22)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--no_amp',        action='store_true',
                        help='Disable AMP (default: AMP enabled, same as training)')
    parser.add_argument('--deploy',        action='store_true',
                        help='Fuse reparam branches → single conv (faster inference)')
    args = parser.parse_args()

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = not args.no_amp and device == 'cuda'

    print(f"\n{'='*70}")
    print(f"  GCNet v3 — Standalone Validation")
    print(f"{'='*70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Val txt:    {args.val_txt}")
    print(f"  Image size: {args.img_h}×{args.img_w}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  AMP:        {use_amp}")
    print(f"  Deploy:     {args.deploy}")
    print(f"  Device:     {device}")
    print(f"{'='*70}\n")

    # ---- Model ----
    if args.model_variant == 'fan_dwsa':
        from model.backbone.model import GCNet
    elif args.model_variant == 'fan_only':
        from model.backbone.fan import GCNet
    else:
        from model.backbone.dwsa import GCNet

    from model.head.segmentation_head import GCNetHead

    C        = 32
    backbone = GCNet(
        in_channels=3, channels=C, ppm_channels=128,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
        align_corners=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        dwsa_reduction=8, deploy=False,
    )
    head = GCNetHead(
        in_channels=C * 4, channels=64, num_classes=NUM_CLASSES,
        align_corners=False, dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
    )

    class Segmentor(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone    = backbone
            self.decode_head = head
        def forward(self, x):
            feat = self.backbone(x)
            return self.decode_head(feat)

    model = Segmentor(backbone, head).to(device)

    # ---- Load checkpoint ----
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = (ckpt.get('model') or ckpt.get('model_state_dict') or
             ckpt.get('state_dict') or ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    expected_missing_markers = ('dwsa', 'alpha', 'in_.', 'foggy', 'loss_', 'fog_')
    unexpected_missing = [k for k in missing
                          if not any(s in k for s in expected_missing_markers)]
    print(f"  Checkpoint loaded from epoch {ckpt.get('epoch', '?')}")
    print(f"  Best mIoU (recorded): {ckpt.get('best_miou', 'N/A')}")
    if unexpected_missing:
        print(f"  ⚠️  Unexpected missing keys ({len(unexpected_missing)}): "
              f"{unexpected_missing[:3]}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected[:3]}")
    print()

    # ---- Deploy mode: fuse reparam branches → single conv ----
    # Phải gọi SAU load_state_dict (cần weight của tất cả branches)
    # và TRƯỚC validate (model.eval() bên trong validate)
    # Lưu ý: switch_to_deploy() KHÔNG ảnh hưởng đến output numerics
    # (chỉ fuse toán học tương đương) → mIoU sẽ giống hệt non-deploy
    if args.deploy:
        model.backbone.switch_to_deploy()
        model.eval()
        # Verify: đếm số Conv2d còn lại (reparam branches đã bị xóa)
        n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        print(f"  ✅ Deploy mode: reparam branches fused")
        print(f"     Conv2d layers after fuse: {n_conv}")
        print()

    # ---- DataLoader — exact same as training val_loader ----
    from data.custom import CityscapesDataset, get_val_transforms

    val_dataset = CityscapesDataset(
        txt_file=args.val_txt,
        transforms=get_val_transforms(img_size=(args.img_h, args.img_w)),
        img_size=(args.img_h, args.img_w),
        label_mapping='train_id',
        dataset_type='foggy',
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=False,    # khớp với train.py (FIX drop_last=False)
    )
    print(f"  Val samples: {len(val_dataset):,} ({len(val_loader)} batches)\n")

    # ---- Run ----
    metrics = validate(model, val_loader, device, use_amp=use_amp)
    print_results(metrics)

    # ---- Sanity check ----
    recorded = ckpt.get('best_miou') or ckpt.get('metrics', {}).get('miou')
    if recorded:
        diff = abs(metrics['miou'] - recorded)
        if diff < 0.001:
            print(f"  ✅ mIoU matches recorded value "
                  f"({recorded:.4f} ± {diff:.4f})")
        else:
            print(f"  ⚠️  mIoU differs from recorded: "
                  f"got {metrics['miou']:.4f}, recorded {recorded:.4f} "
                  f"(diff={diff:.4f})")
            if args.deploy:
                print(f"     → Deploy mode không gây diff — kiểm tra "
                      f"img_h/img_w và batch_size")
            else:
                print(f"     → Kiểm tra img_h/img_w và batch_size có khớp lúc train không")


if __name__ == '__main__':
    main()
