"""
validate_clean.py — Foggy Cityscapes validation (standalone, auditable)

Mục đích: kiểm chứng độc lập con số mIoU.
  - mIoU tính bằng CONFUSION MATRIX (chuẩn mmseg) — khác cách bincount cũ,
    nên nếu ra cùng số => code cũ ĐÚNG, vấn đề ở data/config.
  - In phân bố beta trong val.txt (phát hiện gộp nhiều mức sương).
  - Cờ --no_deploy để test ảnh hưởng của switch_to_deploy + fuse_conv_bn.

Cách chạy:
  python validate_clean.py --ckpt CK.pth --val_txt val.txt
  python validate_clean.py --ckpt CK.pth --val_txt val.txt --no_deploy
  python validate_clean.py --ckpt CK.pth --val_txt val.txt --img_h 1024 --img_w 2048
"""
import argparse, os, sys, re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
NUM_CLASSES, IGNORE_INDEX = 19, 255


# ------------------------------------------------------------------ model
def _fuse_conv_bn(conv, bn):
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(bn.bias.data + (b - bn.running_mean) * scale)
    return conv


def fuse_conv_bn(module):
    for child in module.children():
        fuse_conv_bn(child)
    children = list(module.named_children())
    i = 0
    while i < len(children) - 1:
        (na, ma), (nb, mb) = children[i], children[i + 1]
        if (isinstance(ma, nn.Conv2d) and isinstance(mb, (nn.BatchNorm2d, nn.SyncBatchNorm))
                and ma.out_channels == mb.num_features):
            module._modules[na] = _fuse_conv_bn(ma, mb)
            module._modules[nb] = nn.Identity()
            i += 2
        else:
            i += 1
    return module


class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.decode_head = backbone, head

    def forward(self, x):
        return self.decode_head(self.backbone(x))


def build_model(variant, ckpt_path, device, deploy):
    C = 32
    cfg = dict(in_channels=3, channels=C, ppm_channels=128,
               num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
               align_corners=False, deploy=False,
               norm_cfg=dict(type='BN', requires_grad=True),
               act_cfg=dict(type='ReLU', inplace=True))
    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet; cfg['dwsa_reduction'] = 8
    elif variant == 'fan_only':
        from model.backbone.fan import GCNet
    else:
        from model.backbone.dwsa import GCNet; cfg['dwsa_reduction'] = 8

    model = Segmentor(GCNet(**cfg),
                      GCNetHead(in_channels=C * 4, channels=64, num_classes=NUM_CLASSES,
                                align_corners=False, dropout_ratio=0.0, ignore_index=IGNORE_INDEX,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='ReLU', inplace=True)))

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model') or ck.get('model_state_dict') or ck.get('state_dict') or ck

    # strict=True để LỘ key lệch (sai variant / kiến trúc đổi) — quan trọng để debug
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ⚠️  load_state_dict KHÔNG khớp hoàn toàn:")
        print(f"       missing keys   : {len(missing)}  (vd: {missing[:3]})")
        print(f"       unexpected keys: {len(unexpected)}  (vd: {unexpected[:3]})")
        print(f"       => nếu nhiều key lệch, model đang chạy với weight random!")
    else:
        print(f"  ✅ load_state_dict khớp 100% (mọi key map đúng)")

    recorded = ck.get('best_miou', None)
    print(f"  Loaded {variant} | recorded best_miou trong ckpt: {recorded}")
    print(f"  (LƯU Ý: dòng trên chỉ là số lưu sẵn, KHÔNG phải kết quả validate)")

    if deploy:
        model.backbone.switch_to_deploy()
        nb0 = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)))
        fuse_conv_bn(model)
        nb1 = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)))
        print(f"  DEPLOY=ON : switch_to_deploy + fuse_conv_bn (BN {nb0} → {nb1})")
    else:
        print(f"  DEPLOY=OFF: giữ nguyên model gốc (không fuse) — giống lúc training validate")

    return model.to(device).eval(), recorded


# ------------------------------------------------------------------ diagnostics
def inspect_val_txt(val_txt):
    """In phân bố beta + kiểu label để phát hiện mismatch dữ liệu."""
    betas, label_kind = Counter(), Counter()
    n = 0
    with open(val_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            img, gt = line.split(',')
            m = re.search(r'foggy_beta_([0-9.]+)', img)
            betas[m.group(1) if m else 'none'] += 1
            if 'labelTrainIds' in gt:
                label_kind['labelTrainIds'] += 1
            elif 'labelIds' in gt:
                label_kind['labelIds'] += 1
            else:
                label_kind['other'] += 1

    print(f"\n  ── val.txt diagnostics ──────────────────────────────")
    print(f"  Tổng số dòng: {n}")
    print(f"  Phân bố beta : {dict(betas)}")
    print(f"  Kiểu label   : {dict(label_kind)}")
    if len(betas) > 1:
        print(f"  ⚠️  val.txt GỘP NHIỀU MỨC BETA — beta dày (vd 0.02) kéo mIoU xuống.")
        print(f"      Nếu lúc train chỉ validate 1 beta, đây là lý do số khác nhau.")
    if label_kind.get('labelTrainIds'):
        print(f"  ⚠️  Có file labelTrainIds — nhánh foggy sẽ remap SAI (về 255).")
    print(f"  ─────────────────────────────────────────────────────\n")


# ------------------------------------------------------------------ validate
@torch.no_grad()
def validate(model, val_txt, img_h, img_w, batch_size, num_workers, device, use_amp):
    from data.custom import CityscapesDataset, get_val_transforms

    ds = CityscapesDataset(txt_file=val_txt,
                           transforms=get_val_transforms(img_size=(img_h, img_w)),
                           img_size=(img_h, img_w), label_mapping='train_id',
                           dataset_type='foggy')
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=(device.type == 'cuda'), drop_last=False)
    print(f"  Validating {len(ds)} samples | {len(loader)} batches | res {img_h}×{img_w}\n")

    # CONFUSION MATRIX — gold standard, độc lập với code cũ
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for imgs, masks in tqdm(loader, desc="validate", ncols=90):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
        pred = logits.argmax(1).cpu().numpy().ravel()
        gt = masks.cpu().numpy().ravel()
        valid = (gt >= 0) & (gt < NUM_CLASSES)
        conf += np.bincount(NUM_CLASSES * gt[valid] + pred[valid],
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

    inter = np.diag(conf).astype(np.float64)
    gt_sum = conf.sum(axis=1).astype(np.float64)   # số pixel GT mỗi class
    pr_sum = conf.sum(axis=0).astype(np.float64)   # số pixel pred mỗi class
    union = gt_sum + pr_sum - inter
    present = gt_sum > 0

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    acc = np.where(gt_sum > 0, inter / np.maximum(gt_sum, 1), np.nan)

    miou = float(np.nanmean(iou[present]))
    macc = float(np.nanmean(acc[present]))
    aacc = float(inter.sum() / max(gt_sum.sum(), 1))

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ (confusion-matrix mIoU)")
    print(f"{'='*60}")
    print(f"  aAcc: {aacc:.4f}   mIoU: {miou:.4f}   mAcc: {macc:.4f}")
    print(f"{'='*60}")
    print(f"  {'Class':<16}{'IoU':>8}{'Acc':>8}{'GT px':>14}")
    print(f"  {'-'*46}")
    for k in range(NUM_CLASSES):
        flag = '  ⚠️' if (present[k] and iou[k] < 0.40) else ''
        print(f"  {CLASS_NAMES[k]:<16}{iou[k]:>8.4f}{acc[k]:>8.4f}{int(gt_sum[k]):>14,}{flag}")
    print(f"{'='*60}\n")
    return miou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--val_txt', required=True)
    ap.add_argument('--model_variant', default='fan_dwsa',
                    choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    ap.add_argument('--img_h', type=int, default=512)
    ap.add_argument('--img_w', type=int, default=1024)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--no_deploy', action='store_true',
                    help='Tắt switch_to_deploy + fuse (giống lúc training validate)')
    ap.add_argument('--no_amp', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and device.type == 'cuda'

    print(f"\n{'='*60}")
    print(f"  validate_clean  |  GPU: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  deploy={not args.no_deploy}  amp={use_amp}  res={args.img_h}×{args.img_w}")
    print(f"{'='*60}")

    inspect_val_txt(args.val_txt)
    model, _ = build_model(args.model_variant, args.ckpt, device, deploy=not args.no_deploy)
    validate(model, args.val_txt, args.img_h, args.img_w,
             args.batch_size, args.num_workers, device, use_amp)


if __name__ == '__main__':
    main()
