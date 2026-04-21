"""
evaluate.py — Validate + Benchmark cho Our model (fan_dwsa / fan_only / dwsa_only)

Validate: mIoU, mDice, mAcc, PixelAcc, per-class IoU
Benchmark: FPS, Latency, Params (deploy + fuse_conv_bn, methodology = torch_speed.py)

Cách dùng:
    # Validate + benchmark 1 variant
    python evaluate.py \
        --model_variant fan_dwsa \
        --ckpt ./checkpoints/weighted_run/best.pth \
        --val_txt /kaggle/working/val.txt \
        --dataset_type foggy \
        --img_h 512 --img_w 1024

    # Chỉ validate (không benchmark)
    python evaluate.py --model_variant fan_dwsa --ckpt ... --val_txt ... --no_benchmark

    # Chỉ benchmark (không validate)
    python evaluate.py --model_variant fan_dwsa --ckpt ... --no_validate
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead

# ─── Constants ───────────────────────────────────────────────
CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
NUM_CLASSES = 19


# ─── fuse Conv+BN (deploy, methodology = torch_speed.py) ─────
def _fuse_conv_bn(conv, bn):
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    f = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(w * f.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias   = nn.Parameter((b - bn.running_mean) * f + bn.bias)
    return conv

def fuse_conv_bn(m):
    last, lname = None, None
    try:
        for name, child in m.named_children():
            if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                if last is not None:
                    m._modules[lname] = _fuse_conv_bn(last, child)
                    m._modules[name]  = nn.Identity()
                    last = None
            elif isinstance(child, nn.Conv2d):
                last, lname = child, name
            else:
                fuse_conv_bn(child)
    except Exception:
        pass
    return m


# ─── Model ───────────────────────────────────────────────────
class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        return self.decode_head(self.backbone(x))


def build_model(variant, ckpt_path, device):
    C = 32
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
    elif variant == 'dwsa_only':
        from model.backbone.dwsa import GCNet
        cfg['dwsa_reduction'] = 8
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = Segmentor(
        GCNet(**cfg),
        GCNetHead(
            in_channels=C*4, channels=64, num_classes=NUM_CLASSES,
            align_corners=False, dropout_ratio=0.0, ignore_index=255,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
        )
    )

    ck    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model') or ck.get('model_state_dict') or ck.get('state_dict') or ck
    missing, unexpected = model.load_state_dict(state, strict=False)

    ckpt_miou  = ck.get('best_miou',  ck.get('miou',  None))
    ckpt_mdice = ck.get('best_mdice', ck.get('mdice', None))
    ckpt_macc  = ck.get('best_macc',  ck.get('macc',  None))
    epoch      = ck.get('epoch', '?')

    print(f"  Loaded {variant}: ep={epoch}  "
          f"ckpt_mIoU={ckpt_miou}  missing={len(missing)}")

    return model.to(device).eval(), {
        'miou': ckpt_miou, 'mdice': ckpt_mdice, 'macc': ckpt_macc
    }


# ─── Dataset ─────────────────────────────────────────────────
def build_dataloader(val_txt, dataset_type, img_h, img_w, num_workers=4):
    try:
        from custom import FoggyCityscapesDataset, CityscapesDataset
        DS = FoggyCityscapesDataset if dataset_type == 'foggy' else CityscapesDataset
        dataset = DS(val_txt, img_h=img_h, img_w=img_w, split='val')
    except Exception:
        import cv2
        from torch.utils.data import Dataset

        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        LABEL_MAP = np.full(256, 255, dtype=np.uint8)
        for src, dst in [(7,0),(8,1),(11,2),(12,3),(13,4),(17,5),(19,6),(20,7),
                         (21,8),(22,9),(23,10),(24,11),(25,12),(26,13),(27,14),
                         (28,15),(31,16),(32,17),(33,18)]:
            LABEL_MAP[src] = dst

        class SimpleDataset(Dataset):
            def __init__(self, txt, h, w):
                items_raw = []
                with open(txt) as f:
                    for l in f:
                        l = l.strip()
                        if l:
                            # Support comma-separated và space-separated
                            sep = ',' if ',' in l else None
                            items_raw.append(l.split(sep))
                self.items = items_raw
                self.h, self.w = h, w
                # Detect format
                if len(items_raw) > 0:
                    if len(items_raw[0]) == 1:
                        print(f"  val.txt format: img_only ({len(items_raw)} samples)")
                        print("  WARNING: mask path missing — inferring from img path")
                    else:
                        print(f"  val.txt format: img+mask ({len(items_raw)} samples)")

            def __len__(self): return len(self.items)

            def _get_mask_path(self, img_path):
                """Infer mask path từ img path (Foggy Cityscapes convention)."""
                p = img_path
                for suffix in [
                    '_leftImg8bit_foggy_beta_0.005.png',
                    '_leftImg8bit_foggy_beta_0.01.png',
                    '_leftImg8bit_foggy_beta_0.02.png',
                    '_leftImg8bit.png',
                ]:
                    if p.endswith(suffix):
                        p = p[:-len(suffix)]
                        break
                p = p.replace('/train-city/', '/gt-city/gtFine/')
                p = p.replace('/val-city/',   '/gt-city/gtFine/')
                return p + '_gtFine_labelIds.png'
            def __getitem__(self, idx):
                row = self.items[idx]
                img_path = row[0]
                mask_path = row[1] if len(row) >= 2 else self._get_mask_path(img_path)
                img  = cv2.imread(img_path)
                img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img  = cv2.resize(img, (self.w, self.h))
                img  = (img.astype(np.float32)/255.0 - MEAN) / STD
                img  = torch.from_numpy(img.transpose(2,0,1))
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if any(v > 18 and v != 255 for v in np.unique(mask)):
                    mask = LABEL_MAP[mask]
                return img, torch.from_numpy(mask.astype(np.int64))

        dataset = SimpleDataset(val_txt, img_h, img_w)

    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ─── Validate ────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for imgs, masks in tqdm(loader, desc='  Validating', ncols=80):
        imgs  = imgs.to(device)
        masks = masks.to(device).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)

        logits = model(imgs)
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode='bilinear', align_corners=False)
        pred = logits.argmax(1)

        valid = (masks >= 0) & (masks < NUM_CLASSES)
        p     = pred[valid].cpu().numpy().astype(int)
        t     = masks[valid].cpu().numpy().astype(int)
        count = np.bincount(NUM_CLASSES * t + p, minlength=NUM_CLASSES**2)
        conf += count.reshape(NUM_CLASSES, NUM_CLASSES)

    inter     = np.diag(conf)
    union     = conf.sum(1) + conf.sum(0) - inter
    iou       = inter / (union + 1e-10)
    miou      = float(np.nanmean(iou))
    pred_sum  = conf.sum(0)
    gt_sum    = conf.sum(1)
    dice      = 2 * inter / (pred_sum + gt_sum + 1e-10)
    mdice     = float(np.nanmean(dice))
    acc_cls   = inter / (gt_sum + 1e-10)
    macc      = float(np.nanmean(acc_cls))
    pixel_acc = float(inter.sum() / (conf.sum() + 1e-10))

    return {
        'miou': miou, 'mdice': mdice, 'macc': macc,
        'pixel_acc': pixel_acc,
        'per_class_iou': iou, 'per_class_dice': dice,
    }


# ─── Benchmark ───────────────────────────────────────────────
def benchmark(model, img_h, img_w, device, warmup_runs=2):
    """Methodology = torch_speed.py: deploy + fuse_conv_bn + auto-calibrate + FPS*6."""
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

    # Deploy + fuse
    if hasattr(model.backbone, 'switch_to_deploy'):
        model.backbone.switch_to_deploy()
    fuse_conv_bn(model)
    print(f"  [deploy + fuse_conv_bn applied]")

    inp = torch.randn(1, 3, img_h, img_w, device=device)

    def run():
        out = model(inp)
        F.interpolate(out, size=(img_h, img_w), mode='bilinear', align_corners=False)

    fps_list = []
    for r in range(warmup_runs):
        # Warmup
        with torch.no_grad():
            for _ in range(10): run()

        # Auto-calibrate
        iters, elapsed = 100, 0
        with torch.no_grad():
            while elapsed < 1.0:
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(iters): run()
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                iters *= 2
            iters = int(iters / elapsed * 6)

        # Final measure
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters): run()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        fps = iters / elapsed
        lat = elapsed / iters * 1000
        fps_list.append(fps)
        print(f"  run {r+1}/{warmup_runs}: FPS={fps:.1f}  Latency={lat:.2f}ms  ({iters} iters)")

    avg_fps = sum(fps_list) / len(fps_list)
    avg_lat = 1000 / avg_fps
    params  = sum(p.numel() for p in model.parameters()) / 1e6

    return {'fps': avg_fps, 'latency': avg_lat, 'params': params}


# ─── Print ────────────────────────────────────────────────────
def print_results(variant, val_metrics, bm_metrics, img_h, img_w):
    print(f"\n{'='*62}")
    print(f"  RESULTS: {variant}  @{img_h}×{img_w}")
    print(f"{'='*62}")

    if val_metrics:
        print(f"  Validation:")
        print(f"    mIoU:     {val_metrics['miou']:.4f}  ({val_metrics['miou']*100:.2f}%)")
        print(f"    mDice:    {val_metrics['mdice']:.4f}  ({val_metrics['mdice']*100:.2f}%)")
        print(f"    mAcc:     {val_metrics['macc']:.4f}  ({val_metrics['macc']*100:.2f}%)")
        print(f"    PixelAcc: {val_metrics['pixel_acc']:.4f}  ({val_metrics['pixel_acc']*100:.2f}%)")
        print(f"\n  Per-class IoU:")
        for name, iou in zip(CLASS_NAMES, val_metrics['per_class_iou']):
            bar = '█' * int(iou * 25)
            print(f"    {name:<16} {iou:.4f}  {bar}")

    if bm_metrics:
        print(f"\n  Benchmark (deploy + fuse_conv_bn):")
        print(f"    FPS:      {bm_metrics['fps']:.1f}")
        print(f"    Latency:  {bm_metrics['latency']:.2f} ms")
        print(f"    Params:   {bm_metrics['params']:.2f} M")

    if val_metrics and bm_metrics:
        print(f"\n  {'─'*40}")
        print(f"  Summary  |  mIoU={val_metrics['miou']:.4f}  "
              f"mDice={val_metrics['mdice']:.4f}  "
              f"FPS={bm_metrics['fps']:.1f}  "
              f"Lat={bm_metrics['latency']:.2f}ms  "
              f"Params={bm_metrics['params']:.2f}M")
    print(f"{'='*62}")


# ─── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate (validate + benchmark) Our model")
    parser.add_argument('--model_variant', type=str, default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--ckpt',          type=str, required=True)
    parser.add_argument('--val_txt',       type=str, default=None,
                        help="Required cho validate. Bỏ qua nếu --no_validate")
    parser.add_argument('--dataset_type',  type=str, default='foggy',
                        choices=['foggy', 'normal'])
    parser.add_argument('--img_h',         type=int, default=512)
    parser.add_argument('--img_w',         type=int, default=1024)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--benchmark_runs', type=int, default=2,
                        help="Số lần đo FPS rồi lấy trung bình.")
    parser.add_argument('--no_validate',   action='store_true',
                        help="Bỏ qua validate, chỉ benchmark.")
    parser.add_argument('--no_benchmark',  action='store_true',
                        help="Bỏ qua benchmark, chỉ validate.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Variant: {args.model_variant}  |  {args.img_h}×{args.img_w}")

    print(f"\nBuilding model...")
    model, ckpt_info = build_model(args.model_variant, args.ckpt, device)

    val_metrics = None
    bm_metrics  = None

    # ── Validate ──
    if not args.no_validate:
        if not args.val_txt:
            print("ERROR: --val_txt required for validation. Use --no_validate to skip.")
            return
        print(f"\nLoading val data: {args.val_txt}")
        loader = build_dataloader(args.val_txt, args.dataset_type,
                                  args.img_h, args.img_w, args.num_workers)
        print(f"Val samples: {len(loader.dataset)}")
        val_metrics = validate(model, loader, device)

    # ── Benchmark ──
    if not args.no_benchmark:
        print(f"\nBenchmarking ({args.benchmark_runs} runs)...")
        bm_metrics = benchmark(model, args.img_h, args.img_w,
                               device, warmup_runs=args.benchmark_runs)

    # ── Print ──
    print_results(args.model_variant, val_metrics, bm_metrics,
                  args.img_h, args.img_w)


if __name__ == '__main__':
    main()
