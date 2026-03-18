"""
Evaluation Script: Normal / Flip-TTA / Full-TTA + Speed Benchmark
- Normal    : 1 forward pass  (~60 FPS)
- Flip-TTA  : 2 forward passes (~30 FPS, borderline realtime)
- Full-TTA  : 6 forward passes (~10 FPS, offline only)
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

from model.backbone.model import GCNetWithEnhance
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders


# ============================================================
# SEGMENTOR
# ============================================================
class Segmentor(nn.Module):
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)


# ============================================================
# SPEED BENCHMARK
# ============================================================
def measure_speed(model, input_size=(1, 3, 512, 1024),
                  num_warmup=20, num_iterations=200,
                  device='cuda', mode='normal'):
    """
    Do FPS va latency cho 3 mode inference:
      normal   : 1 forward pass
      flip_tta : 2 forward passes (original + hflip)
      full_tta : 6 forward passes (original + hflip + scale0.75x2 + scale1.25x2)
    """
    model.eval()
    dummy = torch.randn(input_size).to(device)
    H, W = input_size[2], input_size[3]

    def run_once(img):
        if mode == 'normal':
            out = model(img)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            return out

        elif mode == 'flip_tta':
            out = model(img)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            prob = F.softmax(out, dim=1)
            img_f = torch.flip(img, dims=[3])
            out_f = model(img_f)
            out_f = F.interpolate(out_f, size=(H, W), mode='bilinear', align_corners=False)
            out_f = torch.flip(out_f, dims=[3])
            prob_f = F.softmax(out_f, dim=1)
            return (prob + prob_f) / 2

        elif mode == 'full_tta':
            preds = []
            for flip in [False, True]:
                x = torch.flip(img, dims=[3]) if flip else img
                out = model(x)
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                if flip:
                    out = torch.flip(out, dims=[3])
                preds.append(F.softmax(out, dim=1))
            for scale in [0.75, 1.25]:
                h2, w2 = int(H * scale), int(W * scale)
                for flip in [False, True]:
                    x = F.interpolate(img, size=(h2, w2), mode='bilinear', align_corners=False)
                    if flip:
                        x = torch.flip(x, dims=[3])
                    out = model(x)
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                    if flip:
                        out = torch.flip(out, dims=[3])
                    preds.append(F.softmax(out, dim=1))
            return torch.stack(preds).mean(0)

    # Warmup
    print(f"  Warming up [{mode}] ({num_warmup} iters)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            run_once(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc=f"  Speed [{mode}]", leave=False):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_once(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    passes = {'normal': 1, 'flip_tta': 2, 'full_tta': 6}[mode]
    return {
        'fps': 1.0 / times.mean(),
        'latency_ms': times.mean() * 1000,
        'latency_std_ms': times.std() * 1000,
        'forward_passes': passes,
        'realtime': (1.0 / times.mean()) >= 30,
    }


def calculate_flops(model, input_size=(1, 3, 512, 1024), device='cuda'):
    try:
        from thop import profile, clever_format
        model.eval()
        dummy = torch.randn(input_size).to(device)
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy,), verbose=False)
        gflops, pstr = clever_format([flops, params], "%.3f")
        return {'gflops': gflops, 'gflops_raw': flops / 1e9, 'params': pstr}
    except ImportError:
        return {'gflops': '~4.85G (est)', 'gflops_raw': 4.85,
                'params': '~21M (est)', 'note': 'pip install thop for exact values'}


# ============================================================
# METRICS
# ============================================================
class MetricsCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype(int) + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.cm += count.reshape(self.num_classes, self.num_classes)

    def get(self):
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        iou = inter / (union + 1e-10)
        valid = union > 0
        dice = 2 * inter / (self.cm.sum(0) + self.cm.sum(1) + 1e-10)
        return {
            'miou': np.mean(iou[valid]),
            'accuracy': inter.sum() / (self.cm.sum() + 1e-10),
            'dice': np.mean(dice[valid]),
            'per_class_iou': iou,
            'per_class_dice': dice,
        }


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def run_eval(model, dataloader, num_classes, device, mode):
    model.eval()
    calc = MetricsCalculator(num_classes)
    mode_label = {
        'normal': 'Normal (1 pass)',
        'flip_tta': 'Flip-TTA (2 passes)',
        'full_tta': 'Full-TTA (6 passes)',
    }
    pbar = tqdm(dataloader, desc=f"Eval [{mode_label[mode]}]")

    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.cpu().numpy()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        H, W = masks.shape[-2:]

        if mode == 'normal':
            out = model(imgs)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            pred = out.argmax(1).cpu().numpy()

        elif mode == 'flip_tta':
            out = model(imgs)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            prob = F.softmax(out, dim=1)
            imgs_f = torch.flip(imgs, dims=[3])
            out_f = model(imgs_f)
            out_f = F.interpolate(out_f, size=(H, W), mode='bilinear', align_corners=False)
            out_f = torch.flip(out_f, dims=[3])
            prob_f = F.softmax(out_f, dim=1)
            pred = ((prob + prob_f) / 2).argmax(1).cpu().numpy()

        elif mode == 'full_tta':
            preds = []
            for flip in [False, True]:
                x = torch.flip(imgs, dims=[3]) if flip else imgs
                out = model(x)
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                if flip:
                    out = torch.flip(out, dims=[3])
                preds.append(F.softmax(out, dim=1))
            for scale in [0.75, 1.25]:
                h2, w2 = int(H * scale), int(W * scale)
                for flip in [False, True]:
                    x = F.interpolate(imgs, size=(h2, w2), mode='bilinear', align_corners=False)
                    if flip:
                        x = torch.flip(x, dims=[3])
                    out = model(x)
                    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                    if flip:
                        out = torch.flip(out, dims=[3])
                    preds.append(F.softmax(out, dim=1))
            pred = torch.stack(preds).mean(0).argmax(1).cpu().numpy()

        for i in range(pred.shape[0]):
            calc.update(pred[i], masks[i])
        m = calc.get()
        pbar.set_postfix({'mIoU': f"{m['miou']:.4f}"})

    return calc.get()


# ============================================================
# MODEL
# ============================================================
def build_model(num_classes=19, device='cuda', deploy=False):
    backbone = GCNetWithEnhance(
        in_channels=3, channels=32, ppm_channels=128,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
        dwsa_stages=['stage4', 'stage5', 'stage6'],
        dwsa_num_heads=4, dwsa_reduction=4, dwsa_qk_sharing=True,
        dwsa_groups=4, dwsa_drop=0.1, dwsa_alpha=0.1,
        use_multi_scale_context=True, ms_scales=(1, 2),
        ms_branch_ratio=16, ms_alpha=0.1,
        align_corners=False, deploy=deploy,
    ).to(device)

    head = GCNetHead(
        in_channels=128, c4_channels=64, c2_channels=32, c1_channels=32,
        decoder_channels=128, num_classes=num_classes, dropout_ratio=0.1,
        use_gated_fusion=True, use_deep_supervision=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=False), align_corners=False,
    )
    aux_head = GCNetAuxHead(
        in_channels=64, mid_channels=64, num_classes=num_classes,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=False), align_corners=False,
    )
    return Segmentor(backbone, head, aux_head).to(device)


def load_checkpoint(ckpt_path, num_classes, device, deploy=False):
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    is_deployed = any('reparam_3x3' in k for k in state.keys())
    model = build_model(num_classes, device, deploy=is_deployed)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing: {len(missing)} keys (e.g. aux_h4/h2 — OK if new)")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} keys")
    if not is_deployed and deploy:
        count = sum(1 for m in model.modules() if hasattr(m, 'switch_to_deploy')
                    and m.switch_to_deploy() is not None or hasattr(m, 'switch_to_deploy'))
        for m in model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        print(f"  Converted to deploy mode")
    print(f"  OK (deploy={'ON' if (is_deployed or deploy) else 'OFF'})")
    return model


# ============================================================
# PRINT HELPERS
# ============================================================
CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def print_per_class(metrics, title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")
    print(f"  mIoU: {metrics['miou']*100:.2f}%  |  "
          f"Acc: {metrics['accuracy']*100:.2f}%  |  "
          f"Dice: {metrics['dice']*100:.2f}%")
    print(f"  {'─'*60}")
    print(f"  {'Class':<16} {'IoU':>8}  {'Dice':>8}  {'Note'}")
    print(f"  {'─'*60}")
    for i, name in enumerate(CLASS_NAMES):
        iou = metrics['per_class_iou'][i]
        dice = metrics['per_class_dice'][i]
        note = " ← LOW" if iou < 0.50 else (" ← MID" if iou < 0.65 else "")
        print(f"  {name:<16} {iou*100:>7.2f}%  {dice*100:>7.2f}%  {note}")
    print(f"{'='*68}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Eval + Speed: Normal / Flip-TTA / Full-TTA")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--dataset_type", default="foggy", choices=["normal", "foggy"])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--modes", nargs='+',
                        default=['normal', 'flip_tta', 'full_tta'],
                        choices=['normal', 'flip_tta', 'full_tta'])
    parser.add_argument("--num_warmup", type=int, default=20)
    parser.add_argument("--num_speed_iters", type=int, default=200)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only run speed benchmark, skip accuracy eval")
    parser.add_argument("--save_results", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*68}")
    print("  EVALUATION + SPEED BENCHMARK")
    print(f"{'='*68}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Val data   : {args.val_txt} ({args.dataset_type})")
    print(f"  Image size : {args.img_h}x{args.img_w}")
    print(f"  Deploy     : {'ON' if args.deploy else 'OFF'}")
    print(f"  Modes      : {args.modes}")
    print(f"{'='*68}\n")

    # Load
    model = load_checkpoint(args.checkpoint, args.num_classes, device, args.deploy)
    model.eval()

    # Stats
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total/1e6:.2f}M")
    perf = calculate_flops(model, (1, 3, args.img_h, args.img_w), device)
    print(f"GFLOPs    : {perf['gflops']}")
    if 'note' in perf:
        print(f"  ({perf['note']})")

    # ── SPEED BENCHMARK ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  SPEED BENCHMARK  (batch=1, single image, 512x1024)")
    print(f"{'='*68}")
    print(f"  {'Mode':<20} {'FPS':>8}  {'Latency':>14}  {'Passes':>7}  {'Realtime'}")
    print(f"  {'─'*65}")

    speed_results = {}
    for mode in args.modes:
        s = measure_speed(
            model,
            input_size=(1, 3, args.img_h, args.img_w),
            num_warmup=args.num_warmup,
            num_iterations=args.num_speed_iters,
            device=device,
            mode=mode,
        )
        speed_results[mode] = s
        rt = "YES" if s['realtime'] else "NO"
        print(f"  {mode:<20} {s['fps']:>7.1f}  "
              f"{s['latency_ms']:>8.2f}ms ±{s['latency_std_ms']:.1f}  "
              f"{s['forward_passes']:>5}x  {rt}")
    print(f"{'='*68}")

    # ── ACCURACY EVALUATION ──────────────────────────────────────────
    all_metrics = {}
    if not args.skip_eval:
        print(f"\n{'='*68}")
        print("  ACCURACY EVALUATION")
        print(f"{'='*68}\n")

        _, val_loader, _ = create_dataloaders(
            train_txt=args.val_txt,
            val_txt=args.val_txt,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=(args.img_h, args.img_w),
            pin_memory=True,
            compute_class_weights=False,
            dataset_type=args.dataset_type,
        )
        print(f"Val samples: {len(val_loader.dataset)}\n")

        mode_titles = {
            'normal':   'NORMAL (1 forward pass)',
            'flip_tta': 'FLIP-TTA (2 passes — borderline realtime ~30 FPS)',
            'full_tta': 'FULL-TTA (6 passes — offline, ~10 FPS)',
        }
        for mode in args.modes:
            metrics = run_eval(model, val_loader, args.num_classes, device, mode)
            all_metrics[mode] = metrics
            print_per_class(metrics, mode_titles[mode])

    # ── FINAL SUMMARY ────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  SUMMARY")
    print(f"{'='*68}")
    print(f"  {'Mode':<20} {'mIoU':>8}  {'FPS':>8}  {'Realtime':>10}  {'Passes'}")
    print(f"  {'─'*65}")
    for mode in args.modes:
        s = speed_results[mode]
        miou_str = f"{all_metrics[mode]['miou']*100:.2f}%" if mode in all_metrics else "  N/A  "
        rt = "YES" if s['realtime'] else "NO"
        print(f"  {mode:<20} {miou_str:>8}  {s['fps']:>7.1f}  {rt:>10}  {s['forward_passes']}x")
    print(f"{'='*68}\n")

    # Save
    if args.save_results:
        out = {
            'checkpoint': args.checkpoint,
            'deploy': args.deploy,
            'speed': {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                      for k, v in speed_results.items()},
            'accuracy': {
                mode: {
                    'miou': float(all_metrics[mode]['miou']),
                    'accuracy': float(all_metrics[mode]['accuracy']),
                    'per_class_iou': {CLASS_NAMES[i]: float(all_metrics[mode]['per_class_iou'][i])
                                      for i in range(args.num_classes)}
                }
                for mode in args.modes if mode in all_metrics
            }
        }
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Results saved: {args.save_results}")


if __name__ == "__main__":
    main()
