"""
speed_benchmark.py — Benchmark tốc độ tối ưu nhất cho Our model

Áp dụng theo thứ tự:
  1. deploy + fuse_conv_bn  (reparameterize GCBlock + fuse BN vào Conv)
  2. FP16                   (+20-30 FPS, P100 Tensor Cores)
  3. torch.compile()        (+10-20 FPS, kernel fusion)
  4. TensorRT               (+30-50 FPS, nếu torch2trt available)

Cách dùng trên Kaggle:
    python speed_benchmark.py \
        --ckpt ./checkpoints/weighted_run/best.pth \
        --model_variant fan_dwsa \
        --img_h 512 --img_w 1024

    # Tất cả optimizations
    python speed_benchmark.py \
        --ckpt ./checkpoints/weighted_run/best.pth \
        --model_variant fan_dwsa \
        --fp16 --compile --tensorrt

    # So sánh tất cả modes
    python speed_benchmark.py \
        --ckpt ./checkpoints/weighted_run/best.pth \
        --model_variant fan_dwsa \
        --compare_all
"""

import argparse
import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead


# ─── fuse Conv+BN ────────────────────────────────────────────
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
            in_channels=C*4, channels=64, num_classes=19,
            align_corners=False, dropout_ratio=0.0, ignore_index=255,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
        )
    )

    ck    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model') or ck.get('model_state_dict') or ck.get('state_dict') or ck
    model.load_state_dict(state, strict=False)
    miou  = ck.get('best_miou', '?')
    print(f"  Loaded {variant}: mIoU={miou}")
    return model.to(device).eval()


# ─── Optimization pipeline ───────────────────────────────────
def step1_deploy_fuse(model):
    """Step 1: GCBlock reparameterization + Conv-BN fusion."""
    model.backbone.switch_to_deploy()
    fuse_conv_bn(model)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ Step 1: deploy + fuse_conv_bn  ({params:.2f}M params)")
    return model


def step2_fp16(model):
    """Step 2: Convert model weights sang FP16."""
    model.half()
    print(f"  ✓ Step 2: FP16 (model.half())")
    return model


def step3_compile(model):
    """Step 3: torch.compile() — yêu cầu CUDA Capability >= 7.0 (Triton backend)."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            print(f"  ✗ Step 3: torch.compile() skipped — "
                  f"GPU Capability {major}.{minor} < 7.0 (P100=6.0 không hỗ trợ Triton)")
            return model
    try:
        compiled = torch.compile(model, mode='reduce-overhead')
        print(f"  ✓ Step 3: torch.compile(mode='reduce-overhead')")
        return compiled
    except Exception as e:
        print(f"  ✗ Step 3: torch.compile() failed — {e}")
        return model


def step4_tensorrt(model, img_h, img_w, device, fp16=False):
    """Step 4: TensorRT export via torch2trt."""
    try:
        from torch2trt import torch2trt
        dummy = torch.randn(1, 3, img_h, img_w, device=device)
        if fp16:
            dummy = dummy.half()
        trt_model = torch2trt(
            model, [dummy],
            fp16_mode=fp16,
            max_workspace_size=1 << 30,
        )
        print(f"  ✓ Step 4: TensorRT (fp16={fp16})")
        return trt_model, True
    except ImportError:
        print(f"  ✗ Step 4: torch2trt not installed")
        print(f"            pip install torch2trt")
        return model, False
    except Exception as e:
        print(f"  ✗ Step 4: TensorRT failed — {e}")
        return model, False


# ─── Benchmark core ──────────────────────────────────────────
def measure_fps(model, img_h, img_w, device, fp16=False, runs=2):
    """
    Methodology = torch_speed.py:
      warmup 10 → auto-calibrate (elapsed>=1s) → FPS*6 iters → average runs times
    """
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

    inp = torch.randn(1, 3, img_h, img_w, device=device)
    if fp16:
        inp = inp.half()

    def run():
        out = model(inp)
        if out.dtype != torch.float32:
            out = out.float()
        F.interpolate(out, size=(img_h, img_w), mode='bilinear', align_corners=False)

    fps_list = []
    for r in range(runs):
        with torch.no_grad():
            for _ in range(10): run()

        iters, elapsed = 100, 0.0
        with torch.no_grad():
            while elapsed < 1.0:
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(iters): run()
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                iters  *= 2
            iters = int(iters / elapsed * 6)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters): run()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        fps = iters / elapsed
        lat = elapsed / iters * 1000
        fps_list.append(fps)
        print(f"    run {r+1}/{runs}: {fps:.1f} FPS  {lat:.2f}ms  ({iters} iters)")

    avg_fps = sum(fps_list) / len(fps_list)
    avg_lat = 1000 / avg_fps
    return avg_fps, avg_lat


# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Speed Benchmark — Optimal")
    parser.add_argument('--ckpt',          type=str, required=True)
    parser.add_argument('--model_variant', type=str, default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',  type=int, default=512)
    parser.add_argument('--img_w',  type=int, default=1024)
    parser.add_argument('--runs',   type=int, default=2,
                        help="Số lần đo, lấy trung bình.")
    # Optimization flags
    parser.add_argument('--fp16',      action='store_true', help="FP16 inference")
    parser.add_argument('--compile',   action='store_true', help="torch.compile()")
    parser.add_argument('--tensorrt',  action='store_true', help="TensorRT export")
    # So sánh tất cả modes tự động
    parser.add_argument('--compare_all', action='store_true',
                        help="Benchmark tất cả combinations rồi in bảng so sánh")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nGPU:   {torch.cuda.get_device_name(0)}")
    print(f"Input: {args.img_h}×{args.img_w}  |  variant: {args.model_variant}\n")

    if args.compare_all:
        # ── So sánh tất cả modes ──────────────────────────────
        # Check CUDA capability để quyết định modes
        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        supports_compile = (major >= 7)

        modes = [("FP32 (baseline)", False, False, False)]
        if supports_compile:
            modes.append(("FP32 + compile", False, True, False))
        modes.append(("FP16", True, False, False))
        if supports_compile:
            modes.append(("FP16 + compile", True, True, False))
        modes.append(("FP16 + TensorRT", True, False, True))

        if not supports_compile:
            print(f"  Note: torch.compile() skipped (GPU Capability {major}.{minor} < 7.0)")
            print()
        results = []

        for mode_name, use_fp16, use_compile, use_trt in modes:
            print(f"\n{'─'*50}")
            print(f"  Mode: {mode_name}")
            print(f"{'─'*50}")

            # Build fresh model cho mỗi mode
            model = build_model(args.model_variant, args.ckpt, device)
            model = step1_deploy_fuse(model)

            trt_ok = False
            if use_trt:
                model, trt_ok = step4_tensorrt(
                    model, args.img_h, args.img_w, device, fp16=use_fp16)
                if not trt_ok:
                    print("  Skipping this mode.")
                    del model; torch.cuda.empty_cache()
                    continue
            else:
                if use_fp16:
                    model = step2_fp16(model)
                if use_compile:
                    model = step3_compile(model)

            fps, lat = measure_fps(
                model, args.img_h, args.img_w, device,
                fp16=(use_fp16 and not trt_ok), runs=args.runs)
            results.append((mode_name, fps, lat))
            del model; torch.cuda.empty_cache()

        # Print comparison table
        print(f"\n{'='*52}")
        print(f"  SPEED COMPARISON  —  {args.model_variant}  @{args.img_h}×{args.img_w}")
        print(f"{'='*52}")
        baseline_fps = results[0][1] if results else 1
        print(f"  {'Mode':<22} {'FPS':>8} {'Lat(ms)':>9} {'Speedup':>9}")
        print(f"  {'─'*50}")
        for name, fps, lat in results:
            speedup = fps / baseline_fps
            best_mark = ' ◄ BEST' if fps == max(r[1] for r in results) else ''
            print(f"  {name:<22} {fps:>8.1f} {lat:>9.2f} {speedup:>8.2f}x{best_mark}")
        print(f"{'='*52}")

    else:
        # ── Single mode benchmark ─────────────────────────────
        print(f"Building model...")
        model = build_model(args.model_variant, args.ckpt, device)

        print(f"\nApplying optimizations:")
        model = step1_deploy_fuse(model)

        trt_ok = False
        if getattr(args, 'tensorrt', False):
            model, trt_ok = step4_tensorrt(
                model, args.img_h, args.img_w, device,
                fp16=getattr(args, 'fp16', False))
        else:
            if getattr(args, 'fp16', False):
                model = step2_fp16(model)
            if getattr(args, 'compile', False):
                model = step3_compile(model)

        use_fp16 = getattr(args, 'fp16', False) and not trt_ok
        params   = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"\nBenchmarking ({args.runs} runs)...")
        fps, lat = measure_fps(model, args.img_h, args.img_w, device,
                               fp16=use_fp16, runs=args.runs)

        print(f"\n{'='*45}")
        print(f"  {args.model_variant}  @{args.img_h}×{args.img_w}")
        print(f"  FPS:     {fps:.1f}")
        print(f"  Latency: {lat:.2f} ms")
        print(f"  Params:  {params:.2f} M")
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"{'='*45}")


if __name__ == '__main__':
    main()
