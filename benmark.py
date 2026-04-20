"""
benchmark.py — Đo tốc độ inference và so sánh 3 variants GCNet v3

Đo:
  - FPS (frames per second)
  - Latency (ms/frame)
  - Throughput (images/sec)
  - GPU memory usage (MB)
  - Params và FLOPs

Cách dùng:
    # Benchmark tất cả variants
    python benchmark.py \
        --fan_dwsa_ckpt ./checkpoints/weighted_run/best.pth \
        --fan_only_ckpt ./checkpoints/fan_only_run/best.pth \
        --dwsa_only_ckpt ./checkpoints/dwsa_only_run/best.pth

    # Benchmark một variant cụ thể
    python benchmark.py \
        --fan_dwsa_ckpt ./checkpoints/weighted_run/best.pth \
        --img_h 512 --img_w 1024 --batch_size 1 --warmup 50 --runs 200

    # So sánh nhiều resolution
    python benchmark.py \
        --fan_dwsa_ckpt ./checkpoints/weighted_run/best.pth \
        --multi_res
"""

import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def cuda_timer():
    """Context manager đo thời gian CUDA chính xác."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    yield_value = start.elapsed_time(end)  # ms
    return yield_value


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_count_flops(model, input_tensor):
    """Thử đếm FLOPs bằng fvcore hoặc thop nếu có."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9, 'fvcore'
    except ImportError:
        pass
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops / 1e9, 'thop'
    except ImportError:
        pass
    return None, 'N/A'


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.decode_head(feat)


def build_model(variant: str, ckpt_path: str, device: torch.device,
                deploy: bool = False) -> Segmentor:
    C = 32
    backbone_cfg = dict(
        in_channels=3, channels=C, ppm_channels=128,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
        align_corners=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        deploy=deploy,
    )

    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet
        backbone_cfg['dwsa_reduction'] = 8
    elif variant == 'fan_only':
        from model.backbone.model_fan_only import GCNet
        # không có dwsa_reduction
    elif variant == 'dwsa_only':
        from model.backbone.model_dwsa_only import GCNet
        backbone_cfg['dwsa_reduction'] = 8
    else:
        raise ValueError(f"Unknown variant: {variant}")

    backbone = GCNet(**backbone_cfg)
    head = GCNetHead(
        in_channels=C * 4, channels=64, num_classes=19,
        align_corners=False, dropout_ratio=0.1,
        ignore_index=255,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
    )
    model = Segmentor(backbone=backbone, head=head)

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state, strict=False)
        epoch = ckpt.get('epoch', '?')
        miou  = ckpt.get('best_miou', ckpt.get('miou', '?'))
        print(f"  Loaded {variant} from {os.path.basename(ckpt_path)} "
              f"(ep={epoch}, mIoU={miou})")
    else:
        print(f"  {variant}: no checkpoint, using random weights")

    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(model, variant, img_h, img_w, batch_size,
                    warmup, runs, device, deploy=False):
    """Đo FPS, latency, memory cho một model."""

    dummy = torch.randn(batch_size, 3, img_h, img_w, device=device)

    # Switch to deploy mode nếu cần
    if deploy and hasattr(model.backbone, 'switch_to_deploy'):
        model.backbone.switch_to_deploy()
        print(f"    [deploy mode ON]")

    # Warmup
    print(f"    Warmup ({warmup} iters)...", end='', flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    torch.cuda.synchronize()
    print(" done")

    # Measure
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB

    latencies = np.array(latencies)
    lat_mean  = np.mean(latencies)
    lat_std   = np.std(latencies)
    lat_p95   = np.percentile(latencies, 95)
    fps       = 1000 / lat_mean * batch_size

    # FLOPs (single image)
    dummy_single = torch.randn(1, 3, img_h, img_w, device=device)
    gflops, flop_lib = try_count_flops(model, dummy_single)

    total_params, _ = count_params(model)

    return {
        'variant'    : variant,
        'img_h'      : img_h,
        'img_w'      : img_w,
        'batch_size' : batch_size,
        'lat_mean_ms': lat_mean,
        'lat_std_ms' : lat_std,
        'lat_p95_ms' : lat_p95,
        'fps'        : fps,
        'peak_mem_mb': peak_mem,
        'params_m'   : total_params / 1e6,
        'gflops'     : gflops,
        'flop_lib'   : flop_lib,
        'deploy'     : deploy,
    }


def print_result(r):
    deploy_tag = ' [deploy]' if r['deploy'] else ''
    print(f"\n  ── {r['variant']}{deploy_tag}  @{r['img_h']}×{r['img_w']}  bs={r['batch_size']} ──")
    print(f"    Latency:   {r['lat_mean_ms']:.2f} ± {r['lat_std_ms']:.2f} ms  "
          f"(p95={r['lat_p95_ms']:.2f} ms)")
    print(f"    FPS:       {r['fps']:.1f} frames/sec")
    print(f"    GPU Mem:   {r['peak_mem_mb']:.0f} MB")
    print(f"    Params:    {r['params_m']:.2f} M")
    if r['gflops'] is not None:
        print(f"    GFLOPs:    {r['gflops']:.2f}  (via {r['flop_lib']})")


def print_comparison_table(results):
    if not results:
        return
    print(f"\n{'='*75}")
    print("COMPARISON TABLE")
    print(f"{'='*75}")
    header = f"{'Variant':<18} {'Deploy':>7} {'FPS':>8} {'Latency(ms)':>12} {'Mem(MB)':>9} {'Params(M)':>10} {'GFLOPs':>8}"
    print(header)
    print('-'*75)
    for r in results:
        gf = f"{r['gflops']:.2f}" if r['gflops'] else 'N/A'
        dp = '✓' if r['deploy'] else '✗'
        print(f"  {r['variant']:<16} {dp:>7} {r['fps']:>8.1f} "
              f"{r['lat_mean_ms']:>12.2f} {r['peak_mem_mb']:>9.0f} "
              f"{r['params_m']:>10.2f} {gf:>8}")
    print('='*75)

    # Highlight fastest
    fps_vals  = [(r['fps'], r['variant'], r['deploy']) for r in results]
    best_fps  = max(fps_vals, key=lambda x: x[0])
    print(f"\n  Fastest: {best_fps[1]}{'[deploy]' if best_fps[2] else ''} "
          f"@ {best_fps[0]:.1f} FPS")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 Benchmark")
    parser.add_argument("--fan_dwsa_ckpt",  type=str, default=None)
    parser.add_argument("--fan_only_ckpt",  type=str, default=None)
    parser.add_argument("--dwsa_only_ckpt", type=str, default=None)
    parser.add_argument("--img_h",    type=int, default=512)
    parser.add_argument("--img_w",    type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup",   type=int, default=50,
                        help="Số iter warmup trước khi đo.")
    parser.add_argument("--runs",     type=int, default=200,
                        help="Số iter đo chính thức.")
    parser.add_argument("--deploy",   action="store_true",
                        help="Switch sang deploy mode (reparameterize conv).")
    parser.add_argument("--multi_res", action="store_true",
                        help="Benchmark trên nhiều resolutions.")
    parser.add_argument("--no_ckpt",  action="store_true",
                        help="Chạy không cần checkpoint (random weights).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # Danh sách variants cần benchmark
    variants_to_run = []
    if args.fan_dwsa_ckpt or args.no_ckpt:
        variants_to_run.append(('fan_dwsa',  args.fan_dwsa_ckpt))
    if args.fan_only_ckpt or args.no_ckpt:
        variants_to_run.append(('fan_only',  args.fan_only_ckpt))
    if args.dwsa_only_ckpt or args.no_ckpt:
        variants_to_run.append(('dwsa_only', args.dwsa_only_ckpt))

    # Nếu không có arg nào → benchmark tất cả với random weights
    if not variants_to_run:
        print("\nKhông có checkpoint nào được chỉ định.")
        print("Chạy với --no_ckpt để benchmark với random weights, hoặc")
        print("chỉ định checkpoint với --fan_dwsa_ckpt / --fan_only_ckpt / --dwsa_only_ckpt")
        return

    # Resolutions
    if args.multi_res:
        resolutions = [(512, 1024), (384, 768), (256, 512), (720, 1280)]
    else:
        resolutions = [(args.img_h, args.img_w)]

    all_results = []

    for variant, ckpt in variants_to_run:
        print(f"\n{'='*55}")
        print(f"  Benchmarking: {variant}")
        print(f"{'='*55}")

        model = build_model(variant, ckpt, device, deploy=False)

        for h, w in resolutions:
            print(f"\n  Resolution: {h}×{w}")
            r = benchmark_model(
                model, variant, h, w, args.batch_size,
                args.warmup, args.runs, device, deploy=False)
            print_result(r)
            all_results.append(r)

            # Deploy mode
            if args.deploy:
                model_d = build_model(variant, ckpt, device, deploy=True)
                r_d = benchmark_model(
                    model_d, variant, h, w, args.batch_size,
                    args.warmup, args.runs, device, deploy=True)
                print_result(r_d)
                all_results.append(r_d)
                del model_d
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    # Final comparison table
    print_comparison_table(all_results)

    # Save CSV
    import csv
    csv_path = '/kaggle/working/benchmark_results.csv'
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved → {csv_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
