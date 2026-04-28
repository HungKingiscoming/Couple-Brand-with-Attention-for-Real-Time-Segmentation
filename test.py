"""
speed_benchmark.py — Đo FPS inference chuẩn nhất với CUDA Events

Methodology chuẩn cho paper/report:
  - batch=1, pure forward pass (không tính dataload, upsample, postprocess)
  - CUDA Events (không phải time.time) — đồng bộ trực tiếp trên GPU timeline
  - Warmup 50 iters trước khi đo
  - Trim 10% đầu + 5% outlier cao nhất
  - Báo cáo: mean, std, min, P95

Chạy:
    python speed_benchmark.py \
        --ckpt ./checkpoints/best.pth \
        --model_variant fan_dwsa \
        --img_h 512 --img_w 1024
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead


# ============================================================
# BUILD MODEL
# ============================================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        return self.decode_head(self.backbone(x))


def build_model(variant, ckpt_path, device):
    C   = 32
    cfg = dict(
        in_channels=3, channels=C, ppm_channels=128,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
        align_corners=False, deploy=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        dwsa_reduction=8,
    )
    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet
    elif variant == 'fan_only':
        from model.backbone.fan import GCNet
        cfg.pop('dwsa_reduction')
    else:
        from model.backbone.dwsa import GCNet

    model = Segmentor(
        GCNet(**cfg),
        GCNetHead(
            in_channels=C*4, channels=64, num_classes=19,
            align_corners=False, dropout_ratio=0.0,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
        )
    )
    ck    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = (ck.get('model') or ck.get('model_state_dict') or
             ck.get('state_dict') or ck)
    model.load_state_dict(state, strict=False)
    print(f"  Loaded | mIoU recorded: {ck.get('best_miou', '?')}")

    # Deploy: fuse reparam branches → single conv
    model.backbone.switch_to_deploy()
    model = model.to(device).eval()

    n_conv   = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Deploy fused | params: {n_params:.2f}M | Conv2d: {n_conv}")
    return model


# ============================================================
# BENCHMARK CORE — CUDA Events
# ============================================================

def benchmark(
    model,
    img_h: int,
    img_w: int,
    device: torch.device,
    use_amp: bool = True,
    n_warmup: int = 50,
    n_runs:   int = 300,
) -> dict:
    """
    Đo latency bằng CUDA Events — chuẩn nhất cho GPU timing.

    Tại sao CUDA Events tốt hơn time.time():
      - time.time() đo CPU wall time, không phản ánh GPU execution time
      - GPU execute async với CPU → time.time() phụ thuộc vào
        cuda.synchronize() placement và CPU scheduling jitter
      - CUDA Events được insert vào GPU command stream →
        elapsed_time() đo chính xác GPU execution time giữa 2 events

    Tại sao batch=1:
      - Latency đơn ảnh là số liệu chuẩn trong paper segmentation
      - Throughput (batch>1) có thể báo cáo thêm nhưng không phải primary

    Tại sao trim outliers:
      - 10% đầu: GPU clock chưa ổn định ngay sau warmup
      - 5% cuối: thermal throttling spikes, memory transfer outliers
    """
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True   # cuDNN autotuning — set một lần

    inp = torch.randn(1, 3, img_h, img_w, device=device)

    # Warmup — cuDNN autotuning + GPU clock stabilization
    print(f"  Warmup {n_warmup} iters...")
    with torch.no_grad():
        for _ in range(n_warmup):
            with autocast(device_type='cuda', enabled=use_amp):
                _ = model(inp)
    torch.cuda.synchronize()

    # Measure — một Event pair per iteration
    print(f"  Measuring {n_runs} iters (CUDA Events)...")
    latencies = []
    s_evt = torch.cuda.Event(enable_timing=True)
    e_evt = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(n_runs):
            s_evt.record()
            with autocast(device_type='cuda', enabled=use_amp):
                _ = model(inp)
            e_evt.record()
            torch.cuda.synchronize()          # chờ GPU xong trước khi đọc time
            latencies.append(s_evt.elapsed_time(e_evt))   # ms

    lat = np.array(latencies)
    # Trim: bỏ 10% đầu + 5% cao nhất
    lo  = int(n_runs * 0.10)
    hi  = int(n_runs * 0.95)
    lat_trim = np.sort(lat)[lo:hi]

    return {
        'fps'    : 1000.0 / lat_trim.mean(),
        'ms_mean': float(lat_trim.mean()),
        'ms_std' : float(lat_trim.std()),
        'ms_min' : float(lat.min()),
        'ms_p50' : float(np.percentile(lat, 50)),
        'ms_p95' : float(np.percentile(lat, 95)),
        'ms_p99' : float(np.percentile(lat, 99)),
        'mem_mb' : torch.cuda.max_memory_allocated(device) / 1024**2,
        'params_m': sum(p.numel() for p in model.parameters()) / 1e6,
        'n_runs' : n_runs,
        'n_trim' : len(lat_trim),
    }


def print_result(r: dict, variant: str, img_h: int, img_w: int,
                 gpu_name: str, use_amp: bool):
    print(f"\n{'='*60}")
    print(f"  INFERENCE SPEED  —  {variant}  (deploy+fuse)")
    print(f"{'='*60}")
    print(f"  GPU:           {gpu_name}")
    print(f"  Input:         1 × 3 × {img_h} × {img_w}")
    print(f"  AMP:           {use_amp}")
    print(f"  Runs:          {r['n_runs']}  (trimmed: {r['n_trim']})")
    print(f"  {'─'*45}")
    print(f"  FPS:           {r['fps']:>8.1f}")
    print(f"  Latency mean:  {r['ms_mean']:>8.2f} ms")
    print(f"  Latency std:   {r['ms_std']:>8.2f} ms")
    print(f"  Latency min:   {r['ms_min']:>8.2f} ms")
    print(f"  Latency P50:   {r['ms_p50']:>8.2f} ms")
    print(f"  Latency P95:   {r['ms_p95']:>8.2f} ms")
    print(f"  Latency P99:   {r['ms_p99']:>8.2f} ms")
    print(f"  {'─'*45}")
    print(f"  GPU mem peak:  {r['mem_mb']:>8.1f} MB")
    print(f"  Params:        {r['params_m']:>8.2f} M")
    print(f"{'='*60}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 — Speed Benchmark")
    parser.add_argument('--ckpt',          required=True)
    parser.add_argument('--model_variant', default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',    type=int,  default=512)
    parser.add_argument('--img_w',    type=int,  default=1024)
    parser.add_argument('--n_warmup', type=int,  default=50)
    parser.add_argument('--n_runs',   type=int,  default=300)
    parser.add_argument('--no_amp',   action='store_true',
                        help='Tắt AMP (default: bật, dùng FP16 compute)')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    use_amp  = not args.no_amp and str(device) == 'cuda'

    print(f"\n{'='*60}")
    print(f"  GCNet v3 — Speed Benchmark")
    print(f"{'='*60}")
    print(f"  GPU:      {gpu_name}")
    print(f"  Input:    {args.img_h}×{args.img_w}  batch=1")
    print(f"  AMP:      {use_amp}")
    print(f"  Warmup:   {args.n_warmup}  |  Runs: {args.n_runs}")
    print(f"{'='*60}\n")

    torch.cuda.reset_peak_memory_stats(device)

    model  = build_model(args.model_variant, args.ckpt, device)
    result = benchmark(model, args.img_h, args.img_w, device,
                       use_amp=use_amp,
                       n_warmup=args.n_warmup,
                       n_runs=args.n_runs)
    print_result(result, args.model_variant, args.img_h, args.img_w,
                 gpu_name, use_amp)


if __name__ == '__main__':
    main()
