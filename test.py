"""
speed_benchmark.py — Đo FPS inference chuẩn

Methodology:
  - batch=1, pure forward pass
  - Warmup 50 iters (cuDNN autotuning + GPU clock stabilization)
  - Auto-calibrate: chạy đủ để elapsed >= 2s → tính n_iters cho ~6s đo chính
  - Đo N iters LIÊN TIẾP không sync giữa chừng → GPU pipeline đúng thực tế
  - sync MỘT LẦN trước và sau toàn bộ loop → time.time() chính xác
  - Lặp lại 3 lần, lấy median

Tại sao KHÔNG sync sau mỗi iter:
  - Production inference: GPU execute pipelined, không bao giờ sync từng frame
  - sync sau mỗi iter thêm 1-2ms overhead/iter → underestimate FPS ~15-25%
  - File gốc (135.8 FPS) dùng methodology này → đúng

Chạy:
    python speed_benchmark.py \
        --ckpt ./checkpoints/best.pth \
        --model_variant fan_dwsa \
        --img_h 512 --img_w 1024
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead


# ============================================================
# FUSE CONV + BN
# ============================================================

def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w  = conv.weight.data
    b  = conv.bias.data if conv.bias is not None else \
         torch.zeros(conv.out_channels, device=w.device)
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
    conv.bias        = nn.Parameter(bn.bias.data + (b - bn.running_mean) * scale)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """
    Fuse mọi cặp Conv→BN liền kề trong cùng một container.

    FIX: bản cũ track prev_mod across named_children() rồi recurse vào child —
    nếu child cuối của iteration i là Conv và child đầu của iteration i+1 là BN
    (thuộc subtree khác) thì bị fuse nhầm → shape mismatch.

    Bản mới: chỉ fuse khi cả hai là DIRECT children của cùng module
    (Sequential, ModuleList, etc.) — recurse vào subtree TRƯỚC khi check fuse.
    """
    # Recurse vào tất cả children trước
    for child in module.children():
        fuse_conv_bn(child)

    # Sau khi đã recurse xong, fuse các cặp Conv→BN là direct children
    children = list(module.named_children())
    i = 0
    while i < len(children) - 1:
        name_a, mod_a = children[i]
        name_b, mod_b = children[i + 1]
        if isinstance(mod_a, nn.Conv2d) and \
                isinstance(mod_b, (nn.BatchNorm2d, nn.SyncBatchNorm)) and \
                mod_a.out_channels == mod_b.num_features:
            module._modules[name_a] = _fuse_conv_bn(mod_a, mod_b)
            module._modules[name_b] = nn.Identity()
            i += 2   # skip cả hai
        else:
            i += 1
    return module


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


def build_model(variant: str, ckpt_path: str, device: torch.device) -> Segmentor:
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
    state = (ck.get('model') or ck.get('model_state_dict') or
             ck.get('state_dict') or ck)
    model.load_state_dict(state, strict=False)
    print(f"  Loaded {variant} | mIoU recorded: {ck.get('best_miou', '?')}")

    # Step 1: GCBlock reparameterization
    model.backbone.switch_to_deploy()
    # Step 2: Conv-BN fusion
    fuse_conv_bn(model)

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_conv   = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"  deploy + fuse_conv_bn | params: {n_params:.2f}M | Conv2d: {n_conv}")
    return model


# ============================================================
# BENCHMARK — đo nhiều iters liên tiếp, sync một lần
# ============================================================

def _run_iters(model, inp, n: int):
    """Chạy n iters liên tiếp, không sync giữa chừng."""
    for _ in range(n):
        model(inp)


def measure_one_run(model, inp, device, n_iters: int) -> float:
    """
    Đo thời gian của n_iters liên tiếp.
    sync TRƯỚC và SAU toàn bộ loop — không sync giữa chừng.
    Trả về FPS.
    """
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        _run_iters(model, inp, n_iters)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return n_iters / elapsed, elapsed / n_iters * 1000   # FPS, ms/iter


def benchmark(
    model,
    img_h: int,
    img_w: int,
    device: torch.device,
    n_warmup: int = 50,
    n_repeat: int = 3,       # lặp lại bao nhiêu lần, lấy median
    target_sec: float = 6.0, # đo trong ~6 giây mỗi run
) -> dict:
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

    inp = torch.randn(1, 3, img_h, img_w, device=device)

    # Warmup — cuDNN autotuning + GPU clock stabilization
    print(f"  Warmup {n_warmup} iters...")
    with torch.no_grad():
        for _ in range(n_warmup):
            model(inp)
    torch.cuda.synchronize(device)

    # Auto-calibrate: tìm n_iters để elapsed >= target_sec
    print(f"  Auto-calibrating (target {target_sec}s per run)...")
    n_iters = 100
    with torch.no_grad():
        while True:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run_iters(model, inp, n_iters)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                break
            n_iters *= 2
    # Scale lên để đạt target_sec
    n_iters = max(int(n_iters / elapsed * target_sec), 100)
    print(f"  n_iters per run: {n_iters}")

    # Measure n_repeat lần
    fps_list = []
    lat_list = []
    for r in range(n_repeat):
        fps, lat = measure_one_run(model, inp, device, n_iters)
        fps_list.append(fps)
        lat_list.append(lat)
        print(f"    run {r+1}/{n_repeat}: {fps:.1f} FPS  {lat:.2f} ms/iter")

    fps_arr = np.array(fps_list)
    lat_arr = np.array(lat_list)

    # Reset + đo peak memory với 1 forward
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(inp)
    torch.cuda.synchronize(device)
    mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    return {
        'fps_median': float(np.median(fps_arr)),
        'fps_mean'  : float(fps_arr.mean()),
        'fps_std'   : float(fps_arr.std()),
        'ms_median' : float(np.median(lat_arr)),
        'ms_mean'   : float(lat_arr.mean()),
        'ms_std'    : float(lat_arr.std()),
        'mem_mb'    : mem_mb,
        'params_m'  : sum(p.numel() for p in model.parameters()) / 1e6,
        'n_iters'   : n_iters,
        'n_repeat'  : n_repeat,
    }


def print_result(r: dict, variant: str, img_h: int, img_w: int, gpu_name: str):
    print(f"\n{'='*55}")
    print(f"  INFERENCE SPEED  —  {variant}  (deploy+fuse)")
    print(f"{'='*55}")
    print(f"  GPU:           {gpu_name}")
    print(f"  Input:         1 × 3 × {img_h} × {img_w}")
    print(f"  Iters/run:     {r['n_iters']}  ×  {r['n_repeat']} runs")
    print(f"  {'─'*43}")
    print(f"  FPS (median):  {r['fps_median']:>8.1f}")
    print(f"  FPS (mean):    {r['fps_mean']:>8.1f}  ±  {r['fps_std']:.1f}")
    print(f"  {'─'*43}")
    print(f"  ms  (median):  {r['ms_median']:>8.2f} ms")
    print(f"  ms  (mean):    {r['ms_mean']:>8.2f}  ±  {r['ms_std']:.2f} ms")
    print(f"  {'─'*43}")
    print(f"  GPU mem peak:  {r['mem_mb']:>8.1f} MB  (1 forward pass)")
    print(f"  Params:        {r['params_m']:>8.2f} M")
    print(f"{'='*55}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 — Speed Benchmark")
    parser.add_argument('--ckpt',          required=True)
    parser.add_argument('--model_variant', default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',       type=int,   default=512)
    parser.add_argument('--img_w',       type=int,   default=1024)
    parser.add_argument('--n_warmup',    type=int,   default=50)
    parser.add_argument('--n_repeat',    type=int,   default=3,
                        help='Số lần đo, lấy median (default 3)')
    parser.add_argument('--target_sec',  type=float, default=6.0,
                        help='Thời gian đo mỗi run tính bằng giây (default 6)')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    print(f"\n{'='*55}")
    print(f"  GCNet v3 — Speed Benchmark")
    print(f"{'='*55}")
    print(f"  GPU:        {gpu_name}")
    print(f"  Input:      {args.img_h}×{args.img_w}  batch=1")
    print(f"  Warmup:     {args.n_warmup}  |  Repeat: {args.n_repeat}")
    print(f"  Target:     {args.target_sec}s per run")
    print(f"{'='*55}\n")

    model  = build_model(args.model_variant, args.ckpt, device)
    result = benchmark(
        model, args.img_h, args.img_w, device,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
        target_sec=args.target_sec,
    )
    print_result(result, args.model_variant, args.img_h, args.img_w, gpu_name)


if __name__ == '__main__':
    main()
