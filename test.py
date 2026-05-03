
import argparse
import os
import sys
import time
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
NUM_CLASSES  = 19
IGNORE_INDEX = 255


# ============================================================
# FUSE CONV + BN
# ============================================================

def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w     = conv.weight.data
    b     = conv.bias.data if conv.bias is not None else \
            torch.zeros(conv.out_channels, device=w.device)
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
    conv.bias        = nn.Parameter(bn.bias.data + (b - bn.running_mean) * scale)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    # Recurse vào tất cả children trước
    for child in module.children():
        fuse_conv_bn(child)
    # Fuse các cặp Conv→BN là direct children
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
            i += 2
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


def build_model(variant: str, ckpt_path: str, device: torch.device,
                deploy: bool = True) -> Segmentor:
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
            in_channels=C*4, channels=64, num_classes=NUM_CLASSES,
            align_corners=False, dropout_ratio=0.0, ignore_index=IGNORE_INDEX,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
        )
    )
    ck    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = (ck.get('model') or ck.get('model_state_dict') or
             ck.get('state_dict') or ck)
    model.load_state_dict(state, strict=False)
    recorded = ck.get('best_miou', '?')
    print(f"  Loaded {variant} | recorded mIoU: {recorded}")

    if deploy:
        model.backbone.switch_to_deploy()
        print(f"  switch_to_deploy applied (reparam branches fused)")

    model = model.to(device).eval()

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_conv   = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"  params: {n_params:.2f}M | Conv2d layers: {n_conv}")
    return model, recorded


# ============================================================
# VALIDATE
# ============================================================

def _update_accum(pred, target, ti, tu, tp, tl):
    mask = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
    p    = pred[mask].astype(np.int64)
    t    = target[mask].astype(np.int64)
    intersect = p[p == t]
    ai = np.bincount(intersect, minlength=NUM_CLASSES)
    ap = np.bincount(p,         minlength=NUM_CLASSES)
    al = np.bincount(t,         minlength=NUM_CLASSES)
    ti += ai;  tu += ap + al - ai;  tp += ap;  tl += al


def _compute_metrics(ti, tu, tp, tl):
    present = tl > 0
    iou     = ti / (tu + 1e-10)
    acc     = ti / (tl + 1e-10)
    dice    = 2 * ti / (tp + tl + 1e-10)
    aacc    = float(ti[present].sum() / (tl[present].sum() + 1e-10))
    return {
        'aacc'          : aacc,
        'miou'          : float(np.nanmean(iou[present])),
        'macc'          : float(np.nanmean(acc[present])),
        'mdice'         : float(np.nanmean(dice[present])),
        'per_class_iou' : iou,
        'per_class_acc' : acc,
        'per_class_dice': dice,
        'present'       : present,
    }


@torch.no_grad()
def validate(model, val_txt, img_h, img_w, batch_size,
             num_workers, device, use_amp, recorded):
    from data.custom import CityscapesDataset, get_val_transforms

    val_ds = CityscapesDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=(img_h, img_w)),
        img_size=(img_h, img_w),
        label_mapping='train_id',
        dataset_type='foggy',
    )
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    print(f"\n  Val: {len(val_ds):,} samples | {len(loader)} batches\n")

    ti = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_intersect
    tu = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_union
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_pred
    tl = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_label
    total_loss = 0.0
    ce_fn      = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    model.eval()
    pbar = tqdm(loader, desc="Validating", ncols=90)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
            loss   = ce_fn(logits, masks)
        total_loss += loss.item()
        _update_accum(logits.argmax(1).cpu().numpy(),
                      masks.cpu().numpy(), ti, tu, tp, tl)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    m = _compute_metrics(ti, tu, tp, tl)
    m['loss'] = total_loss / len(loader)

    # Print
    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULTS  (GCNet-official methodology)")
    print(f"{'='*65}")
    print(f"  aAcc:   {m['aacc']:.4f}   (overall pixel accuracy)")
    print(f"  mIoU:   {m['miou']:.4f}")
    print(f"  mAcc:   {m['macc']:.4f}   (mean per-class recall)")
    print(f"  mDice:  {m['mdice']:.4f}")
    print(f"  Loss:   {m['loss']:.4f}   (deploy: không so với training log)")
    print(f"{'='*65}")
    print(f"\n  {'Class':<16} {'IoU':>6}  {'Acc':>6}  {'Dice':>6}  Bar")
    print(f"  {'─'*60}")
    for name, i, a, d, p in zip(CLASS_NAMES,
                                 m['per_class_iou'],
                                 m['per_class_acc'],
                                 m['per_class_dice'],
                                 m['present']):
        bar  = '█' * int(i * 20)
        mark = ' ⚠️ ' if i < 0.40 else (' ★' if i > 0.75 else '')
        note = '' if p else ' (no GT)'
        print(f"  {name:<16} {i:>6.4f}  {a:>6.4f}  {d:>6.4f}  {bar}{mark}{note}")

    low = [n for n, v, p in zip(CLASS_NAMES, m['per_class_iou'], m['present'])
           if v < 0.40 and p]
    if low:
        print(f"\n  ⚠️  LOW IoU (<0.40): {low}")
    print(f"{'='*65}")

    if recorded and recorded != '?':
        diff = abs(m['miou'] - float(recorded))
        ok   = '✅' if diff < 0.001 else '⚠️ '
        print(f"\n  {ok} mIoU: {m['miou']:.4f} vs recorded {float(recorded):.4f}"
              f"  (diff={diff:.4f})")
    print()
    return m


# ============================================================
# BENCHMARK
# ============================================================

def _run_iters(model, inp, n):
    for _ in range(n):
        model(inp)


def benchmark(model, img_h, img_w, device,
              n_warmup=50, n_repeat=3, target_sec=6.0):
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

    inp = torch.randn(1, 3, img_h, img_w, device=device)

    print(f"  Warmup {n_warmup} iters...")
    with torch.no_grad():
        for _ in range(n_warmup):
            model(inp)
    torch.cuda.synchronize(device)

    print(f"  Auto-calibrating (target {target_sec}s/run)...")
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
    n_iters = max(int(n_iters / elapsed * target_sec), 100)
    print(f"  n_iters/run: {n_iters}")

    fps_list, lat_list = [], []
    for r in range(n_repeat):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _run_iters(model, inp, n_iters)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        fps = n_iters / elapsed
        lat = elapsed / n_iters * 1000
        fps_list.append(fps);  lat_list.append(lat)
        print(f"    run {r+1}/{n_repeat}: {fps:.1f} FPS  {lat:.2f} ms")

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(inp)
    torch.cuda.synchronize(device)

    fps_arr = np.array(fps_list)
    lat_arr = np.array(lat_list)
    return {
        'fps_median': float(np.median(fps_arr)),
        'fps_mean'  : float(fps_arr.mean()),
        'fps_std'   : float(fps_arr.std()),
        'ms_median' : float(np.median(lat_arr)),
        'ms_mean'   : float(lat_arr.mean()),
        'ms_std'    : float(lat_arr.std()),
        'mem_mb'    : torch.cuda.max_memory_allocated(device) / 1024**2,
        'params_m'  : sum(p.numel() for p in model.parameters()) / 1e6,
        'n_iters'   : n_iters,
        'n_repeat'  : n_repeat,
    }


def print_benchmark(r, variant, img_h, img_w, gpu_name):
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
    parser = argparse.ArgumentParser(description="GCNet v3 — Benchmark + Validate")
    parser.add_argument('--ckpt',          required=True)
    parser.add_argument('--model_variant', default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',       type=int,   default=512)
    parser.add_argument('--img_w',       type=int,   default=1024)
    # Benchmark
    parser.add_argument('--benchmark',   action='store_true',
                        help='Chạy speed benchmark (default: True nếu không có --validate)')
    parser.add_argument('--n_warmup',    type=int,   default=50)
    parser.add_argument('--n_repeat',    type=int,   default=3)
    parser.add_argument('--target_sec',  type=float, default=6.0)
    # Validate
    parser.add_argument('--validate',    action='store_true',
                        help='Chạy validation')
    parser.add_argument('--val_txt',     type=str,   default=None)
    parser.add_argument('--batch_size',  type=int,   default=22)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--no_amp',      action='store_true')
    args = parser.parse_args()

    # Nếu không flag nào được set, default chạy benchmark
    if not args.validate and not args.benchmark:
        args.benchmark = True

    if args.validate and not args.val_txt:
        parser.error("--validate yêu cầu --val_txt")

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    use_amp  = not args.no_amp and device.type == 'cuda'

    print(f"\n{'='*55}")
    print(f"  GCNet v3  —  {'Benchmark' if args.benchmark else ''}"
          f"{'+ ' if args.benchmark and args.validate else ''}"
          f"{'Validate' if args.validate else ''}")
    print(f"{'='*55}")
    print(f"  GPU:    {gpu_name}")
    print(f"  Input:  {args.img_h}×{args.img_w}  |  AMP: {use_amp}")
    print(f"{'='*55}\n")

    model, recorded = build_model(
        args.model_variant, args.ckpt, device, deploy=True)

    if args.validate:
        validate(model, args.val_txt, args.img_h, args.img_w,
                 args.batch_size, args.num_workers, device, use_amp, recorded)

    if args.benchmark:
        print(f"\nRunning benchmark...")
        result = benchmark(model, args.img_h, args.img_w, device,
                           n_warmup=args.n_warmup,
                           n_repeat=args.n_repeat,
                           target_sec=args.target_sec)
        print_benchmark(result, args.model_variant, args.img_h, args.img_w, gpu_name)


if __name__ == '__main__':
    main()
