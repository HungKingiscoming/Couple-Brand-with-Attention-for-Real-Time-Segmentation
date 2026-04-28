"""
validate.py
───────────
Standalone validation — mIoU / mAcc / mDice + speed benchmarks.

Chạy:
    python validate.py \
        --checkpoint ./checkpoints/best.pth \
        --val_txt    /kaggle/working/val.txt \
        --model_variant fan_dwsa

    # + deploy mode + speed:
    python validate.py \
        --checkpoint ./checkpoints/best.pth \
        --val_txt    /kaggle/working/val.txt \
        --deploy --benchmark
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

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
# METRICS từ confusion matrix
# ============================================================

def compute_metrics(conf_matrix: np.ndarray) -> dict:
    """
    Tính mIoU, mAcc, mDice từ confusion matrix (NUM_CLASSES × NUM_CLASSES).

    confusion matrix C[i,j] = số pixel class i được predict là j.
    → C[i,i] = true positives của class i.

    mIoU  = mean( TP_i / (TP_i + FP_i + FN_i) )
    mAcc  = mean( TP_i / (TP_i + FN_i) )          ← per-class recall, mean
    mDice = mean( 2*TP_i / (2*TP_i + FP_i + FN_i) )
    pAcc  = sum(TP_i) / sum(all pixels)            ← pixel accuracy (overall)
    """
    tp = np.diag(conf_matrix).astype(np.float64)           # (C,)
    fn = conf_matrix.sum(axis=1).astype(np.float64) - tp   # row sum - TP
    fp = conf_matrix.sum(axis=0).astype(np.float64) - tp   # col sum - TP

    # mask: chỉ tính class nào có ground-truth pixel
    present = conf_matrix.sum(axis=1) > 0                  # (C,) bool

    iou  = tp / (tp + fp + fn + 1e-10)
    acc  = tp / (tp + fn + 1e-10)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

    miou  = float(np.nanmean(iou[present]))
    macc  = float(np.nanmean(acc[present]))
    mdice = float(np.nanmean(dice[present]))
    pacc  = float(tp.sum() / (conf_matrix.sum() + 1e-10))

    return {
        'miou'         : miou,
        'macc'         : macc,
        'mdice'        : mdice,
        'pacc'         : pacc,
        'per_class_iou': iou,
        'per_class_acc': acc,
        'per_class_dice': dice,
        'present'      : present,
    }


# ============================================================
# SPEED BENCHMARK
# ============================================================

def benchmark_speed(model, device, img_h, img_w,
                    batch_size=1, n_warmup=10, n_runs=50,
                    use_amp=True) -> dict:
    """
    Đo FPS và latency (ms/image) bằng CUDA events.

    Dùng batch_size=1 cho latency thực tế (inference đơn ảnh).
    Dùng batch_size lớn hơn cho throughput (batch inference).

    CUDA events chính xác hơn time.time() vì đồng bộ GPU.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, img_h, img_w, device=device)

    # Warmup — flush cache, JIT compile nếu có
    print(f"  Warming up ({n_warmup} runs, batch={batch_size})...")
    with torch.no_grad():
        for _ in range(n_warmup):
            with autocast(device_type='cuda', enabled=(use_amp and device == 'cuda')):
                _ = model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({n_runs} runs)...")
    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        latencies   = []
        with torch.no_grad():
            for _ in range(n_runs):
                start_event.record()
                with autocast(device_type='cuda', enabled=use_amp):
                    _ = model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))  # ms per batch

        latencies    = np.array(latencies)
        ms_per_batch = float(np.mean(latencies))
        ms_std       = float(np.std(latencies))
        ms_per_img   = ms_per_batch / batch_size
        fps          = 1000.0 / ms_per_img

        # Memory
        mem_alloc    = torch.cuda.memory_allocated(device) / 1024**2    # MB
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2     # MB
    else:
        # CPU fallback với time.time()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy)
        elapsed      = (time.perf_counter() - t0) * 1000   # ms total
        ms_per_batch = elapsed / n_runs
        ms_std       = 0.0
        ms_per_img   = ms_per_batch / batch_size
        fps          = 1000.0 / ms_per_img
        mem_alloc    = 0.0
        mem_reserved = 0.0

    # Params & FLOPs (params chính xác, FLOPs ước tính)
    n_params   = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad) / 1e6

    return {
        'fps'         : fps,
        'ms_per_img'  : ms_per_img,
        'ms_per_batch': ms_per_batch,
        'ms_std'      : ms_std,
        'mem_alloc_mb': mem_alloc,
        'mem_reserved_mb': mem_reserved,
        'params_m'    : n_params,
        'trainable_m' : n_trainable,
        'batch_size'  : batch_size,
        'n_runs'      : n_runs,
    }


def print_speed(speed: dict, deploy: bool):
    print(f"\n{'='*70}")
    print(f"  SPEED BENCHMARK  {'[DEPLOY MODE]' if deploy else '[TRAIN MODE]'}")
    print(f"{'='*70}")
    print(f"  Batch size used:     {speed['batch_size']}")
    print(f"  Runs:                {speed['n_runs']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Latency (ms/image):  {speed['ms_per_img']:>8.2f} ms")
    print(f"  Latency (ms/batch):  {speed['ms_per_batch']:>8.2f} ± {speed['ms_std']:.2f} ms")
    print(f"  FPS:                 {speed['fps']:>8.1f} fps")
    print(f"  ─────────────────────────────────────────")
    print(f"  GPU mem (allocated): {speed['mem_alloc_mb']:>8.1f} MB")
    print(f"  GPU mem (reserved):  {speed['mem_reserved_mb']:>8.1f} MB")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total params:        {speed['params_m']:>8.2f} M")
    print(f"  Trainable params:    {speed['trainable_m']:>8.2f} M")
    print(f"{'='*70}\n")


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()
def validate(model, loader, device, use_amp=True):
    model.eval()

    total_loss  = 0.0
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    ce_loss_fn  = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Wall-clock time (bao gồm dataload + forward + postprocess)
    t_start = time.perf_counter()
    n_imgs  = 0

    pbar = tqdm(loader, desc="Validating", ncols=90)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)

        n_imgs += imgs.shape[0]

        with autocast(device_type='cuda', enabled=use_amp):
            logits      = model(imgs)
            logits_full = F.interpolate(
                logits, size=masks.shape[-2:],
                mode='bilinear', align_corners=False)
            loss = ce_loss_fn(logits_full, masks)

        total_loss += loss.item()

        pred   = logits_full.argmax(1).cpu().numpy()
        target = masks.cpu().numpy()

        # confusion matrix — exact same as train.py
        valid = (target >= 0) & (target < NUM_CLASSES)
        label = NUM_CLASSES * target[valid].astype(int) + pred[valid]
        count = np.bincount(label, minlength=NUM_CLASSES ** 2)
        conf_matrix += count.reshape(NUM_CLASSES, NUM_CLASSES)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    if device == 'cuda':
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t_start

    metrics              = compute_metrics(conf_matrix)
    metrics['loss']      = total_loss / len(loader)
    metrics['wall_fps']  = n_imgs / t_total   # FPS bao gồm dataload
    metrics['wall_ms']   = t_total * 1000 / n_imgs
    metrics['n_imgs']    = n_imgs
    metrics['conf_matrix'] = conf_matrix
    return metrics


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(metrics):
    iou_arr  = metrics['per_class_iou']
    acc_arr  = metrics['per_class_acc']
    dice_arr = metrics['per_class_dice']
    present  = metrics['present']

    print(f"\n{'='*70}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  mIoU:      {metrics['miou']:.4f}")
    print(f"  mAcc:      {metrics['macc']:.4f}   (mean per-class recall)")
    print(f"  mDice:     {metrics['mdice']:.4f}")
    print(f"  pAcc:      {metrics['pacc']:.4f}   (overall pixel accuracy)")
    loss_note = ("  ⚠️  deploy mode — không so sánh với training log"
                 if metrics.get('deploy') else "")
    print(f"  Val Loss:  {metrics['loss']:.4f}{loss_note}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Wall FPS:  {metrics['wall_fps']:.1f} fps  "
          f"({metrics['wall_ms']:.1f} ms/img)")
    print(f"             (bao gồm dataload + forward + postprocess + CPU sync)")
    print(f"             → dùng --benchmark để đo pure GPU latency")
    print(f"  Images:    {metrics['n_imgs']:,}")
    print(f"{'='*70}")

    # Per-class table
    print(f"\n  {'Class':<16} {'IoU':>6}  {'Acc':>6}  {'Dice':>6}  Bar (IoU)")
    print(f"  {'─'*60}")
    for cname, iou, acc, dice, pres in zip(
            CLASS_NAMES, iou_arr, acc_arr, dice_arr, present):
        bar  = '█' * int(iou * 18)
        mark = ' ⚠️ ' if iou < 0.40 else (' ★' if iou > 0.75 else '')
        skip = '' if pres else '  (no GT)'
        print(f"  {cname:<16} {iou:>6.4f}  {acc:>6.4f}  {dice:>6.4f}  {bar}{mark}{skip}")

    low_cls = [n for n, v, p in zip(CLASS_NAMES, iou_arr, present)
               if v < 0.40 and p]
    if low_cls:
        print(f"\n  ⚠️  LOW IoU (<0.40): {low_cls}")
    print(f"{'='*70}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 — Validation + Speed")
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--val_txt',       type=str, required=True)
    parser.add_argument('--model_variant', type=str, default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',         type=int, default=512)
    parser.add_argument('--img_w',         type=int, default=1024)
    parser.add_argument('--batch_size',    type=int, default=22)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--no_amp',        action='store_true')
    parser.add_argument('--deploy',        action='store_true',
                        help='Fuse reparam branches → single conv')
    parser.add_argument('--benchmark',     action='store_true',
                        help='Run speed benchmark (pure GPU, no dataload)')
    parser.add_argument('--bench_batch',   type=int, default=1,
                        help='Batch size for benchmark (default 1 = latency mode)')
    parser.add_argument('--bench_runs',    type=int, default=100,
                        help='Number of benchmark runs (default 100)')
    args = parser.parse_args()

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = not args.no_amp and device == 'cuda'

    print(f"\n{'='*70}")
    print(f"  GCNet v3 — Validation + Speed")
    print(f"{'='*70}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Val txt:      {args.val_txt}")
    print(f"  Image size:   {args.img_h}×{args.img_w}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  AMP:          {use_amp}")
    print(f"  Deploy:       {args.deploy}")
    print(f"  Benchmark:    {args.benchmark}")
    print(f"  Device:       {device}")
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
        def __init__(self, bb, hd):
            super().__init__()
            self.backbone    = bb
            self.decode_head = hd
        def forward(self, x):
            return self.decode_head(self.backbone(x))

    model = Segmentor(backbone, head).to(device)

    # ---- Load checkpoint ----
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = (ckpt.get('model') or ckpt.get('model_state_dict') or
             ckpt.get('state_dict') or ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    skip_markers = ('dwsa', 'alpha', 'in_.', 'foggy', 'loss_', 'fog_')
    unexpected_missing = [k for k in missing
                          if not any(s in k for s in skip_markers)]
    print(f"  Epoch:      {ckpt.get('epoch', '?')}")
    print(f"  Best mIoU (recorded): {ckpt.get('best_miou', 'N/A')}")
    if unexpected_missing:
        print(f"  ⚠️  Missing keys ({len(unexpected_missing)}): "
              f"{unexpected_missing[:3]}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected[:3]}")
    print()

    # ---- Deploy ----
    if args.deploy:
        model.backbone.switch_to_deploy()
        model.eval()
        n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        print(f"  ✅ Deploy: reparam fused | Conv2d count: {n_conv}")
        print()

    # ---- Speed benchmark (pure GPU, no dataload overhead) ----
    if args.benchmark:
        speed = benchmark_speed(
            model, device,
            img_h=args.img_h, img_w=args.img_w,
            batch_size=args.bench_batch,
            n_warmup=10, n_runs=args.bench_runs,
            use_amp=use_amp,
        )
        print_speed(speed, deploy=args.deploy)

    # ---- DataLoader ----
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
        drop_last=False,
    )
    print(f"  Val samples: {len(val_dataset):,} ({len(val_loader)} batches)\n")

    # ---- Validate ----
    metrics = validate(model, val_loader, device, use_amp=use_amp)
    metrics['deploy'] = args.deploy
    print_results(metrics)

    # ---- Sanity check ----
    recorded = ckpt.get('best_miou') or (ckpt.get('metrics') or {}).get('miou')
    if recorded:
        diff = abs(metrics['miou'] - recorded)
        status = '✅' if diff < 0.001 else '⚠️ '
        print(f"  {status} mIoU: got {metrics['miou']:.4f}, "
              f"recorded {recorded:.4f}  (diff={diff:.4f})")
        if diff >= 0.001:
            print(f"     → Kiểm tra --img_h/--img_w và --batch_size")
        print()


if __name__ == '__main__':
    main()
