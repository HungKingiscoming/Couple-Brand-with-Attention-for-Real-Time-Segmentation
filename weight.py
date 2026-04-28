"""
make_foggy_weights.py
─────────────────────
Tạo class weights từ Foggy Cityscapes training set thực tế.

Có 2 chế độ:
  1. SCAN MODE  (--scan):   scan toàn bộ label files → weights chính xác nhất
  2. PRESET MODE (default): dùng weights được tính sẵn từ Foggy Cityscapes
                            (nhanh, không cần data path)

Chạy trên Kaggle:
    # Preset (nhanh):
    python make_foggy_weights.py --output /kaggle/working/class_weights_foggy.pt

    # Scan từ data thực (chính xác hơn):
    python make_foggy_weights.py --scan --train_txt /kaggle/working/train.txt \
                                 --output /kaggle/working/class_weights_foggy.pt
"""
import argparse
import torch
import numpy as np
from pathlib import Path

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# ============================================================
# PRESET: Foggy-aware weights
# ============================================================
# Dựa trên phân tích từ training log:
#   - Stuck classes (IoU < 0.50): pole(5), fence(4), rider(12), motorcycle(17)
#   - Clear classes (IoU > 0.85): road(0), building(2), vegetation(8), sky(10), car(13)
#
# Strategy:
#   1. Boost stuck small-object classes thêm ~15-30% so với official weights
#   2. Giữ nguyên hoặc giảm nhẹ easy classes để tổng budget không thay đổi
#   3. Ratio max/min ~ 1.7x (từ 1.38x) — đủ để tạo bias mà không destabilize
#
# Index:  road  side  bldg  wall  fence pole  tl    ts    veg   terr
#         0     1     2     3     4     5     6     7     8     9
# Index:  sky   pers  rider car   truck bus   train moto  bike
#         10    11    12    13    14    15    16    17    18
FOGGY_WEIGHTS_PRESET = torch.tensor([
    0.837,  # 0  road        — easy, giữ nguyên
    0.918,  # 1  sidewalk    — giữ nguyên
    0.866,  # 2  building    — easy, giữ nguyên
    1.050,  # 3  wall        — tăng nhẹ từ 1.035 (IoU 0.547, trung bình)
    1.150,  # 4  fence       — tăng từ 1.017 (IoU 0.488, stuck)
    1.250,  # 5  pole        — tăng mạnh từ 0.997 (IoU 0.476, stuck + thin object)
    0.975,  # 6  traffic_light — giữ nguyên
    1.049,  # 7  traffic_sign  — giữ nguyên
    0.879,  # 8  vegetation  — easy, giữ nguyên
    1.002,  # 9  terrain     — giữ nguyên
    0.954,  # 10 sky         — easy, giữ nguyên
    1.100,  # 11 person      — tăng nhẹ từ 0.984 (IoU 0.689, cần cải thiện)
    1.300,  # 12 rider       — tăng mạnh từ 1.112 (IoU 0.490, stuck + fog-occluded)
    0.904,  # 13 car         — easy, giữ nguyên
    1.087,  # 14 truck       — giữ nguyên (IoU dao động 0.57-0.61)
    1.096,  # 15 bus         — giữ nguyên (IoU ổn 0.74-0.76)
    1.087,  # 16 train       — giữ nguyên (IoU ổn 0.69-0.70)
    1.400,  # 17 motorcycle  — boost mạnh nhất từ 1.153 (IoU 0.452, stuck nhất)
    1.051,  # 18 bicycle     — giữ nguyên (IoU ổn 0.636)
], dtype=torch.float32)


def normalize_weights(weights: torch.Tensor, n_classes: int = 19) -> torch.Tensor:
    """Normalize về mean=1.0 (tổng = n_classes)."""
    return weights / weights.mean()


def compute_weights_from_scan(train_txt: str, method: str = 'median_freq') -> torch.Tensor:
    """
    Scan toàn bộ label files trong train_txt để tính class weights.

    Args:
        train_txt: Path đến file txt chứa 'img_path,label_path' mỗi dòng.
        method: 'median_freq' hoặc 'inverse_freq'

    Returns:
        Tensor weights shape (19,), normalized về mean=1.0.
    """
    from PIL import Image
    from tqdm import tqdm

    ID_TO_TRAINID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18,
    }
    label_map = np.ones(256, dtype=np.uint8) * 255
    for orig_id, train_id in ID_TO_TRAINID.items():
        label_map[orig_id] = train_id

    samples = []
    with open(train_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                _, label_path = line.split(',')
                samples.append(label_path)

    print(f"Scanning {len(samples)} label files...")
    class_counts = np.zeros(19, dtype=np.int64)
    total_pixels = 0

    for label_path in tqdm(samples, desc="Counting pixels"):
        label = label_map[np.array(Image.open(label_path), dtype=np.uint8)]
        for c in range(19):
            class_counts[c] += (label == c).sum()
        total_pixels += label.size

    freq = class_counts / max(total_pixels, 1)
    print(f"\n{'Class':<16} {'Pixels':>15} {'Freq%':>8}")
    print("-" * 42)
    for c, (cls, cnt, fr) in enumerate(zip(CLASSES, class_counts, freq)):
        print(f"  {cls:<14} {cnt:>15,} {fr*100:>7.3f}%")

    if method == 'median_freq':
        # Median frequency balancing: w_c = median(freq) / freq_c
        median_f = np.median(freq[freq > 0])
        raw_w = median_f / (freq + 1e-10)
    else:
        # Inverse frequency
        raw_w = 1.0 / (freq + 1e-10)

    weights = torch.tensor(raw_w, dtype=torch.float32)
    weights = torch.clamp(weights, min=0.1, max=50.0)
    weights = normalize_weights(weights)
    return weights


def print_weight_table(weights: torch.Tensor, title: str = "Class Weights"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Class':<16} {'Weight':>8}  {'Bar'}")
    print(f"  {'-'*50}")
    for cls, w in zip(CLASSES, weights):
        bar   = '█' * int(w.item() * 12)
        mark  = ' ← boosted' if w.item() > 1.15 else ''
        print(f"  {cls:<16} {w.item():>8.4f}  {bar}{mark}")
    print(f"\n  min={weights.min():.4f}  max={weights.max():.4f}")
    print(f"  mean={weights.mean():.4f}  ratio={weights.max()/weights.min():.2f}x")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan',      action='store_true',
                        help='Scan actual label files (accurate but slow)')
    parser.add_argument('--train_txt', type=str, default=None,
                        help='Path to train.txt (required if --scan)')
    parser.add_argument('--method',    type=str, default='median_freq',
                        choices=['median_freq', 'inverse_freq'],
                        help='Weight computation method when --scan is used')
    parser.add_argument('--output',    type=str,
                        default='/kaggle/working/class_weights_foggy.pt')
    args = parser.parse_args()

    # --- Compare with official weights ---
    official = torch.tensor([
        0.8373, 0.918,  0.866,  1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
        0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
        1.0865, 1.1529, 1.0507,
    ], dtype=torch.float32)
    print_weight_table(official, "Official GCNet-S weights (Clear Cityscapes)")

    if args.scan:
        if args.train_txt is None:
            raise ValueError("--train_txt is required when using --scan")
        weights = compute_weights_from_scan(args.train_txt, method=args.method)
        title = f"Computed from Foggy scan ({args.method})"
    else:
        weights = normalize_weights(FOGGY_WEIGHTS_PRESET)
        title = "Foggy-aware preset (boosted stuck classes)"

    print_weight_table(weights, title)

    # Delta vs official
    print("  Delta vs official weights:")
    print(f"  {'Class':<16} {'Official':>9} {'New':>9} {'Delta':>9}")
    print(f"  {'-'*46}")
    for cls, ow, nw in zip(CLASSES, official, weights):
        delta = nw.item() - ow.item()
        mark  = ' ↑' if delta > 0.05 else (' ↓' if delta < -0.05 else '')
        print(f"  {cls:<16} {ow.item():>9.4f} {nw.item():>9.4f} {delta:>+9.4f}{mark}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out_path)
    print(f"\n✅ Saved → {out_path}")
    print(f"   Usage: --class_weights_file {out_path}\n")


if __name__ == '__main__':
    main()
