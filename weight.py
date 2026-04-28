"""
make_foggy_weights.py
─────────────────────
Tạo class weights từ Foggy Cityscapes training set thực tế.

Có 2 chế độ — chỉnh CONFIG bên dưới rồi chạy:
    python make_foggy_weights.py

SCAN_MODE = True  → scan label files thực (~90s cho 8925 files)
SCAN_MODE = False → dùng preset đã tính sẵn (~1 giây)

FIX: hỗ trợ cả hai loại label file:
  - labelTrainIds: pixel values 0-18 (+ 255 ignore) — phổ biến hơn
  - labelIds:      pixel values gốc 7,8,11... — cần remap
Tự detect qua LABEL_FORMAT config.
"""
import os
import time
import torch
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ============================================================
# ★ CONFIG — chỉnh ở đây ★
# ============================================================

SCAN_MODE    = True
TRAIN_TXT    = '/kaggle/working/train.txt'
METHOD       = 'median_freq'    # 'median_freq' hoặc 'inverse_freq'
WORKERS      = 0                # 0 = auto (min(cpu_count, 8))
OUTPUT       = '/kaggle/working/class_weights_foggy.pt'

# 'train_id' : label files chứa pixel 0-18 + 255 (labelTrainIds) ← kaggle thường dùng
# 'label_id' : label files chứa pixel gốc 7,8,11... (labelIds)
LABEL_FORMAT = 'train_id'

# ============================================================

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# remap cho labelIds format (không dùng nếu LABEL_FORMAT='train_id')
_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_LABEL_MAP = np.full(256, 255, dtype=np.uint8)
for _k, _v in _ID_TO_TRAINID.items():
    _LABEL_MAP[_k] = _v

FOGGY_WEIGHTS_PRESET = torch.tensor([
    0.837,  # 0  road
    0.918,  # 1  sidewalk
    0.866,  # 2  building
    1.050,  # 3  wall
    1.150,  # 4  fence        — stuck IoU 0.488
    1.250,  # 5  pole         — stuck IoU 0.476
    0.975,  # 6  traffic_light
    1.049,  # 7  traffic_sign
    0.879,  # 8  vegetation
    1.002,  # 9  terrain
    0.954,  # 10 sky
    1.100,  # 11 person
    1.300,  # 12 rider        — stuck IoU 0.490
    0.904,  # 13 car
    1.087,  # 14 truck
    1.096,  # 15 bus
    1.087,  # 16 train
    1.400,  # 17 motorcycle   — stuck IoU 0.452
    1.051,  # 18 bicycle
], dtype=torch.float32)


# ============================================================
# WORKER — module-level để multiprocessing pickle được
# ============================================================

def _count_train_id(label_path: str) -> np.ndarray:
    """
    Cho labelTrainIds: pixel values đã là 0-18, 255=ignore.
    Không cần remap — đọc thẳng và bincount.
    """
    from PIL import Image
    raw   = np.array(Image.open(label_path), dtype=np.uint8).ravel()
    valid = raw[raw < 19]                          # loại 255 (ignore)
    return np.bincount(valid, minlength=19).astype(np.int64)


def _count_label_id(label_path: str) -> np.ndarray:
    """
    Cho labelIds: pixel values gốc (7,8,11...) → remap về train_id trước.
    """
    from PIL import Image
    raw    = np.array(Image.open(label_path), dtype=np.uint8)
    mapped = _LABEL_MAP[raw].ravel()
    valid  = mapped[mapped < 19]
    return np.bincount(valid, minlength=19).astype(np.int64)


# ============================================================
# CORE
# ============================================================

def scan_weights(train_txt: str, method: str, workers: int,
                 label_format: str) -> torch.Tensor:
    label_paths = []
    with open(train_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                _, lp = line.split(',')
                label_paths.append(lp)

    n = len(label_paths)
    if workers <= 0:
        workers = min(os.cpu_count() or 4, 8)

    worker_fn = _count_train_id if label_format == 'train_id' else _count_label_id
    print(f"  Files: {n:,}  |  workers: {workers}  |  method: {method}")
    print(f"  Label format: {label_format}  "
          f"({'pixel 0-18 direct' if label_format == 'train_id' else 'orig labelIds → remap'})")

    t0        = time.time()
    chunksize = max(1, n // (workers * 4))

    with Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap(worker_fn, label_paths, chunksize=chunksize),
            total=n, desc="  Scanning", unit="file", ncols=80,
        ))

    class_counts = np.sum(results, axis=0)
    total_valid  = class_counts.sum()
    elapsed      = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({n / elapsed:.0f} files/sec)\n")

    freq = class_counts / max(total_valid, 1)
    print(f"  {'Class':<16} {'Pixels':>15} {'Freq%':>8}")
    print(f"  {'-'*42}")
    for cls, cnt, fr in zip(CLASSES, class_counts, freq):
        print(f"  {cls:<16} {cnt:>15,} {fr*100:>7.3f}%")

    if method == 'median_freq':
        median_f = np.median(freq[freq > 0])
        raw_w    = median_f / (freq + 1e-10)
    else:
        raw_w = 1.0 / (freq + 1e-10)

    weights = torch.tensor(raw_w, dtype=torch.float32)
    weights = torch.clamp(weights, min=0.1, max=10.0)   # clamp max 10x thay vì 50x
    weights = weights / weights.mean()
    return weights


def preset_weights() -> torch.Tensor:
    w = FOGGY_WEIGHTS_PRESET
    return w / w.mean()


def print_table(weights: torch.Tensor, title: str):
    print(f"\n  {'='*60}")
    print(f"  {title}")
    print(f"  {'='*60}")
    print(f"  {'Class':<16} {'Weight':>8}  Bar")
    print(f"  {'-'*58}")
    for cls, w in zip(CLASSES, weights):
        bar  = '█' * int(w.item() * 12)
        mark = ' ← boosted' if w.item() > 1.15 else ''
        print(f"  {cls:<16} {w.item():>8.4f}  {bar}{mark}")
    print(f"\n  min={weights.min():.4f}  max={weights.max():.4f}  "
          f"mean={weights.mean():.4f}  ratio={weights.max()/weights.min():.2f}x")
    print(f"  {'='*60}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    official = torch.tensor([
        0.8373, 0.918,  0.866,  1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
        0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
        1.0865, 1.1529, 1.0507,
    ], dtype=torch.float32)
    print_table(official, "Official GCNet-S (Clear Cityscapes) — reference")

    if SCAN_MODE:
        weights = scan_weights(TRAIN_TXT, METHOD, WORKERS, LABEL_FORMAT)
        title   = f"Scanned from Foggy data ({METHOD}, {LABEL_FORMAT})"
    else:
        weights = preset_weights()
        title   = "Foggy-aware preset (boosted stuck classes)"

    print_table(weights, title)

    print(f"  Delta vs official:")
    print(f"  {'Class':<16} {'Official':>9} {'New':>9} {'Delta':>9}")
    print(f"  {'-'*46}")
    for cls, ow, nw in zip(CLASSES, official, weights):
        delta = nw.item() - ow.item()
        mark  = ' ↑' if delta > 0.05 else (' ↓' if delta < -0.05 else '')
        print(f"  {cls:<16} {ow.item():>9.4f} {nw.item():>9.4f} {delta:>+9.4f}{mark}")

    out_path = Path(OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out_path)
    print(f"\n  ✅ Saved → {out_path}")
    print(f"     Usage: --class_weights_file {out_path}\n")
