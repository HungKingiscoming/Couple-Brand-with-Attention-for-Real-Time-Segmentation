"""
analyze_distribution.py — Phân tích class distribution và tính class weights tối ưu.

Cách dùng:
    python analyze_distribution.py --train_txt /kaggle/working/train.txt
    python analyze_distribution.py --train_txt /kaggle/working/train.txt --save_weights weights.pt
"""

import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

ID_TO_TRAINID = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7,
    21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
    28:15, 31:16, 32:17, 33:18
}


def compute_distribution(txt_file: str, max_samples: int = None):
    label_map = np.ones(256, dtype=np.int32) * 255
    for id_val, train_id in ID_TO_TRAINID.items():
        label_map[id_val] = train_id

    samples = []
    with open(txt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                _, label_path = line.split(',')
                samples.append(label_path)

    if max_samples:
        samples = samples[:max_samples]

    counts = np.zeros(19, dtype=np.int64)
    total  = 0

    for label_path in tqdm(samples, desc="Scanning labels"):
        label = np.array(Image.open(label_path), dtype=np.uint8)
        mapped = label_map[label]
        for c in range(19):
            counts[c] += (mapped == c).sum()
        total += (mapped != 255).sum()

    return counts, total


def compute_weights(counts, total, method='inverse_freq', clip_max=10.0):
    """
    3 phương pháp tính class weights:

    1. inverse_freq: w_c = total / (C * count_c)
       Standard, dùng nhiều nhất. Tỷ lệ nghịch với frequency.

    2. sqrt_inverse: w_c = sqrt(total / (C * count_c))
       Nhẹ hơn inverse_freq — tránh trường hợp class rất hiếm có weight quá lớn
       gây training instability.

    3. median_freq: w_c = median_freq / freq_c
       Dùng median làm anchor thay vì tổng — robust hơn với outliers.
       Khuyến nghị cho Cityscapes vì road/sky chiếm quá nhiều.
    """
    C     = 19
    freqs = counts / total

    if method == 'inverse_freq':
        weights = total / (C * counts.astype(float))
    elif method == 'sqrt_inverse':
        weights = np.sqrt(total / (C * counts.astype(float)))
    elif method == 'median_freq':
        median = np.median(freqs)
        weights = median / freqs
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clip để tránh extreme weights làm gradient explosion
    weights = np.clip(weights, 0.1, clip_max)

    # Normalize: mean = 1.0 để không thay đổi learning rate hiệu quả
    weights = weights / weights.mean()

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt',   required=True)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Giới hạn số samples để chạy nhanh hơn (None = tất cả)')
    parser.add_argument('--save_weights', type=str, default=None,
                        help='Lưu weights tensor ra file .pt để dùng trong training')
    parser.add_argument('--method', default='median_freq',
                        choices=['inverse_freq', 'sqrt_inverse', 'median_freq'])
    parser.add_argument('--clip_max', type=float, default=10.0)
    args = parser.parse_args()

    print(f"\nScanning: {args.train_txt}")
    if args.max_samples:
        print(f"(limited to {args.max_samples} samples for speed)")
    counts, total = compute_distribution(args.train_txt, args.max_samples)

    print(f"\n{'='*75}")
    print(f"{'Class':<16} {'Pixels':>14} {'Freq%':>8} {'inv_freq':>10} {'sqrt_inv':>10} {'med_freq':>10}")
    print('-'*75)

    w_inv  = compute_weights(counts, total, 'inverse_freq', args.clip_max)
    w_sqrt = compute_weights(counts, total, 'sqrt_inverse', args.clip_max)
    w_med  = compute_weights(counts, total, 'median_freq',  args.clip_max)

    for c in range(19):
        freq = counts[c] / total * 100
        print(f"{CITYSCAPES_CLASSES[c]:<16} {counts[c]:>14,} {freq:>7.2f}% "
              f"{w_inv[c]:>10.3f} {w_sqrt[c]:>10.3f} {w_med[c]:>10.3f}")

    print('='*75)
    print(f"\nTotal valid pixels: {total:,}")
    print(f"Imbalance ratio (max/min freq): {counts.max()/counts.min():.1f}x")

    chosen = {'inverse_freq': w_inv, 'sqrt_inverse': w_sqrt, 'median_freq': w_med}[args.method]
    print(f"\nSelected method: {args.method}")
    print(f"Weights (clip_max={args.clip_max}):")
    for c in range(19):
        bar = '█' * int(chosen[c] * 5)
        print(f"  {CITYSCAPES_CLASSES[c]:<16} {chosen[c]:6.3f}  {bar}")

    if args.save_weights:
        weights_tensor = torch.tensor(chosen, dtype=torch.float32)
        torch.save(weights_tensor, args.save_weights)
        print(f"\nWeights saved to: {args.save_weights}")
        print(f"Load in training: weights = torch.load('{args.save_weights}')")


if __name__ == '__main__':
    main()
