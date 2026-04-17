"""
analyze_distribution.py — Tính class weights nhanh dùng DataLoader + GPU bincount.

Cách dùng:
    python analyze_distribution.py --train_txt /kaggle/working/train.txt
    python analyze_distribution.py --train_txt /kaggle/working/train.txt \
        --method sqrt_inverse --clip_max 3.0 \
        --save_weights /kaggle/working/class_weights.pt \
        --batch_size 64 --num_workers 8
"""

import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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


class LabelOnlyDataset(Dataset):
    """Chỉ load label map, không load ảnh RGB → nhanh 3-4x."""

    def __init__(self, txt_file: str, sample_ratio: float = 1.0):
        self.label_map = np.ones(256, dtype=np.uint8) * 255
        for id_val, train_id in ID_TO_TRAINID.items():
            self.label_map[id_val] = train_id

        samples = []
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    _, label_path = line.split(',')
                    samples.append(label_path)

        if sample_ratio < 1.0:
            import random; random.seed(42)
            k = max(1, int(len(samples) * sample_ratio))
            samples = random.sample(samples, k)
            print(f"Sampling {k}/{k+len(samples)-k} labels ({sample_ratio*100:.0f}%)")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = np.array(Image.open(self.samples[idx]), dtype=np.uint8)
        label = self.label_map[label]
        return torch.from_numpy(label).to(torch.int16)


def compute_distribution_fast(txt_file, batch_size=64, num_workers=8,
                               sample_ratio=1.0, device='cuda'):
    dataset = LabelOnlyDataset(txt_file, sample_ratio=sample_ratio)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    dev    = torch.device(device if torch.cuda.is_available() else 'cpu')
    counts = torch.zeros(19, dtype=torch.int64, device=dev)

    for batch in tqdm(loader, desc="Counting pixels"):
        labels = batch.to(dev, non_blocking=True).long()
        valid  = labels[labels != 255]
        if valid.numel() > 0:
            counts += torch.bincount(valid, minlength=19)

    return counts.cpu().numpy(), counts.sum().item()


def compute_weights(counts, total, method='sqrt_inverse', clip_max=3.0):
    C     = 19
    freqs = counts / total

    if method == 'inverse_freq':
        w = total / (C * counts.astype(float))
    elif method == 'sqrt_inverse':
        w = np.sqrt(total / (C * counts.astype(float)))
    elif method == 'median_freq':
        w = np.median(freqs) / freqs
    else:
        raise ValueError(method)

    w = np.clip(w, 0.1, clip_max)
    w = w / w.mean()
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt',    required=True)
    parser.add_argument('--save_weights', default=None)
    parser.add_argument('--method',       default='sqrt_inverse',
                        choices=['inverse_freq', 'sqrt_inverse', 'median_freq'])
    parser.add_argument('--clip_max',     type=float, default=3.0)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='0.3 = dùng 30%% samples, nhanh 3x, accuracy ~99%%')
    parser.add_argument('--device',       default='cuda')
    args = parser.parse_args()

    print(f"\nScanning: {args.train_txt}")
    print(f"  batch_size={args.batch_size}  num_workers={args.num_workers}  "
          f"sample_ratio={args.sample_ratio}  device={args.device}\n")

    counts, total = compute_distribution_fast(
        args.train_txt, args.batch_size, args.num_workers,
        args.sample_ratio, args.device)

    w_inv  = compute_weights(counts, total, 'inverse_freq', args.clip_max)
    w_sqrt = compute_weights(counts, total, 'sqrt_inverse',  args.clip_max)
    w_med  = compute_weights(counts, total, 'median_freq',   args.clip_max)

    print(f"\n{'='*75}")
    print(f"{'Class':<16} {'Pixels':>14} {'Freq%':>8} "
          f"{'inv_freq':>10} {'sqrt_inv':>10} {'med_freq':>10}")
    print('-'*75)
    for c in range(19):
        freq = counts[c] / total * 100
        print(f"{CITYSCAPES_CLASSES[c]:<16} {counts[c]:>14,} {freq:>7.2f}% "
              f"{w_inv[c]:>10.3f} {w_sqrt[c]:>10.3f} {w_med[c]:>10.3f}")
    print('='*75)
    print(f"\nTotal valid pixels: {total:,}")
    print(f"Imbalance ratio: {counts.max()/counts.min():.1f}x")

    chosen = {'inverse_freq': w_inv, 'sqrt_inverse': w_sqrt,
              'median_freq': w_med}[args.method]
    print(f"\nSelected: {args.method} (clip_max={args.clip_max})")
    print(f"min={chosen.min():.3f}  max={chosen.max():.3f}  mean={chosen.mean():.3f}\n")
    for c in range(19):
        print(f"  {CITYSCAPES_CLASSES[c]:<16} {chosen[c]:5.3f}  {'█'*int(chosen[c]*4)}")

    if args.save_weights:
        torch.save(torch.tensor(chosen, dtype=torch.float32), args.save_weights)
        print(f"\nWeights saved → {args.save_weights}")


if __name__ == '__main__':
    main()
