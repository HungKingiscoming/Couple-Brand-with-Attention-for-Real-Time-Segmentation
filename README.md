# Enhancing Semantic Segmentation in Foggy Weather Conditions

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![mIoU](https://img.shields.io/badge/mIoU-67.89%25-orange?style=flat-square)
![FPS](https://img.shields.io/badge/FPS-139-brightgreen?style=flat-square)

**Graduation Thesis — Hung Yen University of Technology and Education, 2026**

*Giang Tuan Hung · Supervisor: Trung Hieu Le, PhD*

</div>

---

## Overview

This repository contains the implementation of our graduation thesis on **fog-aware semantic segmentation**. We propose two lightweight architectural modifications to [GCNet-S](https://arxiv.org/abs/2503.03325) that improve segmentation robustness under foggy conditions without requiring any external dehazing module.

### Key Contributions

| Module | Description | Location |
|--------|-------------|----------|
| **FoggyAwareNorm (FAN)** | Learnable BN↔IN interpolation per channel | `stem_conv1`, `stem_conv2` |
| **Dynamic Weight Self-Attention (DWSA)** | 16× cheaper spatial attention + SE gate from attended features | Semantic branch, Stage 4, 5, 6 |

---

## Results

### Cityscapes-Foggy

| Model | mIoU | mDice | mAcc | Params | FPS |
|-------|------|-------|------|--------|-----|
| GCNet (baseline) | 0.5882 | 0.6911 | 0.7293 | 9.21M | 182.3 |
| BiSeNetV2 | 0.5721 | 0.7143 | 0.6479 | 5.23M | 121.3 |
| PIDNet | 0.5851 | 0.7251 | 0.6885 | 43.83M | 117.0 |
| SCTNet | 0.6396 | 0.7717 | 0.7328 | 12.05M | 162.1 |
| DDRNet | 0.5821 | 0.7236 | 0.6687 | 20.30M | 85.3 |
| RDRNET | 0.5946 | 0.7351 | 0.7028 | 7.30M | 77.8 |
| PSPNet | 0.5237 | 0.6537 | 0.6069 | 24.38M | 65.0 |
| **Ours** | **0.6789** | **0.8074** | **0.7768** | **9.45M** | **139.0** |

### Driving-Foggy (Real-world)

| Model | mIoU | mDice | mAcc |
|-------|------|-------|------|
| GCNet (baseline) | 0.3042 | 0.4254 | 0.4707 |
| BiSeNetV2 | 0.3791 | 0.5123 | 0.5111 |
| PIDNet | 0.3168 | 0.4331 | 0.4494 |
| SCTNet | 0.3739 | 0.5076 | 0.5417 |
| DDRNet | 0.2707 | 0.3828 | 0.3685 |
| RDRNET | 0.3253 | 0.4401 | 0.5139 |
| PSPNet | 0.1633 | 0.2468 | 0.5567 |
| **Ours** | **0.3836** | **0.5158** | **0.5586** |

### vs. Dehazing Pipelines

| Method | mIoU (City) | mIoU (Foggy) | FPS | Latency |
|--------|-------------|--------------|-----|---------|
| CORUN + GCNet | 0.5940 | 0.3405 | 5 | 193ms |
| MB-Taylorformer + GCNet | 0.6207 | 0.3349 | 8 | 118ms |
| FFA-Net + GCNet | 0.6222 | 0.3455 | 0.2 | 5018ms |
| **Ours** | **0.6789** | **0.3836** | **139** | **7.20ms** |

> ✅ Our method is the **only real-time configuration** (≥30 FPS) while achieving the best segmentation quality on both benchmarks.

---

## Architecture

```
Input
  └─ Stage 1: STEM + FAN          ← fog-aware normalization
       └─ Stage 2-3: GCBlock (shared)
            ├─ Semantic Branch
            │    ├─ Stage 4: GCBlock → DWSA → Bilateral Fusion (F)
            │    ├─ Stage 5: GCBlock → DWSA → Bilateral Fusion (F)
            │    └─ Stage 6: GCBlock → DWSA → DAPPM
            └─ Detail Branch
                 ├─ Stage 4: GCBlock → (F)
                 ├─ Stage 5: GCBlock → (F)
                 └─ Stage 6: GCBlock → Head → Output
```

### FoggyAwareNorm (FAN)

Learnable interpolation between BatchNorm and InstanceNorm:

```
FAN(x) = σ(α) · IN(x) + (1 − σ(α)) · BN(x)
```

- `σ(α) → 1`: fog-sensitive channels lean toward IN
- `σ(α) → 0`: semantic channels lean toward BN
- `α ∈ R^(1×C×1×1)`: learnable per-channel, initialized at 0.5

### Dynamic Weight Self-Attention (DWSA)

```
DWSA(x) = x + Conv₁ₓ₁(A · DW(A))
```

- **Spatial reduction**: Pool Q, K, V to H/4 × W/4 → **16× cheaper** than full attention
- **SE gate from A**: Channel weights conditioned on attended features, not raw foggy input
- **Placement**: Semantic branch only, after Bilateral Fusion at Stage 4, 5, 6

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/fog-segmentation.git
cd fog-segmentation

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.23.0
opencv-python>=4.8.0
albumentations[pytorch]>=1.4.0
Pillow>=9.5.0
tqdm
```

---

## Dataset Preparation

### Cityscapes-Foggy
Download from [Cityscapes website](https://www.cityscapes-dataset.com/):
- `leftImg8bit_foggy` — foggy images (β = 0.005, 0.01, 0.02)
- `gtFine` — ground truth labels

Create val.txt:
```
/path/to/foggy_image.png,/path/to/gtFine_labelIds.png
```

### Driving-Foggy
Download from [Foggy Driving benchmark](http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/).

---

## Usage

### Validation on Cityscapes-Foggy

```bash
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --validate \
  --val_txt /path/to/val.txt \
  --img_h 512 --img_w 1024 \
  --batch_size 8
```

### Validation on Driving-Foggy

```bash
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --validate_driving \
  --driving_root /path/to/Foggy_Driving \
  --img_h 512 --img_w 1024
```

### Benchmark Speed

```bash
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --benchmark \
  --img_h 512 --img_w 1024 \
  --n_warmup 50 --n_repeat 3
```

### Video Inference

```bash
# Overlay (original + mask)
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --infer_video \
  --video_input /path/to/video.mp4 \
  --img_h 512 --img_w 1024 \
  --video_alpha 0.55

# Pure segmentation mask
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --infer_video \
  --video_input /path/to/video.mp4 \
  --img_h 512 --img_w 1024 \
  --video_alpha 1.0 \
  --video_save_mask
```

---

## Training

```bash
python train.py \
  --train_txt /path/to/train.txt \
  --val_txt /path/to/val.txt \
  --pretrained /path/to/gcnet_cityscapes.pth \
  --img_h 512 --img_w 1024 \
  --batch_size 4 \
  --epochs 100 \
  --lr 5e-4
```

**Training Configuration:**
- Optimizer: AdamW (lr=5e-4, weight decay=1e-4)
- Scheduler: Cosine annealing
- Loss: OHEM CE + 0.5 × Dice + 0.4 × Auxiliary CE
- Hardware: NVIDIA Tesla P100 16GB (Kaggle)
- Transfer learning: Progressive unfreezing from GCNet-S pretrained on Cityscapes

---

## Qualitative Results

Results on Cityscapes-Foggy at three fog densities (β = 0.005, 0.01, 0.02):

```
Row 1: Input foggy images
Row 2: Ground truth
Row 3: Our predictions
```

---

## Citation

```bibtex
@thesis{giang2026fog,
  title     = {Enhancing Object Detection Performance in Foggy Weather Conditions 
               Based on Semantic Segmentation Techniques},
  author    = {Giang, Tuan Hung},
  school    = {Hung Yen University of Technology and Education},
  year      = {2026},
  type      = {Bachelor's Thesis}
}
```

---

## Acknowledgements

- [GCNet](https://arxiv.org/abs/2503.03325) — backbone architecture
- [Cityscapes-Foggy](https://www.cityscapes-dataset.com/) — training dataset
- [Non-local Networks](https://arxiv.org/abs/1711.07971), [PVT](https://arxiv.org/abs/2102.12122), [SENet](https://arxiv.org/abs/1709.01507) — attention building blocks

---

<div align="center">
<b>Hung Yen University of Technology and Education · 2026</b>
</div>
