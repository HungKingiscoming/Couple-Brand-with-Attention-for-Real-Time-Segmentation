# FD-Net: Real-Time Semantic Segmentation in Foggy Weather Conditions

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![mIoU](https://img.shields.io/badge/Foggy_Cityscapes_mIoU-67.89%25-orange?style=flat-square)
![FPS](https://img.shields.io/badge/Inference-139_FPS-brightgreen?style=flat-square)

**Official PyTorch implementation of FD-Net**  
*A lightweight fog-aware semantic segmentation framework for robust and real-time road-scene perception.*

**Giang Tuan Hung**  
Hung Yen University of Technology and Education, 2026  
Supervised by **Trung Hieu Le, PhD**

</div>

---

## Abstract

Semantic segmentation is a fundamental component of autonomous-driving perception, but its reliability degrades substantially in foggy scenes because atmospheric scattering reduces visibility, weakens object boundaries, and suppresses fine-grained visual cues. A common solution is to cascade an image-dehazing network with a semantic-segmentation model. However, improving low-level image appearance does not necessarily improve semantic understanding, and the additional restoration stage introduces substantial computational overhead.

We propose **FD-Net**, a lightweight semantic-segmentation framework built upon **GCNet** that learns fog-robust representations directly inside the segmentation network. FD-Net introduces two complementary modules: **Foggy-Aware Normalization (FAN)**, which alleviates fog-induced feature-distribution discrepancies in early layers by adaptively combining Batch Normalization and Instance Normalization, and **Dynamic Weight Self-Attention (DWSA)**, which efficiently captures long-range contextual dependencies in the semantic branch through spatially reduced self-attention and dynamic channel weighting. Experiments on **Foggy Cityscapes** and the real-world **Foggy Driving** benchmark demonstrate improved segmentation accuracy while retaining real-time inference efficiency.

---

## Highlights

- **Fog-aware feature normalization.** FAN adaptively interpolates between BN and IN in the first two stem convolutions, where fog-induced appearance shifts are most pronounced.
- **Efficient global context modeling.** DWSA introduces long-range contextual reasoning only in the semantic branch and reduces the spatial size of Q/K/V before attention computation.
- **Real-time performance.** FD-Net reaches **67.89% mIoU** on Foggy Cityscapes with **9.45M parameters** and **139 FPS** in the main inference benchmark.
- **Real-world generalization.** Without using Foggy Driving during training, FD-Net achieves **38.36% mIoU** on the 101-image real-world foggy benchmark.
- **No external dehazing network.** Fog robustness is learned directly in the segmentation model, avoiding the latency of cascaded restoration-and-segmentation pipelines.

---

## Method

FD-Net follows the lightweight dual-branch design of GCNet. The shared shallow stages extract low-level features, after which the network separates into a **Detail Branch** for high-resolution spatial information and a **Semantic Branch** for high-level contextual representation.

The proposed modifications are deliberately lightweight:

1. **FAN** replaces Batch Normalization only in `stem_conv1` and `stem_conv2`.
2. **DWSA** is inserted exclusively in the Semantic Branch at Stages 4, 5, and 6.
3. An auxiliary segmentation head is used during training and removed at inference time.

### Overall Architecture

<img src="assets/architecture.png" width="800"/>

*Overview of the proposed FD-Net architecture. The fog-aware components are introduced into the original lightweight GCNet backbone without adding a separate image-restoration stage.*

### Architecture Summary

```text
Input
  └─ Stage 1: Stem + FAN
       └─ Stage 2–3: Shared GCBlocks
            ├─ Semantic Branch
            │    ├─ Stage 4: GCBlocks + DWSA
            │    ├─ Stage 5: GCBlocks + DWSA
            │    └─ Stage 6: GCBlocks + DWSA → DAPPM
            │
            └─ Detail Branch
                 ├─ Stage 4: GCBlocks
                 ├─ Stage 5: GCBlocks
                 └─ Stage 6: GCBlocks → Segmentation Head

        ↕ bilateral feature interaction between semantic/detail branches
```

### DWSA Placement

| Module | Resolution | Channels |
|---|---:|---:|
| `dwsa_stage4` | 1/16 | 128 (C×4) |
| `dwsa_stage5` | 1/32 | 256 (C×8) |
| `dwsa_stage6` | 1/8 | 128 (C×4) |

---

## Foggy-Aware Normalization (FAN)

Foggy images with different attenuation levels can exhibit heterogeneous feature distributions. Batch Normalization depends on mini-batch statistics and may therefore be sensitive to these cross-sample variations. Instance Normalization is more robust to instance-specific appearance changes, but using it alone may remove useful globally consistent semantic information.

FAN learns a channel-wise interpolation between the two normalization schemes:

```text
FAN(x) = p · IN(x) + (1 − p) · BN(x)
p = σ(α)
```

where `α` is a learnable channel-wise parameter and `σ(·)` is the sigmoid function.

- `p → 1`: FAN approaches **Instance Normalization**.
- `p → 0`: FAN approaches **Batch Normalization**.
- Intermediate values allow the network to adapt its normalization behavior independently for different feature channels.
- FAN is used only in `stem_conv1` and `stem_conv2` to target low-level fog-sensitive features such as intensity, color, texture, and edges.

---

## Dynamic Weight Self-Attention (DWSA)

Small and distant objects in dense fog often have blurred boundaries and weak local appearance cues. To improve contextual reasoning without the cost of full-resolution self-attention, DWSA performs attention on spatially reduced query, key, and value representations.

```text
Q, K, V = Conv1×1(x)
Q', K', V' = AdaptiveAvgPool(Q, K, V)
A = Softmax(Q'ᵀK' / √dₖ)
```

The attended representation is restored to the original spatial resolution and further modulated by a content-adaptive channel gate generated from the attended features. A residual connection preserves the original representation:

```text
DWSA(x) = x + γ · Proj(Context ⊙ ChannelWeight)
```

where `Proj(·)` denotes a `1×1` projection and `γ` is a learnable residual scaling parameter.

The proposed spatial reduction decreases the attention computation by approximately **16×** relative to the unreduced formulation while preserving global contextual interactions.

---

## Training Objective

The network is trained using a combination of hard-pixel supervision, region-overlap optimization, and deep supervision:

```text
L = L_OHEM-CE + 0.5 · L_Dice + 0.4 · L_Aux-CE
```

- **OHEM Cross-Entropy** emphasizes difficult pixels.
- **Dice loss** directly optimizes region-level overlap between predictions and ground truth.
- **Auxiliary Cross-Entropy** provides additional gradients to shallow features through a training-only auxiliary head.

The auxiliary head is discarded during inference and therefore introduces **no additional deployment-time cost**.

---

# Experimental Results

## Foggy Cityscapes

| Model | mIoU | mDice | mAcc | Params | FPS |
|---|---:|---:|---:|---:|---:|
| GCNet | 0.5882 | 0.6911 | 0.7293 | 9.21M | 182.3 |
| BiSeNetV2 | 0.5721 | 0.7143 | 0.6479 | 5.23M | 76.6 |
| PIDNet | 0.5851 | 0.7251 | 0.6885 | 43.83M | 122.0 |
| SCTNet | 0.6396 | 0.7717 | 0.7328 | 12.05M | 162.1 |
| DDRNet | 0.5821 | 0.7236 | 0.6687 | 20.30M | 85.3 |
| RDRNet | 0.5946 | 0.7351 | 0.7028 | 7.30M | 75.0 |
| PSPNet | 0.5234 | 0.6516 | 0.7117 | 24.38M | 65.0 |
| **FD-Net (Ours)** | **0.6789** | **0.8074** | **0.7768** | **9.45M** | **139.0** |

FD-Net achieves **67.89% mIoU**, improving the GCNet baseline by **9.07 percentage points** while retaining real-time inference speed. Weight will be published when our paper is accepted.

## Foggy Driving: Real-World OOD Evaluation

| Model | mIoU | mDice | mAcc |
|---|---:|---:|---:|
| GCNet | 0.3042 | 0.4254 | 0.4707 |
| BiSeNetV2 | 0.3791 | 0.5123 | 0.5111 |
| PIDNet | 0.3168 | 0.4331 | 0.4494 |
| SCTNet | 0.3739 | 0.5076 | 0.5417 |
| DDRNet | 0.2707 | 0.3828 | 0.3685 |
| RDRNet | 0.3253 | 0.4401 | 0.5139 |
| PSPNet | 0.1662 | 0.2473 | 0.5567 |
| **FD-Net (Ours)** | **0.3836** | **0.5158** | **0.5586** |

Foggy Driving is not used during training. The results therefore evaluate the ability of the model to generalize from synthetic fog to unseen real-world foggy scenes.

## Comparison with Dehazing + Segmentation Pipelines

| Method | Foggy Cityscapes mIoU | Foggy Driving mIoU | FPS | Latency |
|---|---:|---:|---:|---:|
| CORUN + GCNet | 0.5940 | 0.3405 | 5 | 193 ms |
| MB-Taylorformer + GCNet | 0.6207 | 0.3349 | 8 | 118 ms |
| FFA-Net + GCNet | 0.6222 | 0.3455 | 0.2 | 5018 ms |
| **FD-Net (Ours)** | **0.6789** | **0.3836** | **139** | **7.20 ms** |

These results show that integrating fog-aware representation learning directly into the segmentation network provides a substantially better accuracy–latency trade-off than cascaded dehazing-and-segmentation pipelines.

---

## Qualitative Results

### Foggy Cityscapes

![Qualitative Results](assets/qualitative_cityscapes.png)

*Qualitative comparison on Foggy Cityscapes. The benchmark contains multiple fog-density levels with attenuation coefficients β = 0.005, 0.01, and 0.02.*

### Foggy Driving

![Qualitative Results Driving](assets/qualitative_driving.png)

*Qualitative results on the real-world Foggy Driving benchmark. The 101 Foggy Driving images are used only for evaluation and are never included in training.*

---

# Getting Started

## Installation

```bash
git clone https://github.com/your-username/fog-segmentation.git
cd fog-segmentation
pip install -r requirements.txt
```

### Requirements

```text
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.23.0
opencv-python>=4.8.0
albumentations[pytorch]>=1.4.0
Pillow>=9.5.0
tqdm
```

---

## Datasets

### Foggy Cityscapes

Foggy Cityscapes is derived from Cityscapes using an atmospheric-scattering model. Three attenuation coefficients are used in the experiments:

- `β = 0.005`
- `β = 0.01`
- `β = 0.02`

The original Cityscapes split contains **2,975 training images** and **500 validation images** at `1024 × 2048` resolution. Considering the three fog-density variants gives **8,925 foggy training images** and **1,500 foggy evaluation images**.

Download the Cityscapes data from:

- [Cityscapes](https://www.cityscapes-dataset.com/)

A validation list can be stored as:

```text
/path/to/foggy_image.png,/path/to/gtFine_labelIds.png
```

### Foggy Driving

Foggy Driving contains **101 real-world foggy road scenes** with semantic annotations. It is used exclusively as an **out-of-distribution test benchmark** and is not involved in model training.

- [Foggy Driving benchmark](http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)

---

## Training

The model is initialized from GCNet weights pretrained on clean Cityscapes and fine-tuned using progressive four-stage unfreezing.

```bash
python train.py \
  --train_txt /path/to/train.txt \
  --val_txt /path/to/val.txt \
  --pretrained_weights /path/to/gcnet_cityscapes.pth \
  --img_h 512 --img_w 1024 \
  --batch_size 4 \
  --epochs 100 \
  --lr 5e-4
```

For higher input-pipeline throughput, add `--persistent_workers` and tune
`--num_workers` for the available CPU. Gradient diagnostics now run every
100 optimizer steps by default (`--gradient_check_interval 0` disables them),
and CUDA allocator cache flushing is disabled unless explicitly requested.

### Faster architecture search

The search reuses one fixed proxy subset, persistent DataLoaders, and one
in-memory pretrained checkpoint across candidates. Completed evaluations are
also recovered from `search_log.jsonl` when all proxy settings match.

Use a cheap screening pass first:

```bash
python search_architecture.py \
  --pretrained_weights /path/to/our_miou_0.6783.pth \
  --train_txt /path/to/train.txt --val_txt /path/to/val.txt \
  --batch_size 16 --img_h 384 --img_w 768 \
  --proxy_data_fraction 0.05 --proxy_epochs 1 \
  --pop_size 6 --max_iter 5 --proxy_num_workers 4 \
  --fast_proxy
```

`--fast_proxy` disables Dice and auxiliary loss only for screening. Re-run
the best candidates without that flag at full proxy resolution before the
final full-data training run.

For the **lightweight/faster-than-baseline** objective on a 12-hour Kaggle
budget, use `efficient_search.py` instead of the unconstrained optimizer:

```bash
python efficient_search.py \
  --pretrained_weights /path/to/our_miou_0.6783.pth \
  --train_txt /path/to/train.txt --val_txt /path/to/val.txt \
  --model_variant fan_dwsa --dataset_type foggy \
  --img_h 512 --img_w 1024 --batch_size 16 \
  --n_candidates 12 --n_promote 3 \
  --stage1_fraction 0.10 --stage1_epochs 1 \
  --stage2_fraction 0.20 --stage2_extra_epochs 2 \
  --proxy_num_workers 4 --seed 42 \
  --max_hours 10 \
  --work_dir efficient_search_runs --out_json best_efficient_arch.json
```

For Kaggle **T4x2**, add `--gpu_ids 0,1 --proxy_num_workers 2` to that command.
The latency gate runs on GPU 0 only, so all candidates are compared against
one T4 baseline. Proxy training then runs two independent candidates at a time
in two spawned processes, one process per GPU; each process reuses its own
checkpoint and DataLoaders. Kaggle T4x2 provides four CPU cores, so two
DataLoader workers per GPU avoid the eight-worker CPU oversubscription of
`--proxy_num_workers 4`. The code also clamps larger worker requests to two
per GPU in T4x2 mode. This parallelizes **architecture evaluations**, not a
single model's training. `train.py` remains single-GPU; use the resulting
architecture JSON/weights to train the selected model afterward.

This search includes the **original checkpoint as a proxy baseline** in both
stages. It first rejects candidates larger than the original training model or
not at least 3% faster in a short deploy-architecture T4 latency benchmark.
It then trains baseline and surviving candidates on identical 10%/1-epoch
subsets, promotes up to three candidates, and continues their saved weights
for two additional epochs on 20% subsets. Stage-2 selection allows at most
0.02 proxy mIoU below the stage-2 baseline and chooses the lowest latency
among candidates meeting that floor. Stage 6 and dropout remain at the
baseline values; stage-4/5 blocks and PPM width are reduced, never expanded.
The process saves per-candidate checkpoints/results under `--work_dir`, and a
re-run with identical inputs/settings reuses completed proxy stages. Use a new
work directory when changing settings. It does **not** prove unchanged full-val
mIoU: the final selected architecture must be fully trained and validated on
all 1,500 images, then benchmarked with `test.py` on the same GPU.
The new search restarts DataLoader workers with matched seeds for each
candidate instead of reusing drifting augmentation RNG state. Its short
latency gate uses randomly initialized deploy-mode models to measure the
architecture, not the final trained model; always re-benchmark the finalist.
`--max_hours` leaves a two-hour buffer in a 12-hour session by not starting
further candidates after ten hours; a candidate already training can still
finish later. Lower `--n_candidates` if early runs reveal a high per-candidate
cost.

The JSON contains `candidate_proxy_checkpoint`, which can be used as
`train.py --pretrained_weights` together with `--arch_json` to continue from
the selected candidate's proxy weights (the optimizer is reinitialized).
If the screening latency is noisy or no candidate passes, reduce
`--min_speedup` and use a **new** `--work_dir`; 3% is a cheap gate, not a claim
that the final model meets a 130-FPS target.

### Fastest fine-tuning recipe that actually beats the existing checkpoint

If architectural compression is not the immediate goal, keep the original
`fan_dwsa` architecture and search only fine-tuning settings. No historical
training-time log is needed: every trial starts from the same 0.6783 checkpoint,
validates on the full 1,500-image list each epoch, and stops as soon as the
all-19 mIoU **strictly exceeds** the recorded baseline plus a configurable
minimum gain. The qualified trial with the shortest elapsed train+validation
time wins. A failed/unqualified trial cannot become the winner.

```bash
python hpo_time_to_miou.py \
  --checkpoint /path/to/our_miou_0.6783.pth \
  --train_txt /path/to/train.txt --val_txt /path/to/val.txt \
  --baseline_miou 0.6783018947437217 --min_gain 0 \
  --gpu_ids 0,1 --workers_per_gpu 2 \
  --batch_size 16 --img_h 512 --img_w 1024 \
  --max_epochs 4 --max_hours 10 --seed 42 \
  --work_dir hpo_bn_scheduler_runs \
  --out_json best_hpo_bn_scheduler.json
```

Two independent `train.py` processes run on the two T4s at once (one GPU per
trial). After six LR-only trials failed to beat 0.6783, the new six-trial
screen locks BatchNorm running statistics and compares a lower LR, cosine vs
polynomial scheduler, and AdamW weight decay on a fixed full-model recipe.
It also tests head-only and attention-plus-head fine-tuning, which may shorten
backward passes. Those scopes freeze parameters, **not** remove layers or
change the inference architecture. Head-only trials disable auxiliary-head
loss because it cannot update the frozen backbone and its output is not used
at inference; the main CE/OHEM+Dice loss remains unchanged. Every trial retains
the same architecture, 512x1024 resolution, data split, batch size and seed.
Each trial's log, checkpoint and result are kept
under `--work_dir`. A rerun with identical settings skips completed trials,
and preserves incomplete attempts in separate directories. If no trial reaches
the full-val threshold within four epochs, the JSON records `best_candidate:
null`; keep the original checkpoint rather than claiming an improvement.
Use a **new** `--work_dir` rather than the LR-only run directory, which remains
untouched; the manifest refuses to mix old and new trial recipes.
`--max_hours` is a hard search wall-time cap: unfinished trials are stopped,
but their epoch checkpoints and logs remain for inspection or retry.
`--min_gain 0` requires any strict full-val mIoU improvement over 0.6783019;
use `0.001` if a small validation fluctuation should not qualify. Parallel trial
training shortens the **search** wall time; it does not use DDP to speed up
one model's training, nor change single-image inference FPS.

Albumentations transform arguments are selected by API signature so both
1.x and 2.x honor ignore-index mask filling (`255`) and the intended fog
range, instead of silently falling back to changed defaults. This changes the
previous Kaggle data-augmentation behavior; compare new trials only against
the fixed checkpoint on the same full validation list.
Seeded HPO additionally requires Albumentations `Compose(seed=...)` support;
the launcher seeds both that pipeline and the training DataLoader generator
so the same sample order and augmentation streams are used across trials.

### Raindrop optimizer for training hyperparameters (fixed GCNet)

`raindrop_hpo.py` uses the repository's actual OBL-ADE-RD optimizer. Each
continuous raindrop is decoded to `lr`, AdamW weight decay, cosine/poly
scheduler, and head-only/attention-plus-head fine-tuning. The architecture,
checkpoint, train/val lists, 512x1024 resolution, loss family and seed stay
fixed. Raindrop **proposes** recipes; `train.py` trains them and returns the
measured full-validation mIoU. This is different from the earlier fixed six
HPO trials and from `search_architecture.py`'s architecture search.

For the Kaggle T4x2 / 12-hour budget, default OBL initialization and two
iterations request about 16 one-epoch proxy evaluations. Each proxy uses all
8,925 train images and all 1,500 validation images but skips model checkpoint
writes. The scheduler still has a four-epoch horizon. The two best proxy
recipes restart from the original checkpoint for up to four full epochs,
stopping early if mIoU exceeds 0.6783019. A candidate is accepted only if
its final deploy/fused checkpoint also exceeds that baseline on the full
validation set. Failed trials are penalized by their *distance below* the
mIoU threshold, so the search still has a signal even when none succeeds.

```bash
python raindrop_hpo.py \
  --checkpoint /kaggle/input/datasets/giangtunhng/our-miou-6783/our_miou_0.6783.pth \
  --train_txt /kaggle/working/train.txt --val_txt /kaggle/working/val.txt \
  --baseline_miou 0.6783018947437217 --min_gain 0 \
  --gpu_ids 0,1 --workers_per_gpu 2 --batch_size 16 \
  --img_h 512 --img_w 1024 --pop_size 4 --max_iter 2 \
  --proxy_epochs 1 --full_epochs 4 --finalists 2 \
  --proxy_hours 6 --max_hours 10 --seed 42 \
  --work_dir /kaggle/working/raindrop_hpo_runs \
  --out_json /kaggle/working/best_raindrop_hpo.json
```

Two independent trials use the two T4s concurrently; this is not DDP.
One-epoch proxy ranking can miss recipes that improve late. The optimizer
cannot guarantee a qualifying model in this budget. If `best_candidate` is
`null`, retain the original checkpoint. The work directory has a manifest
and per-candidate results/logs so the exact same command can reuse completed
evaluations after interruption; keep it separate from prior HPO directories.
The 10-hour limit caps training trials; the final deploy validation may take
a few more minutes.

### Raindrop-guided weight escape (gradient + Raindrop + gradient)

`raindrop_weight_escape.py` tests a different, paper-inspired hypothesis:
whether a bounded weight-space move can help an already-trained checkpoint
fine-tune better. This is **not** the hyperparameter search above, nor a claim
that the 0.6783 checkpoint is trapped in a local minimum. Five existing
tensors are targeted: the final classifier weight/bias and the output
projections of DWSA stages 4–6. Two fixed random directions per tensor give
Raindrop only ten coefficients to search. For target tensor `W_j`, a candidate
uses `W'_j = W_j + s * RMS(W_j) * (a_j0 D_j0 + a_j1 D_j1)`, with
`a_ji` in `[-1, 1]`; all other weights remain unchanged.

Candidate fitness is computed without gradients on 256 deterministic training
images with validation-style transforms. The 1,500-image validation split is
never used inside Raindrop. A candidate must improve proxy mIoU without
raising proxy cross-entropy by more than 5%; otherwise the run stops and the
original checkpoint is retained. If one passes, it and the original checkpoint
receive the **same** two-epoch AdamW fine-tune on separate T4 GPUs with the
same seed and data order. Both use frozen BN statistics and train only the
head, DWSA and FAN parameters. Final deploy/fused mIoU must beat **both** the
original checkpoint and the matched ordinary fine-tune. Deploy parameter
count must match, and median FPS must be at least the original model's
median FPS when benchmarked in the same session. This strict default is
sensitive to measurement noise; set `--fps_tolerance 0.02` only if a 2%
measurement tolerance is acceptable.

Use a new work directory for each run. The original checkpoint is never
overwritten. A `null` `best_candidate` means there was no verified gain.

```bash
python raindrop_weight_escape.py \
  --checkpoint /kaggle/input/datasets/giangtunhng/our-miou-6783/our_miou_0.6783.pth \
  --train_txt /kaggle/working/train.txt \
  --val_txt /kaggle/working/val.txt \
  --baseline_miou 0.6783018947437217 \
  --gpu_ids 0,1 --img_h 512 --img_w 1024 \
  --proxy_samples 256 --pop_size 4 --max_iter 1 \
  --epochs 2 --batch_size 16 --workers_per_gpu 2 \
  --max_hours 10 --seed 42 \
  --work_dir /kaggle/working/raindrop_weight_escape_runs
```

The result is in `work_dir/result.json`, with the ordinary and Raindrop logs
beside it. Proxy evaluations now show progress bars, while the ordinary and
Raindrop training subprocesses stream throttled progress and epoch metrics
to Kaggle stdout while keeping the complete raw output in their `train.log`
files. This pilot does not guarantee mIoU improvement or establish a
publication-worthy contribution by itself; repeated seeds and an independent
test split are needed before making that claim.

### Train and evaluate the selected architecture

`train.py --arch_json` accepts the full `best_gcnet_arch.json` produced by
search, or a JSON file containing just `best_config`. It applies the selected
stage depths, PPM channels, DWSA reduction, and dropout before model creation.
Pretrained weights are loaded with shape matching; the resulting training
checkpoints embed the resolved architecture so evaluation and resume cannot
silently use the default model instead.

```bash
python train.py \
  --arch_json /path/to/best_gcnet_arch.json \
  --pretrained_weights /path/to/our_miou_0.6783.pth \
  --train_txt /path/to/train.txt --val_txt /path/to/val.txt \
  --model_variant fan_dwsa --dataset_type foggy \
  --img_h 512 --img_w 1024 --batch_size 16 \
  --epochs 40 --num_workers 4 --persistent_workers \
  --save_dir /path/to/checkpoints_selected
```

To continue from `last.pth` in another session, use `--resume
/path/to/last.pth --resume_mode continue` with the same training settings.
`--arch_json` is optional for continuation because the architecture is saved
inside the checkpoint. Save the checkpoints outside an ephemeral Kaggle
session before it ends.

```bash
python test.py \
  --ckpt /path/to/checkpoints_selected/best.pth \
  --val_txt /path/to/val.txt --validate \
  --img_h 512 --img_w 1024 --batch_size 16
```

`test.py` automatically reads the architecture embedded in new checkpoints.
For older checkpoints without metadata, pass matching `--arch_json` when
needed. Validation prints both present-class mIoU and all-19-class mIoU (the
latter matches the architecture search's class averaging convention).

### Training Configuration

| Setting | Value |
|---|---|
| Initialization | GCNet pretrained on clean Cityscapes |
| Optimizer | AdamW |
| Initial learning rate | `5 × 10⁻⁴` |
| Weight decay | `1 × 10⁻⁴` |
| Scheduler | Cosine annealing |
| Epochs | 100 |
| Batch size | 4 |
| Input size | `512 × 1024` |
| Augmentation | Random horizontal flip, random scaling, random crop |
| Loss | OHEM CE + 0.5 Dice + 0.4 Auxiliary CE |
| Hardware | NVIDIA Tesla P100 16 GB |
| Transfer strategy | Progressive four-stage unfreezing |

---

## Evaluation

### Foggy Cityscapes

```bash
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --validate \
  --val_txt /path/to/val.txt \
  --img_h 512 --img_w 1024 \
  --batch_size 8
```

### Foggy Driving

```bash
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --validate_driving \
  --driving_root /path/to/Foggy_Driving \
  --img_h 512 --img_w 1024
```

### Speed Benchmark

```bash
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --benchmark \
  --img_h 512 --img_w 1024 \
  --n_warmup 50 --n_repeat 3
```

---

## Video Inference

```bash
# Overlay prediction on the input video
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --infer_video \
  --video_input /path/to/video.mp4 \
  --img_h 512 --img_w 1024 \
  --video_alpha 0.55

# Save the pure semantic segmentation mask
python test.py \
  --ckpt /path/to/checkpoint.pth \
  --infer_video \
  --video_input /path/to/video.mp4 \
  --img_h 512 --img_w 1024 \
  --video_alpha 1.0 \
  --video_save_mask
```

---

# Deployment Benchmark

In addition to the main paper-style experiments, the repository includes an Edge-AI deployment benchmark on a consumer laptop GPU using different inference backends.

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 Laptop GPU (4 GB) |
| CPU | Intel Core i5 laptop processor |
| Input resolution | 512 × 1024 |
| Model | FD-Net, 9.45M parameters |
| Batch size | 1 |

### Backend Comparison

| Backend | Precision | FPS ↑ | Latency ↓ | GPU Memory |
|---|---:|---:|---:|---:|
| PyTorch | FP32 | 60.8 | 16.45 ms | 114.3 MB |
| ONNX Runtime CUDA | FP32 | 55.7 | 17.94 ms | 42.7 MB |
| OpenVINO GPU | FP16 | 29.8 | 33.58 ms | 42.1 MB |
| TensorRT | FP32 | 100.9 | 9.92 ms | 42.7 MB |
| **TensorRT** | **FP16** | **203.4** | **4.92 ms** | **45.4 MB** |

The TensorRT FP16 configuration provides the highest measured throughput in this repository-level deployment benchmark.

---

## Citation

If you use this code or build upon FD-Net in your research, please cite the accompanying manuscript:

```bibtex
@misc{giang2026fdnet,
  title  = {FD-Net: Real-Time Semantic Segmentation in Foggy Weather Conditions},
  author = {Giang, Tuan Hung},
  year   = {2026},
  note   = {Manuscript}
}
```

> The BibTeX entry should be updated to the final journal/conference metadata after publication.

---

## Acknowledgements

This work builds upon and is inspired by the following projects and research directions:

- [GCNet](https://arxiv.org/abs/2503.03325) — baseline real-time semantic segmentation architecture.
- [Foggy Cityscapes / Semantic Foggy Scene Understanding](http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) — synthetic and real-world foggy-scene benchmarks.
- [Cityscapes](https://www.cityscapes-dataset.com/) — urban-scene semantic segmentation dataset.
- [Non-local Networks](https://arxiv.org/abs/1711.07971), [PVT](https://arxiv.org/abs/2102.12122), and [SENet](https://arxiv.org/abs/1709.01507) — contextual and channel-attention foundations related to the DWSA design.

---

## License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

<div align="center">

If this repository is useful for your research, please consider citing the work and starring the repository.

</div>
