"""
debug_label.py — chạy trên Kaggle để tìm chính xác nguồn gốc label corruption

python debug_label.py \
    --train_txt /kaggle/working/train.txt \
    --val_txt   /kaggle/working/val.txt \
    --dataset_type foggy \
    --n_samples 20
"""

import argparse
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


# ============================================================
# STEP 0 — In version info trước hết
# ============================================================
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"albumentations : {A.__version__}")
print(f"opencv         : {cv2.__version__}")
print(f"numpy          : {np.__version__}")
print(f"torch          : {torch.__version__}")
print()


# ============================================================
# HELPER
# ============================================================
VALID_LABELS = set(range(19)) | {255}

def check_label(label_arr: np.ndarray, stage: str, idx: int = 0):
    """In thống kê label và cảnh báo nếu có giá trị ngoài range."""
    uniq = np.unique(label_arr).tolist()
    bad  = [v for v in uniq if v not in VALID_LABELS]
    status = "❌ BAD" if bad else "✓ OK "
    print(f"  [{status}] {stage:45s} | unique={uniq[:12]}{'...' if len(uniq)>12 else ''}"
          + (f" | BAD={bad}" if bad else ""))
    return bad


# ============================================================
# STEP 1 — Đọc raw label file, kiểm tra giá trị gốc
# ============================================================
def step1_raw_file(train_txt: str, n: int = 5):
    print("=" * 70)
    print("STEP 1: RAW LABEL FILES (chưa qua bất kỳ xử lý nào)")
    print("=" * 70)

    samples = []
    with open(train_txt) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                samples.append((parts[0].strip(), parts[1].strip()))

    for i, (_, lp) in enumerate(samples[:n]):
        raw = np.array(Image.open(lp))
        print(f"  Sample {i}: shape={raw.shape} dtype={raw.dtype} "
              f"min={raw.min()} max={raw.max()} unique={np.unique(raw).tolist()[:15]}")

    print()
    return samples


# ============================================================
# STEP 2 — Sau khi áp label_map (id → train_id)
# ============================================================
ID_TO_TRAINID = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7,
    21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
    28:15, 31:16, 32:17, 33:18,
}

def build_label_map(ignore_index=255):
    lm = np.full(256, ignore_index, dtype=np.uint8)
    for orig, train in ID_TO_TRAINID.items():
        lm[orig] = train
    return lm

def step2_after_label_map(samples, n=5):
    print("=" * 70)
    print("STEP 2: SAU KHI ÁP label_map (id → train_id)")
    print("=" * 70)
    lm = build_label_map()
    for i, (_, lp) in enumerate(samples[:n]):
        raw   = np.array(Image.open(lp), dtype=np.uint8)
        label = lm[raw]
        check_label(label, f"sample {i}: {Path(lp).name[:35]}", i)
    print()


# ============================================================
# STEP 3 — Từng transform một, xác định cái nào gây corrupt
# ============================================================
def step3_transform_by_transform(samples, n=5, img_size=(512, 1024)):
    print("=" * 70)
    print("STEP 3: TỪNG TRANSFORM MỘT")
    print("=" * 70)

    lm = build_label_map()

    # Lấy một sample để test
    img_path, lbl_path = samples[0]
    image = np.array(Image.open(img_path).convert('RGB'))
    raw   = np.array(Image.open(lbl_path), dtype=np.uint8)
    label = lm[raw]

    print(f"  Test image: {Path(img_path).name}  shape={image.shape}")
    print(f"  Test label: {Path(lbl_path).name}")
    check_label(label, "BEFORE any transform")
    print()

    # Danh sách transforms thử lần lượt — CUMULATIVE
    transforms_to_test = [
        ("RandomScale INTER_LINEAR (no mask_interpolation)",
         A.RandomScale(scale_limit=0.4, interpolation=cv2.INTER_LINEAR, p=1.0)),

        ("RandomScale + mask_interpolation=INTER_NEAREST",
         A.RandomScale(scale_limit=0.4,
                       interpolation=cv2.INTER_LINEAR,
                       mask_interpolation=cv2.INTER_NEAREST, p=1.0)),

        ("PadIfNeeded BORDER_REFLECT_101",
         A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                       border_mode=cv2.BORDER_REFLECT_101, p=1.0)),

        ("PadIfNeeded BORDER_CONSTANT fill_mask=255",
         A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                       border_mode=cv2.BORDER_CONSTANT,
                       fill=0, fill_mask=255, p=1.0)),

        ("RandomCrop",
         A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0)),

        ("HorizontalFlip",
         A.HorizontalFlip(p=1.0)),  # p=1 để luôn flip
    ]

    np.random.seed(42)
    # Test từng transform một cách độc lập (không cumulative)
    for name, tf in transforms_to_test:
        pipe = A.Compose([tf])
        bads_found = 0
        for _ in range(10):
            r   = pipe(image=image, mask=label)
            bad = [v for v in np.unique(r['mask']) if v not in VALID_LABELS]
            if bad:
                bads_found += 1
        status = f"❌ BAD ({bads_found}/10 runs)" if bads_found else "✓ OK (10/10 runs)"
        print(f"  [{status}] {name}")

    print()

    # Test full geometric pipeline
    print("  --- Full geometric pipeline (10 runs) ---")
    scales_to_test = [
        ("OLD: REFLECT + no mask_interpolation",
         A.Compose([
             A.RandomScale(scale_limit=0.4, interpolation=cv2.INTER_LINEAR, p=1.0),
             A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                           border_mode=cv2.BORDER_REFLECT_101, p=1.0),
             A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),
         ])),
        ("NEW: CONSTANT + INTER_NEAREST mask",
         A.Compose([
             A.RandomScale(scale_limit=0.4,
                           interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST, p=1.0),
             A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                           border_mode=cv2.BORDER_CONSTANT,
                           fill=0, fill_mask=255, p=1.0),
             A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),
         ])),
    ]
    np.random.seed(42)
    for name, pipe in scales_to_test:
        bads_found = 0
        for _ in range(20):
            r   = pipe(image=image, mask=label)
            bad = [v for v in np.unique(r['mask']) if v not in VALID_LABELS]
            if bad:
                bads_found += 1
        status = f"❌ BAD ({bads_found}/20)" if bads_found else "✓ OK  (20/20)"
        print(f"  [{status}] {name}")
    print()


# ============================================================
# STEP 4 — Test full pipeline như trong training
# ============================================================
def step4_full_pipeline(samples, dataset_type='foggy', n=10, img_size=(512,1024)):
    print("=" * 70)
    print(f"STEP 4: FULL TRAINING PIPELINE (dataset_type={dataset_type})")
    print("=" * 70)

    # Import từ custom.py của user
    try:
        sys.path.insert(0, '/kaggle/working/Couple-Brand-with-Attention-for-Real-Time-Segmentation')
        from data.custom import get_train_transforms, CityscapesDataset
        print("  Imported from project data/custom.py")
    except ImportError as e:
        print(f"  ❌ Cannot import custom.py: {e}")
        print("  Skipping step 4.")
        return

    lm = build_label_map()
    tf = get_train_transforms(img_size=img_size, dataset_type=dataset_type)

    all_bad = []
    for i, (img_path, lbl_path) in enumerate(samples[:n]):
        image = np.array(Image.open(img_path).convert('RGB'))
        raw   = np.array(Image.open(lbl_path), dtype=np.uint8)
        label = lm[raw]

        for run in range(5):
            np.random.seed(run * 100 + i)
            r   = tf(image=image, mask=label)
            m   = r['mask']
            # mask có thể là tensor hoặc ndarray tuỳ albumentations version
            if hasattr(m, 'numpy'):
                m = m.numpy()
            bad = [int(v) for v in np.unique(m) if int(v) not in VALID_LABELS]
            if bad:
                all_bad.append((i, run, bad))
                print(f"  ❌ sample {i} run {run}: BAD labels = {bad}")

    if not all_bad:
        print(f"  ✓ All {n} samples × 5 runs = {n*5} checks CLEAN")
    else:
        print(f"\n  ❌ {len(all_bad)} / {n*5} checks had bad labels")
    print()


# ============================================================
# STEP 5 — Kiểm tra Dataset __getitem__ output
# ============================================================
def step5_dataset_getitem(train_txt, dataset_type='foggy', n=10, img_size=(512,1024)):
    print("=" * 70)
    print("STEP 5: Dataset.__getitem__ OUTPUT (như training thật)")
    print("=" * 70)

    try:
        sys.path.insert(0, '/kaggle/working/Couple-Brand-with-Attention-for-Real-Time-Segmentation')
        from data.custom import create_dataloaders
    except ImportError as e:
        print(f"  ❌ Cannot import: {e}")
        return

    # Chỉ load vài batch
    train_loader, _, _ = create_dataloaders(
        train_txt=train_txt,
        val_txt=train_txt,  # dùng train cho cả val để tiết kiệm
        batch_size=4,
        num_workers=0,      # 0 để dễ debug
        img_size=img_size,
        compute_class_weights=False,
        dataset_type=dataset_type,
    )

    all_bad = []
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        if batch_idx >= n:
            break

        # imgs: (B, 3, H, W) float
        # masks: (B, H, W) long
        bad_in_batch = []
        for v in masks.unique().tolist():
            if int(v) not in VALID_LABELS:
                bad_in_batch.append(int(v))

        status = "✓" if not bad_in_batch else "❌"
        print(f"  [{status}] batch {batch_idx:3d} | "
              f"img range=[{imgs.min():.2f}, {imgs.max():.2f}] | "
              f"mask unique={sorted(masks.unique().tolist())} "
              + (f"| BAD={bad_in_batch}" if bad_in_batch else ""))

        if bad_in_batch:
            all_bad.append((batch_idx, bad_in_batch))

    print()
    if not all_bad:
        print(f"  ✓ All {min(n, len(train_loader))} batches CLEAN")
    else:
        print(f"  ❌ {len(all_bad)} batches had bad labels — BUG IS IN DATASET/DATALOADER")
        print(f"  Loss = -log(p) với p≈0 khi model confuse label → loss >> ln(19)=2.94")
    print()


# ============================================================
# STEP 6 — Sanity check loss với fake model
# ============================================================
def step6_loss_sanity():
    print("=" * 70)
    print("STEP 6: LOSS SANITY CHECK")
    print("=" * 70)
    import torch.nn.functional as F

    B, C, H, W = 2, 19, 64, 128

    # Random logits (should give loss ≈ ln(19) ≈ 2.944)
    logits = torch.randn(B, C, H, W)
    labels_ok  = torch.randint(0, 19, (B, H, W))         # valid labels
    labels_bad = torch.randint(0, 256, (B, H, W))        # invalid labels (0-255)

    loss_ok  = F.cross_entropy(logits, labels_ok, ignore_index=255)
    loss_bad = F.cross_entropy(logits, labels_bad, ignore_index=255)

    print(f"  Expected loss (random model) ≈ ln(19) = {np.log(19):.3f}")
    print(f"  Loss với labels hợp lệ  (0-18):   {loss_ok.item():.3f}  {'✓ OK' if abs(loss_ok.item() - np.log(19)) < 0.5 else '❌'}")
    print(f"  Loss với labels corrupt (0-255):   {loss_bad.item():.3f}  {'← nếu cao hơn nhiều = label corrupt' if loss_bad.item() > 3.5 else ''}")
    print()
    print(f"  Trong log của bạn: Train Loss = 4.5~5.4 >> ln(19)=2.94")
    print(f"  → Xác nhận: labels có values ngoài range 0-18 và 255")
    print()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt',    required=True)
    parser.add_argument('--val_txt',      required=True)
    parser.add_argument('--dataset_type', default='foggy', choices=['foggy', 'normal'])
    parser.add_argument('--n_samples',    type=int, default=10)
    parser.add_argument('--img_h',        type=int, default=512)
    parser.add_argument('--img_w',        type=int, default=1024)
    args = parser.parse_args()

    img_size = (args.img_h, args.img_w)

    samples = step1_raw_file(args.train_txt, n=args.n_samples)
    step2_after_label_map(samples, n=args.n_samples)
    step3_transform_by_transform(samples, n=args.n_samples, img_size=img_size)
    step4_full_pipeline(samples, dataset_type=args.dataset_type,
                        n=args.n_samples, img_size=img_size)
    step5_dataset_getitem(args.train_txt, dataset_type=args.dataset_type,
                          n=10, img_size=img_size)
    step6_loss_sanity()

    print("=" * 70)
    print("DEBUG COMPLETE — đọc kết quả từng STEP để xác định nguồn lỗi:")
    print("  STEP 1 BAD → file label không phải Cityscapes format (label ID sai)")
    print("  STEP 2 BAD → label_map bị sai (ID_TO_TRAINID thiếu/sai)")
    print("  STEP 3 BAD → transform cụ thể nào đó gây corrupt")
    print("  STEP 4 BAD → custom.py get_train_transforms có bug")
    print("  STEP 5 BAD → Dataset.__getitem__ hoặc DataLoader có bug")
    print("  STEP 6     → Confirm: loss >> 2.94 = label corrupt")
    print("=" * 70)
