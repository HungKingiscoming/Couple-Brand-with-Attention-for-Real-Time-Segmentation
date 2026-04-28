import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CityscapesDataset(Dataset):
    """
    Universal Dataset for Cityscapes (Normal & Foggy versions)
    """

    ID_TO_TRAINID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18, -1: 255
    }

    def __init__(
        self,
        txt_file: str,
        transforms: Optional[Callable] = None,
        img_size: Tuple[int, int] = (1024, 2048),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        ignore_index: int = 255,
        label_mapping: str = 'train_id',
        dataset_type: str = 'normal'
    ):
        super().__init__()
        self.txt_file      = txt_file
        self.img_size      = img_size
        self.mean          = mean
        self.std           = std
        self.ignore_index  = ignore_index
        self.label_mapping = label_mapping
        self.dataset_type  = dataset_type

        self.create_label_mapping()

        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label_path = line.split(',')
                    self.samples.append((img_path, label_path))

        print(f"📁 Loaded {len(self.samples)} samples from {txt_file}")
        print(f"🏷️  Dataset type: {self.dataset_type.upper()}")
        print(f"🎯 Label mapping mode: {self.label_mapping}")
        print(f"✅ Valid training classes: 19 (0-18)")
        print(f"🚫 Ignore index: {self.ignore_index}")

        if transforms is None:
            self.transforms = self.get_default_transforms()
        else:
            self.transforms = transforms

    def create_label_mapping(self):
        self.label_map = np.ones(256, dtype=np.uint8) * self.ignore_index
        if self.label_mapping == 'train_id':
            for id_val, train_id in self.ID_TO_TRAINID.items():
                if train_id != 255:
                    self.label_map[id_val] = train_id
        else:
            for i in range(256):
                self.label_map[i] = i

    def get_default_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(label_path), dtype=np.uint8)
        label = self.label_map[label]

        transformed = self.transforms(image=image, mask=label)
        image = transformed['image']
        label = transformed['mask']

        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()

        return image, label

    def get_class_distribution(self) -> Dict[int, int]:
        print("📊 Computing class distribution...")
        class_counts = {i: 0 for i in range(19)}
        for idx in tqdm(range(len(self)), desc="Scanning labels"):
            _, label_path = self.samples[idx]
            label = self.label_map[np.array(Image.open(label_path), dtype=np.uint8)]
            for class_id in range(19):
                class_counts[class_id] += (label == class_id).sum()
        return class_counts


# ============================================
# AUGMENTATION PRESETS
# ============================================

def get_train_transforms(
    img_size: Tuple[int, int] = (512, 1024),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    dataset_type: str = 'normal'
) -> A.Compose:

    # ===== GEOMETRIC (CORE) =====
    #
    # FIX 3: scale_limit lower bound -0.25 → -0.10
    # Bản gốc: scale xuống 0.75× → small objects (pole, rider, motorcycle)
    # bị shrink quá mức trước khi vào PadIfNeeded+RandomCrop.
    # -0.10 giới hạn scale down ở 0.90× — đủ diversity nhưng bảo vệ small objects.
    # Quan sát từ log: pole=0.476, rider=0.490, motorcycle=0.454 không cải thiện
    # sau 30 epoch → small object gradient bị mất ở augmentation.
    base_list = [
        A.RandomScale(scale_limit=(-0.10, 0.5), p=0.9),  # FIX 3: -0.25 → -0.10

        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=255,
            p=1.0
        ),

        A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),

        A.HorizontalFlip(p=0.5),

        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=255,
            p=0.2
        ),
    ]

    # ===== DATASET-SPECIFIC =====
    if dataset_type == 'foggy':
        specific = [
            A.CoarseDropout(
                max_holes=4,
                max_height=24,
                max_width=24,
                fill_value=0,
                mask_fill_value=255,
                p=0.25
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.15),

            # FIX 1+2: CLAHE và RandomFog tách thành OneOf để không conflict.
            #
            # Bản gốc:
            #   CLAHE p=0.35  (tăng contrast)
            #   RandomFog p=0.20  (giảm contrast)
            #   → 35%×20% = 7% samples nhận cả hai → triệt tiêu nhau
            #   → chỉ 20% samples có fog augmentation (quá thấp)
            #
            # FIX mới (OneOf):
            #   55% samples nhận MỘT trong hai — không bao giờ cả hai.
            #   CLAHE: tăng local contrast cho low-visibility samples.
            #   RandomFog: simulate fog thực với alpha_coef cao hơn (0.08→0.12)
            #              và fog_coef_upper cao hơn (0.30→0.35) để cover
            #              Cityscapes Foggy beta=0.02 (heavy fog level).
            #   p=0.55: từ 20% fog coverage → 55%×(1/2) ≈ 27.5% fog,
            #           55%×(1/2) ≈ 27.5% CLAHE — balanced và không conflict.
            #
            # Lý do tách: trong log val loss không giảm dù train loss giảm đều
            # (1.31→1.02) — dấu hiệu rõ ràng của train/val distribution gap.
            # Val set là Foggy Cityscapes thực, train chỉ có 20% fog aug → gap.
            A.OneOf([
                A.CLAHE(
                    clip_limit=(1.0, 3.0),   # giảm upper từ 4.0→3.0 (ít aggressive hơn)
                    tile_grid_size=(8, 8),
                    p=1.0
                ),
                A.RandomFog(
                    fog_coef_lower=0.05,
                    fog_coef_upper=0.35,     # FIX 1: 0.30→0.35 (cover heavy fog)
                    alpha_coef=0.12,         # FIX 1: 0.08→0.12 (fog dày hơn, realistic hơn)
                    p=1.0
                ),
            ], p=0.55),                      # FIX 1: tổng p tăng từ 0.20→0.55

            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            ], p=0.3),

            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.02, 0.08),
                p=0.05
            ),
        ]

    else:
        # Normal Cityscapes — không thay đổi
        specific = [
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                mask_fill_value=255,
                p=0.3
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.1),

            A.OneOf([
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
            ], p=0.5),
        ]

    return A.Compose(
        base_list
        + specific
        + [
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    )


def get_val_transforms(
    img_size: Tuple[int, int] = (512, 1024),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1],
                 interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


# ============================================
# DATALOADER CREATION
# ============================================

def create_dataloaders(
    train_txt: str,
    val_txt: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 1024),
    pin_memory: bool = True,
    compute_class_weights: bool = False,
    dataset_type: str = 'normal'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.Tensor]]:

    print(f"\n{'='*60}")
    print(f"🚀 Creating DataLoaders for {dataset_type.upper()} Cityscapes")
    print(f"{'='*60}\n")

    train_dataset = CityscapesDataset(
        txt_file=train_txt,
        transforms=get_train_transforms(img_size=img_size, dataset_type=dataset_type),
        img_size=img_size,
        label_mapping='train_id',
        dataset_type=dataset_type
    )

    print()

    val_dataset = CityscapesDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=img_size),
        img_size=img_size,
        label_mapping='train_id',
        dataset_type=dataset_type
    )

    class_weights = None
    if compute_class_weights:
        print(f"\n{'='*60}")
        print("📊 Computing class weights for balanced training...")
        print(f"{'='*60}\n")

        class_counts = train_dataset.get_class_distribution()
        total_pixels = sum(class_counts.values())
        class_weights = []

        print("\n📈 Class distribution:")
        print(f"{'Class':<8} {'Pixels':<15} {'Frequency':<12} {'Weight':<10}")
        print("-" * 50)

        for class_id in range(19):
            count = class_counts[class_id]
            freq  = count / total_pixels if total_pixels > 0 else 0
            weight = 1.0 / (freq + 1e-8)
            class_weights.append(weight)
            print(f"{class_id:<8} {count:<15,} {freq*100:>6.2f}%      {weight:>8.4f}")

        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = torch.clamp(class_weights, min=0.1, max=50.0)
        class_weights = class_weights / class_weights.sum() * 19
        print(f"\n✅ Class weights normalized (mean=1.0, max clipped to 50x)")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"\n{'='*60}")
    print("✅ DataLoaders Created Successfully")
    print(f"{'='*60}")
    print(f"📦 Train samples: {len(train_dataset):,} ({len(train_loader)} batches)")
    print(f"📦 Val samples:   {len(val_dataset):,} ({len(val_loader)} batches)")
    print(f"🔢 Batch size:    {batch_size}")
    print(f"👷 Workers:       {num_workers}")
    print(f"📐 Image size:    {img_size[0]}x{img_size[1]}")
    print(f"{'='*60}\n")

    return train_loader, val_loader, class_weights
