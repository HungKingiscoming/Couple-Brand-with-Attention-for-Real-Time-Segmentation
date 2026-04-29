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
        self.txt_file    = txt_file
        self.img_size    = img_size
        self.mean        = mean
        self.std         = std
        self.ignore_index = ignore_index
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

    def get_class_distribution(self, num_workers: int = 4) -> Dict[int, int]:
        import multiprocessing as mp
    
        label_map = self.label_map  # copy ref để pickle được
    
        def count_single(label_path: str) -> np.ndarray:
            """Worker function — đếm 1 file label."""
            label = label_map[np.array(Image.open(label_path), dtype=np.uint8).ravel()]
            # bincount nhanh hơn 19× so với loop (label == class_id).sum()
            counts = np.bincount(label, minlength=256)
            return counts[:19]  # chỉ giữ 19 classes (bỏ ignore=255)
    
        label_paths = [p for _, p in self.samples]
    
        print(f"📊 Computing class distribution ({len(label_paths)} files, {num_workers} workers)...")
    
        # Multiprocessing pool — đọc song song
        # chunksize=50: mỗi worker nhận 50 file 1 lần → giảm IPC overhead
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(count_single, label_paths, chunksize=50),
                total=len(label_paths),
                desc="Scanning labels"
            ))
    
        # Stack và sum — nhanh hơn loop cộng dồn
        total_counts = np.stack(results, axis=0).sum(axis=0)
    
        return {i: int(total_counts[i]) for i in range(19)}


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
    # FIX: scale_limit tuple để giới hạn lower bound ở 0.75
    # Bản gốc scale_limit=0.4 → scale có thể xuống 0.6× → sau PadIfNeeded
    # phần lớn crop là padding (mask=255) → gradient thưa, OHEM không hiệu quả
    base_list = [
        A.RandomScale(scale_limit=(-0.25, 0.5), p=0.9),  # 10% samples giữ nguyên scale

        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,   # FIX: CONSTANT thay REFLECT
            value=0,
            mask_value=255,
            p=1.0
        ),

        A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),

        A.HorizontalFlip(p=0.5),

        # GridDistortion: tạo biến thể hình học nhẹ
        # Đặc biệt hiệu quả cho boundary của object nhỏ (rider, motorcycle)
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
            # FIX: giảm holes từ 6 xuống 4 và size từ 32 xuống 24
            # Foggy images đã có nhiều vùng khó (fog-occluded) → dropout quá nhiều
            # làm giảm useful gradient
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

            # FIX: thêm CLAHE để augment low-contrast foggy images
            # CLAHE tăng local contrast → model học được features dù bị fog che
            # Tăng p 0.2→0.35 và clip_limit range để hiệu quả hơn với foggy
            A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.35),

            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            ], p=0.3),

            # FIX: thêm ISONoise — simulates camera noise trong điều kiện sương
            # Giảm ISONoise: fog + noise cùng lúc confuse model
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.02, 0.08),
                p=0.05
            ),

            # Tăng fog_coef_upper 0.15→0.30: cover fog nặng hơn
            # Cityscapes Foggy có 3 beta levels, 0.15 chỉ cover beta=0.005
            A.RandomFog(
                fog_coef_lower=0.05,
                fog_coef_upper=0.30,
                alpha_coef=0.08,
                p=0.2
            ),
        ]

    else:
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
        import hashlib, pathlib
    
        # Cache key dựa trên nội dung train_txt
        # → cùng dataset thì load cache, khác dataset thì tính lại
        with open(train_txt, 'rb') as f:
            txt_hash = hashlib.md5(f.read()).hexdigest()[:8]
        cache_path = pathlib.Path(train_txt).parent / f"class_weights_cache_{txt_hash}.pt"
    
        if cache_path.exists():
            # Load từ cache — gần như instant
            class_weights = torch.load(cache_path, map_location='cpu')
            print(f"✅ Class weights loaded from cache: {cache_path}")
            print(f"   (min={class_weights.min():.3f}, max={class_weights.max():.3f})")
        else:
            print(f"\n{'='*60}")
            print("📊 Computing class weights (first time, will be cached)...")
            print(f"{'='*60}\n")
    
            # Dùng num_workers từ dataloader config
            class_counts = train_dataset.get_class_distribution(
                num_workers=num_workers
            )
            total_pixels = sum(class_counts.values())
    
            print("\n📈 Class distribution:")
            print(f"{'Class':<8} {'Pixels':<15} {'Frequency':<12} {'Weight':<10}")
            print("-" * 50)
    
            raw_weights = []
            for class_id in range(19):
                count = class_counts[class_id]
                freq  = count / total_pixels if total_pixels > 0 else 0
                weight = 1.0 / (freq + 1e-8)
                raw_weights.append(weight)
                print(f"{class_id:<8} {count:<15,} {freq*100:>6.2f}%      {weight:>8.4f}")
    
            class_weights = torch.tensor(raw_weights, dtype=torch.float32)
            class_weights = torch.clamp(class_weights, min=0.1, max=50.0)
            class_weights = class_weights / class_weights.sum() * 19
    
            # Lưu cache
            torch.save(class_weights, cache_path)
            print(f"\n✅ Class weights computed & cached → {cache_path}")
            print(f"   (min={class_weights.min():.3f}, max={class_weights.max():.3f})")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # FIX: drop_last=False cho validation — không nên bỏ samples khi tính metrics
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,   # FIX: False (bản gốc True làm mất vài samples)
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
