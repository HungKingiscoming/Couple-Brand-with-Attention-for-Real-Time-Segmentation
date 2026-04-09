import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Callable, Optional, Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import albumentations.augmentations.functional as AF
import cv2


# =============================================================================
# Fog Synthesis Transform (physics-based atmospheric scattering)
# =============================================================================

class FogSynthesis(A.ImageOnlyTransform):
    """Atmospheric scattering model: I_fog = I * t(d) + A * (1 - t(d))
    t(d) = exp(-beta * d * depth_scale)

    Depth map được sinh theo prior của driving scene:
      - Phần trên ảnh (sky/far) → depth lớn → fog nặng
      - Phần dưới ảnh (road/near) → depth nhỏ → fog nhẹ
    Thêm noise ngẫu nhiên để tránh artifact quá pattern.

    Args:
        beta_range: Scattering coefficient (0.05=nhẹ, 0.5=nặng).
        atm_light_range: Atmospheric light intensity (gần trắng ~0.85-1.0).
        depth_scale: Nhân với depth map trước khi tính transmission.
    """

    def __init__(
        self,
        beta_range: Tuple[float, float] = (0.05, 0.45),
        atm_light_range: Tuple[float, float] = (0.80, 0.98),
        depth_scale: float = 8.0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.beta_range       = beta_range
        self.atm_light_range  = atm_light_range
        self.depth_scale      = depth_scale

    def apply(self, img: np.ndarray, beta: float = 0.15,
              atm_light: float = 0.9, **params) -> np.ndarray:
        H, W = img.shape[:2]
        img_f = img.astype(np.float32) / 255.0

        # Depth prior: tuyến tính từ trên (xa) → dưới (gần)
        y = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None]
        depth = np.tile(y, (1, W))

        # Thêm noise không gian để fog không quá đều
        noise = np.random.uniform(0.0, 0.25, (H, W)).astype(np.float32)
        depth = np.clip(depth + noise, 0.0, 1.0)

        transmission = np.exp(-beta * depth * self.depth_scale)[:, :, None]  # (H,W,1)
        A_val = atm_light
        foggy = img_f * transmission + A_val * (1.0 - transmission)
        return np.clip(foggy * 255.0, 0.0, 255.0).astype(np.uint8)

    def get_params(self) -> dict:
        return {
            'beta':      np.random.uniform(*self.beta_range),
            'atm_light': np.random.uniform(*self.atm_light_range),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ('beta_range', 'atm_light_range', 'depth_scale')


# =============================================================================
# Dataset
# =============================================================================

class CityscapesDataset(Dataset):
    """Cityscapes / Foggy-Cityscapes dataset.

    txt_file format (one sample per line):
        /path/to/image.png,/path/to/label.png

    Label files chứa original Cityscapes IDs (0-33).
    Lookup table ánh xạ sang train_id (0-18, 255=ignore) được tạo 1 lần.
    """

    # id → train_id  (Cityscapes official)
    ID_TO_TRAINID: Dict[int, int] = {
        7:  0,   # road
        8:  1,   # sidewalk
        11: 2,   # building
        12: 3,   # wall
        13: 4,   # fence
        17: 5,   # pole
        19: 6,   # traffic light
        20: 7,   # traffic sign
        21: 8,   # vegetation
        22: 9,   # terrain
        23: 10,  # sky
        24: 11,  # person
        25: 12,  # rider
        26: 13,  # car
        27: 14,  # truck
        28: 15,  # bus
        31: 16,  # train
        32: 17,  # motorcycle
        33: 18,  # bicycle
    }

    def __init__(
        self,
        txt_file: str,
        transforms: Optional[Callable] = None,
        img_size: Tuple[int, int] = (512, 1024),
        mean: List[float] = [0.485, 0.456, 0.406],
        std:  List[float] = [0.229, 0.224, 0.225],
        ignore_index: int = 255,
        dataset_type: str = 'foggy',
    ):
        super().__init__()
        self.img_size     = img_size
        self.mean         = mean
        self.std          = std
        self.ignore_index = ignore_index
        self.dataset_type = dataset_type

        # Lookup table (256 entries, uint8)
        self.label_map = np.full(256, ignore_index, dtype=np.uint8)
        for id_val, train_id in self.ID_TO_TRAINID.items():
            self.label_map[id_val] = train_id

        # Load sample paths
        self.samples: List[Tuple[str, str]] = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    img_p, lbl_p = line.split(',', 1)
                    self.samples.append((img_p.strip(), lbl_p.strip()))

        print(f"Loaded {len(self.samples)} samples | type={dataset_type.upper()} | size={img_size}")

        self.transforms = transforms if transforms is not None \
                          else self._default_val_transforms()

    def _default_val_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1],
                     interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(lbl_path), dtype=np.uint8)
        label = self.label_map[label]   # id → train_id, O(1) lookup

        out   = self.transforms(image=image, mask=label)
        image = out['image']   # (3, H, W) float tensor
        label = out['mask']

        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)
        return image, label.long()

    def get_class_distribution(self) -> Dict[int, int]:
        counts = {i: 0 for i in range(19)}
        for _, lbl_path in tqdm(self.samples, desc='Scanning labels'):
            lbl = self.label_map[np.array(Image.open(lbl_path), dtype=np.uint8)]
            for c in range(19):
                counts[c] += int((lbl == c).sum())
        return counts


# =============================================================================
# Augmentation presets
# =============================================================================

def get_train_transforms(
    img_size: Tuple[int, int] = (512, 1024),
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
    dataset_type: str = 'foggy',
) -> A.Compose:
    """Training augmentation pipeline.

    Foggy-specific changes vs baseline:
      1. FogSynthesis (physics-based) p=0.45  ← quan trọng nhất
      2. RandomFog (albumentations)   p=0.25  ← bổ sung
      3. Desaturation bias (fog giảm sat/contrast)
      4. RandomScale p=0.5 (không phải p=1.0) → model thấy native scale
      5. Loại CoarseDropout (không phù hợp driving)
    """

    # --- Geometric core (cả foggy lẫn normal) ---
    geo = [
        # p=0.5: 50% sample ở native scale, 50% bị rescale
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),

        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_REFLECT_101,
            value=0,
            mask_value=255,
            p=1.0,
        ),
        A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
    ]

    if dataset_type == 'foggy':
        # ------------------------------------------------------------------ #
        # FOG SIMULATION — đây là phần bù đắp quan trọng nhất.               #
        # FoggyAwareNorm và DWSA chỉ học được khi thường xuyên thấy fog.     #
        # ------------------------------------------------------------------ #
        fog_aug = [
            # Physics-based scattering — realistic hơn albumentations built-in
            FogSynthesis(
                beta_range=(0.05, 0.45),
                atm_light_range=(0.80, 0.98),
                depth_scale=8.0,
                p=0.45,
            ),
            # albumentations RandomFog bổ sung (style khác)
            A.RandomFog(
                fog_coef_lower=0.10,
                fog_coef_upper=0.50,
                alpha_coef=0.10,
                p=0.25,
            ),
        ]

        # ------------------------------------------------------------------ #
        # COLOR — bias về low-contrast / desaturate để khớp foggy domain     #
        # ------------------------------------------------------------------ #
        color_aug = [
            A.OneOf([
                # Desaturate + giảm contrast (đặc trưng của fog haze)
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=(-50, 5),   # bias âm → desaturate
                    val_shift_limit=(-20, 15),
                    p=1.0,
                ),
                # Brightness/contrast với bias về tối hơn
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.25, 0.10),
                    contrast_limit=(-0.30, 0.10),
                    p=1.0,
                ),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.02,
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(80, 115), p=1.0),
            ], p=0.65),
        ]

        # ------------------------------------------------------------------ #
        # BLUR — mô phỏng scattering và haze                                  #
        # ------------------------------------------------------------------ #
        blur_aug = [
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.GlassBlur(sigma=0.4, max_delta=1, iterations=1, p=1.0),
            ], p=0.30),
        ]

        # ------------------------------------------------------------------ #
        # NOISE — sensor noise dưới điều kiện tầm nhìn thấp                  #
        # ------------------------------------------------------------------ #
        noise_aug = [
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.04),
                           intensity=(0.05, 0.20), p=1.0),
            ], p=0.20),
        ]

        specific = fog_aug + color_aug + blur_aug + noise_aug

    else:
        # Normal Cityscapes — augment mạnh hơn vì không có domain shift
        specific = [
            A.OneOf([
                A.ColorJitter(brightness=0.15, contrast=0.15,
                              saturation=0.15, hue=0.05, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.25,
                                           contrast_limit=0.25, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.60),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.15),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.15),
        ]

    normalize = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(geo + specific + normalize)


def get_val_transforms(
    img_size: Tuple[int, int] = (512, 1024),
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1],
                 interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# =============================================================================
# DataLoader factory
# =============================================================================

def create_dataloaders(
    train_txt: str,
    val_txt: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 1024),
    pin_memory: bool = True,
    compute_class_weights: bool = False,
    dataset_type: str = 'foggy',
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:

    print(f"\n{'='*60}")
    print(f"Creating DataLoaders — {dataset_type.upper()} Cityscapes")
    print(f"{'='*60}\n")

    train_ds = CityscapesDataset(
        txt_file=train_txt,
        transforms=get_train_transforms(img_size=img_size,
                                         dataset_type=dataset_type),
        img_size=img_size,
        dataset_type=dataset_type,
    )

    val_ds = CityscapesDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=img_size),
        img_size=img_size,
        dataset_type=dataset_type,
    )

    # --- Class weights (inverse-frequency) ---
    class_weights = None
    if compute_class_weights:
        print("Computing class weights...")
        counts   = train_ds.get_class_distribution()
        total_px = sum(counts.values())
        w = []
        for c in range(19):
            freq = counts[c] / (total_px + 1e-8)
            w.append(1.0 / (freq + 1e-8))
        class_weights = torch.tensor(w, dtype=torch.float32)
        class_weights = torch.clamp(class_weights, 1.0, 50.0)
        class_weights = class_weights / class_weights.mean()   # mean ≈ 1
        print("Class weights computed.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,              # OK cho train — ổn định BN
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,             # FIX: False để tính metric trên toàn bộ val set
        persistent_workers=num_workers > 0,
    )

    print(f"\n{'='*60}")
    print(f"Train: {len(train_ds):,} samples | {len(train_loader)} batches")
    print(f"Val:   {len(val_ds):,} samples  | {len(val_loader)} batches")
    print(f"Batch size: {batch_size} | Workers: {num_workers} | Size: {img_size}")
    print(f"{'='*60}\n")

    return train_loader, val_loader, class_weights
