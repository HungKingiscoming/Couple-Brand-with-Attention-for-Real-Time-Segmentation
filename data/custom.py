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
from albumentations.core.transforms_interface import ImageOnlyTransform


# ==============================================================================
# CUSTOM IMAGE-ONLY TRANSFORMS
# ==============================================================================

class PhysicalFog(ImageOnlyTransform):
    """Simulate fog theo Atmospheric Scattering Model (ASM).

    I(x) = J(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))

    ImageOnlyTransform → không bao giờ chạm mask.
    """

    def __init__(self,
                 beta_range=(0.08, 0.25),
                 atm_light_range=(0.80, 0.95),
                 depth_bias=0.7,
                 always_apply=False,
                 p=0.5):
        super().__init__(p=p)
        self.beta_range      = beta_range
        self.atm_light_range = atm_light_range
        self.depth_bias      = depth_bias

    def apply(self, img, beta=0.15, atm_light=0.9, **params):
        img_f = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.copy()
        H, W  = img_f.shape[:2]

        depth = np.linspace(1.0, 0.0, H)[:, np.newaxis]
        noise_range = max(1e-6, 1.0 - self.depth_bias)
        depth = depth * self.depth_bias + np.random.uniform(0, noise_range, (H, 1))
        depth = np.clip(depth, 0.0, 1.0)
        depth = np.tile(depth, (1, W))

        transmission = np.exp(-beta * depth * 3.0)[:, :, np.newaxis]

        atm_r = float(np.clip(atm_light * np.random.uniform(0.98, 1.02), 0, 1))
        atm_g = float(np.clip(atm_light * np.random.uniform(0.96, 1.00), 0, 1))
        atm_b = float(np.clip(atm_light * np.random.uniform(0.90, 0.96), 0, 1))
        A_light = np.array([atm_r, atm_g, atm_b], dtype=np.float32).reshape(1, 1, 3)

        foggy = img_f * transmission + A_light * (1.0 - transmission)
        foggy = np.clip(foggy, 0.0, 1.0)

        return (foggy * 255).astype(np.uint8) if img.dtype == np.uint8 else foggy.astype(np.float32)

    def get_params(self):
        return {
            'beta':      float(np.random.uniform(*self.beta_range)),
            'atm_light': float(np.random.uniform(*self.atm_light_range)),
        }

    def get_transform_init_args_names(self):
        return ('beta_range', 'atm_light_range', 'depth_bias')


class HazeStripe(ImageOnlyTransform):
    """Horizontal haze bands — image-only, không chạm mask."""

    def __init__(self,
                 num_stripes=(1, 3),
                 stripe_width_ratio=(0.05, 0.15),
                 intensity_range=(0.1, 0.4),
                 always_apply=False,
                 p=0.3):
        super().__init__(p=p)
        self.num_stripes        = num_stripes
        self.stripe_width_ratio = stripe_width_ratio
        self.intensity_range    = intensity_range

    def apply(self, img, stripes=None, **params):
        if not stripes:
            return img
        img_f     = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.copy()
        H, W      = img_f.shape[:2]
        fog_color = np.array([0.92, 0.93, 0.95], dtype=np.float32)

        for (y0, y1, intensity) in stripes:
            h    = y1 - y0
            fade = np.ones(h, dtype=np.float32)
            fl   = max(1, h // 4)
            ramp = np.linspace(0, 1, fl)
            fade[:fl]  = ramp
            fade[-fl:] = ramp[::-1]
            fade = fade[:, np.newaxis, np.newaxis] * intensity
            img_f[y0:y1] = img_f[y0:y1] * (1 - fade) + fog_color * fade

        img_f = np.clip(img_f, 0.0, 1.0)
        return (img_f * 255).astype(np.uint8) if img.dtype == np.uint8 else img_f

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        H   = img.shape[0]
        lo, hi = self.num_stripes
        n   = int(np.random.randint(lo, max(lo + 1, hi)))
        stripes = []
        for _ in range(n):
            w  = max(2, int(np.random.uniform(*self.stripe_width_ratio) * H))
            y0 = int(np.random.randint(0, max(1, H - w)))
            y1 = min(H, y0 + w)
            stripes.append((y0, y1, float(np.random.uniform(*self.intensity_range))))
        return {'stripes': stripes}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_stripes', 'stripe_width_ratio', 'intensity_range')


# ==============================================================================
# AUGMENTATION PIPELINES — compatible với albumentations 2.x
# ==============================================================================

def get_foggy_specific_augmentations() -> list:
    """Image-only foggy augmentations. Không bao giờ tác động mask."""
    return [
        # ---- Physical fog (ASM) ----------------------------------------- #
        PhysicalFog(
            beta_range=(0.06, 0.22),
            atm_light_range=(0.78, 0.95),
            depth_bias=0.65,
            p=0.40,
        ),
        HazeStripe(
            num_stripes=(1, 3),
            stripe_width_ratio=(0.04, 0.12),
            intensity_range=(0.08, 0.30),
            p=0.25,
        ),
        # albumentations 2.x: fog_coef_range thay vì fog_coef_lower/upper
        A.RandomFog(
            fog_coef_range=(0.05, 0.18),
            alpha_coef=0.06,
            p=0.15,
        ),
        # ---- Robustness -------------------------------------------------- #
        # albumentations 2.x: num_holes_range, hole_height_range, hole_width_range
        # fill thay vì fill_value; fill_mask thay vì mask_fill_value
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=(200, 200, 205),      # fog-like gray
            fill_mask=255,             # ignore_index
            p=0.30,
        ),
        # ---- Illumination ------------------------------------------------ #
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(88, 112), p=1.0),
        ], p=0.35),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
        # ---- Blur -------------------------------------------------------- #
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.20),
        # ---- Sensor noise ------------------------------------------------ #
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.15),
    ]


def get_train_transforms(
    img_size: Tuple[int, int] = (512, 1024),
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
    dataset_type: str = 'normal',
) -> A.Compose:
    """
    ĐIỂM KHÁC BIỆT CHÍNH so với bản cũ (nguyên nhân gây mIoU=0.06):
    ─────────────────────────────────────────────────────────────────
    1. RandomScale: thêm mask_interpolation=cv2.INTER_NEAREST
       → mask không bị bilinear interpolation → không tạo label trung gian

    2. PadIfNeeded: border_mode=CONSTANT, fill_mask=255
       → vùng padding mask = ignore_index thay vì reflect label ngẫu nhiên

    3. API update cho albumentations 2.x:
       - PadIfNeeded: fill/fill_mask thay vì value/mask_value
       - CoarseDropout: num_holes_range/hole_height_range/... thay vì max_holes/...
       - RandomFog: fog_coef_range thay vì fog_coef_lower/upper
    """

    # ---- Geometric: tác động cả image lẫn mask ---- #
    geometric = [
        A.RandomScale(
            scale_limit=0.4,
            interpolation=cv2.INTER_LINEAR,         # image: linear OK
            mask_interpolation=cv2.INTER_NEAREST,   # mask: NEAREST — không blend label
            p=1.0,
        ),
        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,        # CONSTANT thay REFLECT
            fill=0,                                 # image padding = 0
            fill_mask=255,                          # mask padding = ignore_index
            p=1.0,
        ),
        A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
    ]

    # ---- Dataset-specific image-only augmentations ---- #
    if dataset_type == 'foggy':
        appearance = get_foggy_specific_augmentations()
    else:
        appearance = [
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                fill_mask=255,
                p=0.3,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.1),
            A.OneOf([
                A.ColorJitter(
                    brightness=0.1, contrast=0.1,
                    saturation=0.1, hue=0.05, p=1.0,
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0,
                ),
            ], p=0.5),
        ]

    final = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(geometric + appearance + final)


def get_val_transforms(
    img_size: Tuple[int, int] = (512, 1024),
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    return A.Compose([
        A.Resize(
            height=img_size[0],
            width=img_size[1],
            interpolation=cv2.INTER_LINEAR,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ==============================================================================
# DATASET
# ==============================================================================

class CityscapesDataset(Dataset):
    """Cityscapes dataset (Normal & Foggy)."""

    ID_TO_TRAINID = {
        7: 0,  8: 1,  11: 2,  12: 3,  13: 4,
        17: 5, 19: 6, 20: 7,  21: 8,  22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18,
    }

    def __init__(
        self,
        txt_file: str,
        transforms: Optional[Callable] = None,
        img_size: Tuple[int, int] = (512, 1024),
        mean: List[float] = [0.485, 0.456, 0.406],
        std:  List[float] = [0.229, 0.224, 0.225],
        ignore_index: int = 255,
        label_mapping: str = 'train_id',
        dataset_type: str = 'normal',
        debug_label: bool = False,
    ):
        super().__init__()
        self.txt_file      = txt_file
        self.img_size      = img_size
        self.mean          = mean
        self.std           = std
        self.ignore_index  = ignore_index
        self.label_mapping = label_mapping
        self.dataset_type  = dataset_type
        self.debug_label   = debug_label

        self._build_label_map()

        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    self.samples.append((parts[0].strip(), parts[1].strip()))

        print(f"Loaded {len(self.samples)} samples | type={dataset_type} | mapping={label_mapping}")
        self.transforms = transforms if transforms is not None else self._default_transforms()

    def _build_label_map(self):
        self.label_map = np.full(256, self.ignore_index, dtype=np.uint8)
        if self.label_mapping == 'train_id':
            for orig_id, train_id in self.ID_TO_TRAINID.items():
                if 0 <= orig_id < 256:
                    self.label_map[orig_id] = train_id
        else:
            for i in range(256):
                self.label_map[i] = i

    def _default_transforms(self) -> A.Compose:
        return A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        image     = np.array(Image.open(img_path).convert('RGB'))
        label_raw = np.array(Image.open(label_path), dtype=np.uint8)
        label     = self.label_map[label_raw]   # uint8: {0..18, 255}

        result = self.transforms(image=image, mask=label)
        image  = result['image']                # (3, H, W) float tensor
        label  = result['mask'].long()          # (H, W) int64

        if self.debug_label:
            uniq = label.unique().tolist()
            bad  = [v for v in uniq if v not in list(range(19)) + [255]]
            if bad:
                raise ValueError(
                    f"[DEBUG] idx={idx} — label values ngoài range: {bad}\n"
                    f"  img={img_path}\n  raw unique={np.unique(label_raw).tolist()}"
                )

        return image, label

    def get_class_distribution(self) -> Dict[int, int]:
        print("Computing class distribution...")
        class_counts = {i: 0 for i in range(19)}
        for idx in tqdm(range(len(self)), desc="Scanning labels"):
            _, label_path = self.samples[idx]
            label = self.label_map[np.array(Image.open(label_path), dtype=np.uint8)]
            for c in range(19):
                class_counts[c] += int((label == c).sum())
        return class_counts


# ==============================================================================
# DATALOADER
# ==============================================================================

def create_dataloaders(
    train_txt: str,
    val_txt: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 1024),
    pin_memory: bool = True,
    compute_class_weights: bool = False,
    dataset_type: str = 'normal',
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.Tensor]]:

    print(f"\n{'='*60}")
    print(f"Creating DataLoaders — {dataset_type.upper()} Cityscapes")
    print(f"{'='*60}\n")

    train_dataset = CityscapesDataset(
        txt_file=train_txt,
        transforms=get_train_transforms(img_size=img_size, dataset_type=dataset_type),
        img_size=img_size,
        label_mapping='train_id',
        dataset_type=dataset_type,
    )

    val_dataset = CityscapesDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=img_size),
        img_size=img_size,
        label_mapping='train_id',
        dataset_type=dataset_type,
    )

    class_weights = None
    if compute_class_weights:
        print("Computing class weights...")
        class_counts = train_dataset.get_class_distribution()
        total        = sum(class_counts.values())
        raw          = [1.0 / (class_counts[c] / total + 1e-8) for c in range(19)]
        class_weights = torch.tensor(raw, dtype=torch.float32)
        class_weights = torch.clamp(class_weights, min=0.1, max=50.0)
        class_weights = class_weights / class_weights.sum() * 19
        print(f"Class weights: mean={class_weights.mean():.3f}, "
              f"min={class_weights.min():.3f}, max={class_weights.max():.3f}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )

    print(f"\n{'='*60}")
    print(f"Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_dataset):,} samples ({len(val_loader)} batches)")
    print(f"Batch={batch_size} | Workers={num_workers} | Size={img_size}")
    print(f"{'='*60}\n")

    return train_loader, val_loader, class_weights
