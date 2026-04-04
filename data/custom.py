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

class PhysicalFog(ImageOnlyTransform):
    """Simulate fog theo Atmospheric Scattering Model (ASM).
 
    Công thức: I(x) = J(x) * t(x) + A * (1 - t(x))
    Trong đó:
        I(x) = foggy image (output)
        J(x) = clear image (input)
        A     = atmospheric light (~0.85–1.0 cho foggy driving)
        t(x) = transmission map = exp(-beta * d(x))
        d(x) = depth tại pixel x
        beta  = scattering coefficient (fog density)
 
    Depth được approximate bằng vertical gradient (phần dưới = gần = ít mờ,
    phần trên = xa = mờ hơn) — phù hợp với camera perspective trên xe.
 
    Khác với A.RandomFog (dùng blur đơn giản), PhysicalFog:
    - Tạo depth-dependent attenuation (xa mờ hơn gần)
    - Atmospheric light A ngả vàng/xám nhẹ (realistic hơn pure white)
    - Có thể combine với horizontal bands để simulate fog layers
 
    Args:
        beta_range (tuple): Range của scattering coefficient.
            (0.05, 0.15) = light fog, (0.15, 0.4) = dense fog.
        atm_light_range (tuple): Atmospheric light intensity (per-channel).
        depth_bias (float): Bias vertical gradient (0=uniform, 1=top-heavy).
        always_apply (bool): Always apply. Default: False.
        p (float): Probability. Default: 0.5.
    """
 
    def __init__(self,
                 beta_range=(0.08, 0.25),
                 atm_light_range=(0.80, 0.95),
                 depth_bias=0.7,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.beta_range       = beta_range
        self.atm_light_range  = atm_light_range
        self.depth_bias       = depth_bias
 
    def apply(self, img, beta=0.15, atm_light=0.9, **params):
        """
        Args:
            img: numpy array (H, W, C), dtype uint8 hoặc float32.
        """
        img_f = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.copy()
        H, W  = img_f.shape[:2]
 
        # Depth map: vertical gradient (top=far=1, bottom=near=0)
        # Thêm noise nhỏ để không quá đều
        depth = np.linspace(1.0, 0.0, H)[:, np.newaxis]          # (H, 1)
        depth = depth * self.depth_bias + np.random.uniform(0, 1 - self.depth_bias, (H, 1))
        depth = np.clip(depth, 0.0, 1.0)
        depth = np.tile(depth, (1, W))                             # (H, W)
 
        # Transmission map
        transmission = np.exp(-beta * depth * 3.0)                 # (H, W)
        transmission = transmission[:, :, np.newaxis]              # (H, W, 1)
 
        # Atmospheric light — hơi vàng ấm (RGB: ~0.9, 0.88, 0.85) để realistic
        atm_r = atm_light * np.random.uniform(0.98, 1.02)
        atm_g = atm_light * np.random.uniform(0.96, 1.00)
        atm_b = atm_light * np.random.uniform(0.90, 0.96)
        A_light = np.array([atm_r, atm_g, atm_b], dtype=np.float32).reshape(1, 1, 3)
        A_light = np.clip(A_light, 0.0, 1.0)
 
        # ASM: I = J*t + A*(1-t)
        foggy = img_f * transmission + A_light * (1.0 - transmission)
        foggy = np.clip(foggy, 0.0, 1.0)
 
        if img.dtype == np.uint8:
            return (foggy * 255).astype(np.uint8)
        return foggy.astype(np.float32)
 
    def get_params(self):
        return {
            'beta':      np.random.uniform(*self.beta_range),
            'atm_light': np.random.uniform(*self.atm_light_range),
        }
 
    def get_transform_init_args_names(self):
        return ('beta_range', 'atm_light_range', 'depth_bias')
 
 
class HazeStripe(ImageOnlyTransform):
    """Thêm horizontal haze bands — simulate fog layers ở các độ cao khác nhau.
 
    Foggy driving images thường có fog không đều theo chiều dọc:
    - Có thể có lớp mù dày ở horizon
    - Vùng gần camera thường clear hơn
    Đây là edge case mà model hay fail.
 
    Args:
        num_stripes (tuple): Range số lượng stripe. Default: (1, 3).
        stripe_width_ratio (tuple): Chiều cao stripe / H. Default: (0.05, 0.15).
        intensity_range (tuple): Fog intensity cho mỗi stripe. Default: (0.1, 0.4).
        always_apply (bool): Default: False.
        p (float): Default: 0.3.
    """
 
    def __init__(self,
                 num_stripes=(1, 3),
                 stripe_width_ratio=(0.05, 0.15),
                 intensity_range=(0.1, 0.4),
                 always_apply=False,
                 p=0.3):
        super().__init__(always_apply, p)
        self.num_stripes        = num_stripes
        self.stripe_width_ratio = stripe_width_ratio
        self.intensity_range    = intensity_range
 
    def apply(self, img, stripes=None, **params):
        if stripes is None:
            return img
        img_f = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.copy()
        H, W  = img_f.shape[:2]
 
        for (y0, y1, intensity) in stripes:
            # Fog color: xám sáng pha nhẹ xanh (atmospheric)
            fog_color  = np.array([0.92, 0.93, 0.95], dtype=np.float32)
            # Gaussian fade ở biên stripe để không bị cứng
            fade       = np.ones(y1 - y0, dtype=np.float32)
            fade_len   = max(1, (y1 - y0) // 4)
            fade_curve = np.linspace(0, 1, fade_len)
            fade[:fade_len]  = fade_curve
            fade[-fade_len:] = fade_curve[::-1]
            fade = fade[:, np.newaxis, np.newaxis] * intensity
 
            img_f[y0:y1] = img_f[y0:y1] * (1 - fade) + fog_color * fade
 
        img_f = np.clip(img_f, 0.0, 1.0)
        if img.dtype == np.uint8:
            return (img_f * 255).astype(np.uint8)
        return img_f
 
    def get_params_dependent_on_targets(self, params):
        img   = params['image']
        H, W  = img.shape[:2]
        n     = np.random.randint(*self.num_stripes) if self.num_stripes[0] < self.num_stripes[1] \
                else self.num_stripes[0]
        stripes = []
        for _ in range(n):
            w   = int(np.random.uniform(*self.stripe_width_ratio) * H)
            y0  = np.random.randint(0, max(1, H - w))
            y1  = min(H, y0 + w)
            intensity = np.random.uniform(*self.intensity_range)
            stripes.append((y0, y1, intensity))
        return {'stripes': stripes}
 
    @property
    def targets_as_params(self):
        return ['image']
 
    def get_transform_init_args_names(self):
        return ('num_stripes', 'stripe_width_ratio', 'intensity_range')

def get_foggy_specific_augmentations():
    """Trả về list augmentation thay thế foggy_specific hiện tại.
 
    So với bản cũ:
    - PhysicalFog thay A.RandomFog: ASM đúng, depth-dependent, p=0.4 (tăng 4×)
    - HazeStripe: thêm mới, simulate fog layers
    - CoarseDropout fill_value thay bằng màu fog-like (200, 200, 205)
    - RandomBrightnessContrast tăng limit từ 0.08 lên 0.15
    - Thêm A.CLAHE nhẹ: simulate contrast enhancement (camera auto-exposure)
    - Thêm A.ISONoise nhẹ: simulate noise trong foggy/low-contrast scene
    """
    return [
        # ------------------------------------------------------------------ #
        # 1. Physical fog simulation — QUAN TRỌNG NHẤT                        #
        # ------------------------------------------------------------------ #
        # PhysicalFog: depth-dependent, ASM đúng, p=0.4
        # Tăng mạnh khả năng model thấy foggy samples trong mỗi epoch
        PhysicalFog(
            beta_range=(0.06, 0.22),      # light → moderate fog
            atm_light_range=(0.78, 0.95), # atmospheric light variation
            depth_bias=0.65,              # top-heavy (xa mờ hơn)
            p=0.40,                       # ← tăng từ 0.1 lên 0.4
        ),
 
        # HazeStripe: fog không đồng đều theo chiều dọc, p=0.25
        HazeStripe(
            num_stripes=(1, 3),
            stripe_width_ratio=(0.04, 0.12),
            intensity_range=(0.08, 0.30),
            p=0.25,
        ),
 
        # Giữ A.RandomFog nhẹ như một dạng augmentation bổ sung (không thay thế)
        A.RandomFog(
            fog_coef_lower=0.05,
            fog_coef_upper=0.18,
            alpha_coef=0.06,
            p=0.15,                       # ← giữ nguyên, cộng thêm vào Physical
        ),
 
        # ------------------------------------------------------------------ #
        # 2. Robustness — CoarseDropout với fill màu fog-like                  #
        # ------------------------------------------------------------------ #
        # fill_value=(200, 200, 205) thay vì 0 (đen)
        # Foggy images không có vùng đen tự nhiên — đen là OOD với foggy val set
        A.CoarseDropout(
            max_holes=6,
            max_height=32,
            max_width=32,
            fill_value=(200, 200, 205),   # ← xám sáng thay đen
            mask_fill_value=255,
            p=0.30,
        ),
 
        # ------------------------------------------------------------------ #
        # 3. Illumination — tăng range để cover val distribution rộng hơn    #
        # ------------------------------------------------------------------ #
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,    # ← tăng từ 0.08 lên 0.15
                contrast_limit=0.15,      # ← tăng từ 0.08 lên 0.15
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(88, 112), p=1.0),  # ← mở rộng từ (95,105)
        ], p=0.35),                       # ← tăng từ 0.3 lên 0.35
 
        # CLAHE nhẹ: simulate camera auto-exposure trong foggy condition
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.15,                       # ← thêm mới
        ),
 
        # ------------------------------------------------------------------ #
        # 4. Blur — giữ nguyên, hơi tăng probability                          #
        # ------------------------------------------------------------------ #
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.20),                       # ← tăng từ 0.15 lên 0.20
 
        # ------------------------------------------------------------------ #
        # 5. Sensor noise — thêm mới                                           #
        # ------------------------------------------------------------------ #
        # ISONoise: simulate noise trong điều kiện ánh sáng thấp (foggy = low contrast)
        A.ISONoise(
            color_shift=(0.01, 0.03),
            intensity=(0.05, 0.15),
            p=0.15,                       # ← thêm mới
        ),
    ]
class CityscapesDataset(Dataset):
    """
    Universal Dataset for Cityscapes (Normal & Foggy versions)
    
    Args:
        txt_file (str): Path to txt file containing image and label paths
        transforms (callable, optional): Albumentations transforms
        img_size (tuple): Target image size (H, W)
        mean (list): Normalization mean [R, G, B]
        std (list): Normalization std [R, G, B]
        ignore_index (int): Label index to ignore (default: 255)
        label_mapping (str): Label mapping mode ('train_id' or 'id')
        dataset_type (str): 'foggy' or 'normal' - affects augmentation strategy
    
    Format of txt_file:
        Each line: image_path,label_path
        Example: /path/to/image.png,/path/to/label.png
    
    Cityscapes Label Info:
        - labelIds files contain original IDs (0-33)
        - Only 19 classes are used for training (train_id: 0-18)
        - Other classes are mapped to ignore_index (255)
    """
    
    # Cityscapes label mapping: id -> train_id
    # Reference: https://github.com/mcordts/cityscapesScripts
    ID_TO_TRAINID = {
        7: 0,   # road
        8: 1,   # sidewalk
        11: 2,  # building
        12: 3,  # wall
        13: 4,  # fence
        17: 5,  # pole
        19: 6,  # traffic light
        20: 7,  # traffic sign
        21: 8,  # vegetation
        22: 9,  # terrain
        23: 10, # sky
        24: 11, # person
        25: 12, # rider
        26: 13, # car
        27: 14, # truck
        28: 15, # bus
        31: 16, # train
        32: 17, # motorcycle
        33: 18, # bicycle
        -1: 255 # ignore
    }
    
    def __init__(
        self,
        txt_file: str,
        transforms: Optional[Callable] = None,
        img_size: Tuple[int, int] = (1024, 2048),  # Cityscapes standard
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        ignore_index: int = 255,
        label_mapping: str = 'train_id',  # 'train_id' or 'id'
        dataset_type: str = 'normal'  # 'normal' or 'foggy'
    ):
        super().__init__()
        
        self.txt_file = txt_file
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.label_mapping = label_mapping
        self.dataset_type = dataset_type
        
        # Create lookup table for fast label conversion
        self.create_label_mapping()
        
        # Read file paths
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
        
        # Set transforms
        if transforms is None:
            self.transforms = self.get_default_transforms()
        else:
            self.transforms = transforms
    
    def create_label_mapping(self):
        """Create lookup table: id -> train_id"""
        # Max ID in Cityscapes is 33, create array of 256 for safety
        self.label_map = np.ones(256, dtype=np.uint8) * self.ignore_index
        
        if self.label_mapping == 'train_id':
            # Map valid IDs to train IDs (0-18)
            for id_val, train_id in self.ID_TO_TRAINID.items():
                if train_id != 255:
                    self.label_map[id_val] = train_id
        else:
            # Use original IDs (no mapping)
            for i in range(256):
                self.label_map[i] = i
    
    def get_default_transforms(self) -> A.Compose:
        """Default augmentation pipeline"""
        return A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (3, H, W) normalized tensor
            label: (H, W) long tensor with train_id indices (0-18, 255 for ignore)
        """
        img_path, label_path = self.samples[idx]
        
        # Load image (RGB)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load label (grayscale) - contains original IDs
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        
        # ✅ CRITICAL: Convert ID to train_id using lookup table
        # This maps: 7->0, 8->1, ..., 33->18, others->255
        label = self.label_map[label]
        
        # Apply transforms
        transformed = self.transforms(image=image, mask=label)
        image = transformed['image']  # (3, H, W)
        label = transformed['mask']   # (H, W)
        
        # Convert label to tensor if not already
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        return image, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Compute class distribution in dataset (useful for class weights)
        
        Returns:
            Dict mapping train_id -> pixel count
        """
        print("📊 Computing class distribution...")
        class_counts = {i: 0 for i in range(19)}
        
        for idx in tqdm(range(len(self)), desc="Scanning labels"):
            _, label_path = self.samples[idx]
            label = Image.open(label_path)
            label = np.array(label, dtype=np.uint8)
            label = self.label_map[label]
            
            # Count pixels per class
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
    base_list = [
        A.RandomScale(scale_limit=0.4, p=1.0),

        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_REFLECT_101,
            value=0,
            mask_value=255,
            p=1.0
        ),

        A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),

        A.HorizontalFlip(p=0.5),

        # ❌ REMOVE ROTATE (rất quan trọng cho Cityscapes)
        # A.ShiftScaleRotate(...)
    ]

    # ===== DATASET-SPECIFIC =====
    if dataset_type == 'foggy':
        foggy_specific = get_foggy_specific_augmentations()

    else:
        # NORMAL dataset → augment mạnh hơn
        foggy_specific = [

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
        + foggy_specific
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
        A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_LINEAR),
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
    img_size: Tuple[int, int] = (512, 1024),  # Smaller for training speed
    pin_memory: bool = True,
    compute_class_weights: bool = False,
    dataset_type: str = 'normal'  # 'normal' or 'foggy'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.Tensor]]:
    """
    Create train and validation dataloaders
    
    Args:
        train_txt: Path to training txt file
        val_txt: Path to validation txt file
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Target image size (H, W)
        pin_memory: Whether to pin memory for faster GPU transfer
        compute_class_weights: Whether to compute class weights for balancing
        dataset_type: 'normal' or 'foggy' - affects augmentation strategy
    
    Returns:
        (train_loader, val_loader, class_weights)
    """
    print(f"\n{'='*60}")
    print(f"🚀 Creating DataLoaders for {dataset_type.upper()} Cityscapes")
    print(f"{'='*60}\n")
    
    # Create datasets
    train_dataset = CityscapesDataset(
        txt_file=train_txt,
        transforms=get_train_transforms(img_size=img_size, dataset_type=dataset_type),
        img_size=img_size,
        label_mapping='train_id',  # Use train_id mapping
        dataset_type=dataset_type
    )
    
    print()  # Spacing
    
    val_dataset = CityscapesDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=img_size),
        img_size=img_size,
        label_mapping='train_id',  # Use train_id mapping
        dataset_type=dataset_type
    )
    
    # Compute class weights (optional)
    class_weights = None
    if compute_class_weights:
        print(f"\n{'='*60}")
        print("📊 Computing class weights for balanced training...")
        print(f"{'='*60}\n")
        
        class_counts = train_dataset.get_class_distribution()
        
        # Convert to weights: inverse frequency
        total_pixels = sum(class_counts.values())
        class_weights = []
        
        print("\n📈 Class distribution:")
        print(f"{'Class':<8} {'Pixels':<15} {'Frequency':<12} {'Weight':<10}")
        print("-" * 50)
        
        for class_id in range(19):
            count = class_counts[class_id]
            freq = count / total_pixels if total_pixels > 0 else 0
            weight = 1.0 / (freq + 1e-8)
            class_weights.append(weight)
            print(f"{class_id:<8} {count:<15,} {freq*100:>6.2f}%      {weight:>8.4f}")
        
        # Normalize weights
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = torch.clamp(class_weights, min=0.1, max=50.0)
        class_weights = class_weights / class_weights.sum() * 19
        
        print(f"\n✅ Class weights normalized (mean=1.0, max clipped to 50x)")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
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


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage for both Normal and Foggy Cityscapes
    """
    
    print("\n" + "="*70)
    print("🎯 CITYSCAPES DATASET LOADER - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # ========================================
    # OPTION 1: NORMAL CITYSCAPES
    # ========================================
    print("📌 OPTION 1: Training on NORMAL Cityscapes")
    print("-" * 70)
    
    train_loader_normal, val_loader_normal, weights_normal = create_dataloaders(
        train_txt='data/cityscapes_train.txt',
        val_txt='data/cityscapes_val.txt',
        batch_size=4,
        num_workers=4,
        img_size=(512, 1024),
        compute_class_weights=True,
        dataset_type='normal'  # ← NORMAL dataset
    )
    
    # ========================================
    # OPTION 2: FOGGY CITYSCAPES
    # ========================================
    print("\n📌 OPTION 2: Training on FOGGY Cityscapes")
    print("-" * 70)
    
    train_loader_foggy, val_loader_foggy, weights_foggy = create_dataloaders(
        train_txt='data/cityscapes_foggy_train.txt',
        val_txt='data/cityscapes_foggy_val.txt',
        batch_size=4,
        num_workers=4,
        img_size=(512, 1024),
        compute_class_weights=True,
        dataset_type='foggy'  # ← FOGGY dataset
    )
    
    # ========================================
    # TEST: Load one batch from each
    # ========================================
    print("\n" + "="*70)
    print("🧪 TESTING: Loading sample batches")
    print("="*70 + "\n")
    
    # Test normal dataset
    images, labels = next(iter(train_loader_normal))
    print(f"✅ Normal Cityscapes batch:")
    print(f"   Images: {images.shape} | min: {images.min():.3f}, max: {images.max():.3f}")
    print(f"   Labels: {labels.shape} | unique: {labels.unique().tolist()}")
    
    # Test foggy dataset
    images, labels = next(iter(train_loader_foggy))
    print(f"\n✅ Foggy Cityscapes batch:")
    print(f"   Images: {images.shape} | min: {images.min():.3f}, max: {images.max():.3f}")
    print(f"   Labels: {labels.shape} | unique: {labels.unique().tolist()}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70 + "\n")
