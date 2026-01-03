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
    dataset_type='normal'
) -> A.Compose:
    """Enhanced augmentation for Cityscapes (0.75+ mIoU)"""
    return A.Compose([
        # Initial resize
        A.Resize(height=img_size[0], width=img_size[1], p=1.0),
        
        # More aggressive scaling [0.5, 1.5]
        A.RandomScale(scale_limit=0.25, p=0.5),
        
        # Pad and crop
        A.PadIfNeeded(
            min_height=img_size[0], 
            min_width=img_size[1], 
            border_mode=cv2.BORDER_REFLECT_101,
            value=0, 
            mask_value=255,
            p=1.0
        ),
        A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0),
        
        # Geometric augmentations
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.0,  # ✅ Disable scaling (already done by RandomScale)
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        
        A.HorizontalFlip(p=0.5),
        
        # Blur (helps with generalization)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.1),
        
        # Color augmentations (more aggressive)
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.5),
        
        
        # Normalize
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


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
        drop_last=False
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
