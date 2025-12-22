import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CityscapesCustomDataset(Dataset):
    """
    Custom Dataset for Cityscapes foggy images with label files
    
    Args:
        txt_file (str): Path to txt file containing image and label paths
        transforms (callable, optional): Albumentations transforms
        img_size (tuple): Target image size (H, W)
        mean (list): Normalization mean [R, G, B]
        std (list): Normalization std [R, G, B]
        ignore_index (int): Label index to ignore (default: 255)
        label_mapping (str): Label mapping mode ('train_id' or 'id')
    
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
        label_mapping: str = 'train_id'  # 'train_id' or 'id'
    ):
        super().__init__()
        
        self.txt_file = txt_file
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.label_mapping = label_mapping
        
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
        
        print(f"Loaded {len(self.samples)} samples from {txt_file}")
        print(f"Label mapping mode: {self.label_mapping}")
        print(f"Valid training classes: 19 (0-18)")
        print(f"Ignore index: {self.ignore_index}")
        
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
        print("Computing class distribution...")
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
    img_size: Tuple[int, int] = (1024, 2048),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Training augmentation pipeline with strong augmentations
    
    Args:
        img_size: Target size (H, W)
        mean: Normalization mean
        std: Normalization std
    """
    return A.Compose([
        # Resize
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        # Color augmentations (important for foggy images)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=25,
                g_shift_limit=25,
                b_shift_limit=25,
                p=1.0
            ),
        ], p=0.7),
        
        # Blur and noise (simulate fog variations)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
        
        # Normalize and convert to tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_val_transforms(
    img_size: Tuple[int, int] = (1024, 2048),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Validation/Test augmentation pipeline (no augmentation)
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
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
    compute_class_weights: bool = False
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
    
    Returns:
        (train_loader, val_loader, class_weights)
    """
    # Create datasets
    train_dataset = CityscapesCustomDataset(
        txt_file=train_txt,
        transforms=get_train_transforms(img_size=img_size),
        img_size=img_size,
        label_mapping='train_id'  # Use train_id mapping
    )
    
    val_dataset = CityscapesCustomDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=img_size),
        img_size=img_size,
        label_mapping='train_id'  # Use train_id mapping
    )
    
    # Compute class weights (optional)
    class_weights = None
    if compute_class_weights:
        print("\n📊 Computing class weights for balanced training...")
        class_counts = train_dataset.get_class_distribution()
        
        # Convert to weights: inverse frequency
        total_pixels = sum(class_counts.values())
        class_weights = []
        
        print("\nClass distribution:")
        for class_id in range(19):
            count = class_counts[class_id]
            freq = count / total_pixels if total_pixels > 0 else 0
            weight = 1.0 / (freq + 1e-8)
            class_weights.append(weight)
            print(f"  Class {class_id:2d}: {count:12d} pixels ({freq*100:5.2f}%) -> weight: {weight:.4f}")
        
        # Normalize weights
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * 19  # Normalize to mean=1
        
        print(f"\n✓ Class weights computed: {class_weights.tolist()}")
    
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
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, class_weights


