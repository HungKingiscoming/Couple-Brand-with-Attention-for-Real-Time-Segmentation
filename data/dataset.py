import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Tuple, List
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
    
    Format of txt_file:
        Each line: image_path,label_path
        Example: /path/to/image.png,/path/to/label.png
    """
    
    def __init__(
        self,
        txt_file: str,
        transforms: Optional[Callable] = None,
        img_size: Tuple[int, int] = (1024, 2048),  # Cityscapes standard
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        ignore_index: int = 255
    ):
        super().__init__()
        
        self.txt_file = txt_file
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        
        # Read file paths
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label_path = line.split(',')
                    self.samples.append((img_path, label_path))
        
        print(f"Loaded {len(self.samples)} samples from {txt_file}")
        
        # Set transforms
        if transforms is None:
            self.transforms = self.get_default_transforms()
        else:
            self.transforms = transforms
    
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
            label: (H, W) long tensor with class indices
        """
        img_path, label_path = self.samples[idx]
        
        # Load image (RGB)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load label (grayscale)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.int64)
        
        # Apply transforms
        transformed = self.transforms(image=image, mask=label)
        image = transformed['image']  # (3, H, W)
        label = transformed['mask']   # (H, W)
        
        # Convert label to tensor if not already
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        return image, label


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
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_txt: Path to training txt file
        val_txt: Path to validation txt file
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Target image size (H, W)
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CityscapesCustomDataset(
        txt_file=train_txt,
        transforms=get_train_transforms(img_size=img_size),
        img_size=img_size
    )
    
    val_dataset = CityscapesCustomDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=img_size),
        img_size=img_size
    )
    
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
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


