from .base import BaseSegmentationDataset
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(BaseSegmentationDataset):
    """
    Custom dataset for your own data
    
    Simple structure:
        root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    
    Or with split file:
        root/
        ├── images/
        ├── labels/
        └── splits/
            ├── train.txt
            ├── val.txt
            └── test.txt
    
    Example:
        # Define your classes
        CUSTOM_CLASSES = ('background', 'car', 'person', 'road')
        
        # Create dataset
        dataset = CustomDataset(
            root='/data/my_dataset',
            split='train',
            classes=CUSTOM_CLASSES,
            transform=create_training_pipeline()
        )
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        classes: Optional[Tuple[str, ...]] = None,
        palette: Optional[List[List[int]]] = None,
        image_ext: str = '.jpg',
        label_ext: str = '.png',
        split_file: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            classes: Tuple of class names
            palette: List of RGB colors (auto-generated if None)
            image_ext: Image file extension
            label_ext: Label file extension
            split_file: Optional path to split file (txt with filenames)
        """
        # Set metadata
        if classes is not None:
            num_classes = len(classes)
            if palette is None:
                # Auto-generate palette
                import random
                random.seed(42)
                palette = [[random.randint(0, 255) for _ in range(3)] 
                          for _ in range(num_classes)]
            
            self.METAINFO = {
                'classes': classes,
                'palette': palette,
                'num_classes': num_classes,
                'ignore_index': 255,
            }
        
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.split_file = split_file
        
        super().__init__(root, split, **kwargs)
    
    def _load_samples(self) -> List[Dict]:
        """Load custom dataset samples"""
        samples = []
        
        # Method 1: Use split file
        if self.split_file is not None:
            split_path = self.root / self.split_file
            if not split_path.exists():
                split_path = Path(self.split_file)
            
            with open(split_path, 'r') as f:
                filenames = [line.strip() for line in f.readlines()]
            
            image_dir = self.root / 'images'
            label_dir = self.root / 'labels'
            
            for name in filenames:
                # Remove extension if present
                name = name.replace(self.image_ext, '').replace(self.label_ext, '')
                
                img_path = image_dir / f'{name}{self.image_ext}'
                label_path = label_dir / f'{name}{self.label_ext}'
                
                if img_path.exists() and label_path.exists():
                    samples.append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'name': name
                    })
        
        # Method 2: Use directory structure
        else:
            image_dir = self.root / 'images' / self.split
            label_dir = self.root / 'labels' / self.split
            
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
            for img_path in sorted(image_dir.glob(f'*{self.image_ext}')):
                label_path = label_dir / img_path.name.replace(
                    self.image_ext, self.label_ext
                )
                
                if label_path.exists():
                    samples.append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'name': img_path.stem
                    })
        
        return samples