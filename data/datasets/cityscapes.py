from .base import BaseSegmentationDataset
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CityscapesDataset(BaseSegmentationDataset):
    """
    Cityscapes Dataset
    
    Structure:
        root/
        ├── leftImg8bit/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── gtFine/
            ├── train/
            ├── val/
            └── test/
    
    Example:
        dataset = CityscapesDataset(
            root='/data/cityscapes',
            split='train',
            transform=create_training_pipeline()
        )
    """
    
    METAINFO = {
        'classes': (
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ),
        'palette': [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ],
        'num_classes': 19,
        'ignore_index': 255,
    }
    
    def _load_samples(self) -> List[Dict]:
        """Load Cityscapes samples"""
        samples = []
        
        image_base = self.root / 'leftImg8bit' / self.split
        label_base = self.root / 'gtFine' / self.split
        
        # Check if directories exist
        if not image_base.exists():
            raise FileNotFoundError(f"Image directory not found: {image_base}")
        if not label_base.exists() and self.split != 'test':
            raise FileNotFoundError(f"Label directory not found: {label_base}")
        
        # Cityscapes has city subdirectories
        for city_dir in sorted(image_base.iterdir()):
            if not city_dir.is_dir():
                continue
            
            for img_path in sorted(city_dir.glob('*_leftImg8bit.png')):
                # Find corresponding label
                label_name = img_path.name.replace(
                    '_leftImg8bit.png', '_gtFine_labelTrainIds.png'
                )
                label_path = label_base / city_dir.name / label_name
                
                # For test set, labels might not exist
                if self.split == 'test' or label_path.exists():
                    samples.append({
                        'image_path': img_path,
                        'label_path': label_path if label_path.exists() else None,
                        'name': img_path.stem.replace('_leftImg8bit', ''),
                        'city': city_dir.name
                    })
        
        return samples