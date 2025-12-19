from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BaseSegmentationDataset(Dataset, ABC):
    """
    Abstract base class for semantic segmentation datasets
    
    All custom datasets should inherit from this class and implement:
    - _load_samples()
    - METAINFO (optional)
    
    Attributes:
        METAINFO: Dict containing dataset metadata:
            - classes: Tuple of class names
            - palette: List of RGB colors for visualization
            - num_classes: Number of classes
            - ignore_index: Label to ignore
    """
    
    # Override in subclass
    METAINFO = {
        'classes': (),
        'palette': [],
        'num_classes': 0,
        'ignore_index': 255,
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        cache_images: bool = False,
        return_path: bool = False
    ):
        """
        Args:
            root: Dataset root directory
            split: 'train', 'val', or 'test'
            transform: Transform pipeline
            cache_images: Cache images in memory (use if RAM is sufficient)
            return_path: Return image path in __getitem__
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.cache_images = cache_images
        self.return_path = return_path
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}")
        
        # Load samples
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split '{split}'")
        
        # Cache
        self._image_cache = {} if cache_images else None
        self._label_cache = {} if cache_images else None
        
        print(f"[{self.__class__.__name__}] "
              f"Loaded {len(self.samples)} samples for '{split}' split")
    
    @abstractmethod
    def _load_samples(self) -> List[Dict]:
        """
        Load sample list
        
        Returns:
            List of dicts with keys:
                - image_path: Path to image file
                - label_path: Path to label file
                - name: Sample name/ID
                - **extra: Additional metadata
        
        Example:
            return [
                {
                    'image_path': Path('/data/images/001.jpg'),
                    'label_path': Path('/data/labels/001.png'),
                    'name': '001',
                    'city': 'frankfurt'  # extra metadata
                }
            ]
        """
        raise NotImplementedError
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array"""
        if self._image_cache is not None and path in self._image_cache:
            return self._image_cache[path].copy()
        
        image = np.array(Image.open(path).convert('RGB'))
        
        if self._image_cache is not None:
            self._image_cache[path] = image.copy()
        
        return image
    
    def _load_label(self, path: Path) -> np.ndarray:
        """Load label as numpy array"""
        if self._label_cache is not None and path in self._label_cache:
            return self._label_cache[path].copy()
        
        label = np.array(Image.open(path))
        
        if self._label_cache is not None:
            self._label_cache[path] = label.copy()
        
        return label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - image: Tensor (C, H, W)
                - label: Tensor (H, W)
                - name: str
                - [path]: Optional, if return_path=True
        """
        sample = self.samples[idx]
        
        # Load data
        image = self._load_image(sample['image_path'])
        label = self._load_label(sample['label_path'])
        
        # Apply transform
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        # Prepare output
        output = {
            'image': image,
            'label': label,
            'name': sample['name']
        }
        
        if self.return_path:
            output['image_path'] = str(sample['image_path'])
            output['label_path'] = str(sample['label_path'])
        
        return output
    
    @property
    def classes(self) -> Tuple[str, ...]:
        """Get class names"""
        return self.METAINFO['classes']
    
    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        return self.METAINFO['num_classes']
    
    @property
    def ignore_index(self) -> int:
        """Get ignore index"""
        return self.METAINFO['ignore_index']
    
    @property
    def palette(self) -> List[List[int]]:
        """Get color palette for visualization"""
        return self.METAINFO['palette']