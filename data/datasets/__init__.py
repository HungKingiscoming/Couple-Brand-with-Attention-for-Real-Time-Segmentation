from .base import BaseSegmentationDataset
from .cityscapes import CityscapesDataset
from .camvid import CamVidDataset
from .custom import CustomDataset

# Dataset registry for easy lookup
DATASET_REGISTRY = {
    'cityscapes': CityscapesDataset,
    'camvid': CamVidDataset,
    'custom': CustomDataset,
}

def build_dataset(cfg: dict):
    """
    Build dataset from config
    
    Args:
        cfg: Dataset config dict with keys:
            - name: Dataset name
            - root: Dataset root path
            - split: 'train' / 'val' / 'test'
            - transform: Transform config
            - **kwargs: Additional dataset-specific args
    
    Example:
        cfg = {
            'name': 'cityscapes',
            'root': '/data/cityscapes',
            'split': 'train',
            'transform': {...}
        }
        dataset = build_dataset(cfg)
    """
    name = cfg.pop('name')
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    
    dataset_class = DATASET_REGISTRY[name]
    return dataset_class(**cfg)


__all__ = [
    'BaseSegmentationDataset',
    'CityscapesDataset',
    'PascalVOCDataset',
    'CamVidDataset',
    'CustomDataset',
    'DATASET_REGISTRY',
    'build_dataset',
]