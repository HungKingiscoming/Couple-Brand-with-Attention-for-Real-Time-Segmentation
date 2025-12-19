import torch
from torch.utils.data import DataLoader, Dataset
from datasets.base import BaseSegmentationDataset
from typing import Dict, Optional, Callable
import torch

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Build PyTorch DataLoader
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        collate_fn: Custom collate function
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )


def visualize_sample(sample: Dict, dataset: BaseSegmentationDataset):
    """
    Visualize a sample from dataset
    
    Args:
        sample: Dict from dataset.__getitem__()
        dataset: Dataset instance (for metadata)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    image = sample['image']
    label = sample['label']
    name = sample['name']
    
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    # Create colored label
    palette = np.array(dataset.palette)
    colored_label = palette[label]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title(f'Image: {name}')
    axes[0].axis('off')
    
    axes[1].imshow(label, cmap='tab20')
    axes[1].set_title('Label (ID)')
    axes[1].axis('off')
    
    axes[2].imshow(colored_label.astype(np.uint8))
    axes[2].set_title('Label (Colored)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_dataset(dataset: Dataset):
    """
    Analyze dataset statistics
    
    Prints:
        - Number of samples
        - Image size statistics
        - Class distribution
        - Memory usage estimate
    """
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {dataset.__class__.__name__}")
    print(f"{'='*60}")
    
    print(f"\nBasic Info:")
    print(f"  Split: {dataset.split}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Classes: {dataset.num_classes}")
    
    if hasattr(dataset, 'classes'):
        print(f"  Class names: {dataset.classes[:5]}..." if len(dataset.classes) > 5 
              else f"  Class names: {dataset.classes}")
    
    print(f"\nSample Info:")
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label shape: {sample['label'].shape}")
    print(f"  Label unique: {torch.unique(sample['label']).tolist()[:10]}")
    
    print(f"\n{'='*60}\n")