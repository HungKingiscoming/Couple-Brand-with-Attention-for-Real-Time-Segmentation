"""
Label mapping utilities for Cityscapes dataset
Reference: https://github.com/mcordts/cityscapesScripts
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch


# ============================================
# CITYSCAPES LABEL DEFINITIONS
# ============================================

class CityscapesLabels:
    """Complete Cityscapes label information"""
    
    # Full label information (id -> properties)
    LABELS = {
        0:  {'name': 'unlabeled',     'train_id': 255, 'color': (0, 0, 0)},
        1:  {'name': 'ego vehicle',   'train_id': 255, 'color': (0, 0, 0)},
        2:  {'name': 'rectification', 'train_id': 255, 'color': (0, 0, 0)},
        3:  {'name': 'out of roi',    'train_id': 255, 'color': (0, 0, 0)},
        4:  {'name': 'static',        'train_id': 255, 'color': (0, 0, 0)},
        5:  {'name': 'dynamic',       'train_id': 255, 'color': (111, 74, 0)},
        6:  {'name': 'ground',        'train_id': 255, 'color': (81, 0, 81)},
        7:  {'name': 'road',          'train_id': 0,   'color': (128, 64, 128)},
        8:  {'name': 'sidewalk',      'train_id': 1,   'color': (244, 35, 232)},
        9:  {'name': 'parking',       'train_id': 255, 'color': (250, 170, 160)},
        10: {'name': 'rail track',    'train_id': 255, 'color': (230, 150, 140)},
        11: {'name': 'building',      'train_id': 2,   'color': (70, 70, 70)},
        12: {'name': 'wall',          'train_id': 3,   'color': (102, 102, 156)},
        13: {'name': 'fence',         'train_id': 4,   'color': (190, 153, 153)},
        14: {'name': 'guard rail',    'train_id': 255, 'color': (180, 165, 180)},
        15: {'name': 'bridge',        'train_id': 255, 'color': (150, 100, 100)},
        16: {'name': 'tunnel',        'train_id': 255, 'color': (150, 120, 90)},
        17: {'name': 'pole',          'train_id': 5,   'color': (153, 153, 153)},
        18: {'name': 'polegroup',     'train_id': 255, 'color': (153, 153, 153)},
        19: {'name': 'traffic light', 'train_id': 6,   'color': (250, 170, 30)},
        20: {'name': 'traffic sign',  'train_id': 7,   'color': (220, 220, 0)},
        21: {'name': 'vegetation',    'train_id': 8,   'color': (107, 142, 35)},
        22: {'name': 'terrain',       'train_id': 9,   'color': (152, 251, 152)},
        23: {'name': 'sky',           'train_id': 10,  'color': (70, 130, 180)},
        24: {'name': 'person',        'train_id': 11,  'color': (220, 20, 60)},
        25: {'name': 'rider',         'train_id': 12,  'color': (255, 0, 0)},
        26: {'name': 'car',           'train_id': 13,  'color': (0, 0, 142)},
        27: {'name': 'truck',         'train_id': 14,  'color': (0, 0, 70)},
        28: {'name': 'bus',           'train_id': 15,  'color': (0, 60, 100)},
        29: {'name': 'caravan',       'train_id': 255, 'color': (0, 0, 90)},
        30: {'name': 'trailer',       'train_id': 255, 'color': (0, 0, 110)},
        31: {'name': 'train',         'train_id': 16,  'color': (0, 80, 100)},
        32: {'name': 'motorcycle',    'train_id': 17,  'color': (0, 0, 230)},
        33: {'name': 'bicycle',       'train_id': 18,  'color': (119, 11, 32)},
        -1: {'name': 'license plate', 'train_id': 255, 'color': (0, 0, 142)},
    }
    
    @classmethod
    def get_train_id_mapping(cls) -> Dict[int, int]:
        """Get id -> train_id mapping"""
        return {k: v['train_id'] for k, v in cls.LABELS.items()}
    
    @classmethod
    def get_train_id_info(cls) -> Dict[int, Dict]:
        """Get train_id -> info mapping (only valid classes)"""
        train_id_info = {}
        for id_val, info in cls.LABELS.items():
            train_id = info['train_id']
            if train_id != 255:
                train_id_info[train_id] = {
                    'name': info['name'],
                    'color': info['color'],
                    'original_id': id_val
                }
        return train_id_info
    
    @classmethod
    def get_color_map(cls) -> np.ndarray:
        """Get color map for visualization (train_id -> RGB)"""
        color_map = np.zeros((256, 3), dtype=np.uint8)
        color_map[255] = [0, 0, 0]  # Ignore class = black
        
        for id_val, info in cls.LABELS.items():
            train_id = info['train_id']
            if train_id != 255:
                color_map[train_id] = info['color']
        
        return color_map
    
    @classmethod
    def print_mapping_info(cls):
        """Print complete label mapping information"""
        print("="*80)
        print("CITYSCAPES LABEL MAPPING")
        print("="*80)
        print(f"{'ID':>3} | {'Train ID':>8} | {'Name':20} | {'Color (R,G,B)'}")
        print("-"*80)
        
        for id_val in sorted(cls.LABELS.keys()):
            if id_val < 0:
                continue
            info = cls.LABELS[id_val]
            train_id = info['train_id']
            name = info['name']
            color = info['color']
            
            train_id_str = str(train_id) if train_id != 255 else 'IGNORE'
            print(f"{id_val:3d} | {train_id_str:>8} | {name:20} | {color}")
        
        print("="*80)
        print(f"Total classes: 34 (IDs 0-33)")
        print(f"Training classes: 19 (Train IDs 0-18)")
        print(f"Ignored classes: 15 (mapped to 255)")
        print("="*80)


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def visualize_label_mapping(
    label_path: str,
    output_path: str = None,
    show_original: bool = True
):
    """
    Visualize how labels are mapped from ID to train_ID
    
    Args:
        label_path: Path to *_labelIds.png file
        output_path: Where to save visualization
        show_original: Show original IDs alongside train IDs
    """
    # Load label
    label_id = np.array(Image.open(label_path), dtype=np.uint8)
    
    # Create mapping
    id_to_trainid = CityscapesLabels.get_train_id_mapping()
    label_map = np.ones(256, dtype=np.uint8) * 255
    for id_val, train_id in id_to_trainid.items():
        if id_val >= 0:
            label_map[id_val] = train_id
    
    # Convert to train_id
    label_train_id = label_map[label_id]
    
    # Get color maps
    color_map = CityscapesLabels.get_color_map()
    
    # Convert to RGB
    label_train_id_rgb = color_map[label_train_id]
    
    # Create figure
    if show_original:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original IDs (raw visualization)
        axes[0].imshow(label_id, cmap='tab20')
        axes[0].set_title('Original Label IDs (0-33)', fontsize=14)
        axes[0].axis('off')
        
        # Train IDs (raw)
        axes[1].imshow(label_train_id, cmap='tab20')
        axes[1].set_title('Train IDs (0-18, 255=ignore)', fontsize=14)
        axes[1].axis('off')
        
        # Train IDs (colored)
        axes[2].imshow(label_train_id_rgb)
        axes[2].set_title('Train IDs (Colored)', fontsize=14)
        axes[2].axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(label_train_id_rgb)
        ax.set_title('Train IDs (Colored)', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    unique_ids = np.unique(label_id)
    unique_train_ids = np.unique(label_train_id)
    
    print("\nLabel Statistics:")
    print(f"  Unique original IDs: {unique_ids.tolist()}")
    print(f"  Unique train IDs: {unique_train_ids.tolist()}")
    print(f"  Valid training pixels: {(label_train_id != 255).sum():,}")
    print(f"  Ignored pixels: {(label_train_id == 255).sum():,}")


def verify_label_file(label_path: str):
    """
    Verify a label file and show what IDs it contains
    
    Args:
        label_path: Path to label file
    """
    print("="*80)
    print(f"VERIFYING LABEL FILE: {label_path}")
    print("="*80)
    
    # Load label
    label = np.array(Image.open(label_path), dtype=np.uint8)
    unique_ids = np.unique(label)
    
    print(f"\nImage size: {label.shape}")
    print(f"Unique IDs found: {len(unique_ids)}")
    print(f"IDs: {unique_ids.tolist()}")
    
    # Check each ID
    labels_info = CityscapesLabels.LABELS
    
    print("\nID Breakdown:")
    print(f"{'ID':>3} | {'Train ID':>8} | {'Name':20} | {'Pixel Count':>12}")
    print("-"*80)
    
    for id_val in sorted(unique_ids):
        if id_val in labels_info:
            info = labels_info[id_val]
            train_id = info['train_id']
            name = info['name']
            count = (label == id_val).sum()
            
            train_id_str = str(train_id) if train_id != 255 else 'IGNORE'
            print(f"{id_val:3d} | {train_id_str:>8} | {name:20} | {count:12,}")
        else:
            print(f"{id_val:3d} | UNKNOWN")
    
    print("="*80)
    
    # Apply mapping
    id_to_trainid = CityscapesLabels.get_train_id_mapping()
    label_map = np.ones(256, dtype=np.uint8) * 255
    for id_val, train_id in id_to_trainid.items():
        if id_val >= 0:
            label_map[id_val] = train_id
    
    label_train_id = label_map[label]
    unique_train_ids = np.unique(label_train_id)
    
    print("\nAfter Mapping to Train IDs:")
    print(f"Unique train IDs: {unique_train_ids.tolist()}")
    print(f"Valid pixels (0-18): {(label_train_id < 19).sum():,}")
    print(f"Ignored pixels (255): {(label_train_id == 255).sum():,}")
    print("="*80)


def create_legend(output_path: str = 'cityscapes_legend.png'):
    """Create a legend showing all train_id classes and their colors"""
    train_id_info = CityscapesLabels.get_train_id_info()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    y_pos = 0.95
    for train_id in sorted(train_id_info.keys()):
        info = train_id_info[train_id]
        name = info['name']
        color = np.array(info['color']) / 255.0
        
        # Draw color patch
        rect = plt.Rectangle((0.1, y_pos - 0.03), 0.05, 0.03, 
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Draw text
        ax.text(0.17, y_pos - 0.015, f"Train ID {train_id:2d}: {name}", 
               fontsize=12, verticalalignment='center')
        
        y_pos -= 0.05
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Cityscapes Training Classes', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved legend to: {output_path}")


# ============================================
# MAIN UTILITIES
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cityscapes label utilities')
    parser.add_argument('--mode', type=str, default='info',
                        choices=['info', 'verify', 'visualize', 'legend'],
                        help='Operation mode')
    parser.add_argument('--label_path', type=str,
                        help='Path to label file (for verify/visualize)')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output path for visualization')
    args = parser.parse_args()
    
    if args.mode == 'info':
        # Print label mapping information
        CityscapesLabels.print_mapping_info()
    
    elif args.mode == 'verify':
        if args.label_path is None:
            raise ValueError("--label_path required for verify mode")
        verify_label_file(args.label_path)
    
    elif args.mode == 'visualize':
        if args.label_path is None:
            raise ValueError("--label_path required for visualize mode")
        visualize_label_mapping(args.label_path, args.output)
    
    elif args.mode == 'legend':
        create_legend(args.output)