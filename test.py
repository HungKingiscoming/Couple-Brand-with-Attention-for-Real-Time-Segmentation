import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import json

# ===================== MODEL =====================
from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead

# ===================== PALETTE =====================
CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# ===================== DATASET =====================
class ValidationDataset(Dataset):
    """Dataset for validation with ground truth labels"""
    
    def __init__(
        self,
        img_dir: str,
        gt_dir: str,
        img_size: Tuple[int, int] = (512, 1024),
        ignore_index: int = 255
    ):
        self.img_dir = Path(img_dir)
        self.gt_dir = Path(gt_dir)
        self.img_size = img_size
        self.ignore_index = ignore_index
        
        # Get image list
        self.img_paths = sorted(list(self.img_dir.glob('*.png')) + 
                               list(self.img_dir.glob('*.jpg')))
        
        # Build transform
        self.transform = A.Compose([
            A.Resize(*img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.gt_transform = A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST)
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Load ground truth
        gt_name = img_path.stem.replace('_leftImg8bit', '_gtFine_labelIds')
        gt_path = self.gt_dir / f'{gt_name}.png'
        
        if not gt_path.exists():
            # Try alternative naming
            gt_path = self.gt_dir / f'{img_path.stem}.png'
        
        gt = np.array(Image.open(gt_path))
        
        # Apply transforms
        img_transformed = self.transform(image=img)['image']
        gt_transformed = self.gt_transform(image=gt)['image']
        
        return {
            'image': img_transformed,
            'gt': torch.from_numpy(gt_transformed).long(),
            'img_path': str(img_path),
            'original_size': img.shape[:2]
        }

# ===================== MODEL WRAPPER =====================
class GCNetSegmentor(nn.Module):
    def __init__(self, num_classes, backbone_cfg, head_cfg):
        super().__init__()
        self.backbone = GCNetImproved(**backbone_cfg)
        self.decode_head = GCNetHead(
            in_channels=head_cfg['in_channels'],
            channels=head_cfg['channels'],
            num_classes=num_classes,
            decode_enabled=True,
            decoder_channels=head_cfg['decoder_channels'],
            skip_channels=head_cfg['skip_channels'],
            use_gated_fusion=True,
            dropout_ratio=0.1,
            align_corners=False
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)

# ===================== METRICS =====================
class SegmentationMetrics:
    """Calculate IoU, Dice, Pixel Accuracy for semantic segmentation"""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update confusion matrix
        
        Args:
            pred: (H, W) predicted labels
            target: (H, W) ground truth labels
        """
        # Flatten
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignore index
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            self.confusion_matrix[t, p] += 1
        
        self.total_samples += 1
    
    def compute_iou(self) -> Dict[str, float]:
        """Compute IoU (Intersection over Union) per class"""
        iou_per_class = []
        
        for i in range(self.num_classes):
            # True Positive
            tp = self.confusion_matrix[i, i]
            
            # False Positive + False Negative
            fp_fn = self.confusion_matrix[i, :].sum() + \
                    self.confusion_matrix[:, i].sum() - tp
            
            if fp_fn == 0:
                iou = float('nan')
            else:
                iou = tp / fp_fn
            
            iou_per_class.append(iou)
        
        # Mean IoU (ignore nan)
        iou_per_class = np.array(iou_per_class)
        valid_iou = iou_per_class[~np.isnan(iou_per_class)]
        mean_iou = valid_iou.mean() if len(valid_iou) > 0 else 0.0
        
        return {
            'mIoU': mean_iou,
            'IoU_per_class': iou_per_class
        }
    
    def compute_dice(self) -> Dict[str, float]:
        """Compute Dice coefficient per class"""
        dice_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = 2 * tp + fp + fn
            
            if denominator == 0:
                dice = float('nan')
            else:
                dice = (2 * tp) / denominator
            
            dice_per_class.append(dice)
        
        # Mean Dice
        dice_per_class = np.array(dice_per_class)
        valid_dice = dice_per_class[~np.isnan(dice_per_class)]
        mean_dice = valid_dice.mean() if len(valid_dice) > 0 else 0.0
        
        return {
            'mDice': mean_dice,
            'Dice_per_class': dice_per_class
        }
    
    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def compute_mean_accuracy(self) -> float:
        """Compute mean class accuracy"""
        acc_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            total = self.confusion_matrix[i, :].sum()
            
            if total == 0:
                acc = float('nan')
            else:
                acc = tp / total
            
            acc_per_class.append(acc)
        
        acc_per_class = np.array(acc_per_class)
        valid_acc = acc_per_class[~np.isnan(acc_per_class)]
        
        return valid_acc.mean() if len(valid_acc) > 0 else 0.0
    
    def get_results(self) -> Dict:
        """Get all metrics"""
        iou_results = self.compute_iou()
        dice_results = self.compute_dice()
        
        return {
            'mIoU': iou_results['mIoU'],
            'mDice': dice_results['mDice'],
            'Pixel_Accuracy': self.compute_pixel_accuracy(),
            'Mean_Accuracy': self.compute_mean_accuracy(),
            'IoU_per_class': iou_results['IoU_per_class'],
            'Dice_per_class': dice_results['Dice_per_class']
        }

# ===================== TESTER =====================
class Tester:
    """Validation tester with comprehensive metrics"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        num_classes: int = 19,
        ignore_index: int = 255,
        use_amp: bool = True,
        sliding_window: bool = False,
        window_size: Tuple[int, int] = (512, 1024),
        stride: Tuple[int, int] = (256, 512)
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_amp = use_amp
        self.sliding_window = sliding_window
        self.window_size = window_size
        self.stride = stride
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes, ignore_index)
    
    @torch.no_grad()
    def predict(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Predict single image
        
        Args:
            img_tensor: (1, 3, H, W)
        
        Returns:
            pred: (H, W) uint8 array
        """
        img_tensor = img_tensor.to(self.device)
        
        with autocast(enabled=self.use_amp):
            if not self.sliding_window:
                logits = self.model(img_tensor)
            else:
                logits = self._sliding_window_inference(img_tensor)
        
        # Get predictions
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        return pred
    
    def _sliding_window_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding window inference for large images"""
        B, C, H, W = x.shape
        wh, ww = self.window_size
        sh, sw = self.stride
        
        logits_sum = torch.zeros((1, self.num_classes, H, W), 
                                device=self.device)
        count = torch.zeros((1, 1, H, W), device=self.device)
        
        for y in range(0, H - wh + 1, sh):
            for x0 in range(0, W - ww + 1, sw):
                patch = x[:, :, y:y+wh, x0:x0+ww]
                
                with autocast(enabled=self.use_amp):
                    out = self.model(patch)
                
                logits_sum[:, :, y:y+wh, x0:x0+ww] += out
                count[:, :, y:y+wh, x0:x0+ww] += 1
        
        return logits_sum / count.clamp(min=1)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_dir: Optional[str] = None,
        save_visualizations: bool = False
    ) -> Dict:
        """
        Evaluate on validation set
        
        Args:
            dataloader: Validation dataloader
            save_dir: Directory to save results
            save_visualizations: Whether to save prediction visualizations
        
        Returns:
            Dict of metrics
        """
        self.metrics.reset()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if save_visualizations:
                vis_dir = save_dir / 'visualizations'
                vis_dir.mkdir(exist_ok=True)
        
        print("Starting evaluation...")
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            img = batch['image']  # (B, 3, H, W)
            gt = batch['gt']      # (B, H, W)
            
            # Predict
            pred = self.predict(img)  # (H, W)
            
            # Update metrics
            gt_np = gt.squeeze(0).numpy()
            self.metrics.update(pred, gt_np)
            
            # Save visualizations
            if save_visualizations and save_dir:
                img_path = batch['img_path'][0]
                img_name = Path(img_path).stem
                
                # Colorize prediction
                vis = self.visualize(pred)
                vis_img = Image.fromarray(vis)
                vis_img.save(vis_dir / f'{img_name}_pred.png')
                
                # Colorize ground truth
                gt_vis = self.visualize(gt_np)
                gt_vis_img = Image.fromarray(gt_vis)
                gt_vis_img.save(vis_dir / f'{img_name}_gt.png')
        
        # Compute final metrics
        results = self.metrics.get_results()
        
        # Print results
        self.print_results(results)
        
        # Save results
        if save_dir:
            self.save_results(results, save_dir)
        
        return results
    
    @staticmethod
    def visualize(mask: np.ndarray) -> np.ndarray:
        """Convert label mask to RGB visualization"""
        h, w = mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, color in enumerate(CITYSCAPES_PALETTE):
            out[mask == i] = color
        
        return out
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n{'Metric':<30} {'Value':>10}")
        print("-"*60)
        print(f"{'mIoU (Mean IoU)':<30} {results['mIoU']:>10.4f}")
        print(f"{'mDice (Mean Dice)':<30} {results['mDice']:>10.4f}")
        print(f"{'Pixel Accuracy':<30} {results['Pixel_Accuracy']:>10.4f}")
        print(f"{'Mean Class Accuracy':<30} {results['Mean_Accuracy']:>10.4f}")
        
        print("\n" + "-"*60)
        print("Per-Class IoU:")
        print("-"*60)
        
        for i, (cls_name, iou) in enumerate(zip(CITYSCAPES_CLASSES, 
                                                results['IoU_per_class'])):
            if not np.isnan(iou):
                print(f"{cls_name:<30} {iou:>10.4f}")
        
        print("="*60 + "\n")
    
    def save_results(self, results: Dict, save_dir: Path):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            'mIoU': float(results['mIoU']),
            'mDice': float(results['mDice']),
            'Pixel_Accuracy': float(results['Pixel_Accuracy']),
            'Mean_Accuracy': float(results['Mean_Accuracy']),
            'IoU_per_class': {
                cls_name: float(iou) if not np.isnan(iou) else None
                for cls_name, iou in zip(CITYSCAPES_CLASSES, 
                                        results['IoU_per_class'])
            },
            'Dice_per_class': {
                cls_name: float(dice) if not np.isnan(dice) else None
                for cls_name, dice in zip(CITYSCAPES_CLASSES, 
                                         results['Dice_per_class'])
            }
        }
        
        # Save to JSON
        json_path = save_dir / 'eval_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"Results saved to {json_path}")

# ===================== LOAD MODEL =====================
def load_model(ckpt_path: str, device: str) -> nn.Module:
    """Load trained model from checkpoint"""
    backbone_cfg = {
        'in_channels': 3,
        'channels': 32,
        'ppm_channels': 128,
        'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
        'use_flash_attention': True,
        'flash_attn_stage': 4,
        'flash_attn_layers': 2,
        'flash_attn_heads': 8,
        'use_se': True,
        'deploy': True
    }

    head_cfg = {
        'in_channels': 64,
        'channels': 128,
        'decoder_channels': 128,
        'skip_channels': [64, 32, 32]
    }

    model = GCNetSegmentor(19, backbone_cfg, head_cfg)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    return model

# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser(description='Validate GCNet Segmentation Model')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data args
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Validation images directory')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Ground truth labels directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 1024],
                       help='Image size (H W)')
    
    # Inference args
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--sliding', action='store_true',
                       help='Use sliding window inference')
    parser.add_argument('--window_size', type=int, nargs=2, default=[512, 1024],
                       help='Sliding window size')
    parser.add_argument('--stride', type=int, nargs=2, default=[256, 512],
                       help='Sliding window stride')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--num_classes', type=int, default=19,
                       help='Number of classes')
    parser.add_argument('--ignore_index', type=int, default=255,
                       help='Ignore index in ground truth')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Create dataset
    print(f"Loading validation dataset from {args.img_dir}...")
    val_dataset = ValidationDataset(
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        img_size=tuple(args.img_size),
        ignore_index=args.ignore_index
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Found {len(val_dataset)} validation images")
    
    # Create tester
    tester = Tester(
        model=model,
        device=device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        use_amp=args.amp,
        sliding_window=args.sliding,
        window_size=tuple(args.window_size),
        stride=tuple(args.stride)
    )
    
    # Evaluate
    results = tester.evaluate(
        dataloader=val_loader,
        save_dir=args.output_dir,
        save_visualizations=args.save_vis
    )
    
    print("\n✓ Evaluation completed!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
