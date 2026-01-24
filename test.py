# ============================================
# EVALUATION SCRIPT - Compute mIoU, Dice, Accuracy
# ============================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# Import from your training code
from model.backbone.model import GCNetWithEnhance
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders
from model.model_utils import replace_bn_with_gn

# ============================================
# SEGMENTOR (same as train.py)
# ============================================
class Segmentor(nn.Module):
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)

# ============================================
# METRICS COMPUTATION
# ============================================
class MetricsCalculator:
    def __init__(self, num_classes, ignore_index=255, device='cuda'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_dice = 0.0
        self.total_samples = 0
    
    def update(self, pred, target):
        """Update confusion matrix and dice score"""
        # Confusion matrix for mIoU and accuracy
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
        
        # Dice score
        dice_score = self.compute_dice_per_sample(pred, target)
        self.total_dice += dice_score
        self.total_samples += 1
    
    def compute_dice_per_sample(self, pred, target):
        """Compute dice score for a single sample"""
        dice_scores = []
        
        for cls in range(self.num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union == 0:
                continue  # Skip classes not present
            
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice)
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def get_metrics(self):
        """Compute final metrics"""
        # Per-class IoU
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        
        # Mean IoU
        valid_classes = union > 0
        miou = np.mean(iou[valid_classes])
        
        # Pixel Accuracy
        acc = intersection.sum() / (self.confusion_matrix.sum() + 1e-10)
        
        # Mean Dice
        mean_dice = self.total_dice / max(self.total_samples, 1)
        
        return {
            'miou': miou,
            'accuracy': acc,
            'dice': mean_dice,
            'per_class_iou': iou,
            'confusion_matrix': self.confusion_matrix
        }

# ============================================
# EVALUATION FUNCTION
# ============================================
@torch.no_grad()
def evaluate_model(model, dataloader, num_classes, ignore_index=255, 
                   use_multiscale=False, device='cuda'):
    """
    Evaluate model on validation/test set
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation/test
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation
        use_multiscale: Whether to use multi-scale testing
        device: Device to run evaluation
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator(num_classes, ignore_index, device)
    
    scales = [0.75, 1.0, 1.25] if use_multiscale else [1.0]
    desc = f"Evaluating (MS={len(scales)} scales)" if use_multiscale else "Evaluating"
    
    pbar = tqdm(dataloader, desc=desc)
    
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.cpu().numpy()
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        H, W = masks.shape[-2:]
        
        # Multi-scale prediction
        if use_multiscale:
            final_pred = torch.zeros(imgs.size(0), num_classes, H, W).to(device)
            
            for scale in scales:
                h, w = int(H * scale), int(W * scale)
                img_scaled = F.interpolate(imgs, size=(h, w), mode='bilinear', align_corners=False)
                
                logits_scaled = model(img_scaled)
                logits_scaled = F.interpolate(logits_scaled, size=(H, W), mode='bilinear', align_corners=False)
                
                final_pred += F.softmax(logits_scaled, dim=1)
            
            final_pred /= len(scales)
            pred = final_pred.argmax(1).cpu().numpy()
        else:
            # Single scale
            logits = model(imgs)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            pred = logits.argmax(1).cpu().numpy()
        
        # Update metrics
        for i in range(pred.shape[0]):
            metrics_calc.update(pred[i], masks[i])
        
        # Progress update
        current_metrics = metrics_calc.get_metrics()
        pbar.set_postfix({
            'mIoU': f"{current_metrics['miou']:.4f}",
            'Dice': f"{current_metrics['dice']:.4f}",
            'Acc': f"{current_metrics['accuracy']:.4f}"
        })
    
    return metrics_calc.get_metrics()

# ============================================
# MODEL BUILDING
# ============================================
def build_model(num_classes=19, device='cuda'):
    """Build model with same config as train.py"""
    cfg = {
        'backbone': {
            'in_channels': 3,
            'channels': 32,
            'ppm_channels': 128,
            'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
            'dwsa_stages': ['stage5', 'stage6'],
            'dwsa_num_heads': 4,
            'dwsa_reduction': 4,
            'dwsa_qk_sharing': True,
            'dwsa_groups': 4,
            'dwsa_drop': 0.1,
            'dwsa_alpha': 0.1,
            'use_multi_scale_context': True,
            'ms_scales': (1, 2),
            'ms_branch_ratio': 8,
            'ms_alpha': 0.1,
            'align_corners': False,
            'deploy': False
        }
    }
    
    # Build backbone
    backbone = GCNetWithEnhance(**cfg['backbone'])
    
    # Detect channels - FIX: Use eval mode and batch_size=2
    backbone.eval()
    backbone = backbone.to(device)
    with torch.no_grad():
        sample = torch.randn(2, 3, 512, 1024).to(device)
        feats = backbone(sample)
        
        if isinstance(feats, tuple):
            detected_channels = {
                'c4': feats[0].shape[1] if len(feats) > 0 else 128,
                'c5': feats[1].shape[1] if len(feats) > 1 else 128
            }
        elif isinstance(feats, dict):
            detected_channels = {k: v.shape[1] for k, v in feats.items()}
        else:
            detected_channels = {'c5': feats.shape[1]}
    
    # Move backbone back to CPU before building full model
    backbone = backbone.cpu()
    
    # Build heads
    head = GCNetHead(
        in_channels=detected_channels.get('c5', 128),
        c1_channels=detected_channels.get('c1', 32),
        c2_channels=detected_channels.get('c2', 64),
        decoder_channels=128,
        num_classes=num_classes,
        dropout_ratio=0.1,
        use_gated_fusion=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=False),
        align_corners=False
    )
    
    aux_head = GCNetAuxHead(
        in_channels=detected_channels.get('c4', 128),
        channels=96,
        num_classes=num_classes,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=False),
        align_corners=False
    )
    
    # Build model
    model = Segmentor(backbone=backbone, head=head, aux_head=aux_head)
    
    # Replace BN with GN BEFORE moving to device
    model = replace_bn_with_gn(model)
    
    # Move to device AFTER replace
    model = model.to(device)
    
    return model

# ============================================
# MAIN EVALUATION
# ============================================
def main():
    parser = argparse.ArgumentParser(description="🎯 Model Evaluation - Compute mIoU, Dice, Accuracy")
    
    # Model & Checkpoint
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    
    # Dataset
    parser.add_argument("--val_txt", required=True, help="Path to validation txt file")
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    
    # Evaluation settings
    parser.add_argument("--use_multiscale", action="store_true", 
                       help="Use multi-scale testing (0.75, 1.0, 1.25)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save evaluation results (JSON)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("🎯 MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📊 Dataset: {args.val_txt}")
    print(f"🖼️  Image size: {args.img_h}x{args.img_w}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📐 Multi-scale: {args.use_multiscale}")
    print(f"🎯 Device: {device}")
    print(f"{'='*70}\n")
    
    # Create dataloader
    print("📦 Loading dataset...")
    _, val_loader, _ = create_dataloaders(
        train_txt=args.val_txt,  # Use same file for both (we only need val)
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        pin_memory=True,
        compute_class_weights=False,
        dataset_type=args.dataset_type
    )
    print(f"✅ Loaded {len(val_loader.dataset)} samples\n")
    
    # Build model
    print("🏗️  Building model...")
    model = build_model(num_classes=args.num_classes, device=device)
    print("✅ Model built\n")
    
    # Load checkpoint
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    print("✅ Checkpoint loaded\n")
    
    # Evaluate
    print("🚀 Starting evaluation...\n")
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        use_multiscale=args.use_multiscale,
        device=device
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"🎯 Mean IoU (mIoU):    {metrics['miou']:.4f} ({metrics['miou']*100:.2f}%)")
    print(f"🎲 Dice Score:         {metrics['dice']:.4f} ({metrics['dice']*100:.2f}%)")
    print(f"✅ Pixel Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Per-class IoU
    print("📋 Per-Class IoU:")
    print("-" * 70)
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    for i, (name, iou) in enumerate(zip(class_names, metrics['per_class_iou'])):
        if i < args.num_classes:
            print(f"  {i:2d}. {name:15s}: {iou:.4f} ({iou*100:.2f}%)")
    print("=" * 70)
    
    # Save results
    if args.save_results:
        results = {
            'checkpoint': args.checkpoint,
            'dataset': args.val_txt,
            'num_samples': len(val_loader.dataset),
            'multiscale': args.use_multiscale,
            'metrics': {
                'miou': float(metrics['miou']),
                'dice': float(metrics['dice']),
                'accuracy': float(metrics['accuracy']),
            },
            'per_class_iou': {
                class_names[i]: float(metrics['per_class_iou'][i])
                for i in range(min(len(class_names), args.num_classes))
            }
        }
        
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_path}")
    
    print("\n✅ Evaluation completed!\n")

if __name__ == "__main__":
    main()
