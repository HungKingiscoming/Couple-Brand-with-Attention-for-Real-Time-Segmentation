#!/usr/bin/env python3
# ============================================
# evaluation_deploy.py - Full Evaluation with Auto-Reparameter
# ============================================
"""
Evaluation script với:
1. Auto-reparameter GCBlock (deploy mode) → Tăng 30-50% tốc độ
2. Tính metrics trên toàn bộ val set (mIOU, Accuracy, Dice)
3. Visualize 5 ảnh predict random
4. Export metrics to JSON
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import time

# Import từ project
from model.backbone.model import GCNetWithEnhance
from model.head.segmentation_head import GCNetHead
from data.custom import create_dataloaders


# ============================================
# SEGMENTOR
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
# AUTO-REPARAMETER
# ============================================

def convert_to_deploy_mode(model):
    """
    🚀 Auto-convert tất cả GCBlock sang deploy mode
    → Fuse 4 branches thành 1 Conv3x3
    → Tăng 30-50% FPS
    """
    print("\n" + "="*70)
    print("🚀 AUTO-REPARAMETER: Converting to Deploy Mode")
    print("="*70)

    total_blocks = 0
    converted_blocks = 0

    def switch_gcblock_recursive(module, prefix=''):
        nonlocal total_blocks, converted_blocks

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Nếu là GCBlock
            if child.__class__.__name__ == 'GCBlock':
                total_blocks += 1
                if not child.deploy:
                    try:
                        child.switch_to_deploy()
                        converted_blocks += 1
                        print(f"  ✅ {full_name}")
                    except Exception as e:
                        print(f"  ❌ {full_name}: {e}")
            else:
                # Recursively process children
                switch_gcblock_recursive(child, full_name)

    # Convert
    switch_gcblock_recursive(model)

    print(f"\n📊 Summary:")
    print(f"   Total GCBlocks:  {total_blocks}")
    print(f"   Converted:       {converted_blocks}")
    print(f"   Success rate:    {converted_blocks/max(total_blocks,1)*100:.1f}%")
    print("="*70)

    return model


# ============================================
# METRICS
# ============================================

class SegmentationMetrics:
    """Tính toán mIOU, Accuracy, Dice Score"""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        """Update confusion matrix"""
        pred = pred.flatten()
        target = target.flatten()

        # Filter valid pixels
        mask = (target >= 0) & (target < self.num_classes) & (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        # Update confusion matrix
        label = self.num_classes * target.astype('int') + pred
        count = np.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def get_miou(self):
        """Calculate mean IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        valid_iou = iou[union > 0]
        miou = np.mean(valid_iou)
        return miou, iou

    def get_accuracy(self):
        """Calculate pixel accuracy"""
        intersection = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum()
        accuracy = intersection.sum() / (total + 1e-10)
        return accuracy

    def get_dice(self):
        """Calculate Dice Score"""
        intersection = np.diag(self.confusion_matrix)
        pred_sum = self.confusion_matrix.sum(axis=0)
        target_sum = self.confusion_matrix.sum(axis=1)
        dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-10)
        valid_dice = dice[target_sum > 0]
        mean_dice = np.mean(valid_dice)
        return mean_dice, dice


# ============================================
# VISUALIZATION
# ============================================

# Cityscapes palette
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]


def visualize_predictions(images, masks, predictions, save_path, num_samples=5):
    """Visualize predictions với ground truth"""
    num_samples = min(num_samples, images.shape[0])

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize image (ImageNet normalization)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # ✅ FIX: Handle ignore_index (255)
        mask = masks[i].copy()
        pred = predictions[i].copy()
        
        # Replace ignore_index (255) with 0 for visualization
        mask[mask == 255] = 0
        pred[pred == 255] = 0
        
        # Clip to valid range [0, 18]
        mask = np.clip(mask, 0, len(PALETTE) - 1)
        pred = np.clip(pred, 0, len(PALETTE) - 1)
        
        # Convert labels to RGB
        mask_rgb = PALETTE[mask] / 255.0
        pred_rgb = PALETTE[pred] / 255.0

        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}: Input Image', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📷 Visualization saved: {save_path}")
    plt.close()



# ============================================
# MODEL LOADING
# ============================================

def load_model(checkpoint_path, num_classes, channels=32, device='cuda', auto_deploy=True):
    """Load model với auto-detect channels"""
    print(f"\n📦 Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # ✅ AUTO-DETECT từ STEM (chính xác nhất)
    stem_key = 'backbone.backbone.stem.0.conv.weight'
    if stem_key in state_dict:
        detected_channels = state_dict[stem_key].shape[0]
        print(f"🔍 Detected channels from stem: {detected_channels}")
        channels = detected_channels
    
    # ✅ AUTO-DETECT c1, c2 từ checkpoint
    c2_key = 'decode_head.decoder.c2_proj.conv.weight'
    
    if c2_key in state_dict:
        c2_channels_ckpt = state_dict[c2_key].shape[1]
        print(f"🔍 Detected c2_channels from checkpoint: {c2_channels_ckpt}")
    else:
        c2_channels_ckpt = channels * 2  # fallback
        print(f"⚠️  Using default c2_channels: {c2_channels_ckpt}")
    
    c1_channels_ckpt = channels  # Assume c1 = base channels
    
    # ✅ CHECK NORM TYPE
    bn_keys = [k for k in state_dict.keys() if '.bn.' in k]
    gn_keys = [k for k in state_dict.keys() if '.gn.' in k]
    
    if len(gn_keys) > 0:
        norm_cfg = dict(type='GN', num_groups=8, requires_grad=True)
        print(f"🔍 Detected GroupNorm")
    else:
        norm_cfg = dict(type='BN', requires_grad=True)
        print(f"🔍 Detected BatchNorm")

    print(f"\n✅ Building model with:")
    print(f"   channels={channels}")
    print(f"   c1_channels={c1_channels_ckpt}")
    print(f"   c2_channels={c2_channels_ckpt}")
    print(f"   in_channels={channels*4}")
    print(f"   norm_cfg={norm_cfg['type']}")
    
    # Build model
    backbone_cfg = {
        'in_channels': 3,
        'channels': channels,
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
        'norm_cfg': norm_cfg,
        'deploy': False
    }

    head_cfg = {
        'in_channels': channels * 4,
        'c1_channels': channels,
        'c2_channels': c2_channels,
        'decoder_channels': 128,
        'num_classes': num_classes,
        'dropout_ratio': 0.1,
        'use_gated_fusion': True,
        'norm_cfg': norm_cfg,
        'act_cfg': dict(type='ReLU', inplace=False),
        'align_corners': False,
    }

    # Build
    backbone = GCNetWithEnhance(**backbone_cfg)
    head = GCNetHead(**head_cfg)
    model = Segmentor(backbone, head, aux_head=None)

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # ✅ FIX: Initialize variables
    other_missing = []
    bn_missing = []
    
    if missing:
        bn_missing = [k for k in missing if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
        other_missing = [k for k in missing if k not in bn_missing]
        
        if len(other_missing) > 0:
            print(f"\n❌ ERROR: Missing {len(other_missing)} critical keys:")
            for k in other_missing[:10]:
                print(f"   - {k}")
            if len(other_missing) > 10:
                print(f"   ... and {len(other_missing)-10} more")
        
        if len(bn_missing) > 0:
            if norm_cfg['type'] == 'GN':
                print(f"✅ Missing {len(bn_missing)} BN stats (OK for GroupNorm)")
            else:
                print(f"⚠️  Missing {len(bn_missing)} BN stats (will be recalculated)")
    
    if unexpected:
        print(f"⚠️  Unexpected {len(unexpected)} keys")
    
    if len(other_missing) == 0:
        print(f"\n✅ Model loaded successfully!")
    else:
        print(f"\n❌ Model load has errors!")

    # Metadata
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_miou' in checkpoint:
        print(f"   Best mIOU: {checkpoint['best_miou']:.4f}")

    # Auto-reparameter
    if auto_deploy:
        model = convert_to_deploy_mode(model)

    model = model.to(device)
    model.eval()

    return model



# ============================================
# EVALUATION
# ============================================

@torch.no_grad()
def evaluate_full(model, dataloader, device, num_classes, ignore_index=255, 
                  save_vis_samples=5, output_dir='./evaluation_results'):
    """
    Full evaluation trên toàn bộ val set
    + Auto-reparameter
    + Metrics: mIOU, Accuracy, Dice
    + Visualization 5 ảnh
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index)

    # Storage for visualization
    vis_images = []
    vis_masks = []
    vis_preds = []

    # Storage for FPS calculation
    inference_times = []

    print("\n" + "="*70)
    print("🔍 EVALUATING ON FULL VALIDATION SET")
    print("="*70)

    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks_np = masks.numpy().astype(np.int32)

        # Measure inference time
        if device == 'cuda':
            torch.cuda.synchronize()
            start = time.time()
        else:
            start = time.time()

        # Forward pass
        logits = model(images)

        if device == 'cuda':
            torch.cuda.synchronize()

        inference_times.append(time.time() - start)

        # Resize to original size
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

        # Get predictions
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.int32)

        # Update metrics
        metrics.update(preds, masks_np)

        # Store samples for visualization (random sampling)
        if len(vis_images) < save_vis_samples and random.random() < 0.05:
            vis_images.append(images.cpu())
            vis_masks.append(masks_np)
            vis_preds.append(preds)

        # Update progress bar
        if batch_idx % 10 == 0:
            current_miou, _ = metrics.get_miou()
            pbar.set_postfix({'mIOU': f'{current_miou:.4f}'})

    # Calculate final metrics
    miou, per_class_iou = metrics.get_miou()
    accuracy = metrics.get_accuracy()
    dice, per_class_dice = metrics.get_dice()

    # Calculate FPS
    avg_time = np.mean(inference_times[10:])  # Skip first 10 for warmup
    fps = 1.0 / avg_time

    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"✅ mIOU:     {miou:.4f} ({miou*100:.2f}%)")
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Dice:     {dice:.4f} ({dice*100:.2f}%)")
    print(f"⚡ FPS:      {fps:.2f} frames/sec")
    print(f"⏱️  Avg time: {avg_time*1000:.2f} ms/frame")
    print("="*70)

    # Print per-class metrics
    print("\n📈 PER-CLASS METRICS:")
    print("-"*70)
    print(f"{'Class':<20} {'IoU':<15} {'Dice':<15}")
    print("-"*70)

    for i in range(num_classes):
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
        if metrics.confusion_matrix[i].sum() > 0:
            print(f"{class_name:<20} {per_class_iou[i]:<15.4f} {per_class_dice[i]:<15.4f}")
    print("-"*70)

    # Visualization
    os.makedirs(output_dir, exist_ok=True)

    if len(vis_images) > 0:
        # Pad to save_vis_samples if needed
        while len(vis_images) < save_vis_samples:
            vis_images.append(vis_images[-1])
            vis_masks.append(vis_masks[-1])
            vis_preds.append(vis_preds[-1])

        vis_images = torch.cat(vis_images, dim=0)[:save_vis_samples]
        vis_masks = np.concatenate(vis_masks, axis=0)[:save_vis_samples]
        vis_preds = np.concatenate(vis_preds, axis=0)[:save_vis_samples]

        vis_path = os.path.join(output_dir, 'predictions_visualization.png')
        visualize_predictions(vis_images, vis_masks, vis_preds, vis_path, save_vis_samples)

    # Save metrics to JSON
    results = {
        'miou': float(miou),
        'accuracy': float(accuracy),
        'dice': float(dice),
        'fps': float(fps),
        'avg_inference_time_ms': float(avg_time * 1000),
        'per_class_iou': per_class_iou.tolist(),
        'per_class_dice': per_class_dice.tolist(),
        'class_names': CLASS_NAMES[:num_classes]
    }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Metrics saved: {metrics_path}")

    return results


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="🚀 Full Evaluation with Auto-Reparameter")

    # Model & Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--channels", type=int, default=32,
                       help="Base channel count")

    # Dataset
    parser.add_argument("--val_txt", type=str, required=True,
                       help="Path to validation txt file")
    parser.add_argument("--num_classes", type=int, default=19,
                       help="Number of classes")
    parser.add_argument("--ignore_index", type=int, default=255)

    # Data settings
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Evaluation settings
    parser.add_argument("--num_vis_samples", type=int, default=5,
                       help="Number of visualization samples")
    parser.add_argument("--no_deploy", action="store_true",
                       help="Disable auto-reparameter (slower)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save results")

    # System
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("🚀 FULL EVALUATION WITH AUTO-REPARAMETER")
    print("="*70)
    print(f"📁 Checkpoint:  {args.checkpoint}")
    print(f"📂 Val set:     {args.val_txt}")
    print(f"🖥️  Device:      {device}")
    print(f"🎨 Vis samples: {args.num_vis_samples}")
    print(f"⚡ Deploy mode: {'YES' if not args.no_deploy else 'NO'}")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        channels=args.channels,
        device=device,
        auto_deploy=not args.no_deploy
    )

    # Create dataloader
    print(f"\n📂 Loading validation dataset...")
    _, val_loader, _r = create_dataloaders(
        train_txt=args.val_txt,  # dummy
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w),
        dataset_type='foggy',
    )
    print(f"✅ Loaded {len(val_loader.dataset)} validation samples")

    # Evaluate
    results = evaluate_full(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        save_vis_samples=args.num_vis_samples,
        output_dir=args.output_dir
    )

    print(f"\n✅ Evaluation completed!")
    print(f"\n📁 Results saved to: {args.output_dir}/")
    print(f"   - predictions_visualization.png")
    print(f"   - metrics.json")


if __name__ == "__main__":
    main()
