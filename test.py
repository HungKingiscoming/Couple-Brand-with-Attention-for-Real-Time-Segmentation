# ============================================
# ENHANCED EVALUATION SCRIPT
# - Compute mIoU, Dice, Accuracy
# - Support Deploy Mode (Reparameterization)
# - Measure GFLOPs, FPS, Latency
# ============================================
import os
import time
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
# PERFORMANCE METRICS
# ============================================
def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def calculate_flops(model, input_size=(1, 3, 512, 1024), device='cuda'):
    """Calculate GFLOPs using thop library"""
    try:
        from thop import profile, clever_format
        
        model.eval()
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        print("⚠️  'thop' not installed. Install with: pip install thop")
        return "N/A", "N/A"

def measure_inference_time(model, input_size=(1, 3, 512, 1024), 
                          num_warmup=10, num_iterations=100, device='cuda'):
    """Measure FPS and Latency"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    print(f"🔥 Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    print(f"⏱️  Measuring inference time ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Timing"):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
    
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000
    
    return {
        'fps': fps,
        'latency_ms': latency_ms,
        'latency_std_ms': std_time * 1000,
        'avg_time_s': avg_time,
    }

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
    
    def update(self, pred, target):
        """Update confusion matrix"""
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
    
    def get_metrics(self):
        """Compute final metrics from confusion matrix"""
        # Per-class IoU
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        
        # Mean IoU
        valid_classes = union > 0
        miou = np.mean(iou[valid_classes])
        
        # Pixel Accuracy
        acc = intersection.sum() / (self.confusion_matrix.sum() + 1e-10)
        
        # Dice Score
        pred_total = self.confusion_matrix.sum(0)
        target_total = self.confusion_matrix.sum(1)
        
        dice_per_class = (2.0 * intersection) / (pred_total + target_total + 1e-10)
        mean_dice = np.mean(dice_per_class[valid_classes])
        
        return {
            'miou': miou,
            'accuracy': acc,
            'dice': mean_dice,
            'per_class_iou': iou,
            'per_class_dice': dice_per_class,
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
def build_model(num_classes=19, device='cuda', deploy=False):
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
            'deploy': deploy  # 🔥 Enable deploy mode
        }
    }
    
    # Build backbone
    backbone = GCNetWithEnhance(**cfg['backbone'])
    
    # Detect channels
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

def switch_to_deploy(model):
    """Convert model to deploy mode (reparameterization)"""
    print("\n🔧 Switching to deploy mode (reparameterization)...")
    
    count = 0
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
            count += 1
    
    print(f"✅ Converted {count} modules to deploy mode")
    return model

# ============================================
# MAIN EVALUATION
# ============================================
def main():
    parser = argparse.ArgumentParser(description="🎯 Enhanced Model Evaluation")
    
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
    parser.add_argument("--deploy", action="store_true",
                       help="Enable deploy mode (reparameterization)")
    parser.add_argument("--skip_accuracy", action="store_true",
                       help="Skip accuracy evaluation (only measure speed)")
    
    # Performance measurement
    parser.add_argument("--measure_speed", action="store_true",
                       help="Measure GFLOPs, FPS, Latency")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=100)
    
    # Output
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save evaluation results (JSON)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("🎯 ENHANCED MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📊 Dataset: {args.val_txt}")
    print(f"🖼️  Image size: {args.img_h}x{args.img_w}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📐 Multi-scale: {args.use_multiscale}")
    print(f"🚀 Deploy mode: {args.deploy}")
    print(f"⚡ Measure speed: {args.measure_speed}")
    print(f"🎯 Device: {device}")
    print(f"{'='*70}\n")
    
    # Build model
    print("🏗️  Building model...")
    model = build_model(num_classes=args.num_classes, device=device, deploy=args.deploy)
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
    
    # Switch to deploy mode if requested
    if args.deploy:
        model = switch_to_deploy(model)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    performance_metrics = {}
    
    # Measure speed if requested
    if args.measure_speed:
        print(f"\n{'='*70}")
        print("⚡ PERFORMANCE MEASUREMENT")
        print(f"{'='*70}")
        
        # Calculate FLOPs
        flops, params = calculate_flops(
            model, 
            input_size=(1, 3, args.img_h, args.img_w),
            device=device
        )
        print(f"📊 GFLOPs: {flops}")
        print(f"📊 Params: {params}")
        
        # Measure inference time
        timing = measure_inference_time(
            model,
            input_size=(1, 3, args.img_h, args.img_w),
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            device=device
        )
        
        print(f"\n{'='*70}")
        print("⏱️  INFERENCE SPEED")
        print(f"{'='*70}")
        print(f"🚀 FPS: {timing['fps']:.2f}")
        print(f"⏱️  Latency: {timing['latency_ms']:.2f} ms (±{timing['latency_std_ms']:.2f} ms)")
        print(f"{'='*70}\n")
        
        performance_metrics = {
            'gflops': flops,
            'params': params,
            'fps': timing['fps'],
            'latency_ms': timing['latency_ms'],
            'latency_std_ms': timing['latency_std_ms'],
        }
    
    accuracy_metrics = {}
    
    # Evaluate accuracy if not skipped
    if not args.skip_accuracy:
        # Create dataloader
        print("📦 Loading dataset...")
        _, val_loader, _ = create_dataloaders(
            train_txt=args.val_txt,
            val_txt=args.val_txt,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=(args.img_h, args.img_w),
            pin_memory=True,
            compute_class_weights=False,
            dataset_type=args.dataset_type
        )
        print(f"✅ Loaded {len(val_loader.dataset)} samples\n")
        
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
        print("📊 ACCURACY RESULTS")
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
        
        for i, (name, iou, dice) in enumerate(zip(class_names, metrics['per_class_iou'], metrics['per_class_dice'])):
            if i < args.num_classes:
                print(f"  {i:2d}. {name:15s}: IoU={iou:.4f} ({iou*100:.2f}%)  |  Dice={dice:.4f} ({dice*100:.2f}%)")
        print("=" * 70)
        
        accuracy_metrics = {
            'miou': float(metrics['miou']),
            'dice': float(metrics['dice']),
            'accuracy': float(metrics['accuracy']),
            'per_class_iou': {
                class_names[i]: float(metrics['per_class_iou'][i])
                for i in range(min(len(class_names), args.num_classes))
            }
        }
    
    # Save results
    if args.save_results:
        results = {
            'checkpoint': args.checkpoint,
            'dataset': args.val_txt if not args.skip_accuracy else "N/A",
            'num_samples': len(val_loader.dataset) if not args.skip_accuracy else 0,
            'multiscale': args.use_multiscale,
            'deploy_mode': args.deploy,
            'model_stats': {
                'total_params': int(total_params),
                'total_params_M': float(total_params / 1e6),
                'trainable_params': int(trainable_params),
            },
            'performance': performance_metrics,
            'accuracy': accuracy_metrics,
        }
        
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Deploy Mode: {'✅ ON' if args.deploy else '❌ OFF'}")
    
    if performance_metrics:
        print(f"\n⚡ Performance:")
        print(f"   GFLOPs: {performance_metrics.get('gflops', 'N/A')}")
        print(f"   FPS: {performance_metrics.get('fps', 'N/A'):.2f}")
        print(f"   Latency: {performance_metrics.get('latency_ms', 'N/A'):.2f} ms")
    
    if accuracy_metrics:
        print(f"\n🎯 Accuracy:")
        print(f"   mIoU: {accuracy_metrics.get('miou', 'N/A'):.4f}")
        print(f"   Dice: {accuracy_metrics.get('dice', 'N/A'):.4f}")
        print(f"   Pixel Acc: {accuracy_metrics.get('accuracy', 'N/A'):.4f}")
    
    print(f"{'='*70}\n")
    print("✅ Evaluation completed!\n")

if __name__ == "__main__":
    main()
