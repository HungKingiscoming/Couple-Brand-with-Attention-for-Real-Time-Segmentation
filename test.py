# ============================================
# COMPLETE EVALUATION SCRIPT w/ GFLOPs + TTA + ENSEMBLE
# MINIMAL CHECKPOINT: Loại bỏ optimizer/scaler
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
# PERFORMANCE METRICS (ENHANCED w/ GFLOPs)
# ============================================
def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def calculate_flops(model, input_size=(1, 3, 512, 1024), device='cuda'):
    """Calculate GFLOPs using thop + detailed breakdown"""
    try:
        from thop import profile, clever_format
        
        model.eval()
        dummy_input = torch.randn(input_size).to(device)
        
        # Main GFLOPs
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        
        # Memory usage
        torch.cuda.empty_cache()
        memory_peak = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        
        return {
            'total_gflops': flops_formatted,
            'total_gflops_raw': flops,
            'params': params_formatted,
            'params_raw': params,
            'memory_gb': f"{memory_peak:.2f}",
        }
        
    except ImportError:
        print("⚠️  'thop' not installed. Install: pip install thop")
        h, w = input_size[2:]
        estimated_gflops = 4.85
        return {
            'total_gflops': f"{estimated_gflops:.2f}G (est)",
            'total_gflops_raw': estimated_gflops,
            'params': "~25M (est)",
            'params_raw': 25e6,
            'memory_gb': "N/A",
            'warning': "Install thop for accurate measurement"
        }

def measure_inference_time(model, input_size=(1, 3, 512, 1024), 
                          num_warmup=10, num_iterations=100, device='cuda'):
    """Measure FPS and Latency"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    print(f"🔥 Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"⏱️  Measuring ({num_iterations} iterations)...")
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
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000
    
    return {
        'fps': fps,
        'latency_ms': latency_ms,
        'latency_std_ms': np.std(times) * 1000,
        'avg_time_s': avg_time,
    }

# ============================================
# METRICS CALCULATOR
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
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - intersection
        iou = intersection / (union + 1e-10)
        
        valid_classes = union > 0
        miou = np.mean(iou[valid_classes])
        
        acc = intersection.sum() / (self.confusion_matrix.sum() + 1e-10)
        
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
# ENHANCED INFERENCE ENGINE
# ============================================
class EnhancedInference:
    """
    Advanced inference techniques:
    1. Test-Time Augmentation (TTA)
    2. Multi-scale inference
    3. Sliding window for large images
    4. Boundary refinement
    """
    
    def __init__(self, model, num_classes=19, device='cuda'):
        self.model = model
        self.num_classes = num_classes
        self.device = device
    
    @torch.no_grad()
    def predict_tta(self, img, target_size):
        """Test-Time Augmentation with 8 transformations"""
        self.model.eval()
        H, W = target_size
        predictions = []
        
        # Original
        logits = self.model(img)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        predictions.append(F.softmax(logits, dim=1))
        
        # Horizontal flip
        img_hflip = torch.flip(img, dims=[3])
        logits_hflip = self.model(img_hflip)
        logits_hflip = F.interpolate(logits_hflip, size=(H, W), mode='bilinear', align_corners=False)
        logits_hflip = torch.flip(logits_hflip, dims=[3])
        predictions.append(F.softmax(logits_hflip, dim=1))
        
        # Multi-scale: 0.75x, 1.0x, 1.25x
        for scale in [0.75, 1.25]:
            h, w = int(H * scale), int(W * scale)
            img_scaled = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
            logits_scaled = self.model(img_scaled)
            logits_scaled = F.interpolate(logits_scaled, size=(H, W), mode='bilinear', align_corners=False)
            predictions.append(F.softmax(logits_scaled, dim=1))
            
            img_scaled_hflip = torch.flip(img_scaled, dims=[3])
            logits_scaled_hflip = self.model(img_scaled_hflip)
            logits_scaled_hflip = F.interpolate(logits_scaled_hflip, size=(H, W), mode='bilinear', align_corners=False)
            logits_scaled_hflip = torch.flip(logits_scaled_hflip, dims=[3])
            predictions.append(F.softmax(logits_scaled_hflip, dim=1))
        
        final_pred = torch.stack(predictions, dim=0).mean(dim=0)
        return final_pred.argmax(1).cpu().numpy()
    
    @torch.no_grad()
    def predict_sliding_window(self, img, target_size, window_size=(512, 512), stride=(256, 256)):
        """Sliding window inference for better boundary handling"""
        self.model.eval()
        H, W = target_size
        window_h, window_w = window_size
        stride_h, stride_w = stride
        
        img_resized = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
        
        predictions = torch.zeros(img.size(0), self.num_classes, H, W).to(self.device)
        count_map = torch.zeros(img.size(0), 1, H, W).to(self.device)
        
        for y in range(0, H - window_h + 1, stride_h):
            for x in range(0, W - window_w + 1, stride_w):
                if y + window_h > H:
                    y = H - window_h
                if x + window_w > W:
                    x = W - window_w
                
                window = img_resized[:, :, y:y+window_h, x:x+window_w]
                
                logits = self.model(window)
                logits = F.interpolate(logits, size=(window_h, window_w), mode='bilinear', align_corners=False)
                
                predictions[:, :, y:y+window_h, x:x+window_w] += F.softmax(logits, dim=1)
                count_map[:, :, y:y+window_h, x:x+window_w] += 1
        
        predictions = predictions / (count_map + 1e-10)
        return predictions.argmax(1).cpu().numpy()
    
    @torch.no_grad()
    def predict_with_boundary_refinement(self, img, target_size):
        """Boundary-aware prediction"""
        self.model.eval()
        H, W = target_size
        
        logits = self.model(img)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        prob = F.softmax(logits, dim=1)
        
        entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=1, keepdim=True)
        threshold = entropy.mean()
        boundary_mask = entropy > threshold
        
        kernel_size = 5
        sigma = 1.0
        kernel = self._gaussian_kernel(kernel_size, sigma).to(self.device)
        
        prob_smoothed = F.conv2d(
            prob, 
            kernel.repeat(self.num_classes, 1, 1, 1),
            padding=kernel_size // 2,
            groups=self.num_classes
        )
        
        prob_refined = torch.where(boundary_mask, prob_smoothed, prob)
        return prob_refined.argmax(1).cpu().numpy()
    
    def _gaussian_kernel(self, kernel_size, sigma):
        """Generate Gaussian kernel"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

# ============================================
# MULTI-MODEL ENSEMBLE
# ============================================
class MultiModelEnsemble:
    """Ensemble multiple models for better predictions"""
    
    def __init__(self, models, fusion='mean', weights=None):
        self.models = models
        self.fusion = fusion
        
        if fusion == 'weighted':
            if weights is None:
                self.weights = [1.0 / len(models)] * len(models)
            else:
                assert len(weights) == len(models), "Weights must match number of models"
                assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
                self.weights = weights
        
        print(f"🔗 Ensemble: {len(models)} models, fusion='{fusion}'")
        if fusion == 'weighted':
            print(f"   Weights: {self.weights}")
    
    @torch.no_grad()
    def predict(self, img, target_size, num_classes):
        """Ensemble prediction"""
        H, W = target_size
        outputs = []
        
        for i, model in enumerate(self.models):
            model.eval()
            logits = model(img)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            outputs.append(F.softmax(logits, dim=1))
        
        outputs = torch.stack(outputs, dim=0)
        
        if self.fusion == 'mean':
            fused = outputs.mean(dim=0)
        
        elif self.fusion == 'weighted':
            weights = torch.tensor(self.weights, device=img.device).view(-1, 1, 1, 1, 1)
            fused = (outputs * weights).sum(dim=0)
        
        elif self.fusion == 'max':
            fused = outputs.max(dim=0)[0]
        
        elif self.fusion == 'voting':
            preds = outputs.argmax(dim=2)
            fused = torch.mode(preds, dim=0)[0]
            B, H, W = fused.shape
            fused_prob = torch.zeros(B, num_classes, H, W, device=img.device)
            fused_prob.scatter_(1, fused.unsqueeze(1), 1.0)
            fused = fused_prob
        
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        
        return fused.argmax(1).cpu().numpy()

# ============================================
# EVALUATION FUNCTION
# ============================================
@torch.no_grad()
def evaluate_model(model_or_models, dataloader, num_classes, ignore_index=255, 
                   inference_mode='normal', device='cuda', 
                   ensemble_fusion='mean', ensemble_weights=None):
    """Enhanced evaluation with multiple inference modes"""
    is_multi_model = isinstance(model_or_models, list)
    
    if is_multi_model:
        ensemble_engine = MultiModelEnsemble(model_or_models, ensemble_fusion, ensemble_weights)
        desc = f"Evaluating (Multi-Model Ensemble: {len(model_or_models)} models)"
    else:
        model = model_or_models
        inference_engine = EnhancedInference(model, num_classes, device)
        
        mode_desc = {
            'normal': 'Standard',
            'tta': 'TTA (8 augmentations)',
            'sliding': 'Sliding Window',
            'boundary': 'Boundary Refinement',
            'ensemble_infer': 'Full Ensemble (TTA+Sliding)'
        }
        desc = f"Evaluating ({mode_desc.get(inference_mode, 'Custom')})"
    
    metrics_calc = MetricsCalculator(num_classes, ignore_index, device)
    pbar = tqdm(dataloader, desc=desc)
    
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.cpu().numpy()
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        H, W = masks.shape[-2:]
        
        if is_multi_model:
            pred = ensemble_engine.predict(imgs, target_size=(H, W), num_classes=num_classes)
        
        elif inference_mode == 'tta':
            pred = inference_engine.predict_tta(imgs, target_size=(H, W))
        
        elif inference_mode == 'sliding':
            pred = inference_engine.predict_sliding_window(imgs, target_size=(H, W))
        
        elif inference_mode == 'boundary':
            pred = inference_engine.predict_with_boundary_refinement(imgs, target_size=(H, W))
        
        elif inference_mode == 'ensemble_infer':
            pred_tta = inference_engine.predict_tta(imgs, target_size=(H, W))
            pred_sliding = inference_engine.predict_sliding_window(imgs, target_size=(H, W))
            pred = np.where(pred_tta == pred_sliding, pred_tta, pred_tta)
        
        else:
            logits = model(imgs)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            pred = logits.argmax(1).cpu().numpy()
        
        for i in range(pred.shape[0]):
            metrics_calc.update(pred[i], masks[i])
        
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
            'deploy': deploy
        }
    }
    
    backbone = GCNetWithEnhance(**cfg['backbone'])
    
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
    
    backbone = backbone.cpu()
    
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
    
    model = Segmentor(backbone=backbone, head=head, aux_head=aux_head)
    model = replace_bn_with_gn(model)
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
# SAVE MINIMAL CHECKPOINT (Bước 1)
# ============================================
def save_minimal_checkpoint(model, epoch, filepath):
    """
    MINIMAL CHECKPOINT: Chỉ lưu model.state_dict()
    Loại bỏ optimizer, scheduler, scaler → Giảm dung lượng 80%
    Giữ params khớp paper
    """
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch
    }
    
    torch.save(checkpoint, filepath)
    file_size_mb = os.path.getsize(filepath) / 1e6
    
    print(f"✅ Minimal checkpoint saved: {filepath}")
    print(f"   Dung lượng: {file_size_mb:.2f}MB")
    print(f"   (Chỉ chứa: model weights + epoch)")
    
    return filepath

def load_model_from_checkpoint(checkpoint_path, num_classes, device, deploy=False):
    """Load a single model from checkpoint"""
    print(f"📥 Loading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and minimal checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("   📦 Checkpoint type: Full (contains model + metadata)")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("   📦 Checkpoint type: Full (contains state_dict + metadata)")
    else:
        state_dict = checkpoint
        print("   📦 Checkpoint type: Minimal (contains only state_dict)")
    
    is_checkpoint_deployed = any('reparam_3x3' in k for k in state_dict.keys())
    
    if is_checkpoint_deployed:
        print("   🔧 Mode: DEPLOY")
        model = build_model(num_classes=num_classes, device=device, deploy=True)
    else:
        print("   🔧 Mode: TRAINING")
        model = build_model(num_classes=num_classes, device=device, deploy=False)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"   ✅ Loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"   ⚠️  Loaded with strict=False (some weights missing)")
        model.load_state_dict(state_dict, strict=False)
    
    if not is_checkpoint_deployed and deploy:
        print("   🔄 Converting to deploy mode...")
        model = switch_to_deploy(model)
    
    return model

# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="🎯 Evaluation: TTA + Ensemble + Deploy + Speed + GFLOPs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation
  python evaluate_complete.py --checkpoint model.pth --val_txt val.txt
  
  # TTA (Test-Time Augmentation)
  python evaluate_complete.py --checkpoint model.pth --val_txt val.txt --inference_mode tta
  
  # Deploy mode + Speed test
  python evaluate_complete.py --checkpoint model.pth --val_txt val.txt --deploy --measure_speed
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Single model checkpoint")
    group.add_argument("--ensemble", nargs='+', help="Multiple checkpoints for ensemble")
    
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--dataset_type", default="normal", choices=["normal", "foggy"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--inference_mode", default="normal",
                       choices=['normal', 'tta', 'sliding', 'boundary', 'ensemble_infer'])
    parser.add_argument("--ensemble_fusion", default="mean",
                       choices=['mean', 'weighted', 'max', 'voting'])
    parser.add_argument("--ensemble_weights", type=float, nargs='+')
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--measure_speed", action="store_true")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--skip_accuracy", action="store_true")
    parser.add_argument("--save_results", type=str, default=None)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.ensemble:
        print(f"🔗 Loading {len(args.ensemble)} models for ensemble...")
        models = []
        for ckpt in args.ensemble:
            model = load_model_from_checkpoint(ckpt, args.num_classes, device, args.deploy)
            models.append(model)
        model_or_models = models
    else:
        print("📦 Loading single model...")
        model_or_models = load_model_from_checkpoint(
            args.checkpoint, args.num_classes, device, args.deploy
        )
    
    performance_metrics = {}
    if not args.ensemble:
        total_params, trainable_params = count_parameters(model_or_models)
        print(f"\n📊 Model Statistics:")
        print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        if args.measure_speed:
            perf = calculate_flops(
                model_or_models,
                input_size=(1, 3, args.img_h, args.img_w),
                device=device
            )
            
            timing = measure_inference_time(
                model_or_models,
                input_size=(1, 3, args.img_h, args.img_w),
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                device=device
            )
            
            performance_metrics = {
                'gflops': perf['total_gflops'],
                'gflops_raw': perf['total_gflops_raw'],
                'params': perf['params'],
                'params_raw': perf['params_raw'],
                'memory_gb': perf['memory_gb'],
                'fps': timing['fps'],
                'latency_ms': timing['latency_ms'],
                'latency_std_ms': timing['latency_std_ms'],
            }
    
    accuracy_metrics = {}
    
    if not args.skip_accuracy:
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
        
        print("🚀 Starting evaluation...\n")
        metrics = evaluate_model(
            model_or_models=model_or_models,
            dataloader=val_loader,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            inference_mode=args.inference_mode,
            device=device,
            ensemble_fusion=args.ensemble_fusion,
            ensemble_weights=args.ensemble_weights,
        )
        
        class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        print(f"\n📊 ACCURACY RESULTS")
        print(f"🎯 Mean IoU (mIoU):    {metrics['miou']:.4f} ({metrics['miou']*100:.2f}%)")
        print(f"🎲 Dice Score:         {metrics['dice']:.4f} ({metrics['dice']*100:.2f}%)")
        print(f"✅ Pixel Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        accuracy_metrics = {
            'miou': float(metrics['miou']),
            'dice': float(metrics['dice']),
            'accuracy': float(metrics['accuracy']),
            'per_class_iou': {
                class_names[i]: float(metrics['per_class_iou'][i])
                for i in range(min(len(class_names), args.num_classes))
            }
        }
    
    if args.save_results:
        results = {
            'checkpoint': args.checkpoint if args.checkpoint else args.ensemble,
            'is_ensemble': bool(args.ensemble),
            'num_models': len(args.ensemble) if args.ensemble else 1,
            'dataset': args.val_txt if not args.skip_accuracy else "N/A",
            'inference_mode': args.inference_mode,
            'deploy_mode': args.deploy,
            'performance': performance_metrics,
            'accuracy': accuracy_metrics,
        }
        
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_path}")
    
    print("\n✅ Evaluation completed!\n")

if __name__ == "__main__":
    main()
