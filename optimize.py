#!/usr/bin/env python3
# ============================================
# optimize_weight_size_FIXED.py - FIXED BatchNorm stats
# ============================================
"""
FIXED VERSION - Giữ đầy đủ BatchNorm running_mean/var
"""

import torch
import argparse
from pathlib import Path


def analyze_checkpoint(checkpoint_path):
    """Phân tích kích thước các thành phần trong checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("\n" + "="*70)
    print("📦 CHECKPOINT ANALYSIS")
    print("="*70)

    total_size = Path(checkpoint_path).stat().st_size / (1024*1024)
    print(f"Total file size: {total_size:.2f} MB")
    print("\nComponents:")

    component_sizes = {}

    for key, value in ckpt.items():
        if isinstance(value, dict):
            size = sum(v.numel() * v.element_size() for v in value.values() 
                      if isinstance(v, torch.Tensor)) / (1024*1024)
            component_sizes[key] = size
            print(f"  - {key:<20} {size:>8.2f} MB")
        elif isinstance(value, torch.Tensor):
            size = value.numel() * value.element_size() / (1024*1024)
            component_sizes[key] = size
            print(f"  - {key:<20} {size:>8.2f} MB")
        else:
            print(f"  - {key:<20} (metadata)")

    # Model breakdown
    if 'model' in ckpt:
        print("\n  Model components breakdown:")
        model_params = ckpt['model']

        backbone_size = sum(v.numel() * v.element_size() for k, v in model_params.items() 
                           if 'backbone' in k) / (1024*1024)
        head_size = sum(v.numel() * v.element_size() for k, v in model_params.items() 
                       if 'decode_head' in k or 'head' in k) / (1024*1024)
        aux_size = sum(v.numel() * v.element_size() for k, v in model_params.items() 
                      if 'aux' in k) / (1024*1024)

        print(f"    • backbone:     {backbone_size:>8.2f} MB")
        print(f"    • decode_head:  {head_size:>8.2f} MB")
        print(f"    • aux_head:     {aux_size:>8.2f} MB")

        # ✅ Check BatchNorm stats
        bn_keys = [k for k in model_params.keys() 
                   if 'running_mean' in k or 'running_var' in k]
        print(f"    • BN stats:     {len(bn_keys)} keys")

    print("="*70)

    return component_sizes


def optimize_checkpoint_size(input_path, output_path, 
                             keep_optimizer=False,
                             use_fp16=False,
                             remove_aux=True):
    """
    ✅ FIXED: Giữ đầy đủ BatchNorm running_mean/var

    Args:
        input_path: Đường dẫn checkpoint gốc
        output_path: Đường dẫn lưu checkpoint tối ưu
        keep_optimizer: Giữ optimizer state (False = remove)
        use_fp16: Convert sang FP16 (True = half size)
        remove_aux: Remove aux_head (chỉ dùng training)
    """
    print("\n" + "="*70)
    print("🔧 OPTIMIZING CHECKPOINT SIZE (FIXED)")
    print("="*70)

    # Load checkpoint
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    original_size = Path(input_path).stat().st_size / (1024*1024)
    print(f"Original size: {original_size:.2f} MB")

    # Create optimized checkpoint
    optimized = {}

    # 1. Process model weights
    if 'model' in ckpt:
        model_state = {}  # ← NEW: Create new dict instead of copy

        # ✅ FIX: Iterate and preserve ALL keys (including BN stats)
        for key, param in ckpt['model'].items():
            # Skip aux_head if requested
            if remove_aux and ('aux_head' in key or 'auxhead' in key):
                continue

            # ✅ FIX: Convert to FP16 but KEEP BN running stats as FP32
            if use_fp16:
                # Check if it's a trainable parameter (weight/bias)
                if param.dtype == torch.float32:
                    # Keep BN running_mean/var as FP32 for stability
                    if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                        model_state[key] = param  # ← Keep FP32
                    else:
                        model_state[key] = param.half()  # ← Convert to FP16
                else:
                    model_state[key] = param
            else:
                # FP32 mode - keep everything as is
                model_state[key] = param

        # Count what we kept
        bn_stats_kept = len([k for k in model_state.keys() 
                            if 'running_mean' in k or 'running_var' in k])
        removed_aux = len(ckpt['model']) - len(model_state)

        optimized['model'] = model_state

        print(f"✅ Model weights: {len(model_state)} keys")
        print(f"   • BN stats kept: {bn_stats_kept}")
        if removed_aux > 0:
            print(f"   • Removed aux_head: {removed_aux} keys")

    # 2. Keep metadata
    for key in ['epoch', 'best_miou', 'metrics']:
        if key in ckpt:
            optimized[key] = ckpt[key]

    # 3. Optional: keep optimizer
    if keep_optimizer and 'optimizer' in ckpt:
        optimized['optimizer'] = ckpt['optimizer']
        print("✅ Kept optimizer state")
    else:
        print("✅ Removed optimizer state")

    # 4. Remove training-only components
    removed = []
    for key in ['scheduler', 'scaler', 'global_step']:
        if key in ckpt and key not in optimized:
            removed.append(key)

    if removed:
        print(f"✅ Removed: {', '.join(removed)}")

    # Add optimization info
    optimized['optimized'] = True
    optimized['fp16'] = use_fp16
    optimized['aux_removed'] = remove_aux

    # Save
    torch.save(optimized, output_path)
    new_size = Path(output_path).stat().st_size / (1024*1024)

    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Original size:  {original_size:>8.2f} MB")
    print(f"Optimized size: {new_size:>8.2f} MB")
    print(f"Reduction:      {original_size - new_size:>8.2f} MB ({(1-new_size/original_size)*100:.1f}%)")

    if use_fp16:
        print("\n⚠️  FP16 Mode:")
        print("   • Weights/bias: FP16")
        print("   • BN running stats: FP32 (for stability)")

    print("="*70)

    return new_size, original_size


def main():
    parser = argparse.ArgumentParser(description="🔧 Optimize Checkpoint File Size (FIXED)")

    parser.add_argument("--input", type=str, required=True,
                       help="Input checkpoint path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output checkpoint path (default: input_inference.pth)")

    # Optimization options
    parser.add_argument("--fp16", action="store_true",
                       help="Convert to FP16 (half size, ~0.1%% accuracy loss)")
    parser.add_argument("--keep-optimizer", action="store_true",
                       help="Keep optimizer state (for resume training)")
    parser.add_argument("--keep-aux", action="store_true",
                       help="Keep aux_head (usually not needed for inference)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze, don't optimize")

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        suffix = "_fp16" if args.fp16 else "_inference"
        args.output = str(input_path.parent / f"{input_path.stem}{suffix}.pth")

    print("\n" + "="*70)
    print("🔧 CHECKPOINT SIZE OPTIMIZER (FIXED)")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Analyze
    analyze_checkpoint(args.input)

    if not args.analyze_only:
        # Optimize
        optimize_checkpoint_size(
            input_path=args.input,
            output_path=args.output,
            keep_optimizer=args.keep_optimizer,
            use_fp16=args.fp16,
            remove_aux=not args.keep_aux
        )

        print(f"\n✅ Optimized checkpoint saved: {args.output}")

        # Usage instructions
        print("\n💡 USAGE:")
        if args.fp16:
            print("   # Load FP16 checkpoint:")
            print("   model.load_state_dict(checkpoint['model'])")
            print("   model.half()  # Convert model to FP16")
            print("   images = images.half()  # Input as FP16")
        else:
            print("   # Load FP32 checkpoint (normal):")
            print("   model.load_state_dict(checkpoint['model'])")

        print("\n   python evaluation_deploy.py \\")
        print(f"       --checkpoint {args.output}")


if __name__ == "__main__":
    main()
