"""
Quantize model for faster inference with minimal accuracy loss
"""
import torch
import torch.quantization
from pathlib import Path
import argparse
import time
import numpy as np

from convert_to_deploy import build_model

def quantize_dynamic(model):
    """Dynamic quantization (easiest, ~2x speedup)"""
    print("\n🔧 Applying dynamic quantization...")
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    print("✅ Dynamic quantization applied")
    return quantized_model

def main():
    parser = argparse.ArgumentParser(description="Quantize model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--method", default="dynamic", choices=["dynamic"])
    
    args = parser.parse_args()
    
    device = "cpu"  # Quantization works best on CPU
    
    print(f"\n{'='*70}")
    print("🚀 MODEL QUANTIZATION")
    print(f"{'='*70}\n")
    
    # Load model
    print("📦 Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = build_model(num_classes=args.num_classes, deploy=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Get original size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    
    # Quantize
    if args.method == "dynamic":
        quantized_model = quantize_dynamic(model)
    
    # Get quantized size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1e6
    
    # Save
    print(f"\n💾 Saving quantized model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model': quantized_model.state_dict(),
        'num_classes': args.num_classes,
        'quantized': True,
        'method': args.method,
    }, output_path)
    
    print(f"✅ Saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print(f"Original size:  {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Reduction:      {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
