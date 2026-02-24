"""
Export PyTorch model to ONNX format
"""
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import argparse

from convert_to_deploy import build_model

def export_to_onnx(model, output_path, input_size=(1, 3, 512, 1024), 
                   opset_version=13, simplify=True, dynamic_axes=False):
    """
    Export model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_size: Input tensor size
        opset_version: ONNX opset version (11, 13, or 16)
        simplify: Whether to simplify ONNX graph
        dynamic_axes: Whether to use dynamic input size
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)
    
    # Dynamic axes config
    if dynamic_axes:
        dynamic_config = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    else:
        dynamic_config = None
    
    print(f"\n🔧 Exporting to ONNX...")
    print(f"   Input size: {input_size}")
    print(f"   Opset version: {opset_version}")
    print(f"   Dynamic axes: {dynamic_axes}")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_config,
        verbose=False
    )
    
    print(f"✅ ONNX exported to: {output_path}")
    
    # Verify ONNX model
    print("\n🔍 Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Simplify if requested
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            print("\n🔧 Simplifying ONNX graph...")
            
            simplified_model, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print("✅ ONNX graph simplified")
            else:
                print("⚠️  Simplification failed, keeping original")
        
        except ImportError:
            print("⚠️  onnx-simplifier not installed. Install: pip install onnx-simplifier")
    
    # Test inference with ONNX Runtime
    print("\n🔍 Testing ONNX Runtime inference...")
    
    ort_session = ort.InferenceSession(
        str(output_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # Prepare input
    input_data = dummy_input.cpu().numpy()
    
    # Run ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Run PyTorch
    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()
    
    # Compare
    diff = np.abs(ort_output - torch_output).max()
    print(f"   Output shape: {ort_output.shape}")
    print(f"   Max difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("   ✅ ONNX outputs match PyTorch! (diff < 1e-4)")
    elif diff < 1e-3:
        print("   ⚠️  Small difference (1e-4 < diff < 1e-3)")
    else:
        print("   ❌ Large difference! (diff > 1e-3)")
    
    return ort_session

def benchmark_onnx(ort_session, input_size=(1, 3, 512, 1024), num_iterations=100):
    """Benchmark ONNX model"""
    import time
    
    input_data = np.random.randn(*input_size).astype(np.float32)
    input_name = ort_session.get_inputs()[0].name
    
    # Warmup
    print(f"\n🔥 Warming up...")
    for _ in range(10):
        _ = ort_session.run(None, {input_name: input_data})
    
    # Benchmark
    print(f"⏱️  Benchmarking ({num_iterations} iterations)...")
    times = []
    
    for _ in range(num_iterations):
        start = time.time()
        _ = ort_session.run(None, {input_name: input_data})
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000
    
    print(f"\n📊 ONNX Runtime Performance:")
    print(f"   FPS:     {fps:.1f}")
    print(f"   Latency: {latency_ms:.2f} ms (±{np.std(times)*1000:.2f}ms)")
    
    return fps, latency_ms

def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Deploy checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=13, choices=[11, 13, 16])
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("🚀 ONNX EXPORT")
    print(f"{'='*70}")
    print(f"📥 Input:  {args.checkpoint}")
    print(f"💾 Output: {args.output}")
    print(f"{'='*70}\n")
    
    # Load deploy model
    print("📦 Loading deploy model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = build_model(num_classes=args.num_classes, deploy=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded\n")
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ort_session = export_to_onnx(
        model,
        output_path,
        input_size=(1, 3, args.img_h, args.img_w),
        opset_version=args.opset,
        simplify=args.simplify,
        dynamic_axes=args.dynamic
    )
    
    # Benchmark
    if args.benchmark:
        fps, latency = benchmark_onnx(
            ort_session,
            input_size=(1, 3, args.img_h, args.img_w),
            num_iterations=100
        )
    
    print(f"\n{'='*70}")
    print("✅ Export completed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
