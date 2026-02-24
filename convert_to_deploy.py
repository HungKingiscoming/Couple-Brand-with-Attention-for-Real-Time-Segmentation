"""
FIXED: Convert trained model to deploy mode with reparameterization
"""
import torch
import torch.nn as nn
from pathlib import Path
import argparse

from model.backbone.model import GCNetWithEnhance
from model.head.segmentation_head import GCNetHead, GCNetAuxHead

class Segmentor(nn.Module):
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)

def auto_detect_dwsa_stages(state_dict):
    """
    Auto-detect which DWSA stages are present in checkpoint
    """
    stages = []
    if any('dwsa4' in k for k in state_dict.keys()):
        stages.append('stage4')
    if any('dwsa5' in k for k in state_dict.keys()):
        stages.append('stage5')
    if any('dwsa6' in k for k in state_dict.keys()):
        stages.append('stage6')
    
    return stages if stages else ['stage5', 'stage6']  # Default

def build_model(num_classes=19, deploy=False, dwsa_stages=None):
    """Build model with deploy mode support"""
    
    # Auto-detect or use provided
    if dwsa_stages is None:
        dwsa_stages = ['stage4', 'stage5', 'stage6']
    
    cfg = {
        'in_channels': 3,
        'channels': 32,
        'ppm_channels': 128,
        'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
        'dwsa_stages': dwsa_stages,  # ✅ Configurable
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
        'deploy': deploy,
        'norm_cfg': {'type': 'BN', 'requires_grad': True},
        'act_cfg': {'type': 'ReLU', 'inplace': False}
    }
    
    backbone = GCNetWithEnhance(**cfg)
    
    # Detect channels
    backbone.eval()
    with torch.no_grad():
        sample = torch.randn(1, 3, 512, 1024)
        feats = backbone(sample)
    
    # Handle different output formats
    if isinstance(feats, dict):
        c1_channels = feats.get('c1', torch.zeros(1, 32, 1, 1)).shape[1]
        c2_channels = feats.get('c2', torch.zeros(1, 64, 1, 1)).shape[1]
        c4_channels = feats.get('c4', torch.zeros(1, 128, 1, 1)).shape[1]
        c5_channels = feats.get('c5', torch.zeros(1, 128, 1, 1)).shape[1]
    else:
        c1_channels = 32
        c2_channels = 64
        c4_channels = 128
        c5_channels = 128
    
    head = GCNetHead(
        in_channels=c5_channels,
        c1_channels=c1_channels,
        c2_channels=c2_channels,
        decoder_channels=128,
        num_classes=num_classes,
        dropout_ratio=0.1,
        use_gated_fusion=True,
        norm_cfg={'type': 'BN', 'requires_grad': True},
        act_cfg={'type': 'ReLU', 'inplace': False},
        align_corners=False
    )
    
    aux_head = GCNetAuxHead(
        in_channels=c4_channels,
        channels=96,
        num_classes=num_classes,
        dropout_ratio=0.1,
        norm_cfg={'type': 'BN', 'requires_grad': True},
        act_cfg={'type': 'ReLU', 'inplace': False},
        align_corners=False
    )
    
    return Segmentor(backbone=backbone, head=head, aux_head=aux_head)

def switch_to_deploy(model):
    """
    Convert all GCBlock modules to deploy mode
    This fuses BN into Conv for faster inference
    """
    print("\n🔧 Converting to deploy mode...")
    count = 0
    
    def convert_module(module):
        nonlocal count
        if hasattr(module, 'switch_to_deploy'):
            try:
                module.switch_to_deploy()
                count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to convert module: {e}")
        
        for child in module.children():
            convert_module(child)
    
    convert_module(model)
    
    print(f"✅ Converted {count} modules to deploy mode")
    return model

def main():
    parser = argparse.ArgumentParser(description="Convert model to deploy mode")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", required=True, help="Path to save deploy model")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--verify", action="store_true", help="Verify output match")
    parser.add_argument("--skip_convert", action="store_true", help="Skip deploy conversion (just save)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("🚀 MODEL DEPLOYMENT CONVERTER")
    print(f"{'='*70}")
    print(f"📥 Input:  {args.checkpoint}")
    print(f"💾 Output: {args.output}")
    print(f"{'='*70}\n")
    
    # Load training checkpoint
    print("📦 Loading training checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect DWSA stages
    dwsa_stages = auto_detect_dwsa_stages(state_dict)
    print(f"🔍 Detected DWSA stages: {dwsa_stages}")
    
    # Build model in training mode
    print("🏗️  Building model (training mode)...")
    model_train = build_model(
        num_classes=args.num_classes, 
        deploy=False,
        dwsa_stages=dwsa_stages
    )
    
    # Try to load state dict
    try:
        model_train.load_state_dict(state_dict, strict=True)
        print("✅ State dict loaded (strict=True)")
    except RuntimeError as e:
        print(f"⚠️  Failed with strict=True: {str(e)[:200]}")
        print("    Trying with strict=False...")
        missing, unexpected = model_train.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"    Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        
        print("✅ State dict loaded (strict=False)")
    
    model_train = model_train.to(device)
    model_train.eval()
    
    print("✅ Training model loaded\n")
    
    # Verify before conversion
    if args.verify:
        print("🔍 Testing training model...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 512, 1024).to(device)
            try:
                output_train = model_train(test_input)
                print(f"   Output shape: {output_train.shape}")
                print(f"   Output range: [{output_train.min():.4f}, {output_train.max():.4f}]\n")
            except Exception as e:
                print(f"   ❌ Forward failed: {e}\n")
                return
    
    # Convert to deploy mode
    if not args.skip_convert:
        model_deploy = switch_to_deploy(model_train)
    else:
        model_deploy = model_train
        print("\n⏭️  Skipping deploy conversion (--skip_convert)\n")
    
    # Verify after conversion
    if args.verify and not args.skip_convert:
        print("\n🔍 Verifying deploy model...")
        with torch.no_grad():
            try:
                output_deploy = model_deploy(test_input)
                
                print(f"   Output shape: {output_deploy.shape}")
                print(f"   Output range: [{output_deploy.min():.4f}, {output_deploy.max():.4f}]")
                
                # Check difference
                diff = (output_train - output_deploy).abs().max().item()
                print(f"   Max difference: {diff:.6f}")
                
                if diff < 1e-4:
                    print("   ✅ Outputs match! (diff < 1e-4)")
                elif diff < 1e-3:
                    print("   ⚠️  Small difference (1e-4 < diff < 1e-3)")
                else:
                    print("   ❌ Large difference! (diff > 1e-3)")
            except Exception as e:
                print(f"   ❌ Deploy forward failed: {e}")
    
    # Save deploy checkpoint
    print(f"\n💾 Saving deploy model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    deploy_checkpoint = {
        'model': model_deploy.state_dict(),
        'num_classes': args.num_classes,
        'dwsa_stages': dwsa_stages,
        'deploy_mode': not args.skip_convert,
        'original_checkpoint': args.checkpoint,
    }
    
    if 'epoch' in checkpoint:
        deploy_checkpoint['epoch'] = checkpoint['epoch']
    if 'best_miou' in checkpoint:
        deploy_checkpoint['best_miou'] = checkpoint['best_miou']
    
    torch.save(deploy_checkpoint, output_path)
    
    print(f"✅ Deploy model saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    
    train_params = sum(p.numel() for p in model_train.parameters())
    deploy_params = sum(p.numel() for p in model_deploy.parameters())
    
    print(f"Training params: {train_params:,} ({train_params/1e6:.2f}M)")
    print(f"Deploy params:   {deploy_params:,} ({deploy_params/1e6:.2f}M)")
    if train_params > deploy_params:
        print(f"Reduction:       {(train_params-deploy_params)/train_params*100:.1f}%")
    
    if args.verify and 'diff' in locals():
        print(f"\nOutput difference: {diff:.6f}")
    
    print(f"{'='*70}\n")
    print("✅ Conversion completed!\n")

if __name__ == "__main__":
    main()
