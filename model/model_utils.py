import torch
import torch.nn as nn


def replace_bn_with_gn(model, ch_per_group=8, eps=1e-5):
    """✅ PRODUCTION READY - Your logic + fixes"""
    bn_count = gn_count = 0
    
    def convert(m):
        nonlocal bn_count, gn_count
        
        for name, child in m.named_children():
            if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                bn_count += 1
                C = child.num_features
                
                num_groups = max(1, C // ch_per_group)
                
                # ✅ Ensure divisible (optimized)
                original_g = num_groups
                while num_groups > 0 and C % num_groups != 0:
                    num_groups -= 1
                
                gn = nn.GroupNorm(num_groups, C, eps=eps)
                
                # ✅ SAFE copy (check exists)
                if hasattr(child, 'weight') and child.weight is not None:
                    gn.weight.data.copy_(child.weight.data)
                if hasattr(child, 'bias') and child.bias is not None:
                    gn.bias.data.copy_(child.bias.data)
                
                setattr(m, name, gn)
                gn_count += 1
                
                ratio = C // num_groups
                print(f"✅ BN{C} → GN{num_groups} ({ratio}ch)")
            
            else:
                convert(child)
    
    convert(model)
    
    # ✅ FINAL VERIFY
    bn_left = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)))
    assert bn_left == 0, f"❌ {bn_left} BN left!"
    
    print(f"\n🎯 SUCCESS: {bn_count}→{gn_count} GN | 0 BN remaining")
    return model 


def init_weights(module):
    """
    Apply robust Kaiming (He) initialization for training from scratch.
    
    This is CRITICAL for from-scratch training without pretrained weights.
    Default initialization is too weak. Kaiming init jumpstarts learning.
    
    Args:
        module: PyTorch module (usually apply with model.apply(init_weights))
    
    Example:
        >>> model.apply(init_weights)
        >>> # Now model has proper Kaiming initialization
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        # Kaiming Normal (He Init) for ReLU/GeLU networks
        # Fan-out mode: good for conv layers
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize bias to 0
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        # Normalization layers: weight=1.0, bias=0.0
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def count_parameters(model):
    """
    Count total trainable parameters in the model.
    
    Args:
        model: PyTorch module
    
    Returns:
        int: Number of trainable parameters
    
    Example:
        >>> total = count_parameters(model)
        >>> print(f"Model has {total:,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_health(model):
    """
    Verify model is ready for training with small batches.
    Checks that BatchNorm has been replaced with GroupNorm.
    
    Args:
        model: PyTorch module
    
    Returns:
        bool: True if model is healthy (no BatchNorm), False otherwise
    
    Example:
        >>> is_healthy = check_model_health(model)
        >>> if not is_healthy:
        >>>     model = replace_bn_with_gn(model)
    """
    has_bn = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            has_bn = True
            break
    
    if has_bn:
        print("⚠️  WARNING: BatchNorm detected!")
        print("   Recommended to replace with GroupNorm for batch_size < 16.")
        return False
    else:
        print("✅ Model healthy: No BatchNorm detected (GroupNorm active).")
        return True


# Example usage in train.py:
# 
# from model_utils import replace_bn_with_gn, init_weights, check_model_health
#
# # After creating model
# model = Segmentor(...)
#
# # Fix for small batch training
# if args.batch_size < 16:
#     print("Optimizing model for small batch size...")
#     model = replace_bn_with_gn(model)
#
# # Fix for from-scratch training
# if args.from_scratch:
#     print("Applying Kaiming initialization...")
#     model.apply(init_weights)
#
# # Verify
# check_model_health(model)
