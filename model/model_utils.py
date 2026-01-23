import torch
import torch.nn as nn


def replace_bn_with_gn(model, group_size=8, eps=1e-5):
    """✅ DYNAMIC: num_groups = channels // group_size"""
    gn_count = bn_count = 0
    
    def _convert_recursive(m):
        nonlocal gn_count, bn_count
        
        for child_name, child in m.named_children():
            if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                bn_count += 1
                C = child.num_features
                
                # ✅ DYNAMIC groups: C//8 (min 16, max C)
                num_groups = max(16, min(C, C // group_size))
                
                gn = nn.GroupNorm(num_groups, C, eps=eps)
                
                # Copy params
                if child.weight is not None: gn.weight.data.copy_(child.weight.data)
                if child.bias is not None:   gn.bias.data.copy_(child.bias.data)
                
                # Replace
                setattr(m, child_name, gn)
                gn_count += 1
                
                print(f"✅ BN{C} → GN{num_groups} ({C//num_groups}ch/group)")
            
            else:
                _convert_recursive(child)
    
    _convert_recursive(model)
    
    print(f"\n🎯 DYNAMIC BN→GN: {bn_count}→{gn_count}")
    assert gn_count == bn_count, "❌ Incomplete conversion!"
    
    # Final verify
    remaining_bn = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)))
    print(f"✅ {remaining_bn} BN remaining")
    return gn_count


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
