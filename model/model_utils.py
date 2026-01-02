import torch
import torch.nn as nn


def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm.
    
    This is CRITICAL for training with batch_size < 16.
    BatchNorm becomes unreliable when batch size is small.
    GroupNorm works perfectly even with batch_size=1.
    
    Args:
        module: PyTorch module (model)
        num_groups: Number of groups for GroupNorm (default 32)
    
    Returns:
        Module with GroupNorm instead of BatchNorm
    
    Example:
        >>> model = replace_bn_with_gn(model)
        >>> print(model)  # Should not have BatchNorm2d anymore
    """
    # If the module itself is BatchNorm, replace it
    if isinstance(module, nn.BatchNorm2d):
        num_channels = module.num_features
        
        # Ensure num_groups divides num_channels evenly
        current_groups = num_groups
        while num_channels % current_groups != 0:
            current_groups //= 2
        
        # Create GroupNorm with same number of channels
        return nn.GroupNorm(current_groups, num_channels)
    
    # Otherwise, recursively iterate over children
    for name, child in module.named_children():
        module.add_module(name, replace_bn_with_gn(child, num_groups))
    
    return module


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
