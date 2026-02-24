"""
analyze_checkpoint.py - Phân tích chi tiết checkpoint
"""
import torch
from collections import defaultdict

def analyze_checkpoint(checkpoint_path):
    """Phân tích toàn bộ structure của checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"🔍 ANALYZING CHECKPOINT")
    print(f"{'='*70}")
    print(f"File: {checkpoint_path}\n")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 1. TOP-LEVEL KEYS
    print("📦 Top-level keys:")
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"   - {key}: dict ({len(ckpt[key])} items)")
        elif isinstance(ckpt[key], (int, float, str)):
            print(f"   - {key}: {ckpt[key]}")
        else:
            print(f"   - {key}: {type(ckpt[key])}")
    print()
    
    # 2. GET STATE DICT
    if 'model' in ckpt:
        state_dict = ckpt['model']
        dict_key = 'model'
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        dict_key = 'state_dict'
    else:
        state_dict = ckpt
        dict_key = 'root'
    
    print(f"📦 Using state_dict from: '{dict_key}'")
    print(f"   Total keys: {len(state_dict)}\n")
    
    # 3. GROUP BY MODULE
    print("🏗️  MODEL STRUCTURE (grouped by module):")
    print("-" * 70)
    
    module_groups = defaultdict(list)
    
    for key in state_dict.keys():
        # Get top-level module name
        parts = key.split('.')
        if len(parts) >= 2:
            module_name = f"{parts[0]}.{parts[1]}"
        else:
            module_name = parts[0]
        
        module_groups[module_name].append(key)
    
    # Sort and display
    for module_name in sorted(module_groups.keys()):
        keys = module_groups[module_name]
        print(f"\n{module_name}: ({len(keys)} params)")
        
        # Show first 3 and last 3 keys
        if len(keys) <= 6:
            for k in keys:
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'scalar'
                print(f"   - {k}: {shape}")
        else:
            for k in keys[:3]:
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'scalar'
                print(f"   - {k}: {shape}")
            print(f"   ... ({len(keys) - 6} more)")
            for k in keys[-3:]:
                shape = state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'scalar'
                print(f"   - {k}: {shape}")
    
    # 4. CHECK DWSA MODULES
    print(f"\n{'='*70}")
    print("🎯 DWSA MODULES DETECTED:")
    print("-" * 70)
    
    dwsa_modules = set()
    for key in state_dict.keys():
        if 'dwsa' in key.lower():
            # Extract module name (e.g., 'dwsa4', 'dwsa5', 'dwsa6')
            parts = key.split('.')
            for part in parts:
                if 'dwsa' in part.lower():
                    dwsa_modules.add(part)
    
    if dwsa_modules:
        for module in sorted(dwsa_modules):
            # Count params in this module
            module_keys = [k for k in state_dict.keys() if f'.{module}.' in k or k.startswith(f'{module}.')]
            print(f"✅ {module}: {len(module_keys)} parameters")
            
            # Show structure
            param_types = defaultdict(int)
            for k in module_keys:
                param_name = k.split('.')[-1]  # weight, bias, etc.
                param_types[param_name] += 1
            
            print(f"   Structure: {dict(param_types)}")
    else:
        print("❌ No DWSA modules found!")
    
    # 5. CHECK NORMALIZATION LAYERS
    print(f"\n{'='*70}")
    print("🔍 NORMALIZATION LAYERS:")
    print("-" * 70)
    
    bn_count = sum(1 for k in state_dict.keys() if '.bn.' in k or k.endswith('.weight') and 'bn' in k)
    gn_count = sum(1 for k in state_dict.keys() if '.gn.' in k or 'group_norm' in k.lower())
    ln_count = sum(1 for k in state_dict.keys() if '.ln.' in k or 'layer_norm' in k.lower())
    
    print(f"BatchNorm:  {bn_count} params")
    print(f"GroupNorm:  {gn_count} params")
    print(f"LayerNorm:  {ln_count} params")
    
    # 6. BACKBONE vs HEAD
    print(f"\n{'='*70}")
    print("📊 PARAMETER COUNT:")
    print("-" * 70)
    
    backbone_params = sum(p.numel() for k, p in state_dict.items() if k.startswith('backbone.'))
    head_params = sum(p.numel() for k, p in state_dict.items() if k.startswith('decode_head.'))
    aux_params = sum(p.numel() for k, p in state_dict.items() if k.startswith('aux_head.'))
    other_params = sum(p.numel() for k, p in state_dict.items() 
                      if not any(k.startswith(x) for x in ['backbone.', 'decode_head.', 'aux_head.']))
    
    total = backbone_params + head_params + aux_params + other_params
    
    print(f"Backbone:    {backbone_params:>12,} ({backbone_params/1e6:.2f}M) | {100*backbone_params/total:.1f}%")
    print(f"Decode Head: {head_params:>12,} ({head_params/1e6:.2f}M) | {100*head_params/total:.1f}%")
    print(f"Aux Head:    {aux_params:>12,} ({aux_params/1e6:.2f}M) | {100*aux_params/total:.1f}%")
    if other_params > 0:
        print(f"Other:       {other_params:>12,} ({other_params/1e6:.2f}M) | {100*other_params/total:.1f}%")
    print(f"{'─'*70}")
    print(f"TOTAL:       {total:>12,} ({total/1e6:.2f}M)")
    
    # 7. GENERATE CONFIG
    print(f"\n{'='*70}")
    print("🔧 SUGGESTED CONFIG:")
    print("-" * 70)
    
    print("""
# Based on checkpoint analysis:
"backbone": {
    "dwsa_stages": [""")
    
    for module in sorted(dwsa_modules):
        stage_num = module.replace('dwsa', '')
        print(f"        'stage{stage_num}',")
    
    print("""    ],
    # ... other config ...
}
""")
    
    print(f"{'='*70}\n")
    
    return {
        'state_dict_key': dict_key,
        'total_params': len(state_dict),
        'dwsa_modules': sorted(dwsa_modules),
        'has_bn': bn_count > 0,
        'has_gn': gn_count > 0,
        'backbone_params': backbone_params,
        'head_params': head_params,
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    analyze_checkpoint(checkpoint_path)
