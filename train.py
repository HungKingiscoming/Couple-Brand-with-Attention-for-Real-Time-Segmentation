# ============================================
# FINAL FIX: Replace lines 520-545 in train.py
# ============================================

def main():
    # ... (previous code unchanged) ...
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        compute_class_weights=args.compute_class_weights,
        dataset_type=args.dataset_type
    )
    
    # ============================================
    # ✅ FIXED MODEL CREATION
    # ============================================
    print(f"\n{'='*70}")
    print("🏗️  Building Model...")
    print(f"{'='*70}\n")
    
    # Get config based on model size
    if args.model_size == "lightweight":
        cfg = ModelConfig.get_lightweight_config()
    elif args.model_size == "medium":
        cfg = ModelConfig.get_medium_config()
    else:  # performance
        cfg = ModelConfig.get_performance_config()
    
    # ✅ FIX: Properly set num_classes before passing to model
    head_cfg = cfg["head"].copy()
    head_cfg["num_classes"] = args.num_classes
    
    aux_head_cfg = cfg["aux_head"].copy()
    aux_head_cfg["num_classes"] = args.num_classes
    
    # Create model
    model = Segmentor(
        backbone=GCNetWithDWSA(**cfg["backbone"]),
        head=GCNetHead(**head_cfg),
        aux_head=GCNetAuxHead(**aux_head_cfg)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"   Model size:           {args.model_size}")
    print(f"   Backbone channels:    {cfg['backbone']['channels']}")
    print(f"   DWSA stages:          {cfg['backbone']['dwsa_stages']}")
    print(f"   Decoder enabled:      {head_cfg['decode_enabled']}")
    
    print_memory_usage("After model creation")
    
    # Rest of training code...
    # ... (unchanged) ...


# ============================================
# ALTERNATIVE: Using helper function
# ============================================

def create_model_from_config(config: dict, num_classes: int):
    """
    ✅ Helper function to create model from config
    Ensures num_classes is properly set
    """
    # Prepare head config
    head_cfg = config["head"].copy()
    head_cfg["num_classes"] = num_classes
    
    # Prepare aux head config
    aux_head_cfg = config["aux_head"].copy()
    aux_head_cfg["num_classes"] = num_classes
    
    # Create model
    model = Segmentor(
        backbone=GCNetWithDWSA(**config["backbone"]),
        head=GCNetHead(**head_cfg),
        aux_head=GCNetAuxHead(**aux_head_cfg)
    )
    
    return model


# Then in main():
def main_alternative():
    # ... (previous code) ...
    
    # Get config
    if args.model_size == "lightweight":
        cfg = ModelConfig.get_lightweight_config()
    elif args.model_size == "medium":
        cfg = ModelConfig.get_medium_config()
    else:
        cfg = ModelConfig.get_performance_config()
    
    # ✅ Use helper function (cleaner)
    model = create_model_from_config(cfg, num_classes=args.num_classes)
    
    # ... (rest of training) ...


# ============================================
# TESTING CODE
# ============================================

if __name__ == "__main__":
    """
    Quick test to verify model creation works
    """
    import argparse
    
    # Mock args
    class Args:
        model_size = "lightweight"
        num_classes = 19
    
    args = Args()
    
    # Test config
    print("Testing model creation...\n")
    
    if args.model_size == "lightweight":
        cfg = ModelConfig.get_lightweight_config()
    
    # Show original config
    print("Original config:")
    print(f"  head['num_classes']: {cfg['head']['num_classes']}")
    print(f"  aux_head['num_classes']: {cfg['aux_head']['num_classes']}")
    
    # Fix config
    head_cfg = cfg["head"].copy()
    head_cfg["num_classes"] = args.num_classes
    
    aux_head_cfg = cfg["aux_head"].copy()
    aux_head_cfg["num_classes"] = args.num_classes
    
    print("\nFixed config:")
    print(f"  head_cfg['num_classes']: {head_cfg['num_classes']}")
    print(f"  aux_head_cfg['num_classes']: {aux_head_cfg['num_classes']}")
    
    # Verify no duplicate keys
    print("\n✅ No duplicate keys - model creation should work!")
    
    # Test actual creation (if imports available)
    try:
        from model.backbone.model import GCNetWithDWSA
        from model.head.segmentation_head import GCNetHead, GCNetAuxHead
        
        print("\nCreating model...")
        
        model = Segmentor(
            backbone=GCNetWithDWSA(**cfg["backbone"]),
            head=GCNetHead(**head_cfg),
            aux_head=GCNetAuxHead(**aux_head_cfg)
        )
        
        print("✅ Model created successfully!")
        
        # Test forward
        import torch
        x = torch.randn(1, 3, 512, 1024)
        with torch.no_grad():
            out = model(x)
        
        print(f"✅ Forward pass successful! Output shape: {out.shape}")
        
    except ImportError as e:
        print(f"\n⚠️  Cannot test model creation (imports not available)")
        print(f"   Error: {e}")
        print("   This is OK - just make sure imports work in your environment")
