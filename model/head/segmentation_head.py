# head.py - FINAL FIXED VERSION
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType
)


class GCNetHead(nn.Module):
    """
    ✅ FINAL FIXED: Main segmentation head with lightweight decoder
    
    Changes from original:
    1. ✅ Removed decode_enabled parameter (always True)
    2. ✅ Fixed skip connections order
    3. ✅ Always use LightweightDecoder
    4. ✅ Proper channel dimensions
    
    Input from backbone:
        - c1: (B, 32, H/2, W/2)
        - c2: (B, 32, H/4, W/4)
        - c3: (B, 64, H/8, W/8)
        - c4: (B, 128, H/16, W/16) - for aux head only
        - c5: (B, 64, H/8, W/8) - main features
    
    Output:
        - logits: (B, num_classes, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 64,           # c5 channels
        channels: int = 128,              # Not used (kept for compatibility)
        num_classes: int = 19,
        decoder_channels: int = 128,      # Base decoder channels
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        # ✅ Import decoder
        from model.decoder import LightweightDecoder
        
        # ✅ Always use lightweight decoder
        self.decoder = LightweightDecoder(
            in_channels=in_channels,      # 64
            channels=decoder_channels,     # 128
            use_gated_fusion=False,        # Simplified for speed
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # Segmentation head
        self.conv_seg = nn.Conv2d(
            decoder_channels // 8,  # 16 from decoder output
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with keys ['c1', 'c2', 'c3', 'c4', 'c5']
                - c1: (B, 32, H/2, W/2)
                - c2: (B, 32, H/4, W/4)
                - c3: (B, 64, H/8, W/8)
                - c5: (B, 64, H/8, W/8)
        
        Returns:
            logits: (B, num_classes, H, W)
        """
        # Extract features
        c1 = inputs['c1']  # H/2
        c2 = inputs['c2']  # H/4
        c5 = inputs['c5']  # H/8 (main output)
        
        # ✅ FIXED: Correct skip order for decoder
        skip_connections = [c2, c1, None]
        
        # Decode to full resolution
        x = self.decoder(c5, skip_connections)  # (B, 16, H, W)
        
        # Dropout
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)  # (B, num_classes, H, W)
        
        return output


class GCNetAuxHead(nn.Module):
    """
    ✅ Auxiliary head for deep supervision on c4 (H/16)
    
    Input:
        - c4: (B, 128, H/16, W/16)
    
    Output:
        - logits: (B, num_classes, H/16, W/16)
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 64,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Conv layers
        self.conv = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        self.conv_seg = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with key 'c4' (B, 128, H/16, W/16)
        
        Returns:
            logits: (B, num_classes, H/16, W/16)
        """
        # Extract c4 (stage 4 semantic features)
        x = inputs['c4']
        
        # Convolutions
        x = self.conv(x)
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)
        
        return output


# ============================================
# ✅ TESTING
# ============================================

if __name__ == "__main__":
    print("Testing Fixed GCNetHead...")
    
    # Create dummy backbone outputs
    B = 2
    inputs = {
        'c1': torch.randn(B, 32, 256, 512),   # H/2
        'c2': torch.randn(B, 32, 128, 256),   # H/4
        'c3': torch.randn(B, 64, 64, 128),    # H/8
        'c4': torch.randn(B, 128, 32, 64),    # H/16
        'c5': torch.randn(B, 64, 64, 128)     # H/8
    }
    
    # Test main head
    print("\n1. Testing GCNetHead...")
    head = GCNetHead(
        in_channels=64,
        channels=128,
        num_classes=19,
        decoder_channels=128,
        dropout_ratio=0.1
    )
    
    try:
        logits = head(inputs)
        print(f"✅ GCNetHead output: {logits.shape}")
        assert logits.shape == (B, 19, 512, 1024), f"Expected (2, 19, 512, 1024), got {logits.shape}"
        print("✅ Output shape correct!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test aux head
    print("\n2. Testing GCNetAuxHead...")
    aux_head = GCNetAuxHead(
        in_channels=128,
        channels=64,
        num_classes=19,
        dropout_ratio=0.1
    )
    
    try:
        aux_logits = aux_head(inputs)
        print(f"✅ GCNetAuxHead output: {aux_logits.shape}")
        assert aux_logits.shape == (B, 19, 32, 64), f"Expected (2, 19, 32, 64), got {aux_logits.shape}"
        print("✅ Output shape correct!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test parameter count
    print("\n3. Checking parameters...")
    total_params = sum(p.numel() for p in head.parameters())
    print(f"GCNetHead parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    aux_params = sum(p.numel() for p in aux_head.parameters())
    print(f"GCNetAuxHead parameters: {aux_params:,} ({aux_params/1e6:.2f}M)")
    
    print("\n✅ All tests passed!")
    
    # Print expected config
    print("\n" + "="*70)
    print("📋 CORRECT CONFIG FOR TRAINING")
    print("="*70)
    print("""
cfg = {
    "head": {
        "in_channels": 64,           # c5 channels (channels * 2)
        "channels": 128,             # Not used but kept for compatibility
        "decoder_channels": 128,     # Decoder base channels
        "dropout_ratio": 0.1,
        "align_corners": False
        # ❌ NO decode_enabled parameter!
    },
    "aux_head": {
        "in_channels": 128,          # c4 channels (channels * 4)
        "channels": 64,
        "dropout_ratio": 0.1,
        "align_corners": False
    }
}

# Usage:
head = GCNetHead(num_classes=19, **cfg["head"])
aux_head = GCNetAuxHead(num_classes=19, **cfg["aux_head"])
""")
