import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

# Lưu code vào file: custom_components.py

# Sau đó import:
# Sau đó import:
from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    BaseDecodeHead,
    OptConfigType,
    SampleList
)


# ============================================
# LIGHTWEIGHT DECODER
# ============================================

class DecoderStage(nn.Module):
    """
    Optimized decoder stage with proper channel handling
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_gated_fusion: bool = True,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        self.use_gated_fusion = use_gated_fusion
        
        # Upsample decoder features
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # ✅ CRITICAL: Project skip to match decoder channels
        # This enables gated fusion to work properly
        if skip_channels != out_channels:
            self.skip_proj = ConvModule(
                in_channels=skip_channels,
                out_channels=out_channels,  # Match decoder output!
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        else:
            self.skip_proj = nn.Identity()
        
        # Gated fusion (works because channels match now)
        if use_gated_fusion:
            self.fusion = GatedFusion(
                channels=out_channels,
                norm_cfg=norm_cfg
            )
        else:
            self.fusion = None
        
        # Refinement convs
        self.refine = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
    
    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Decoder features (B, C_in, H, W)
            skip: Skip connection (B, C_skip, H_skip, W_skip)
        
        Returns:
            Refined features (B, C_out, H*2, W*2)
        """
        # Upsample decoder features
        x = self.upsample(x)
        
        # Process skip connection
        if skip is not None:
            # Project skip to match decoder channels
            skip = self.skip_proj(skip)
            
            # Resize skip if spatial size doesn't match
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Fusion (channels now match!)
            if self.fusion is not None:
                x = self.fusion(skip, x)
            else:
                x = x + skip
        
        # Refinement
        x = self.refine(x)
        
        return x


class GatedFusion(nn.Module):
    """
    Gated fusion for adaptive feature selection
    
    Now works properly because skip and decoder features
    have matching channels after projection
    """
    
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True)
    ):
        super().__init__()
        
        self.gate_conv = ConvModule(
            in_channels=channels * 2,
            out_channels=1,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, enc_feat: Tensor, dec_feat: Tensor) -> Tensor:
        """
        Both features must have same channels (ensured by skip_proj)
        """
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        gate = self.sigmoid(self.gate_conv(concat))
        return gate * enc_feat + (1 - gate) * dec_feat


class LightweightDecoder(nn.Module):
    """
    ✅ NEW: Lightweight decoder for memory efficiency
    
    Architecture:
        Input (H/8) → H/4 → H/2 → H (output)
        
    Features:
    - Progressive upsampling
    - Thin channels: 128 → 64 → 32 → 16
    - Simple skip fusion (concat + 1x1)
    - Low memory overhead (~1.5GB additional)
    """
    
    def __init__(
        self,
        in_channels: int = 64,      # c5 channels
        channels: int = 128,         # Base decoder channels
        use_gated_fusion: bool = False,  # Simplified: always False
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        from components.components import ConvModule
        
        # Stage 1: H/8 → H/4 (128 channels)
        self.up1 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Fusion with c2 (skip from H/4)
        self.fusion1 = ConvModule(
            in_channels=channels + 32,  # 128 + c2_channels
            out_channels=channels // 2,  # 64
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Stage 2: H/4 → H/2 (64 channels)
        self.up2 = nn.Sequential(
            ConvModule(
                in_channels=channels // 2,
                out_channels=channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Fusion with c1 (skip from H/2)
        self.fusion2 = ConvModule(
            in_channels=channels // 2 + 32,  # 64 + c1_channels
            out_channels=channels // 4,  # 32
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Stage 3: H/2 → H (32 channels)
        self.up3 = nn.Sequential(
            ConvModule(
                in_channels=channels // 4,
                out_channels=channels // 4,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Final refinement
        self.final = ConvModule(
            in_channels=channels // 4,
            out_channels=channels // 8,  # 16
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
    
    def forward(self, x: Tensor, skip_connections: List[Tensor]) -> Tensor:
        """
        Args:
            x: (B, 64, H/8, W/8) - c5 from backbone
            skip_connections: [c2, c1, None]
                - c2: (B, 32, H/4, W/4)
                - c1: (B, 32, H/2, W/2)
        
        Returns:
            out: (B, 16, H, W)
        """
        c2, c1, _ = skip_connections
        
        # Stage 1: H/8 → H/4
        x = self.up1(x)  # (B, 128, H/4, W/4)
        
        if c2 is not None:
            x = torch.cat([x, c2], dim=1)  # (B, 160, H/4, W/4)
        x = self.fusion1(x)  # (B, 64, H/4, W/4)
        
        # Stage 2: H/4 → H/2
        x = self.up2(x)  # (B, 64, H/2, W/2)
        
        if c1 is not None:
            x = torch.cat([x, c1], dim=1)  # (B, 96, H/2, W/2)
        x = self.fusion2(x)  # (B, 32, H/2, W/2)
        
        # Stage 3: H/2 → H
        x = self.up3(x)  # (B, 32, H, W)
        x = self.final(x)  # (B, 16, H, W)
        
        return x
