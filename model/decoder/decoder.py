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


class LightweightDecoder(BaseModule):
    """
    ✅ OPTIMIZED: Proper channel handling + Gated fusion
    
    Architecture:
    - Input: c5 (64 channels @ H/8)
    - Stage 1: H/8 → H/4 (128 → 64) + skip c2 (32→64)
    - Stage 2: H/4 → H/2 (64 → 32) + skip c1 (32→32)
    - Stage 3: H/2 → H (32 → 16) + no skip
    - Output: 16 channels @ H
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        channels: int = 128,
        num_stages: int = 3,
        use_gated_fusion: bool = True,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        init_cfg: OptConfigType = None
    ):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_stages = num_stages
        
        # Input projection: 64 → 128
        self.input_proj = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Decoder stages
        self.stages = nn.ModuleList()
        
        # ✅ Stage 1: H/8 → H/4
        # Input: 128, Skip: c2 (32), Output: 64
        # Skip projection: 32 → 64 (inside DecoderStage)
        self.stages.append(
            DecoderStage(
                in_channels=channels,           # 128
                skip_channels=32,               # c2 will be projected to 64
                out_channels=channels // 2,     # 64
                use_gated_fusion=use_gated_fusion,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # ✅ Stage 2: H/4 → H/2
        # Input: 64, Skip: c1 (32), Output: 32
        # Skip already matches (32 == 32), no projection needed
        self.stages.append(
            DecoderStage(
                in_channels=channels // 2,      # 64
                skip_channels=32,               # c1 (already 32)
                out_channels=channels // 4,     # 32
                use_gated_fusion=use_gated_fusion,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # ✅ Stage 3: H/2 → H
        # Input: 32, Skip: None, Output: 16
        self.stages.append(
            DecoderStage(
                in_channels=channels // 4,      # 32
                skip_channels=32,               # Dummy (won't be used)
                out_channels=channels // 8,     # 16
                use_gated_fusion=False,         # No skip, no fusion
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
    
    def forward(
        self, 
        x: Tensor, 
        skip_connections: Optional[List[Tensor]] = None
    ) -> Tensor:
        """
        Args:
            x: c5 features (B, 64, H/8, W/8)
            skip_connections: [c2, c1, None] at [H/4, H/2, H]
        
        Returns:
            Decoded features (B, 16, H, W)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Progressive decoding
        for i, stage in enumerate(self.stages):
            skip = None
            if skip_connections is not None and i < len(skip_connections):
                skip = skip_connections[i]
            
            x = stage(x, skip)
        
        return x
