import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType
)

# ============================================
# ENHANCED FUSION MODULES
# ============================================

class GatedFusion(nn.Module):
    """
    ✅ UPGRADED: Gated Fusion with improved feature selection
    - Learns to weight skip connections vs decoder features
    - Prevents information loss through selective gating
    """
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        # Gate mechanism: [skip, decoder] -> gate weights
        self.gate_conv = nn.Sequential(
            ConvModule(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=None,  # No norm on gate output
                act_cfg=dict(type='Sigmoid')
            )
        )
    
    def forward(self, skip_feat: Tensor, dec_feat: Tensor) -> Tensor:
        """
        Args:
            skip_feat: Features from encoder skip connection
            dec_feat: Features from decoder upsampling
        Returns:
            Fused features with learned gating
        """
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        
        # Adaptive fusion: gate * skip + (1-gate) * dec
        out = gate * skip_feat + (1 - gate) * dec_feat
        return out


class DWConvModule(nn.Module):
    """
    ✅ NEW: Depthwise Separable Convolution for efficient upsampling
    - Reduces computation while maintaining receptive field
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        # Depthwise convolution
        self.dw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # Depthwise
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        
        # Pointwise convolution
        self.pw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    ✅ NEW: Residual block for decoder stability
    - Skip connection prevents gradient vanishing
    - Improves feature propagation
    """
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        
        self.act = build_activation_layer(act_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


# ============================================
# UPGRADED DECODER FOR ENHANCED BACKBONE
# ============================================

class EnhancedDecoder(nn.Module):
    """
    ✅ UPGRADED: Decoder optimized for channels=48 backbone
    
    Design:
    - c5: 96 channels (48*2) → 128 (decoder)
    - c2: 96 channels (48*2) → project to 96 for skip fusion
    - c1: 48 channels → project to 48 for skip fusion
    
    Features:
    - Gated fusion for adaptive feature selection
    - Residual blocks for better gradient flow
    - Depthwise separable convolutions for efficiency
    - Progressive upsampling with channel reduction
    """
    
    def __init__(
        self,
        in_channels: int = 96,  # c5 = channels * 2 = 48 * 2
        decoder_channels: int = 128,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True
    ):
        super().__init__()
        
        self.use_gated_fusion = use_gated_fusion
        self.decoder_channels = decoder_channels
        
        # ======================================
        # STAGE 1: c5 (H/16) → H/8
        # ======================================
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.refine1 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Skip fusion: c2 (96 channels)
        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels + 96,  # decoder + c2
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        
        # ======================================
        # STAGE 2: H/8 → H/4
        # ======================================
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Skip fusion: c1 (48 channels)
        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(
                in_channels=(decoder_channels // 2) + 48,  # decoder + c1
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        
        # ======================================
        # STAGE 3: H/4 → H/2
        # ======================================
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.refine3 = nn.Sequential(
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        
        # ======================================
        # FINAL: Output projection
        # ======================================
        self.final_proj = ConvModule(
            in_channels=decoder_channels // 2,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
    
    def forward(
        self,
        c5: Tensor,
        c2: Tensor,
        c1: Tensor
    ) -> Tensor:
        """
        Args:
            c5: (B, 96, H/16, W/16) - bottleneck features
            c2: (B, 96, H/4, W/4) - skip connection from stage2
            c1: (B, 48, H/2, W/2) - skip connection from stage1
        Returns:
            (B, 64, H/2, W/2) - decoder output
        """
        
        # Stage 1: H/16 → H/8
        x = self.up1(c5)  # (B, 96, H/8, W/8)
        x = self.refine1(x)  # (B, 128, H/8, W/8)
        
        if self.use_gated_fusion:
            x = self.fusion1_gate(c2, x)
        else:
            x = torch.cat([x, c2], dim=1)
            x = self.fusion1(x)
        
        # Stage 2: H/8 → H/4
        x = self.up2(x)  # (B, 128, H/4, W/4)
        x = self.refine2(x)  # (B, 64, H/4, W/4)
        
        if self.use_gated_fusion:
            x = self.fusion2_gate(c1, x)
        else:
            x = torch.cat([x, c1], dim=1)
            x = self.fusion2(x)
        
        # Stage 3: H/4 → H/2
        x = self.up3(x)  # (B, 64, H/2, W/2)
        x = self.refine3(x)  # (B, 64, H/2, W/2)
        
        # Final projection
        x = self.final_proj(x)
        x = self.dropout(x)
        
        return x


# ============================================
# AUXILIARY HEAD
# ============================================

class GCNetAuxHead(nn.Module):
    """
    ✅ UPDATED: Auxiliary head for early supervision
    - Applied to c4 features for multi-scale training
    - Improves gradient flow in early stages
    """
    def __init__(
        self,
        in_channels: int = 192,  # c4 = channels * 4 = 48 * 4
        channels: int = 96,
        num_classes: int = 19,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        align_corners: bool = False
    ):
        super().__init__()
        
        self.align_corners = align_corners
        
        # Feature extraction
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Segmentation head
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.conv_seg(x)


# ============================================
# MAIN SEGMENTATION HEAD
# ============================================

class GCNetHead(nn.Module):
    """
    ✅ UPDATED: Main segmentation head for enhanced backbone
    
    Pipeline:
    c5 (96ch) → Decoder → (64ch, H/2) → Segmentation
    
    Components:
    - Enhanced decoder with gated fusion
    - Residual blocks for stability
    - Dropout for regularization
    - Proper channel handling for channels=48 backbone
    """
    
    def __init__(
        self,
        in_channels: int = 96,  # c5 = channels * 2 = 48 * 2
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        align_corners: bool = False,
        use_gated_fusion: bool = True
    ):
        super().__init__()
        
        self.align_corners = align_corners
        
        # Decoder: c5 → output features
        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
            use_gated_fusion=use_gated_fusion
        )
        
        # Segmentation head
        output_channels = decoder_channels // 2  # 64
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(output_channels, num_classes, kernel_size=1)
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dictionary containing:
                c1: (B, 48, H/2, W/2)
                c2: (B, 96, H/4, W/4)
                c5: (B, 96, H/16, W/16)
        Returns:
            Segmentation logits (B, num_classes, H/2, W/2)
        """
        c1 = inputs['c1']
        c2 = inputs['c2']
        c5 = inputs['c5']
        
        # Decode
        x = self.decoder(c5, c2, c1)  # (B, 64, H/2, W/2)
        
        # Segment
        x = self.conv_seg(x)  # (B, num_classes, H/2, W/2)
        
        return x
