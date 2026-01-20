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
# ENHANCED DECODER WITH FLEXIBLE CHANNELS
# ============================================

class EnhancedDecoder(nn.Module):
    """
    ✅ FIXED: Decoder with flexible channel handling
    - No longer assumes specific c1, c2 channels
    - Auto-projects backbone outputs to decoder needs
    - Zero performance impact
    """
    
    def __init__(
        self,
        in_channels: int = 128,        # c5 channels
        c2_channels: int = 64,        # ✅ NEW: c2 from backbone
        c1_channels: int = 32,        # ✅ NEW: c1 from backbone
        decoder_channels: int = 128,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False),
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
        
        # ✅ NEW: Project c2 to decoder_channels (flexible)
        self.c2_proj = ConvModule(
            in_channels=c2_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        ) if c2_channels != decoder_channels else nn.Identity()
        
        # Skip fusion: project c2 before fusion
        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels + decoder_channels,
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
        
        # ✅ NEW: Project c1 to decoder_channels//2 (flexible)
        self.c1_proj = ConvModule(
            in_channels=c1_channels,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        ) if c1_channels != (decoder_channels // 2) else nn.Identity()
        
        # Skip fusion: project c1 before fusion
        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(
                in_channels=(decoder_channels // 2) + (decoder_channels // 2),
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
    
    def forward(self, c5: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        """
        Args:
            c5: (B, in_channels, H/16, W/16)
            c2: (B, c2_channels, H/4, W/4) - flexible channels
            c1: (B, c1_channels, H/2, W/2) - flexible channels
        Returns:
            (B, 64, H/2, W/2)
        """
        
        # Stage 1: H/16 → H/8
        x = self.up1(c5)
        x = self.refine1(x)
        
        # ✅ Project c2 before fusion
        c2_proj = self.c2_proj(c2)
        
        if self.use_gated_fusion:
            x = self.fusion1_gate(c2_proj, x)
        else:
            x = torch.cat([x, c2_proj], dim=1)
            x = self.fusion1(x)
        
        # Stage 2: H/8 → H/4
        x = self.up2(x)
        x = self.refine2(x)
        
        # ✅ Project c1 before fusion
        c1_proj = self.c1_proj(c1)
        
        if self.use_gated_fusion:
            x = self.fusion2_gate(c1_proj, x)
        else:
            x = torch.cat([x, c1_proj], dim=1)
            x = self.fusion2(x)
        
        # Stage 3: H/4 → H/2
        x = self.up3(x)
        x = self.refine3(x)
        
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
        """Handle both dict and tensor input"""
        if isinstance(x, dict):
            x = x['c4']
        
        x = self.conv1(x)
        return self.conv_seg(x)


# ============================================
# MAIN SEGMENTATION HEAD
# ============================================

class GCNetHead(nn.Module):
    """
    ✅ FINAL VERSION: Main segmentation head with flexible channels
    
    Pipeline:
    c5 (96ch) → Decoder → (64ch, H/2) → Segmentation
    
    Components:
    - Enhanced decoder with gated fusion
    - Residual blocks for stability
    - Dropout for regularization
    - Flexible channel handling for any backbone
    """
    
    def __init__(
        self,
        in_channels: int = 128,  # c5
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        align_corners: bool = False,
        use_gated_fusion: bool = True,
        # ✅ NEW: Accept flexible c1, c2 channels from backbone
        c1_channels: int = 32,
        c2_channels: int = 64
    ):
        super().__init__()
        
        self.align_corners = align_corners
        
        # ✅ Pass detected channels to decoder
        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            c2_channels=c2_channels,      # From backbone
            c1_channels=c1_channels,      # From backbone
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
            use_gated_fusion=use_gated_fusion
        )
        
        # Segmentation head
        output_channels = decoder_channels // 2
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(output_channels, num_classes, kernel_size=1)
        )
    
    def forward(self, feats: Dict[str, Tensor] | tuple | Tensor) -> Tensor:
        if isinstance(feats, dict):
            # ✅ Explicit keys (SAFE)
            if not all(k in feats for k in ['c1', 'c2', 'c5']):
                raise KeyError(
                    f"GCNetHead expects keys ['c1','c2','c5'], "
                    f"but got {list(feats.keys())}"
                )
    
            c1 = feats['c1']
            c2 = feats['c2']
            c5 = feats['c5']
            x = self.decoder(c5, c2, c1)
    
        elif isinstance(feats, tuple):
            if len(feats) == 2 and self.training:
                # (aux_feat, final_feat)
                final_feat = feats[1]  # usually H/8
                x = F.interpolate(
                    final_feat,
                    scale_factor=4,
                    mode='bilinear',
                    align_corners=self.align_corners
                )
            elif len(feats) >= 3:
                # (c1, c2, c5, ...)
                c1, c2, c5 = feats[0], feats[1], feats[2]
                x = self.decoder(c5, c2, c1)
            else:
                raise ValueError(f"Unsupported tuple length: {len(feats)}")
    
        else:
            # Single tensor fallback
            x = F.interpolate(
                feats,
                scale_factor=4,
                mode='bilinear',
                align_corners=self.align_corners
            )
    
        return self.conv_seg(x)
