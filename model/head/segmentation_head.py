import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional

from components.components import ConvModule, OptConfigType, build_activation_layer

# ============================================
# RESIDUAL BLOCK (Helper for auxiliary head)
# ============================================

class ResidualBlock(nn.Module):
    """Simple residual block for feature refinement"""
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
# UPGRADED GENET HEAD
# ============================================

class GCNetHead(nn.Module):
    """
    ✅ UPGRADED: Main segmentation head for enhanced backbone
    
    Features:
    - Gated fusion from upgraded decoder
    - Proper channel handling for channels=48
    - Dropout for regularization
    - Final segmentation layer
    
    Input:
        - c1: (B, 48, H/2, W/2)
        - c2: (B, 96, H/4, W/4)
        - c5: (B, 96, H/16, W/16)
    
    Output:
        - logits: (B, num_classes, H/2, W/2)
    """
    
    def __init__(
        self,
        in_channels: int = 96,  # c5 = channels * 2 = 48 * 2
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        use_gated_fusion: bool = True,
        c1_channels: int = 48,
        c2_channels: int = 48
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        self.align_corners = align_corners
        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            c2_channels=c2_channels,
            c1_channels=c1_channels,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
            use_gated_fusion=use_gated_fusion
        )
        output_channels = decoder_channels // 2
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(output_channels, num_classes, kernel_size=1)
        )
        self.use_gated_fusion = use_gated_fusion
        
        # ======================================
        # DECODER STAGE 1: c5 (H/16) → H/8
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
        
        # Fusion with c2 (96 channels)
        if use_gated_fusion:
            self.fusion1_gate = self._build_gated_fusion(decoder_channels, norm_cfg, act_cfg)
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels + 96,
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        
        # ======================================
        # DECODER STAGE 2: H/8 → H/4
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
        
        # Fusion with c1 (48 channels)
        if use_gated_fusion:
            self.fusion2_gate = self._build_gated_fusion(decoder_channels // 2, norm_cfg, act_cfg)
        else:
            self.fusion2 = ConvModule(
                in_channels=(decoder_channels // 2) + 48,
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        
        # ======================================
        # DECODER STAGE 3: H/4 → H/2
        # ======================================
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.refine3 = nn.Sequential(
            self._build_dw_conv(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            self._build_dw_conv(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        
        # Final projection
        self.final_proj = ConvModule(
            in_channels=decoder_channels // 2,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Dropout before segmentation
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # ======================================
        # SEGMENTATION HEAD
        # ======================================
        # Output channels = decoder_channels // 2 = 128 // 2 = 64
        self.conv_seg = nn.Conv2d(
            decoder_channels // 2,
            num_classes,
            kernel_size=1
        )
    
    def _build_gated_fusion(self, channels, norm_cfg, act_cfg):
        """Build gated fusion module"""
        return nn.Sequential(
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
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid')
            )
        )
    
    def _build_dw_conv(self, channels, kernel_size, norm_cfg, act_cfg):
        """Build depthwise separable convolution"""
        padding = kernel_size // 2
        return nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                norm_cfg=norm_cfg,
                act_cfg=None
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with keys 'c1', 'c2', 'c5'
                c1: (B, 48, H/2, W/2)
                c2: (B, 96, H/4, W/4)
                c5: (B, 96, H/16, W/16)
        
        Returns:
            logits: (B, num_classes, H/2, W/2)
        """
        c1 = inputs['c1']
        c2 = inputs['c2']
        c5 = inputs['c5']
        
        # Stage 1: H/16 → H/8
        x = self.up1(c5)
        x = self.refine1(x)
        
        if self.use_gated_fusion:
            # Gated fusion: gate * skip + (1-gate) * decoder
            concat = torch.cat([c2, x], dim=1)
            gate = self.fusion1_gate(concat)
            x = gate * c2 + (1 - gate) * x
        else:
            x = torch.cat([x, c2], dim=1)
            x = self.fusion1(x)
        
        # Stage 2: H/8 → H/4
        x = self.up2(x)
        x = self.refine2(x)
        
        if self.use_gated_fusion:
            concat = torch.cat([c1, x], dim=1)
            gate = self.fusion2_gate(concat)
            x = gate * c1 + (1 - gate) * x
        else:
            x = torch.cat([x, c1], dim=1)
            x = self.fusion2(x)
        
        # Stage 3: H/4 → H/2
        x = self.up3(x)
        x = self.refine3(x)
        
        # Final projection and dropout
        x = self.final_proj(x)
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)
        
        return output


# ============================================
# AUXILIARY HEAD - DEEP SUPERVISION
# ============================================

class GCNetAuxHead(nn.Module):
    """
    ✅ UPGRADED: Auxiliary head for deep supervision
    
    - Applied to c4 features (H/16) for multi-scale training
    - Improves gradient flow in early stages
    - Expected: +1-2% mIoU improvement
    
    Input:
        c4: (B, 192, H/16, W/16) [channels=48: 48*4=192]
    
    Output:
        logits: (B, num_classes, H/16, W/16)
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
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Feature refinement: reduce channels gradually
        self.refine = nn.Sequential(
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
            ),
            ResidualBlock(
                channels=channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # Final segmentation layer
        self.conv_seg = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with key 'c4'
                c4: (B, 192, H/16, W/16)
        
        Returns:
            logits: (B, num_classes, H/16, W/16)
        """
        x = inputs['c4']
        
        # Feature refinement
        x = self.refine(x)
        
        # Regularization
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)
        
        return output
