import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict

from components.components import (
    ConvModule,
    build_activation_layer,
    OptConfigType
)


class GatedFusion(nn.Module):
    """Learned sigmoid gate: output = gate*skip + (1-gate)*dec"""
    def __init__(self, channels, norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.gate_conv = nn.Sequential(
            ConvModule(channels*2, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1, norm_cfg=None,
                       act_cfg=dict(type='Sigmoid')))

    def forward(self, skip, dec):
        gate = self.gate_conv(torch.cat([skip, dec], dim=1))
        return gate * skip + (1.0 - gate) * dec

    def forward(self, skip_feat: Tensor, dec_feat: Tensor) -> Tensor:
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        out = gate * skip_feat + (1 - gate) * dec_feat
        return out


class DWConvModule(nn.Module):
    """Depthwise-separable conv: ~8-9x cheaper than regular Conv3x3."""
    def __init__(self, channels, kernel_size=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.dw = ConvModule(channels, channels, kernel_size,
                             padding=kernel_size//2, groups=channels,
                             norm_cfg=norm_cfg, act_cfg=None)
        self.pw = ConvModule(channels, channels, 1,
                             norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=None)
        self.act   = build_activation_layer(act_cfg)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)


class EnhancedDecoder(nn.Module):
    """
    FPN-style decoder with GatedFusion skip connections.

    Input:
        c5 : (B, C*4, H/8, W/8)  = (B, 128, H/8, W/8)
        c4 : (B, C*2, H/8, W/8)  = (B,  64, H/8, W/8)  same spatial as c5
        c2 : (B, C,   H/4, W/4)  = (B,  32, H/4, W/4)
        c1 : (B, C,   H/2, W/2)  = (B,  32, H/2, W/2)
    Output:
        (B, D//2, H/2, W/2) = (B, 64, H/2, W/2)

    Stage 0 (H/8 → H/8): refine c5 with DWConv (cheaper than ResidualBlock)
                          then fuse c4 via GatedFusion with LayerScale on c4
    Stage 1 (H/8 → H/4): upsample, ResidualBlock refine, fuse c2
    Stage 2 (H/4 → H/2): upsample, DWConv×2 refine, fuse c1
    """
    def __init__(self, in_channels=128, c4_channels=64,
                 c2_channels=32, c1_channels=32,
                 decoder_channels=128,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout_ratio=0.1, use_gated_fusion=True):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        D  = decoder_channels       # 128
        D2 = decoder_channels // 2  # 64

        # ── Stage 0: refine c5 at H/8 with DWConv (not ResidualBlock)
        # Reason: H/8 is the largest spatial in decoder — ResidualBlock
        # (2× Conv3x3) costs ~2× more than DWConv for negligible accuracy gain
        # since c5 already carries rich features from backbone+SPP.
        self.refine0 = nn.Sequential(
            DWConvModule(in_channels, 3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels, D, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # LayerScale on c4 before GatedFusion
        # c4 (detail branch) and c5 (semantic+SPP) have different semantic levels.
        # LayerScale init ~1.0 lets the model learn how much c4 contributes
        # without breaking initialisation (identical to original at step 0).
        self.c4_proj = (ConvModule(c4_channels, D, 1, norm_cfg=norm_cfg, act_cfg=None)
                        if c4_channels != D else nn.Identity())
        self.c4_scale = nn.Parameter(torch.ones(D))   # per-channel scale for c4

        if use_gated_fusion:
            self.fusion0 = GatedFusion(D, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion0 = ConvModule(D*2, D, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 1: H/8 → H/4, fuse c2
        self.up1     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(D, D2, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c2_proj = (ConvModule(c2_channels, D2, 1, norm_cfg=norm_cfg, act_cfg=None)
                        if c2_channels != D2 else nn.Identity())
        if use_gated_fusion:
            self.fusion1 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(D2*2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 2: H/4 → H/2, fuse c1
        self.up2     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = nn.Sequential(
            DWConvModule(D2, 3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(D2, 3, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c1_proj = (ConvModule(c1_channels, D2, 1, norm_cfg=norm_cfg, act_cfg=None)
                        if c1_channels != D2 else nn.Identity())
        if use_gated_fusion:
            self.fusion2 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(D2*2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.final_proj = ConvModule(D2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout    = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, c5, c4, c2, c1):
        # Stage 0: refine c5, apply LayerScale to c4, fuse at H/8
        x   = self.refine0(c5)
        c4p = self.c4_proj(c4) * self.c4_scale.view(1, -1, 1, 1)
        x   = self.fusion0(c4p, x) if self.use_gated_fusion \
              else self.fusion0(torch.cat([c4p, x], dim=1))

        # Stage 1: H/8 → H/4
        x   = self.refine1(self.up1(x))
        c2p = self.c2_proj(c2)
        x   = self.fusion1(c2p, x) if self.use_gated_fusion \
              else self.fusion1(torch.cat([c2p, x], dim=1))

        # Stage 2: H/4 → H/2
        x   = self.refine2(self.up2(x))
        c1p = self.c1_proj(c1)
        x   = self.fusion2(c1p, x) if self.use_gated_fusion \
              else self.fusion2(torch.cat([c1p, x], dim=1))

        return self.dropout(self.final_proj(x))   # (B, D2, H/2, W/2)
