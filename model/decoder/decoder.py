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
    def __init__(self, channels, norm_cfg, act_cfg):
        super().__init__()
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
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid')
            )
        )

    def forward(self, skip_feat, dec_feat):
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        return dec_feat + gate * skip_feat


class DWConvModule(nn.Module):
    def __init__(self, channels, kernel_size=3, norm_cfg=None, act_cfg=None):
        super().__init__()
        padding = kernel_size // 2

        self.dw = ConvModule(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        self.pw = ConvModule(
            channels, channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_cfg, act_cfg):
        super().__init__()
        self.conv1 = ConvModule(
            channels, channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.conv2 = ConvModule(
            channels, channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=None
        )
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)


class EnhancedDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        c2_channels,
        c1_channels,
        decoder_channels=128,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=False),
        dropout_ratio=0.1,
        use_gated_fusion=True
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion

        # ===== Stage 1 =====
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.refine1 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg, act_cfg),
            ConvModule(
                in_channels, decoder_channels,
                kernel_size=3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        )

        self.c2_proj = ConvModule(
            c2_channels, decoder_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None
        ) if c2_channels != decoder_channels else nn.Identity()

        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(decoder_channels, norm_cfg, act_cfg)
        else:
            self.fusion1 = ConvModule(
                decoder_channels * 2, decoder_channels,
                kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg
            )

        # ===== Stage 2 =====
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg, act_cfg),
            ConvModule(
                decoder_channels, decoder_channels // 2,
                kernel_size=3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        )

        self.c1_proj = ConvModule(
            c1_channels, decoder_channels // 2,
            kernel_size=1, norm_cfg=norm_cfg, act_cfg=None
        ) if c1_channels != decoder_channels // 2 else nn.Identity()

        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(decoder_channels // 2, norm_cfg, act_cfg)
        else:
            self.fusion2 = ConvModule(
                decoder_channels, decoder_channels // 2,
                kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg
            )

        # ===== Final refine =====
        self.refine3 = nn.Sequential(
            DWConvModule(decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        self.final_proj = ConvModule(
            decoder_channels // 2,
            decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, c5: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        # ===== Stage 1 =====
        x = self.up1(c5)
        x = self.refine1(x)

        skip = self.c2_proj(c2)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        if self.use_gated_fusion:
            x = self.fusion1_gate(skip, x)
        else:
            x = self.fusion1(torch.cat([skip, x], dim=1))

        # ===== Stage 2 =====
        x = self.up2(x)
        x = self.refine2(x)

        skip = self.c1_proj(c1)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        if self.use_gated_fusion:
            x = self.fusion2_gate(skip, x)
        else:
            x = self.fusion2(torch.cat([skip, x], dim=1))

        # ===== Final =====
        x = self.refine3(x)
        x = self.final_proj(x)
        x = self.dropout(x)

        return x
