import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

from components.components import (
    ConvModule,
    build_activation_layer,
    resize,
    OptConfigType,
)

# ===================== FUSION & BUILDING BLOCKS =====================

class GatedFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='GELU', inplace=False),
    ):
        super().__init__()
        self.gate_conv = nn.Sequential(
            ConvModule(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid'),
            ),
        )

    def forward(self, skip_feat: Tensor, dec_feat: Tensor) -> Tensor:
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        return gate * skip_feat + (1.0 - gate) * dec_feat


class DWConvModule(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='GELU', inplace=False),
    ):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.pw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='GELU', inplace=False),
    ):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


# ===================== DECODER =====================

class EnhancedDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 96,   # c5
        c2_channels: int = 48,   # c2
        c1_channels: int = 48,   # c1
        decoder_channels: int = 128,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='GELU', inplace=False),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion

        # Stage 1: c5 (H/8) -> up (H/4)
        self.refine1 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )
        self.c2_proj = (
            ConvModule(
                in_channels=c2_channels,
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
            if c2_channels != decoder_channels
            else nn.Identity()
        )
        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(
                decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels * 2,
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        # Stage 2: (H/4) -> up (H/2)
        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )
        self.c1_proj = (
            ConvModule(
                in_channels=c1_channels,
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
            if c1_channels != (decoder_channels // 2)
            else nn.Identity()
        )
        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(
                decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        else:
            self.fusion2 = ConvModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        # Stage 3: (H/2) -> refine
        self.refine3 = nn.Sequential(
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.final_proj = ConvModule(
            in_channels=decoder_channels // 2,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )

    def forward(self, c5: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        # Assume c5: H/8, c2: H/4, c1: H/2
        # Stage 1: up c5 -> H/4
        x = resize(c5, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.refine1(x)
        c2_proj = self.c2_proj(c2)
        if self.use_gated_fusion:
            x = self.fusion1_gate(c2_proj, x)
        else:
            x = self.fusion1(torch.cat([x, c2_proj], dim=1))

        # Stage 2: up -> H/2
        x = resize(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.refine2(x)
        c1_proj = self.c1_proj(c1)
        if self.use_gated_fusion:
            x = self.fusion2_gate(c1_proj, x)
        else:
            x = self.fusion2(torch.cat([x, c1_proj], dim=1))

        # Stage 3: refine at H/2
        x = self.refine3(x)
        x = self.final_proj(x)
        x = self.dropout(x)
        return x


# ===================== AUX HEAD =====================

class GCNetAuxHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,  # c4 = 48*4
        channels: int = 96,
        num_classes: int = 19,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='GELU', inplace=False),
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.conv_seg(x)


# ===================== MAIN HEAD =====================

class GCNetHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 96,   # c5
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='GELU', inplace=False),
        align_corners: bool = False,
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
            use_gated_fusion=use_gated_fusion,
        )
        out_ch = decoder_channels // 2
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(out_ch, num_classes, kernel_size=1),
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        inputs:
            c1: (B, 48, H/2, W/2)
            c2: (B, 48, H/4, W/4)
            c5: (B, 96, H/8, W/8)
        """
        c1 = inputs["c1"]
        c2 = inputs["c2"]
        c5 = inputs["c5"]
        x = self.decoder(c5, c2, c1)
        x = self.conv_seg(x)
        return x
