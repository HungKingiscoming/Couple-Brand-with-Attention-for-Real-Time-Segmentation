import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType,
)


# ===========================
# GatedFusion
# Weight keys:
#   gate_conv.0.conv.weight  gate_conv.0.bn.{weight,bias,running_mean,running_var,...}
#   gate_conv.1.conv.weight  gate_conv.1.conv.bias
# ===========================

class GatedFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
    ):
        super().__init__()

        self.gate_conv = nn.Sequential(
            ConvModule(              # [0]  has BN + ReLU
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(              # [1]  no norm, Sigmoid activation
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
        return gate * skip_feat + (1 - gate) * dec_feat


# ===========================
# DWConvModule
# Weight keys:
#   dw_conv.conv.weight  dw_conv.bn.{weight,bias,running_mean,running_var,...}
#   pw_conv.conv.weight  pw_conv.bn.{weight,bias,running_mean,running_var,...}
# ===========================

class DWConvModule(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
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


# ===========================
# ResidualBlock
# Weight keys:
#   conv1.conv.weight  conv1.bn.{weight,bias,running_mean,running_var,...}
#   conv2.conv.weight  conv2.bn.{weight,bias,running_mean,running_var,...}
# ===========================

class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
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
        out = self.act(out)
        return out


# ===========================
# EnhancedDecoder
#
# Weight keys produced (decoder_channels=128, c2_channels=32, c1_channels=32):
#
#   refine1.0.conv1.*  refine1.0.conv2.*           ← ResidualBlock
#   refine1.1.conv.*   refine1.1.bn.*              ← ConvModule 128→128
#   c2_proj.conv.*     c2_proj.bn.*                ← ConvModule 32→128
#   fusion1_gate.gate_conv.0.*  fusion1_gate.gate_conv.1.*
#
#   refine2.0.conv1.*  refine2.0.conv2.*           ← ResidualBlock
#   refine2.1.conv.*   refine2.1.bn.*              ← ConvModule 128→64
#   c1_proj.conv.*     c1_proj.bn.*                ← ConvModule 32→64
#   fusion2_gate.gate_conv.0.*  fusion2_gate.gate_conv.1.*
#
#   refine3.0.dw_conv.* refine3.0.pw_conv.*        ← DWConvModule
#   refine3.1.dw_conv.* refine3.1.pw_conv.*        ← DWConvModule
#
#   final_proj.conv.*   final_proj.bn.*
# ===========================

class EnhancedDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        c2_channels: int = 32,
        c1_channels: int = 32,
        decoder_channels: int = 128,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True,
    ):
        super().__init__()

        self.use_gated_fusion = use_gated_fusion
        self.decoder_channels = decoder_channels

        # Stage 1: H/16 → H/8
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.refine1 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),  # [0]
            ConvModule(                                                        # [1]
                in_channels=in_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        # c2 projection: 32 → 128
        self.c2_proj = ConvModule(
            in_channels=c2_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        ) if c2_channels != decoder_channels else nn.Identity()

        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels * 2,
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        # Stage 2: H/8 → H/4
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),  # [0]
            ConvModule(                                                              # [1]
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        # c1 projection: 32 → 64
        self.c1_proj = ConvModule(
            in_channels=c1_channels,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        ) if c1_channels != (decoder_channels // 2) else nn.Identity()

        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        # Stage 3: H/4 → H/2
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.refine3 = nn.Sequential(
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),  # [0]
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),  # [1]
        )

        # Final projection
        self.final_proj = ConvModule(
            in_channels=decoder_channels // 2,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, c5: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        # Stage 1: H/16 → H/8
        x = self.up1(c5)
        x = self.refine1(x)

        c2_proj = self.c2_proj(c2)

        if self.use_gated_fusion:
            x = self.fusion1_gate(c2_proj, x)
        else:
            x = torch.cat([x, c2_proj], dim=1)
            x = self.fusion1(x)

        # Stage 2: H/8 → H/4
        x = self.up2(x)
        x = self.refine2(x)

        c1_proj = self.c1_proj(c1)

        if self.use_gated_fusion:
            x = self.fusion2_gate(c1_proj, x)
        else:
            x = torch.cat([x, c1_proj], dim=1)
            x = self.fusion2(x)

        # Stage 3: H/4 → H/2
        x = self.up3(x)
        x = self.refine3(x)

        x = self.final_proj(x)
        x = self.dropout(x)

        return x


# ===========================
# GCNetAuxHead
# Weight keys:
#   conv1.conv.weight  conv1.bn.{weight,bias,running_mean,running_var,...}
#   conv_seg.1.weight  conv_seg.1.bias
# ===========================

class GCNetAuxHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,       # c4 = channels * 2 = 32*2 = 64
        channels: int = 96,
        num_classes: int = 19,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()

        self.align_corners = align_corners

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),  # [0]
            nn.Conv2d(channels, num_classes, kernel_size=1),                        # [1]
        )

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, dict):
            x = x['c4']
        x = self.conv1(x)
        return self.conv_seg(x)


# ===========================
# GCNetHead  (= decode_head in checkpoint)
# Weight keys:
#   decoder.*    ← EnhancedDecoder
#   conv_seg.1.weight  conv_seg.1.bias
# ===========================

class GCNetHead(nn.Module):
    """
    Main segmentation head.
    Checkpoint key prefix: decode_head.*

    Default values match the saved weight shapes:
        in_channels=128   (c5 channels = C*4 = 32*4)
        c2_channels=32    (stem[1] output = C = 32)
        c1_channels=32    (stem[0] output = C = 32)
        decoder_channels=128
        num_classes=19
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        align_corners: bool = False,
        use_gated_fusion: bool = True,
        c1_channels: int = 32,
        c2_channels: int = 32,
    ):
        super().__init__()

        self.align_corners = align_corners

        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            c2_channels=c2_channels,
            c1_channels=c1_channels,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
            use_gated_fusion=use_gated_fusion,
        )

        output_channels = decoder_channels // 2   # 64
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),  # [0]
            nn.Conv2d(output_channels, num_classes, kernel_size=1),                # [1]
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        if isinstance(inputs, dict):
            c1 = inputs['c1']
            c2 = inputs['c2']
            c5 = inputs['c5']
        else:
            c1, c2, c5 = inputs[0], inputs[1], inputs[2]

        x = self.decoder(c5, c2, c1)
        x = self.conv_seg(x)
        return x
