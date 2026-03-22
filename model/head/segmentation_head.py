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


# ===================== GATED FUSION (IMPROVED) =====================
class GatedFusion(nn.Module):
    def __init__(self, channels, norm_cfg, act_cfg):
        super().__init__()
        self.gate_conv = nn.Sequential(
            ConvModule(channels * 2, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1, norm_cfg=None, act_cfg=dict(type='Sigmoid')),
        )

    def forward(self, skip, dec):
        if skip.shape[-2:] != dec.shape[-2:]:
            dec = F.interpolate(dec, size=skip.shape[-2:], mode='bilinear', align_corners=True)

        gate = self.gate_conv(torch.cat([skip, dec], dim=1))
        return dec + gate * skip   # ✅ residual fusion (tốt hơn)


# ===================== LIGHT BLOCK =====================
class DWConvModule(nn.Module):
    def __init__(self, channels, norm_cfg, act_cfg):
        super().__init__()
        self.dw = ConvModule(channels, channels, 3, padding=1,
                             groups=channels, norm_cfg=norm_cfg, act_cfg=None)
        self.pw = ConvModule(channels, channels, 1,
                             norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_cfg, act_cfg):
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)


# ===================== DECODER =====================
class EnhancedDecoder(nn.Module):
    def __init__(
        self,
        in_channels=128,
        c4_channels=64,
        c2_channels=32,
        c1_channels=32,
        decoder_channels=96,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        dropout_ratio=0.1,
        use_gated_fusion=True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        D, D2 = decoder_channels, decoder_channels // 2

        self.refine0 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg, act_cfg),
            ConvModule(in_channels, D, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        self.c4_proj = ConvModule(c4_channels, D, 1, norm_cfg=norm_cfg, act_cfg=None)

        self.fusion0 = GatedFusion(D, norm_cfg, act_cfg) if use_gated_fusion else \
            ConvModule(D * 2, D, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg, act_cfg),
            ConvModule(D, D2, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        self.c2_proj = ConvModule(c2_channels, D2, 1, norm_cfg=norm_cfg, act_cfg=None)

        self.fusion1 = GatedFusion(D2, norm_cfg, act_cfg) if use_gated_fusion else \
            ConvModule(D2 * 2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.refine2 = nn.Sequential(
            DWConvModule(D2, norm_cfg, act_cfg),
            DWConvModule(D2, norm_cfg, act_cfg),
        )

        self.c1_proj = ConvModule(c1_channels, D2, 1, norm_cfg=norm_cfg, act_cfg=None)

        self.fusion2 = GatedFusion(D2, norm_cfg, act_cfg) if use_gated_fusion else \
            ConvModule(D2 * 2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.final_proj = ConvModule(D2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, c5, c4, c2, c1, return_intermediate=False):
        # Stage 0
        x = self.refine0(c5)
        c4p = self.c4_proj(c4)
        x = self.fusion0(c4p, x) if self.use_gated_fusion else \
            self.fusion0(torch.cat([c4p, x], dim=1))

        # Stage 1
        x = self.up1(x)
        x = self.refine1(x)
        c2p = self.c2_proj(c2)
        x = self.fusion1(c2p, x) if self.use_gated_fusion else \
            self.fusion1(torch.cat([c2p, x], dim=1))
        feat_h4 = x

        # Stage 2
        x = self.up2(x)
        x = self.refine2(x)
        c1p = self.c1_proj(c1)
        x = self.fusion2(c1p, x) if self.use_gated_fusion else \
            self.fusion2(torch.cat([c1p, x], dim=1))
        feat_h2 = x

        x = self.final_proj(x)
        x = self.dropout(x)

        return (x, feat_h4, feat_h2) if return_intermediate else x


# ===================== HEAD =====================
class GCNetHead(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_classes=19,
        decoder_channels=128,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        use_gated_fusion=True,
        c4_channels=64,
        c2_channels=32,
        c1_channels=32,
        use_deep_supervision=True,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.use_deep_supervision = use_deep_supervision

        self.decoder = EnhancedDecoder(
            in_channels, c4_channels, c2_channels, c1_channels,
            decoder_channels, norm_cfg, act_cfg,
            dropout_ratio, use_gated_fusion
        )

        out_ch = decoder_channels // 2

        self.conv_seg = nn.Conv2d(out_ch, num_classes, 1)

        if use_deep_supervision:
            self.aux_h4 = nn.Conv2d(out_ch, num_classes, 1)
            self.aux_h2 = nn.Conv2d(out_ch, num_classes, 1)

    def forward(self, feats: Dict[str, Tensor], return_aux=False, img_size=None):
        c1, c2, c4, c5 = feats['c1'], feats['c2'], feats['c4'], feats['c5']

        if return_aux and self.use_deep_supervision:
            x, h4, h2 = self.decoder(c5, c4, c2, c1, True)

            main = self.conv_seg(x)
            aux1 = self.aux_h4(h4)
            aux2 = self.aux_h2(h2)

            # resize về cùng size
            aux1 = F.interpolate(aux1, size=main.shape[-2:], mode='bilinear', align_corners=self.align_corners)

            return main, aux1, aux2

        x = self.decoder(c5, c4, c2, c1)
        return self.conv_seg(x)
