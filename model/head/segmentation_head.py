import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List

from components.components import (
    ConvModule,
    BaseDecodeHead,
    resize,
    OptConfigType,
)

# ============================================================
# 1️⃣ ASPP-LITE (Depthwise, rất nhẹ)
# ============================================================

class ASPPLite(nn.Module):
    """
    ASPP-Lite:
    - 3x3 DW conv (rate 1, 2, 3)
    - Global pooling branch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: OptConfigType,
        act_cfg: OptConfigType,
    ):
        super().__init__()

        self.branches = nn.ModuleList([
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=in_channels,      # depthwise
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=in_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=in_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        ])

        # Global context
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )

        self.project = ConvModule(
            out_channels * 4,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[2:]

        feats = [branch(x) for branch in self.branches]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)

        feats.append(gp)

        x = torch.cat(feats, dim=1)
        return self.project(x)


# ============================================================
# 2️⃣ CLASS-AWARE CONTEXT MODULE (NHẸ – RẤT QUAN TRỌNG)
# ============================================================

class ClassAwareContext(nn.Module):
    """
    Learn per-class importance and reweight features
    """

    def __init__(self, channels: int, num_classes: int):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

        self.class_embed = nn.Embedding(num_classes, channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, _, _ = x.shape

        pooled = self.pool(x)                 # (B, C, 1, 1)
        gates = self.fc(pooled)               # (B, num_classes, 1, 1)

        class_weights = self.class_embed.weight   # (num_classes, C)
        class_weights = class_weights.unsqueeze(0)  # (1, num_classes, C)

        # Weighted sum over classes
        weights = (gates.view(B, -1, 1) * class_weights).sum(dim=1)
        weights = weights.view(B, C, 1, 1)

        return x * weights


# ============================================================
# 3️⃣ GCNet HEAD V2 (DROP-IN REPLACEMENT)
# ============================================================

class GCNetHeadV2(BaseDecodeHead):
    """
    ✅ GCNet Head V2:
    - ASPP-Lite
    - Class-Aware Context
    - Lightweight Decoder (giữ nguyên)
    """

    def __init__(
        self,
        decoder_channels: int = 128,
        use_gated_fusion: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        from model.decoder.lightweight_decoder import LightweightDecoder

        # ASPP-Lite
        self.aspp = ASPPLite(
            in_channels=self.in_channels,      # c5
            out_channels=decoder_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # Class-aware context
        self.class_aware = ClassAwareContext(
            channels=decoder_channels,
            num_classes=self.num_classes,
        )

        # Decoder (giữ nguyên kiến trúc bạn đã làm)
        self.decoder = LightweightDecoder(
            in_channels=decoder_channels,
            channels=decoder_channels,
            use_gated_fusion=use_gated_fusion,
        )

        self.conv_seg = nn.Conv2d(
            decoder_channels // 8,
            self.num_classes,
            kernel_size=1
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        inputs: dict {c1, c2, c3, c4, c5}
        """
        c1 = inputs["c1"]
        c2 = inputs["c2"]
        c5 = inputs["c5"]

        # 1️⃣ ASPP-Lite
        x = self.aspp(c5)

        # 2️⃣ Class-aware reweighting
        x = self.class_aware(x)

        # 3️⃣ Decoder
        x = self.decoder(x, [c2, c1, None])

        x = self.dropout(x)
        return self.conv_seg(x)
