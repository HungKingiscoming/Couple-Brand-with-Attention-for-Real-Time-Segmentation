import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional

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
            nn.Sequential(
                # Depthwise
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    dilation=1,
                    groups=in_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                # Pointwise
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            ),
            nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
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
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            ),
            nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=3,
                    dilation=3,
                    groups=in_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
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
# 3️⃣ GCNet HEAD V2 (FIXED VERSION)
# ============================================================

class GCNetHead(BaseDecodeHead):
    """
    ✅ GCNet Head V2 - FIXED:
    - Properly filter kwargs before passing to BaseDecodeHead
    - ASPP-Lite for multi-scale context
    - Class-Aware Context for handling imbalanced classes
    - Optional lightweight decoder
    """

    def __init__(
        self,
        # Custom parameters (not for BaseDecodeHead)
        decoder_channels: int = 128,
        decode_enabled: bool = False,
        skip_channels: Optional[List[int]] = None,
        use_gated_fusion: bool = True,
        # BaseDecodeHead parameters
        **kwargs
    ):
        # ✅ Filter out custom kwargs before passing to super().__init__()
        # BaseDecodeHead only accepts: in_channels, channels, num_classes, 
        # dropout_ratio, conv_cfg, norm_cfg, act_cfg, in_index, input_transform,
        # loss_decode, ignore_index, sampler, align_corners, init_cfg
        
        super().__init__(**kwargs)
        
        # Store custom parameters
        self.decoder_channels = decoder_channels
        self.decode_enabled = decode_enabled
        self.skip_channels = skip_channels or [64, 32, 32]
        self.use_gated_fusion = use_gated_fusion

        # ASPP-Lite for multi-scale context
        self.aspp = ASPPLite(
            in_channels=self.in_channels,      # c5 channels
            out_channels=decoder_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # Class-aware context module
        self.class_aware = ClassAwareContext(
            channels=decoder_channels,
            num_classes=self.num_classes,
        )

        # Optional decoder
        if self.decode_enabled:
            try:
                from model.decoder.lightweight_decoder import LightweightDecoder
                
                self.decoder = LightweightDecoder(
                    in_channels=decoder_channels,
                    channels=decoder_channels,
                    use_gated_fusion=use_gated_fusion,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                seg_in_channels = decoder_channels // 8
            except ImportError:
                print("⚠️  LightweightDecoder not found, decoder disabled")
                self.decoder = None
                seg_in_channels = decoder_channels
        else:
            self.decoder = None
            seg_in_channels = decoder_channels

        # Segmentation head
        self.conv_seg = nn.Conv2d(
            seg_in_channels,
            self.num_classes,
            kernel_size=1
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with keys ['c1', 'c2', 'c3', 'c4', 'c5']
        
        Returns:
            Segmentation logits
        """
        c5 = inputs["c5"]

        # 1️⃣ ASPP-Lite for multi-scale context
        x = self.aspp(c5)

        # 2️⃣ Class-aware reweighting
        x = self.class_aware(x)

        # 3️⃣ Optional decoder
        if self.decoder is not None and self.decode_enabled:
            c1 = inputs.get("c1")
            c2 = inputs.get("c2")
            skip_connections = [c2, c1, None]
            x = self.decoder(x, skip_connections)

        # 4️⃣ Dropout and segmentation
        x = self.dropout(x)
        logits = self.conv_seg(x)
        
        return logits


# ============================================================
# 4️⃣ AUXILIARY HEAD (UNCHANGED)
# ============================================================

class GCNetAuxHead(BaseDecodeHead):
    """
    Auxiliary head for deep supervision
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Sequential(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with backbone features
        
        Returns:
            Auxiliary segmentation logits
        """
        # Use c3 or c4 for auxiliary supervision
        x = inputs.get("c3", inputs.get("c4"))
        if x is None:
            # Fallback to any available feature
            x = list(inputs.values())[-2]
        x = self.conv(x)
        return x
