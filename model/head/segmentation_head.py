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

# ============================================================
# Backbone output channels (channels = 32 mặc định):
#   c1  : H/2,  C    = 32   — stem layer 0 (conv stride2)
#   c2  : H/4,  C    = 32   — stem layer 1 (conv stride2)
#   c4  : H/8,  C*2  = 64   — detail branch stage4 output
#   c5  : H/8,  C*4  = 128  — detail + semantic fused output
#
# Decoder flow:
#   c5 (H/8, 128) → up×2 → fuse c4 (H/8, 64) → refine
#                 → up×2 → fuse c2 (H/4,  32) → refine
#                 → up×2 → fuse c1 (H/2,  32) → refine
#                 → H/2, decoder_channels//2 = 64
#                 → cls head → logits
#
# Auxiliary head:
#   c4 (H/8, 64) → conv → logits  (supervises trực tiếp detail branch)
# ============================================================


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """
    Learned gating giữa skip connection và decoder feature.
    gate = sigmoid(concat(skip, dec)) → output = gate*skip + (1-gate)*dec

    Tương đương Attention Gate trong U-Net++, nhưng nhẹ hơn.
    """
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
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

    def forward(self, skip: Tensor, dec: Tensor) -> Tensor:
        gate = self.gate_conv(torch.cat([skip, dec], dim=1))
        return gate * skip + (1.0 - gate) * dec


class DWConvModule(nn.Module):
    """Depthwise-Separable Conv: nhẹ hơn 3×3 conv thường ~8-9×."""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
    ):
        super().__init__()
        self.dw = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.pw = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    """Standard pre-act residual block (BN→ReLU→Conv×2 + identity)."""
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
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
        return self.act(self.conv2(self.conv1(x)) + x)


# ─────────────────────────────────────────────────────────────
# Main Decoder
# ─────────────────────────────────────────────────────────────

class EnhancedDecoder(nn.Module):
    """
    FPN-style decoder với GatedFusion skip connections.

    Input channels (channels=32 mặc định):
        c5 : H/8,  C*4 = 128  — fused backbone output
        c4 : H/8,  C*2 = 64   — detail branch stage4 (skip)
        c2 : H/4,  C   = 32   — stem stage2 (skip)
        c1 : H/2,  C   = 32   — stem stage1 (skip)

    Pipeline:
        c5 (H/8)  → refine1 → proj → 128ch
                             ↕ GatedFusion(c4_proj, dec)   ← c4 dùng ở đây
        up×2 (H/4) → refine2 → proj → 64ch
                             ↕ GatedFusion(c2_proj, dec)
        up×2 (H/8) → refine3 → DWConv → 64ch              ← WAIT, sai
        ...

    Đúng phải là:
        c5 (H/8)   → ResidualBlock → ConvModule → dec_ch=128
                   fuse c4 (H/8)   → GatedFusion            [same resolution, no upsample]
        up×2→ H/4  → ResidualBlock → ConvModule → dec_ch//2=64
                   fuse c2 (H/4)   → GatedFusion
        up×2→ H/2  → DWConv×2     → 64ch
                   fuse c1 (H/2)   → GatedFusion
        → final_proj → dropout → 64ch output
    """

    def __init__(
        self,
        # c5 channels = C*4 = 128
        in_channels: int = 128,
        # c4 channels = C*2 = 64  (detail branch stage4)
        c4_channels: int = 64,
        # c2 channels = C = 32    (stem layer 1)
        c2_channels: int = 32,
        # c1 channels = C = 32    (stem layer 0)
        c1_channels: int = 32,
        decoder_channels: int = 128,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        D = decoder_channels      # 128
        D2 = decoder_channels // 2  # 64

        # ── Stage 0: c5 (H/8) → refine, fuse c4 (H/8) ────────────
        # Không upsample vì c5 và c4 cùng H/8
        self.refine0 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels, D, kernel_size=3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c4_proj = ConvModule(c4_channels, D, kernel_size=1,
                                  norm_cfg=norm_cfg, act_cfg=None) \
                       if c4_channels != D else nn.Identity()
        if use_gated_fusion:
            self.fusion0 = GatedFusion(D, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion0 = ConvModule(D * 2, D, kernel_size=1,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 1: H/8 → H/4, fuse c2 ──────────────────────────
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(D, D2, kernel_size=3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c2_proj = ConvModule(c2_channels, D2, kernel_size=1,
                                  norm_cfg=norm_cfg, act_cfg=None) \
                       if c2_channels != D2 else nn.Identity()
        if use_gated_fusion:
            self.fusion1 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(D2 * 2, D2, kernel_size=1,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 2: H/4 → H/2, fuse c1 ──────────────────────────
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = nn.Sequential(
            DWConvModule(D2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(D2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c1_proj = ConvModule(c1_channels, D2, kernel_size=1,
                                  norm_cfg=norm_cfg, act_cfg=None) \
                       if c1_channels != D2 else nn.Identity()
        if use_gated_fusion:
            self.fusion2 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(D2 * 2, D2, kernel_size=1,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Final projection ───────────────────────────────────────
        self.final_proj = ConvModule(D2, D2, kernel_size=1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, c5: Tensor, c4: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        """
        Args:
            c5: (B, C*4, H/8, W/8)   = (B, 128, H/8, W/8)
            c4: (B, C*2, H/8, W/8)   = (B,  64, H/8, W/8)  ← same spatial as c5
            c2: (B, C,   H/4, W/4)   = (B,  32, H/4, W/4)
            c1: (B, C,   H/2, W/2)   = (B,  32, H/2, W/2)
        Returns:
            (B, D//2, H/2, W/2) = (B, 64, H/2, W/2)
        """
        # Stage 0: refine c5, fuse c4 (cùng resolution H/8)
        x = self.refine0(c5)
        c4p = self.c4_proj(c4)
        if self.use_gated_fusion:
            x = self.fusion0(c4p, x)
        else:
            x = self.fusion0(torch.cat([c4p, x], dim=1))

        # Stage 1: H/8 → H/4, fuse c2
        x = self.up1(x)
        x = self.refine1(x)
        c2p = self.c2_proj(c2)
        if self.use_gated_fusion:
            x = self.fusion1(c2p, x)
        else:
            x = self.fusion1(torch.cat([c2p, x], dim=1))

        # Stage 2: H/4 → H/2, fuse c1
        x = self.up2(x)
        x = self.refine2(x)
        c1p = self.c1_proj(c1)
        if self.use_gated_fusion:
            x = self.fusion2(c1p, x)
        else:
            x = self.fusion2(torch.cat([c1p, x], dim=1))

        x = self.final_proj(x)
        x = self.dropout(x)
        return x   # (B, 64, H/2, W/2)


# ─────────────────────────────────────────────────────────────
# Auxiliary Head  (supervises c4 — detail branch stage4)
# ─────────────────────────────────────────────────────────────

class GCNetAuxHead(nn.Module):
    """
    Auxiliary segmentation head applied to c4 (H/8, C*2=64ch).

    Tại sao c4?
    - c4 là output của detail branch sau stage4 bilateral fusion.
    - Supervising c4 trực tiếp tạo gradient flow sớm vào detail branch,
      giúp DWSA4 học được useful spatial attention từ đầu training.
    - Không supervise c5 vì c5 đã được main head supervise qua decoder.
    """
    def __init__(
        self,
        in_channels: int = 64,    # C*2 = 64 với channels=32
        mid_channels: int = 64,
        num_classes: int = 19,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            feats: backbone output dict, phải có key 'c4'
        Returns:
            logits at H/8 resolution (sẽ được upsample trong loss computation)
        """
        x = feats['c4'] if isinstance(feats, dict) else feats
        return self.conv_seg(self.conv1(x))


# ─────────────────────────────────────────────────────────────
# Main Segmentation Head
# ─────────────────────────────────────────────────────────────

class GCNetHead(nn.Module):
    """
    Main segmentation head.

    Nhận dict từ GCNetWithEnhance:
        {c1, c2, c4, c5}

    Pipeline:
        c5 (H/8,  128) ─┐
        c4 (H/8,   64) ─┤→ EnhancedDecoder → (H/2, 64) → conv_seg → logits
        c2 (H/4,   32) ─┤
        c1 (H/2,   32) ─┘

    channels=32 (default):
        in_channels  = C*4 = 128
        c4_channels  = C*2 = 64
        c2_channels  = C   = 32
        c1_channels  = C   = 32
    """

    def __init__(
        self,
        in_channels: int = 128,      # c5 = C*4
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
        align_corners: bool = False,
        use_gated_fusion: bool = True,
        c4_channels: int = 64,       # C*2
        c2_channels: int = 32,       # C
        c1_channels: int = 32,       # C
    ):
        super().__init__()
        self.align_corners = align_corners

        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            c4_channels=c4_channels,
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
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(output_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            feats: dict với keys {c1, c2, c4, c5}
        Returns:
            logits: (B, num_classes, H/2, W/2)
            — sẽ được interpolate lên full resolution trong Trainer
        """
        c1 = feats['c1']
        c2 = feats['c2']
        c4 = feats['c4']
        c5 = feats['c5']

        # c4 được đưa vào decoder như skip connection thực sự
        x = self.decoder(c5, c4, c2, c1)
        return self.conv_seg(x)
