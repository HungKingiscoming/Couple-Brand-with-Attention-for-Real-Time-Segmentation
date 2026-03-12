"""
segmentation_head.py — Best-of-Both Merge

Kết hợp:
  ✅ Từ file của bạn (doc 11):
     - 4-skip EnhancedDecoder: c5 + c4 (H/8) + c2 (H/4) + c1 (H/2)
     - GatedFusion trong decoder (learned attention gate)
     - GCNetHead nhận đủ {c1, c2, c4, c5}
     - GCNetAuxHead supervise c4 trực tiếp

  ✅ Từ model.py session này:
     - act_cfg inplace=True (nhất quán với backbone)
     - BN requires_grad=True (đảm bảo fine-tune được)
     - Residual block pre-act pattern

Backbone output channels (channels=32 mặc định):
  c1 : H/2,  C    = 32  — stem layer 0 (conv stride2)
  c2 : H/4,  C    = 32  — stem layer 1 (conv stride2)
  c4 : H/8,  C*2  = 64  — detail branch stage4 output
  c5 : H/8,  C*4  = 128 — fused detail+semantic output

Decoder flow:
  c5 (H/8, 128) → Stage0: refine → fuse c4 (H/8, 64)   ← SAME RESOLUTION
               → Stage1: ×2 → H/4 → fuse c2 (H/4, 32)
               → Stage2: ×2 → H/2 → fuse c1 (H/2, 32)
               → H/2, 64ch → conv_seg → logits (H/2)
               → interpolate → full resolution trong Trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict

from components.components import (
    ConvModule,
    build_activation_layer,
    OptConfigType,
)


# =============================================================================
# Building Blocks
# =============================================================================

class GatedFusion(nn.Module):
    """
    Learned gate fusion giữa skip connection và decoder feature.

    gate = sigmoid( Conv1x1(BN(ReLU( concat[skip, dec] ))) )
    out  = gate * skip + (1 - gate) * dec

    Tại sao tốt hơn simple add/concat:
    - Model tự học khi nào trust skip vs decoder feature
    - skip feature ở early stages có thể noisy → gate suppress tự động
    - Gradient flow đến skip features mượt hơn (gate bounded [0,1])
    - Tương đương Attention Gate trong U-Net++ nhưng lightweight hơn
    """
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg:  OptConfigType = dict(type='ReLU', inplace=True),
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
    """
    Depthwise-Separable Conv: ~8-9× nhẹ hơn Conv3x3 thường.
    DW(C,k) → BN → PW(C,1) → BN → ReLU
    Dùng ở Stage2 (H/2) — resolution lớn nhất → tiết kiệm compute.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg:  OptConfigType = dict(type='ReLU', inplace=True),
    ):
        super().__init__()
        self.dw = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=None,        # activation sau PW, không phải DW
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
    """
    Standard residual block: Conv3x3→BN→ReLU → Conv3x3→BN + identity → ReLU

    Dùng ở Stage0 và Stage1 (resolution thấp H/8, H/4) nơi
    regular conv3x3 vẫn affordable và cần capacity cao.
    """
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg:  OptConfigType = dict(type='ReLU', inplace=True),
    ):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3, padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3, padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


# =============================================================================
# EnhancedDecoder — 4-skip FPN với GatedFusion
# =============================================================================

class EnhancedDecoder(nn.Module):
    """
    FPN-style decoder với 4 skip connections và GatedFusion.

    Input channels (channels=32 mặc định):
        c5 : H/8,  C*4 = 128  — fused backbone output (main input)
        c4 : H/8,  C*2 = 64   — detail branch stage4 (skip, CÙNG H/8 với c5)
        c2 : H/4,  C   = 32   — stem stage2 (skip)
        c1 : H/2,  C   = 32   — stem stage1 (skip)

    Pipeline:
        Stage0 (H/8):  c5 → ResidualBlock → Conv → D(=128)ch
                            GatedFusion(c4_proj, dec)    ← NO upsample, same res
        Stage1 (H/4):  ×2 upsample → ResidualBlock → Conv → D/2(=64)ch
                            GatedFusion(c2_proj, dec)
        Stage2 (H/2):  ×2 upsample → DWConv×2 → D/2(=64)ch
                            GatedFusion(c1_proj, dec)
        → final_proj → dropout → D/2(=64)ch output

    Tại sao c4 ở Stage0 KHÔNG upsample:
        c4 và c5 đều ở H/8 → same spatial resolution
        → fuse trực tiếp, không cần interpolate
        → tốt hơn: giữ nguyên spatial alignment, không artifacts
        → c4 chứa detail branch info (edge, texture) bổ sung cho c5 (semantic)
    """

    def __init__(
        self,
        in_channels:     int = 128,   # c5 = C*4
        c4_channels:     int = 64,    # c4 = C*2
        c2_channels:     int = 32,    # c2 = C
        c1_channels:     int = 32,    # c1 = C
        decoder_channels: int = 128,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg:  OptConfigType = dict(type='ReLU', inplace=True),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        D  = decoder_channels        # 128
        D2 = decoder_channels // 2   # 64

        # ── Stage 0: c5 (H/8) → refine, fuse c4 (H/8) ─────────────────────
        # Không upsample vì c4 và c5 cùng H/8
        self.refine0 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels, D, kernel_size=3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c4_proj = (
            ConvModule(c4_channels, D, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=None)
            if c4_channels != D else nn.Identity()
        )
        if use_gated_fusion:
            self.fusion0 = GatedFusion(D, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion0_conv = ConvModule(D * 2, D, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 1: H/8 → H/4, fuse c2 ────────────────────────────────────
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(D, D2, kernel_size=3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c2_proj = (
            ConvModule(c2_channels, D2, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=None)
            if c2_channels != D2 else nn.Identity()
        )
        if use_gated_fusion:
            self.fusion1 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1_conv = ConvModule(D2 * 2, D2, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 2: H/4 → H/2, fuse c1 ────────────────────────────────────
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # DWConv×2 thay ResidualBlock: H/2 lớn → tiết kiệm compute
        self.refine2 = nn.Sequential(
            DWConvModule(D2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(D2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.c1_proj = (
            ConvModule(c1_channels, D2, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=None)
            if c1_channels != D2 else nn.Identity()
        )
        if use_gated_fusion:
            self.fusion2 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2_conv = ConvModule(D2 * 2, D2, kernel_size=1,
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Final ────────────────────────────────────────────────────────────
        self.final_proj = ConvModule(D2, D2, kernel_size=1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = (nn.Dropout2d(dropout_ratio)
                        if dropout_ratio > 0 else nn.Identity())

    def forward(self, c5: Tensor, c4: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        """
        Args:
            c5: (B, C*4, H/8, W/8)  = (B, 128, H/8, W/8)
            c4: (B, C*2, H/8, W/8)  = (B,  64, H/8, W/8)  ← same spatial as c5
            c2: (B, C,   H/4, W/4)  = (B,  32, H/4, W/4)
            c1: (B, C,   H/2, W/2)  = (B,  32, H/2, W/2)
        Returns:
            (B, D//2, H/2, W/2) = (B, 64, H/2, W/2)
        """
        # Stage 0: refine c5, fuse c4 (cùng H/8 → không upsample)
        x   = self.refine0(c5)
        c4p = self.c4_proj(c4)
        if self.use_gated_fusion:
            x = self.fusion0(c4p, x)
        else:
            x = self.fusion0_conv(torch.cat([c4p, x], dim=1))

        # Stage 1: H/8 → H/4, fuse c2
        x   = self.up1(x)
        x   = self.refine1(x)
        c2p = self.c2_proj(c2)
        if self.use_gated_fusion:
            x = self.fusion1(c2p, x)
        else:
            x = self.fusion1_conv(torch.cat([c2p, x], dim=1))

        # Stage 2: H/4 → H/2, fuse c1
        x   = self.up2(x)
        x   = self.refine2(x)
        c1p = self.c1_proj(c1)
        if self.use_gated_fusion:
            x = self.fusion2(c1p, x)
        else:
            x = self.fusion2_conv(torch.cat([c1p, x], dim=1))

        x = self.final_proj(x)
        x = self.dropout(x)
        return x   # (B, 64, H/2, W/2)


# =============================================================================
# GCNetAuxHead — supervise c4 (detail branch stage4)
# =============================================================================

class GCNetAuxHead(nn.Module):
    """
    Auxiliary segmentation head applied to c4 (H/8, C*2 = 64ch).

    Tại sao supervise c4:
    - c4 = detail branch sau stage4 bilateral fusion
    - Gradient trực tiếp vào detail branch từ sớm trong training
    - Giúp DWSA4 học spatial attention có ý nghĩa từ đầu
    - Không supervise c5 vì c5 đã được main head supervise qua decoder
    - Aux weight được decay theo epoch: aux_w * (1 - ep/total)^0.9
      → aux mạnh lúc đầu (guide), yếu dần cuối (main head dominant)
    """
    def __init__(
        self,
        in_channels:  int = 64,    # C*2 = 64 với channels=32
        mid_channels: int = 64,
        num_classes:  int = 19,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg:  OptConfigType = dict(type='ReLU', inplace=True),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3, padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            feats: backbone output dict, cần key 'c4'
        Returns:
            logits (B, num_classes, H/8, W/8)
            → sẽ được upsample lên full resolution trong Trainer
        """
        x = feats['c4'] if isinstance(feats, dict) else feats
        return self.conv_seg(self.conv1(x))


# =============================================================================
# GCNetHead — Main segmentation head
# =============================================================================

class GCNetHead(nn.Module):
    """
    Main segmentation head.

    Nhận dict từ GCNetWithEnhance: {c1, c2, c4, c5}

    Pipeline:
        c5 (H/8,  128) ─┐
        c4 (H/8,   64) ─┤→ EnhancedDecoder → (H/2, 64) → conv_seg → logits
        c2 (H/4,   32) ─┤
        c1 (H/2,   32) ─┘

    channels=32 mặc định:
        in_channels  = C*4 = 128
        c4_channels  = C*2 = 64
        c2_channels  = C   = 32
        c1_channels  = C   = 32
    """

    def __init__(
        self,
        in_channels:      int = 128,   # c5 = C*4
        c4_channels:      int = 64,    # c4 = C*2
        c2_channels:      int = 32,    # c2 = C
        c1_channels:      int = 32,    # c1 = C
        num_classes:      int = 19,
        decoder_channels: int = 128,
        dropout_ratio:    float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg:  OptConfigType = dict(type='ReLU', inplace=True),
        align_corners: bool = False,
        use_gated_fusion: bool = True,
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

        out_ch = decoder_channels // 2   # 64
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(out_ch, num_classes, kernel_size=1),
        )

    def forward(self, feats: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            feats: dict với keys {c1, c2, c4, c5}
        Returns:
            logits: (B, num_classes, H/2, W/2)
            → interpolate lên full resolution trong Trainer
        """
        c1 = feats['c1']
        c2 = feats['c2']
        c4 = feats['c4']
        c5 = feats['c5']

        x = self.decoder(c5, c4, c2, c1)
        return self.conv_seg(x)
