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
    FPN-style decoder với GatedFusion skip connections.

    Input:
        c5 : (B, C*4, H/8,  W/8)  = (B, 128, H/8,  W/8)   semantic+detail fused
        c4 : (B, C*2, H/8,  W/8)  = (B,  64, H/8,  W/8)   detail branch raw
        c2 : (B, C,   H/4,  W/4)  = (B,  32, H/4,  W/4)   early detail
        c1 : (B, C,   H/2,  W/2)  = (B,  32, H/2,  W/2)   stem features

    Output:
        (B, D2, H/2, W/2) = (B, 64, H/2, W/2)

    Luồng xử lý:
        Stage 0 (H/8  → H/8):  refine c5, scale c4, GatedFusion → D=128
        Stage 1 (H/8  → H/4):  upsample, ResidualBlock, GatedFusion với c2, THEN giảm channel
        Stage 2 (H/4  → H/2):  upsample, DWConv×2, GatedFusion với c1
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

        # ── Stage 0: refine c5 tại H/8 ───────────────────────────────────────
        # Dùng DWConv thay ResidualBlock vì:
        #   - H/8 là spatial lớn nhất trong decoder → tốn compute nhất
        #   - c5 đã mang rich features từ backbone+SPP → không cần refine mạnh
        #   - DWConv tiết kiệm ~8x FLOPs so với Conv3x3 thông thường
        self.refine0 = nn.Sequential(
            DWConvModule(in_channels, 3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels, D, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        # in_channels=128, D=128 → Conv1x1 không thay đổi channel
        # nhưng thêm BN+ReLU để normalize trước khi fuse

        # ── c4 projection + LayerScale ────────────────────────────────────────
        # c4 = detail branch features (C*2=64) cần project lên D=128
        # để có cùng channel với x sau refine0
        self.c4_proj = (
            ConvModule(c4_channels, D, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            if c4_channels != D
            else nn.Identity()
        )

        # [FIX] LayerScale init = 0.1 thay vì 1.0
        # Lý do: lúc đầu training, c4 (detail raw) có thể overwhelm
        # x (semantic đã được normalize nhiều lần). Init nhỏ để:
        #   - Residual path (x) chiếm ưu thế lúc đầu → training stable
        #   - Model dần học tăng c4_scale khi thấy c4 hữu ích
        #   - Tránh gradient explosion ở giai đoạn đầu
        self.c4_scale = nn.Parameter(torch.full((D,), 0.1))

        # GatedFusion hoặc simple concat fusion cho stage 0
        if use_gated_fusion:
            self.fusion0 = GatedFusion(D, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion0 = ConvModule(D*2, D, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 1: H/8 → H/4, fuse với c2 ─────────────────────────────────
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # [FIX] Tách refine1 thành 2 bước riêng biệt:
        #   Bước cũ: ResidualBlock(D) → Conv(D→D2) → fuse với c2(D2)
        #            → c2 chỉ thấy compressed D2 features
        #   Bước mới: ResidualBlock(D) → fuse với c2(D) → Conv(D→D2)
        #             → c2 interact với đầy đủ D=128 channel features
        self.refine1 = ResidualBlock(D, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # ResidualBlock giữ nguyên D=128 channels

        # c2 (C=32) cần project lên D=128 để fuse ở full channel
        # [FIX] project c2 lên D thay vì D2
        self.c2_proj = (
            ConvModule(c2_channels, D, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            if c2_channels != D
            else nn.Identity()
        )

        # GatedFusion ở D=128 channels (full channel, trước khi nén)
        if use_gated_fusion:
            self.fusion1 = GatedFusion(D, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(D*2, D, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # [FIX] Giảm channel SAU fusion: D=128 → D2=64
        # Tách ra khỏi refine1 để fusion xảy ra ở D channels
        self.proj1 = ConvModule(D, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Stage 2: H/4 → H/2, fuse với c1 ─────────────────────────────────
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # DWConv×2 đủ mạnh ở stage cuối vì:
        #   - Feature đã được refine kỹ ở stage 0 và 1
        #   - H/4 spatial nhỏ → DWConv rẻ
        self.refine2 = nn.Sequential(
            DWConvModule(D2, 3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(D2, 3, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # c1 (C=32) project lên D2=64
        self.c1_proj = (
            ConvModule(c1_channels, D2, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            if c1_channels != D2
            else nn.Identity()
        )

        # GatedFusion ở D2=64 channels
        if use_gated_fusion:
            self.fusion2 = GatedFusion(D2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(D2*2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ── Output projection ─────────────────────────────────────────────────
        self.final_proj = ConvModule(D2, D2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout    = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()


    def forward(self, c5, c4, c2, c1):


        x   = self.refine0(c5)

        c4p = self.c4_proj(c4) * self.c4_scale.view(1, -1, 1, 1)

        if self.use_gated_fusion:
            x = self.fusion0(c4p, x)
            # GatedFusion: gate = sigmoid(Conv(concat(c4p, x)))
            # c5 = gate*c4p + (1-gate)*x
        else:
            x = self.fusion0(torch.cat([c4p, x], dim=1))
        # x : (B, D=128, H/8, W/8)

        # ── Stage 1: upsample H/8→H/4, fuse c2 tại D=128, THEN giảm channel ─
        x = self.up1(x)
        # x : (B, D=128, H/4, W/4)

        x = self.refine1(x)
        # x : (B, D=128, H/4, W/4)  — ResidualBlock giữ nguyên channel

        # [FIX] c2 project lên D=128 (không phải D2=64 như trước)
        c2p = self.c2_proj(c2)
        # c2p : (B, D=128, H/4, W/4)

        if self.use_gated_fusion:
            x = self.fusion1(c2p, x)
            # Fusion xảy ra ở D=128 → c2 interact với đầy đủ features
        else:
            x = self.fusion1(torch.cat([c2p, x], dim=1))
        # x : (B, D=128, H/4, W/4)

        # [FIX] Giảm channel SAU fusion (không phải trước như cũ)
        x = self.proj1(x)
        # x : (B, D2=64, H/4, W/4)

        # ── Stage 2: upsample H/4→H/2, fuse c1 tại D2=64 ────────────────────
        x = self.up2(x)
        # x : (B, D2=64, H/2, W/2)

        x = self.refine2(x)
        # x : (B, D2=64, H/2, W/2)  — DWConv×2

        c1p = self.c1_proj(c1)
        # c1p : (B, D2=64, H/2, W/2)

        if self.use_gated_fusion:
            x = self.fusion2(c1p, x)
        else:
            x = self.fusion2(torch.cat([c1p, x], dim=1))
        # x : (B, D2=64, H/2, W/2)

       
        return self.dropout(self.final_proj(x))
        
