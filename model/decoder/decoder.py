import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict

# Lưu code vào file: decoder.py

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType
)

# ============================================
# GATED FUSION (Cải tiến để tránh mất thông tin)
# ============================================

class GatedFusion(nn.Module):
    """
    Gated fusion giúp mô hình chọn lọc thông tin từ Skip và Decoder
    """
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True)
    ):
        super().__init__()
        self.gate_conv = ConvModule(
            in_channels=channels * 2,
            out_channels=channels, # Nâng từ 1 lên bằng số channels
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='Sigmoid')
        )
    
    def forward(self, enc_feat: Tensor, dec_feat: Tensor) -> Tensor:
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        return gate * enc_feat + (1 - gate) * dec_feat


class LightweightDecoder(nn.Module):    
    def __init__(self, in_channels: int = 64, channels: int = 128,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 act_cfg: dict = dict(type='ReLU', inplace=False)):
        super().__init__()
        
        # ✅ Project c2: 32 -> 64 channels
        self.c2_proj = ConvModule(
            in_channels=32, out_channels=64, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        
        self.up1 = nn.Sequential(
            ConvModule(in_channels, channels, kernel_size=3, padding=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # ✅ Fusion1: 128 + 64 = 192 -> 128
        self.fusion1 = ConvModule(
            in_channels=channels + 64,  # Changed from 32
            out_channels=channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        
        self.up2 = nn.Sequential(
            ConvModule(channels, channels // 2, kernel_size=3, padding=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.fusion2 = ConvModule(
            in_channels=(channels // 2) + 32, out_channels=channels // 2,
            kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        
        self.up3 = nn.Sequential(
            ConvModule(channels // 2, channels // 2, kernel_size=3, padding=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.final = ConvModule(
            channels // 2, channels // 2, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        
        # ✅ Dropout only during training
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: Tensor, skip_connections: List[Tensor]) -> Tensor:
        c2, c1, _ = skip_connections
        
        x = self.up1(x)
        if c2 is not None:
            c2 = self.c2_proj(c2)  # ✅ 32 -> 64
            x = torch.cat([x, c2], dim=1)
            x = self.fusion1(x)
        
        x = self.up2(x)
        if c1 is not None:
            x = torch.cat([x, c1], dim=1)
            x = self.fusion2(x)
        
        x = self.up3(x)
        x = self.final(x)
        x = self.dropout(x) if self.training else x  # ✅ Training only
        
        return x


# ============================================
# CẬP NHẬT HEAD ĐỂ TƯƠNG THÍCH VỚI CHANNELS MỚI
# ============================================

class GCNetHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 19,
        decoder_channels: int = 128,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        self.decoder = LightweightDecoder(
            in_channels=in_channels,
            channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # ✅ conv_seg bây giờ nhận 64 channels thay vì 16
        self.conv_seg = nn.Conv2d(
            decoder_channels // 2, # 64
            num_classes,
            kernel_size=1
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        c1, c2, c5 = inputs['c1'], inputs['c2'], inputs['c5']
        
        x = self.decoder(c5, [c2, c1, None])
        return self.conv_seg(x)
