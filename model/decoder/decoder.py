import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

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

# ============================================
# LIGHTWEIGHT DECODER (BẢN SỬA ĐỔI MẠNH MẼ)
# ============================================

class LightweightDecoder(nn.Module):
    """
    FIXED VERSION:
    1. Tăng channels ở đầu ra cuối (16 -> 64) để chứa đủ thông tin 19 classes.
    2. Thêm Dropout nội bộ để tránh overfitting.
    3. Cấu trúc channels: 128 -> 128 -> 64 -> 64
    """
    
    def __init__(
        self,
        in_channels: int = 64,      # c5 channels
        channels: int = 128,         # Base decoder channels
        use_gated_fusion: bool = False,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        # Tầng 1: H/8 -> H/4 (Giữ nguyên 128 channels để học feature sâu)
        self.up1 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Fusion 1 (c2: H/4)
        self.fusion1 = ConvModule(
            in_channels=channels + 32,  # 128 + 32
            out_channels=channels,      # Giữ 128 channels ở scale H/4
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Tầng 2: H/4 -> H/2 (Hạ xuống 64 channels)
        self.up2 = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels // 2, # 64
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Fusion 2 (c1: H/2)
        self.fusion2 = ConvModule(
            in_channels=(channels // 2) + 32,  # 64 + 32
            out_channels=channels // 2,        # 64
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Tầng 3: H/2 -> H (Scale ảnh về gốc)
        self.up3 = nn.Sequential(
            ConvModule(
                in_channels=channels // 2,
                out_channels=channels // 2, # Giữ 64 channels thay vì nén xuống 32
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # ✅ FINAL REFINEMENT: Nâng từ 16 lên 64 channels
        # Điều này cực kỳ quan trọng để mô hình phân biệt được 19 classes phức tạp
        self.final = nn.Sequential(
            ConvModule(
                in_channels=channels // 2,
                out_channels=channels // 2, # 64 channels
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            nn.Dropout2d(0.1) # Thêm dropout nhẹ để bền bỉ hơn
        )
    
    def forward(self, x: Tensor, skip_connections: List[Tensor]) -> Tensor:
        """
        x: c5 (H/8)
        skip_connections: [c2 (H/4), c1 (H/2), None]
        """
        c2, c1, _ = skip_connections
        
        # Stage 1: H/8 -> H/4
        x = self.up1(x)
        if c2 is not None:
            x = torch.cat([x, c2], dim=1)
            x = self.fusion1(x)
        
        # Stage 2: H/4 -> H/2
        x = self.up2(x)
        if c1 is not None:
            x = torch.cat([x, c1], dim=1)
            x = self.fusion2(x)
            
        # Stage 3: H/2 -> H
        x = self.up3(x)
        x = self.final(x) # Output: (B, 64, H, W)
        
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
