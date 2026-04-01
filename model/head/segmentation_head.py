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

class LiteGatedFusion(nn.Module):
    """Giảm số lượng Conv từ 2 xuống 1 để tăng tốc độ xử lý gate"""
    def __init__(
            self, 
            channels: int, 
            norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
            **kwargs # Đón tất cả các tham số thừa để không bị lỗi TypeError
        ):
        super().__init__()
        _, norm_layer = build_norm_layer(norm_cfg, channels)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, skip: Tensor, dec: Tensor) -> Tensor:
        # Resize skip connection về cùng kích thước với decoder feature nếu cần
        if skip.shape[-2:] != dec.shape[-2:]:
            skip = F.interpolate(skip, size=dec.shape[2:], mode='bilinear', align_corners=False)
        gate = self.gate_conv(torch.cat([skip, dec], dim=1))
        return gate * skip + (1.0 - gate) * dec

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


class EnhancedDecoder(nn.Module):
    def __init__(
        self, 
        in_channels=128, 
        c4_channels=64, 
        c2_channels=32, 
        c1_channels=32, 
        decoder_channels=128,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        **kwargs
    ):
        super().__init__()
        D = decoder_channels # 128
        D2 = D // 2          # 64

        # Stage 0: Cần đưa c4 từ 64 lên 128 trước khi Fuse
        self.refine0 = ConvModule(in_channels, D, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # ĐÂY LÀ LỚP QUAN TRỌNG: Đưa c4 (64) -> D (128)
        self.c4_proj = ConvModule(c4_channels, D, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        
        self.fusion0 = LiteGatedFusion(D, norm_cfg=norm_cfg)

        # Stage 1: H/8 -> H/4
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = ConvModule(D, D2, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # Đưa c2 (32) -> D2 (64)
        self.c2_proj = ConvModule(c2_channels, D2, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)

        # Stage 2: H/4 -> H/2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = nn.Sequential(
            nn.Conv2d(D2, D2, kernel_size=3, padding=1, groups=D2),
            nn.Conv2d(D2, D2, kernel_size=1),
            build_norm_layer(norm_cfg, D2)[1],
            build_activation_layer(act_cfg)
        )
        # Đưa c1 (32) -> D2 (64)
        self.c1_proj = ConvModule(c1_channels, D2, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        
        self.final_proj = ConvModule(D2, D2, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, c5, c4, c2, c1):
        # Stage 0
        x = self.refine0(c5)          # ra 128 channels
        c4_p = self.c4_proj(c4)       # từ 64 lên 128 channels
        x = self.fusion0(c4_p, x)     # 128 + 128 = 256 (Khớp với weight)

        # Stage 1
        x = self.up1(x)
        x = self.refine1(x)           # ra 64 channels
        c2_p = self.c2_proj(c2)       # từ 32 lên 64 channels
        x = x + c2_p                  # Cộng trực tiếp (phải cùng 64 channels)

        # Stage 2
        x = self.up2(x)
        x = self.refine2(x)           # ra 64 channels
        c1_p = self.c1_proj(c1)       # từ 32 lên 64 channels
        x = x + c1_p                  # Cộng trực tiếp

        return self.final_proj(x)



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
        mid_channels: int = 64,
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
        c4_channels: int = 64,
        c2_channels: int = 32,
    ):
        super().__init__()

        self.align_corners = align_corners

        self.decoder = EnhancedDecoder(
            in_channels=in_channels,
            c4_channels=c4_channels,
            c2_channels=c2_channels,
            c1_channels=c1_channels,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg, # Bây giờ tham số này đã được EnhancedDecoder chấp nhận
            act_cfg=act_cfg,   # Đảm bảo truyền cả act_cfg nếu có
            dropout_ratio=dropout_ratio
        )

        output_channels = decoder_channels // 2   # 64
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),  # [0]
            nn.Conv2d(output_channels, num_classes, kernel_size=1),                # [1]
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # 1. Trích xuất đầy đủ 4 tầng đặc trưng
        if isinstance(inputs, dict):
            c1 = inputs['c1']
            c2 = inputs['c2']
            c4 = inputs['c4'] # Thêm c4
            c5 = inputs['c5']
        else:
            # Nếu đầu vào là list/tuple, hãy đảm bảo thứ tự truyền vào khớp với backbone
            c1, c2, c4, c5 = inputs[0], inputs[1], inputs[2], inputs[3]

        # 2. Truyền ĐỦ 4 tham số vào decoder (theo đúng thứ tự khai báo của EnhancedDecoder)
        # Thứ tự thường là: c5 (sâu nhất) -> c4 -> c2 -> c1 (nông nhất)
        x = self.decoder(c5, c4, c2, c1) 
        
        x = self.conv_seg(x)
        return x
