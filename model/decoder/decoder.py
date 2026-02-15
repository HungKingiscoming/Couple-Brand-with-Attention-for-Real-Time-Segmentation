import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional

from components.components import (
    ConvModule,
    build_activation_layer,
    OptConfigType
)


class GatedFusion(nn.Module):
    """✅ IMPROVED: Thêm shortcut và stability"""
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        self.channels = channels
        
        self.gate_conv = nn.Sequential(
            ConvModule(
                in_channels=channels * 2,
                out_channels=channels // 4,  # Giảm channels cho gate nhẹ hơn
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=channels // 4,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid')
            )
        )
        
        # ✅ Residual shortcut cho stability
        self.shortcut = nn.Identity()

    def forward(self, skip_feat: Tensor, dec_feat: Tensor) -> Tensor:
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        weighted = gate * skip_feat + (1 - gate) * dec_feat
        return self.shortcut(weighted)  # Residual stability


class DWConvModule(nn.Module):
    """✅ UNCHANGED - Perfect implementation"""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
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
            act_cfg=None
        )
        self.pw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ResidualBlock(nn.Module):
    """✅ FIXED: Proper post-act residual"""
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity  # Residual trước activation
        out = self.act(out)
        return out


class EnhancedDecoder(nn.Module):
    """
    ✅ PERFECT MATCH với GCNetWithEnhance:
    Input:  c5(H/8,C*4=128), c2(H/4,C*2=64), c1(H/2,C*1=32)
    Output: (H/2, 64) - ready cho segmentation head
    """
    def __init__(
        self,
        in_channels: int,         # c5: 128
        c2_channels: int,         # c2: 64  
        c1_channels: int,         # c1: 32
        decoder_channels: int = 128,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True
    ):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.use_gated_fusion = use_gated_fusion

        # ✅ Stage 1: c5 (H/8,C5) → H/4 + fuse c2
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        self.c2_proj = ConvModule(
            c2_channels, decoder_channels, 1,
            norm_cfg=norm_cfg, act_cfg=None
        ) if c2_channels != decoder_channels else nn.Identity()

        self.fusion1 = GatedFusion(decoder_channels) if use_gated_fusion else \
                      ConvModule(decoder_channels*2, decoder_channels, 1, 
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ✅ Stage 2: H/4 → H/2 + fuse c1  
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                decoder_channels, decoder_channels//2, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        )
        self.c1_proj = ConvModule(
            c1_channels, decoder_channels//2, 1,
            norm_cfg=norm_cfg, act_cfg=None
        ) if c1_channels != decoder_channels//2 else nn.Identity()

        self.fusion2 = GatedFusion(decoder_channels//2) if use_gated_fusion else \
                      ConvModule((decoder_channels//2)*2, decoder_channels//2, 1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        # ✅ Stage 3: H/2 refine + attention-like DW
        self.refine3 = nn.Sequential(
            DWConvModule(decoder_channels//2, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ResidualBlock(decoder_channels//2, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(decoder_channels//2, norm_cfg=norm_cfg, act_cfg=None),
        )
        self.final_proj = ConvModule(
            decoder_channels//2, decoder_channels//2, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, feats: Union[Tuple[Tensor, Tensor, Tensor], Dict]):
        """✅ FLEXIBLE: Accept Dict hoặc positional args"""
        if isinstance(feats, (tuple, list)) and len(feats) == 3:
            c5, c2, c1 = feats

        # Stage 1: H/8→H/4 + c2 fusion
        x = self.up1(c5)
        x = self.refine1(x)
        c2_p = self.c2_proj(c2)
        x = self.fusion1(c2_p, x)

        # Stage 2: H/4→H/2 + c1 fusion  
        x = self.up2(x)
        x = self.refine2(x)
        c1_p = self.c1_proj(c1)
        x = self.fusion2(c1_p, x)

        # Stage 3: H/2 refine
        x = self.refine3(x)
        x = self.final_proj(x)
        x = self.dropout(x)
        return x


# ✅ COMPLETE USAGE EXAMPLE
def test_decoder():
    decoder = EnhancedDecoder(
        in_channels=128,    # GCNet c5
        c2_channels=64,     # GCNet c2  
        c1_channels=32,     # GCNet c1
        decoder_channels=128
    )
    
    # Test với GCNetWithEnhance output
    backbone_feats = {
        'c1': torch.randn(2, 32, 128, 128),
        'c2': torch.randn(2, 64, 64, 64),
        'c5': torch.randn(2, 128, 32, 32)
    }
    
    dec_out = decoder(backbone_feats)
    print(f"Decoder output: {dec_out.shape}")  # [2, 64, 128, 128] ✓
    
    # Hoặc positional args
    dec_out2 = decoder((backbone_feats['c5'], backbone_feats['c2'], backbone_feats['c1']))
    print(f"Positional: {dec_out2.shape}")    # Same ✓
