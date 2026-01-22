import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Union, Optional, Tuple, Any

from components.components import (
    ConvModule,
    build_activation_layer,
    OptConfigType
)


class GatedFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        self.gate_conv = nn.Sequential(
            ConvModule(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid')
            )
        )

    def forward(self, skip_feat: Tensor, dec_feat: Tensor) -> Tensor:
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        out = gate * skip_feat + (1 - gate) * dec_feat
        return out


class DWConvModule(nn.Module):
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
        out = out + identity
        out = self.act(out)
        return out


class EnhancedDecoder(nn.Module):
    """
    Decoder for GCNet backbone:
      - c5: (B, in_channels, H/8, W/8)
      - c2: (B, c2_channels, H/4, W/4)
      - c1: (B, c1_channels, H/2, W/2)
    Output:
      - (B, decoder_channels//2, H/2, W/2)
    """
    def __init__(
        self,
        in_channels: int,
        c2_channels: int,
        c1_channels: int,
        decoder_channels: int = 128,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        use_gated_fusion: bool = True
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion

        # Stage 1: H/8 (c5) → H/4, fuse with c2
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
            in_channels=c2_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        ) if c2_channels != decoder_channels else nn.Identity()
        
        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels * 2,
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )

        # Stage 2: H/4 → H/2, fuse with c1
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        self.c1_proj = ConvModule(
            in_channels=c1_channels,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        ) if c1_channels != decoder_channels // 2 else nn.Identity()
        
        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(decoder_channels // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.fusion2 = ConvModule(
                in_channels=(decoder_channels // 2) * 2,
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )

        # Stage 3: H/2 → H/2 (refine)
        self.refine3 = nn.Sequential(
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
            DWConvModule(decoder_channels // 2, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        
        self.final_proj = ConvModule(
            in_channels=decoder_channels // 2,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, c5: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        # Stage 1: H/8 → H/4
        x = self.up1(c5)  # (B, in_channels, H/4, W/4)
        x = self.refine1(x)  # (B, decoder_channels=128, H/4, W/4)
        c2_proj = self.c2_proj(c2)  # (B, decoder_channels=128, H/4, W/4)
        if self.use_gated_fusion:
            x = self.fusion1_gate(c2_proj, x)  # (B, 128, H/4, W/4)
        else:
            x = self.fusion1(torch.cat([x, c2_proj], dim=1))  # (B, 128, H/4, W/4)

        # Stage 2: H/4 → H/2
        x = self.up2(x)  # (B, 128, H/2, W/2)
        x = self.refine2(x)  # (B, 64, H/2, W/2)
        c1_proj = self.c1_proj(c1)  # (B, 64, H/2, W/2)
        if self.use_gated_fusion:
            x = self.fusion2_gate(c1_proj, x)  # (B, 64, H/2, W/2)
        else:
            x = self.fusion2(torch.cat([x, c1_proj], dim=1))  # (B, 64, H/2, W/2)

        # Stage 3: refine H/2
        x = self.refine3(x)  # (B, 64, H/2, W/2)
        x = self.final_proj(x)  # (B, 64, H/2, W/2)
        x = self.dropout(x)  # (B, 64, H/2, W/2)
        
        return x

class GCNetAuxHead(nn.Module):
    """✅ PERFECT - Không cần sửa"""
    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 96,
        num_classes: int = 19,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        dropout_ratio: float = 0.1,
        align_corners: bool = False
    ):
        super().__init__()
        self.align_corners = align_corners
        
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )
    
    def forward(self, x: Union[Dict[str, Tensor], Tensor]) -> Tensor:
        if isinstance(x, dict):
            x = x.get('c4', x['c4'])  # Flexible c4 access
        x = self.conv1(x)
        return self.conv_seg(x)


class GCNetHead(nn.Module):
    def __init__(self, backbone_channels=32, num_classes=19, decoder_channels=128,
                 in_channels=None, c1_channels=None, c2_channels=None, **kwargs):
        super().__init__()
        
        # Channels
        self.in_channels = in_channels or backbone_channels*4  # 128
        self.c2_channels = c2_channels or backbone_channels*2  # 64
        self.c1_channels = c1_channels or backbone_channels    # 32
        
        # ✅ DECODER
        decoder_args = {
            'in_channels': self.in_channels,
            'c2_channels': self.c2_channels,
            'c1_channels': self.c1_channels,
            'decoder_channels': decoder_channels,
            'norm_cfg': kwargs.get('norm_cfg'),
            'act_cfg': kwargs.get('act_cfg'),
            'dropout_ratio': kwargs.get('dropout_ratio'),
            'use_gated_fusion': kwargs.get('use_gated_fusion', True)
        }
        self.decoder = EnhancedDecoder(**decoder_args)
        
        # ✅ PROJECTION layers cho fake features (training mode)
        self.c2_fake_proj = nn.Conv2d(self.in_channels, self.c2_channels, 1, bias=False)
        self.c1_fake_proj = nn.Conv2d(self.c2_channels, self.c1_channels, 1, bias=False)
        
        # Seg head
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(kwargs.get('dropout_ratio', 0.1)),
            nn.Conv2d(decoder_channels//2, num_classes, 1)
        )
    
    def forward(self, feats: Union[Dict[str, Tensor], Tuple[Any, ...]]) -> Tensor:
        """✅ FIXED: Handle Dict và Tuple correctly"""
        
        # ✅ CASE 1: Dict (inference/eval mode)
        if isinstance(feats, dict):
            c1 = feats['c1']  # (B, 32, H/2, W/2)
            c2 = feats['c2']  # (B, 64, H/4, W/4)
            c5 = feats['c5']  # (B, 128, H/8, W/8)
        
        # ✅ CASE 2: Tuple training (c4, c5_enhanced)
        elif isinstance(feats, tuple) and len(feats) == 2:
            c4, c5 = feats  # Both H/8, 128ch
            
            # Fake c2 (H/4)
            c2 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
            c2 = self.c2_fake_proj(c2)  # ✅ Use registered layer
            
            # Fake c1 (H/2)
            c1 = F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)
            c1 = self.c1_fake_proj(c1)  # ✅ Use registered layer
            
        # ✅ CASE 3: Full tuple (c1, c2, c5)
        elif isinstance(feats, tuple) and len(feats) >= 3:
            c1, c2, c5 = feats[0], feats[1], feats[2]
        
        else:
            raise ValueError(
                f"Unsupported feats type: {type(feats)}, "
                f"len={len(feats) if isinstance(feats, tuple) else 'N/A'}"
            )
        
        # Decoder + segmentation
        dec_feat = self.decoder(c5, c2, c1)
        return self.conv_seg(dec_feat)

