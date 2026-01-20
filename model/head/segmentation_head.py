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
    """
    ✅ FIXED VERSION: Perfect compatibility với GCNetWithEnhance
    
    Supported inputs:
    1. Dict: {'c1':..., 'c2':..., 'c5':...} 
    2. Tuple training: (c4_dict/tensor, main_dict)  ← GCNet style
    3. Tuple direct: (c1, c2, c5)
    """
    def __init__(
        self,
        backbone_channels: int = 32,  # GCNet channels param
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        use_gated_fusion: bool = True,
        align_corners: bool = False
    ):
        super().__init__()
        self.align_corners = align_corners
        self.backbone_channels = backbone_channels
        
        # ✅ Auto-compute GCNet channel sizes
        c1_ch = backbone_channels      # 32
        c2_ch = backbone_channels * 2  # 64
        c5_ch = backbone_channels * 4  # 128
        
        # Decoder với exact channel matching
        self.decoder = EnhancedDecoder(
            in_channels=c5_ch,           # 128
            c2_channels=c2_ch,           # 64
            c1_channels=c1_ch,           # 32
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
            use_gated_fusion=use_gated_fusion
        )
        
        # Segmentation head (64 → num_classes)
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(decoder_channels//2, num_classes, kernel_size=1)
        )
    
    def forward(self, feats: Union[Dict[str, Tensor], Tuple[Any, ...]]) -> Tensor:
        """✅ ROBUST input parsing cho tất cả cases"""
        
        # CASE 1: Dict input (inference)
        if isinstance(feats, dict):
            required_keys = {'c1', 'c2', 'c5'}
            if not required_keys.issubset(feats.keys()):
                raise KeyError(
                    f"Missing keys {required_keys - set(feats.keys())}. "
                    f"Expected: {required_keys}"
                )
            c1, c2, c5 = feats['c1'], feats['c2'], feats['c5']
        
        # CASE 2: Training tuple (c4_aux, main_feats)
        elif isinstance(feats, tuple) and len(feats) == 2:
            aux, main = feats
            
            # GCNet training format: (c4_dict/tensor, main_dict)
            if isinstance(main, dict) and {'c1', 'c2', 'c5'}.issubset(main.keys()):
                c1, c2, c5 = main['c1'], main['c2'], main['c5']
            elif isinstance(aux, dict) and {'c1', 'c2', 'c5'}.issubset(aux.keys()):
                c1, c2, c5 = aux['c1'], aux['c2'], aux['c5']
            else:
                # Fallback: treat as positional (c1, c2, c5, ...)
                raise ValueError(
                    "Training tuple must contain dict with ['c1','c2','c5']"
                )
        
        # CASE 3: Direct positional tensors
        elif isinstance(feats, tuple) and len(feats) >= 3:
            c1, c2, c5 = feats[0], feats[1], feats[2]
        
        else:
            raise TypeError(
                f"Unsupported input type: {type(feats)}. "
                f"Expected Dict['c1,c2,c5'] or Tuple with valid format."
            )
        
        # ✅ Decoder forward
        dec_feat = self.decoder(c5, c2, c1)
        seg_logit = self.conv_seg(dec_feat)
        
        return seg_logit
