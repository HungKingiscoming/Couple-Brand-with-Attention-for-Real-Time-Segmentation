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
    """✅ STABLE: Reduced gate complexity + residual"""
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        # ✅ Giảm channels cho gate (lighter computation)
        self.gate_conv = nn.Sequential(
            ConvModule(
                in_channels=channels * 2,
                out_channels=channels // 4,  # ← Giảm từ channels xuống channels//4
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False  # ✅ No bias với BN
            ),
            ConvModule(
                in_channels=channels // 4,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=None,  # ✅ No BN trước Sigmoid
                act_cfg=dict(type='Sigmoid'),
                bias=True  # ✅ Bias for Sigmoid
            )
        )
        
        # ✅ Learnable residual weight (start small)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, skip_feat: Tensor, dec_feat: Tensor) -> Tensor:
        concat = torch.cat([skip_feat, dec_feat], dim=1)
        gate = self.gate_conv(concat)
        
        # ✅ Weighted fusion with stability
        out = gate * skip_feat + (1 - gate) * dec_feat
        
        # ✅ Residual connection (prefer skip_feat initially)
        out = self.alpha * out + (1 - self.alpha) * skip_feat
        return out


class DWConvModule(nn.Module):
    """
    🔥 CRITICAL FIX: Proper initialization for depthwise convolutions
    
    Problem: DWConv với groups=channels có variance rất cao
    Solution: Custom initialization với std nhỏ hơn nhiều
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        init_scale: float = 0.01  # ✅ Thêm parameter để control init
    ):
        super().__init__()
        padding = kernel_size // 2
        self.init_scale = init_scale

        # ✅ Depthwise Conv (CRITICAL: No bias with BN)
        self.dw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # ← Depthwise
            norm_cfg=norm_cfg,
            act_cfg=None,  # ✅ No activation in DW
            bias=False  # ✅ CRITICAL: No bias when using BN
        )
        
        # ✅ Pointwise Conv
        self.pw_conv = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False  # ✅ No bias with BN
        )
        
        # ✅ Initialize weights PROPERLY
        self._init_weights()

    def _init_weights(self):
        """
        🔥 CRITICAL: Custom initialization for DWConv
        
        Standard Kaiming init assumes fan_in = kernel_size^2 * in_channels
        But for DWConv, fan_in = kernel_size^2 (since groups=channels)
        → Need MUCH smaller std
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels:  # Depthwise
                    # ✅ CRITICAL: Very small initialization
                    # Standard: std ~ 0.1-0.3
                    # DWConv: std ~ 0.001-0.01
                    nn.init.normal_(m.weight, mean=0.0, std=self.init_scale)
                    
                    # ✅ Alternative: Uniform init
                    # bound = self.init_scale
                    # nn.init.uniform_(m.weight, -bound, bound)
                    
                else:  # Pointwise or regular conv
                    nn.init.kaiming_normal_(
                        m.weight, 
                        mode='fan_out', 
                        nonlinearity='relu'
                    )
                
                # ✅ Ensure no bias (should already be None)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # ✅ BN init: weight=1, bias=0
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        # ✅ Simple forward (initialization does the heavy lifting)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ResidualBlock(nn.Module):
    """✅ STABLE: Proper residual block"""
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
            act_cfg=act_cfg,
            bias=False
        )
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,  # ✅ No act before residual add
            bias=False
        )
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity  # ✅ Add then activate
        out = self.act(out)
        return out


class EnhancedDecoder(nn.Module):
    """
    ✅ FIXED VERSION with proper initialization
    
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
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        
        # ================== Stage 1: H/8 → H/4 ==================
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False
            )
        )
        
        self.c2_proj = ConvModule(
            in_channels=c2_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias=False
        ) if c2_channels != decoder_channels else nn.Identity()
        
        if use_gated_fusion:
            self.fusion1_gate = GatedFusion(
                decoder_channels, 
                norm_cfg=norm_cfg, 
                act_cfg=act_cfg
            )
        else:
            self.fusion1 = ConvModule(
                in_channels=decoder_channels * 2,
                out_channels=decoder_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False
            )

        # ================== Stage 2: H/4 → H/2 ==================
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = nn.Sequential(
            ResidualBlock(decoder_channels, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False
            )
        )
        
        self.c1_proj = ConvModule(
            in_channels=c1_channels,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias=False
        ) if c1_channels != decoder_channels // 2 else nn.Identity()
        
        if use_gated_fusion:
            self.fusion2_gate = GatedFusion(
                decoder_channels // 2, 
                norm_cfg=norm_cfg, 
                act_cfg=act_cfg
            )
        else:
            self.fusion2 = ConvModule(
                in_channels=(decoder_channels // 2) * 2,
                out_channels=decoder_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False
            )

        # ================== Stage 3: H/2 Refinement ==================
        # ✅ CRITICAL: DWConv with proper initialization
        self.refine3 = nn.Sequential(
            DWConvModule(
                decoder_channels // 2, 
                kernel_size=3, 
                norm_cfg=norm_cfg, 
                act_cfg=act_cfg,
            ),
            DWConvModule(
                decoder_channels // 2, 
                kernel_size=3, 
                norm_cfg=norm_cfg, 
                act_cfg=None, 
            ),
        )
        
        self.final_proj = ConvModule(
            in_channels=decoder_channels // 2,
            out_channels=decoder_channels // 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # ✅ Apply global initialization
        self._init_global_weights()

    def _init_global_weights(self):
        """
        ✅ Global initialization for non-DWConv layers
        (DWConv handles its own initialization)
        """
        for m in self.modules():
            if isinstance(m, DWConvModule):
                continue
            
            if isinstance(m, nn.Conv2d):
                # Regular conv: Kaiming init
                if m.groups == 1:  # Not depthwise
                    nn.init.kaiming_normal_(
                        m.weight, 
                        mode='fan_out', 
                        nonlinearity='relu'
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, c5: Tensor, c2: Tensor, c1: Tensor) -> Tensor:
        # Stage 1: H/8 → H/4
        x = self.up1(c5)
        x = self.refine1(x)
        c2_proj = self.c2_proj(c2)
        
        if self.use_gated_fusion:
            x = self.fusion1_gate(c2_proj, x)
        else:
            x = self.fusion1(torch.cat([x, c2_proj], dim=1))

        # Stage 2: H/4 → H/2
        x = self.up2(x)
        x = self.refine2(x)
        c1_proj = self.c1_proj(c1)
        
        if self.use_gated_fusion:
            x = self.fusion2_gate(c1_proj, x)
        else:
            x = self.fusion2(torch.cat([x, c1_proj], dim=1))

        # Stage 3: H/2 refinement
        x = self.refine3(x)
        x = self.final_proj(x)
        x = self.dropout(x)
        
        return x


class GCNetAuxHead(nn.Module):
    """✅ UNCHANGED - Already correct"""
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
            act_cfg=act_cfg,
            bias=False
        )
        
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )
        
        # ✅ Init segmentation head
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # ✅ Smaller init for final classifier
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Union[Dict[str, Tensor], Tensor]) -> Tensor:
        if isinstance(x, dict):
            x = x.get('c4', x['c4'])
        x = self.conv1(x)
        return self.conv_seg(x)


class GCNetHead(nn.Module):
    """✅ FIXED: Proper initialization + fake projections"""
    def __init__(
        self, 
        backbone_channels=32, 
        num_classes=19, 
        decoder_channels=128,
        in_channels=None, 
        c1_channels=None, 
        c2_channels=None, 
        **kwargs
    ):
        super().__init__()
        
        # Resolve channels
        in_channels = in_channels or backbone_channels * 4
        c2_channels = c2_channels or backbone_channels * 2
        c1_channels = c1_channels or backbone_channels
        
        # ✅ Decoder with proper init
        decoder_args = {
            'in_channels': in_channels,
            'c2_channels': c2_channels,
            'c1_channels': c1_channels,
            'decoder_channels': decoder_channels,
            'norm_cfg': kwargs.get('norm_cfg', dict(type='BN', requires_grad=True)),
            'act_cfg': kwargs.get('act_cfg', dict(type='ReLU', inplace=False)),
            'dropout_ratio': kwargs.get('dropout_ratio', 0.1),
            'use_gated_fusion': kwargs.get('use_gated_fusion', True),
        }
        
        self.decoder = EnhancedDecoder(**decoder_args)
        
        # ✅ Segmentation head with BN
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(kwargs.get('dropout_ratio', 0.1)),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )
        
        # ✅ Fake projections WITH BatchNorm
        self.fake_c2_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2)
        )
        self.fake_c1_proj = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4)
        )
        
        # ✅ Init conv_seg
        self._init_seg_head()
    
    def _init_seg_head(self):
        """Initialize final segmentation head"""
        for m in self.conv_seg.modules():
            if isinstance(m, nn.Conv2d):
                # ✅ Very small init for classifier
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, feats: Union[Dict[str, Tensor], Tuple[Any, ...]]) -> Tensor:
        # CASE 1: Dict from inference
        if isinstance(feats, dict):
            c1, c2, c5 = feats['c1'], feats['c2'], feats['c5']
        
        # CASE 2: Tuple (c4, c5) from training
        elif isinstance(feats, tuple) and len(feats) == 2:
            c4, c5 = feats
            
            # ✅ Fake c2, c1 with BN
            c2 = F.interpolate(c4, scale_factor=2, mode='bilinear', align_corners=False)
            c2 = self.fake_c2_proj(c2)
            
            c1 = F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)
            c1 = self.fake_c1_proj(c1)
        
        # CASE 3: Full tuple (c1, c2, c5)
        elif isinstance(feats, tuple) and len(feats) >= 3:
            c1, c2, c5 = feats[:3]
        
        else:
            raise ValueError(f"Unsupported feats type: {type(feats)}")
        
        # Decode
        dec_feat = self.decoder((c5, c2, c1))
        return self.conv_seg(dec_feat)
