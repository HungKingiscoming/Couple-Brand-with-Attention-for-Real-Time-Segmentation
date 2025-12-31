# head.py - FIXED VERSION
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

# Sau đó import:
from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    BaseDecodeHead,
    OptConfigType,
    SampleList
)


class GCNetHead(nn.Module):
    """
    ✅ FIXED: Corrected skip connections
    
    Changes:
    1. ✅ Proper skip order: [c2, c1, None]
    2. ✅ Always enable decoder (lightweight version)
    3. ✅ Correct channel dimensions
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        channels: int = 128,
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        from components.components import ConvModule
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        # ✅ FIXED: Always use lightweight decoder
        self.decoder = LightweightDecoder(
            in_channels=in_channels,
            channels=decoder_channels,
            use_gated_fusion=False,  # Simplified
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # Segmentation head
        self.conv_seg = nn.Conv2d(
            decoder_channels // 8,  # 16 from decoder
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with keys ['c1', 'c2', 'c3', 'c4', 'c5']
        
        Returns:
            logits: (B, num_classes, H, W)
        """
        c1 = inputs['c1']  # (B, 32, H/2, W/2)
        c2 = inputs['c2']  # (B, 32, H/4, W/4)
        c5 = inputs['c5']  # (B, 64, H/8, W/8)
        
        # ✅ FIXED: Correct skip order
        skip_connections = [c2, c1, None]
        
        # Decode to full resolution
        x = self.decoder(c5, skip_connections)  # (B, 16, H, W)
        
        # Dropout
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)  # (B, num_classes, H, W)
        
        return output


class GCNetAuxHead(nn.Module):
    """
    ✅ Auxiliary head (unchanged, already correct)
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 64,
        num_classes: int = 19,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        from components.components import ConvModule
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Conv layers
        self.conv = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        self.conv_seg = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with key 'c4'
        
        Returns:
            logits: (B, num_classes, H/16, W/16)
        """
        x = inputs['c4']
        x = self.conv(x)
        x = self.dropout(x)
        output = self.conv_seg(x)
        return output
