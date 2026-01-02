import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional

from components.components import ConvModule, OptConfigType
from model.decoder import LightweightDecoder


class GCNetHead(nn.Module):
    """
    ✅ Main segmentation head with lightweight decoder
    
    Input:
        - c1: (B, 32, H/2, W/2)
        - c2: (B, 32, H/4, W/4)
        - c5: (B, 64, H/8, W/8)
    
    Output:
        - logits: (B, num_classes, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 64,  # c5 channels
        num_classes: int = 19,
        decoder_channels: int = 128,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_cfg: dict = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Decoder: takes c5 and skip connections
        self.decoder = LightweightDecoder(
            in_channels=in_channels,
            channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Dropout before final conv
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # ✅ FIX: Calculate correct input channels for final conv
        # LightweightDecoder outputs (decoder_channels // 2) channels
        # If decoder_channels=128, then output is 64
        self.conv_seg = nn.Conv2d(
            decoder_channels // 2,  # <--- THIS WAS THE FIX (Changed from 16 to 64)
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict with keys 'c1', 'c2', 'c5'
        Returns:
            logits: (B, num_classes, H, W)
        """
        c1 = inputs['c1']
        c2 = inputs['c2']
        c5 = inputs['c5']
        
        # Decoder output: (B, 64, H, W)
        x = self.decoder(c5, [c2, c1, None])
        
        # Dropout
        x = self.dropout(x)
        
        # Segmentation: (B, 64, H, W) -> (B, num_classes, H, W)
        output = self.conv_seg(x)
        
        return output


class GCNetAuxHead(nn.Module):
    """
    ✅ Auxiliary head for deep supervision on c4 (H/16)
    Input: c4 (B, 128, H/16, W/16)
    Output: logits (B, num_classes, H/16, W/16)
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
        
        # Final segmentation layer
        self.conv_seg = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        x = inputs['c4']
        x = self.conv(x)
        x = self.dropout(x)
        output = self.conv_seg(x)
        return output
