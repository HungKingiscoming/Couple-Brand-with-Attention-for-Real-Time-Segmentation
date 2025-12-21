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


class GCNetHead(BaseDecodeHead):
    """
    ✅ FIXED: Main segmentation head khớp với sơ đồ
    
    Input from backbone:
        - c1: H/2
        - c2: H/4
        - c3: H/8
        - c4: H/16 (for aux head only)
        - c5: H/8 (main features)
    
    Args:
        decode_enabled (bool): Use decoder or simple fusion
        decoder_channels (int): Base decoder channels
        skip_channels (List[int]): Skip connection channels [c3, c2, c1]
        use_gated_fusion (bool): Use gated fusion in decoder
    """
    
    def __init__(
        self,
        decode_enabled: bool = True,
        decoder_channels: int = 128,
        skip_channels: list = [64, 32, 32],  # [c3, c2, c1] from backbone
        use_gated_fusion: bool = True,
        **kwargs
    ):

        
        super().__init__(**kwargs)
        
        self.decode_enabled = decode_enabled
        
        if decode_enabled:
            # ✅ Use decoder with proper skip channels
            from decoder import LightweightDecoder
            
            self.decoder = LightweightDecoder(
                in_channels=self.in_channels,     # c5 channels (64)
                channels=decoder_channels,         # 128
                skip_channels=skip_channels,       # [64, 32, 32]
                use_gated_fusion=use_gated_fusion,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            
            # Output from decoder: channels // 8 = 16
            conv_in_channels = decoder_channels // 8
        else:
            # Simple fusion without decoder
            self.decoder = None
            
            # Fuse c1, c2, c3, c5 (all resized to H/8)
            total_channels = sum([32, 32, 64, 64])  # c1 + c2 + c3 + c5
            
            self.fusion = ConvModule(
                in_channels=total_channels,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            
            conv_in_channels = self.channels
        
        # Segmentation conv
        self.conv_seg = nn.Conv2d(
            conv_in_channels,
            self.num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        ✅ FIXED: Handle dict inputs from backbone
        
        Args:
            inputs: Dict with keys ['c1', 'c2', 'c3', 'c4', 'c5']
        
        Returns:
            Segmentation logits (B, num_classes, H/8 or H, W/8 or W)
        """
        # Extract features
        c1 = inputs['c1']  # H/2
        c2 = inputs['c2']  # H/4
        c3 = inputs['c3']  # H/8
        c5 = inputs['c5']  # H/8 (main output)
        
        if self.decode_enabled and self.decoder is not None:
            # ✅ Use decoder with skip connections [c3, c2, c1]
            skip_connections = [c3, c2, c1]
            
            x = self.decoder(c5, skip_connections)  # -> (B, 16, H, W)
        else:
            # Simple multi-scale fusion at H/8 resolution
            target_size = c5.shape[2:]  # H/8, W/8
            
            # Resize c1, c2 to H/8
            c1_resized = resize(c1, size=target_size, mode='bilinear',
                               align_corners=self.align_corners)
            c2_resized = resize(c2, size=target_size, mode='bilinear',
                               align_corners=self.align_corners)
            
            # Concatenate [c1, c2, c3, c5]
            x = torch.cat([c1_resized, c2_resized, c3, c5], dim=1)
            x = self.fusion(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)
        
        return output



class GCNetAuxHead(BaseDecodeHead):
    """
    ✅ Auxiliary head for deep supervision on c4 (H/16)
    
    Args:
        Same as BaseDecodeHead
    """
    
    def __init__(self, **kwargs):
        if 'input_transform' not in kwargs:
            kwargs['input_transform'] = None
        
        super().__init__(**kwargs)
        
        # Simple conv layers
        self.conv = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )
        
        # Segmentation conv
        self.conv_seg = nn.Conv2d(
            self.channels,
            self.num_classes,
            kernel_size=1
        )
    
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        ✅ FIXED: Extract c4 from dict
        
        Args:
            inputs: Dict with key 'c4' (B, C, H/16, W/16)
        
        Returns:
            Segmentation logits (B, num_classes, H/16, W/16)
        """
        # Extract c4 (stage 4 semantic features)
        x = inputs['c4']
        
        # Convolutions
        x = self.conv(x)
        x = self.dropout(x)
        
        # Segmentation
        output = self.conv_seg(x)
        
        return output
