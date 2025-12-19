import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


# ============================================
# LIGHTWEIGHT DECODER
# ============================================

class DecoderStage(nn.Module):
    """
    Single decoder stage with gated skip connection
    
    Args:
        in_channels (int): Input channels from encoder
        skip_channels (int): Skip connection channels
        out_channels (int): Output channels
        use_gated_fusion (bool): Whether to use gated fusion
        norm_cfg (dict): Normalization config
        act_cfg (dict): Activation config
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_gated_fusion: bool = True,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True)
    ):
        super().__init__()
        
        self.use_gated_fusion = use_gated_fusion
        
        # Upsample
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Skip connection processing
        if skip_channels != out_channels:
            self.skip_conv = ConvModule(
                in_channels=skip_channels,
                out_channels=out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
        else:
            self.skip_conv = nn.Identity()
        
        # Gated fusion or simple addition
        if use_gated_fusion:
            self.fusion = GatedFusion(
                channels=out_channels,
                norm_cfg=norm_cfg
            )
        else:
            self.fusion = None
        
        # Refinement
        self.refine = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
    
    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input from previous decoder stage (B, C_in, H, W)
            skip: Skip connection from encoder (B, C_skip, H*2, W*2)
        
        Returns:
            out: Decoded features (B, C_out, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)
        
        # Process skip connection if provided
        if skip is not None:
            skip = self.skip_conv(skip)
            
            # Fusion
            if self.fusion is not None:
                x = self.fusion(skip, x)
            else:
                x = x + skip
        
        # Refinement
        x = self.refine(x)
        
        return x


class GatedFusion(nn.Module):
    """Gated fusion for adaptive feature selection"""
    
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True)
    ):
        super().__init__()
        
        self.gate_conv = ConvModule(
            in_channels=channels * 2,
            out_channels=1,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, enc_feat: Tensor, dec_feat: Tensor) -> Tensor:
        """
        Args:
            enc_feat: Encoder features (skip connection)
            dec_feat: Decoder features
        
        Returns:
            Fused features
        """
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        gate = self.sigmoid(self.gate_conv(concat))
        return gate * enc_feat + (1 - gate) * dec_feat


class LightweightDecoder(BaseModule):
    """
    Lightweight decoder for GCNet Improved
    
    Features:
    - Progressive upsampling (4 stages)
    - Gated skip connections
    - Depthwise separable convolutions for efficiency
    
    Args:
        in_channels (int): Input channels from backbone
        channels (int): Base decoder channels
        num_stages (int): Number of decoder stages (default: 4)
        use_gated_fusion (bool): Whether to use gated fusion
        norm_cfg (dict): Normalization config
        act_cfg (dict): Activation config
        init_cfg (dict): Initialization config
    """
    
    def __init__(
        self,
        in_channels: int = 64,  # channels * 2 from backbone
        channels: int = 128,
        num_stages: int = 4,
        use_gated_fusion: bool = True,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
        init_cfg: OptConfigType = None
    ):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_stages = num_stages
        
        # Input projection
        self.input_proj = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Decoder stages
        self.stages = nn.ModuleList()
        
        # Stage 1: H/8 -> H/4 (channels -> channels//2)
        self.stages.append(
            DecoderStage(
                in_channels=channels,
                skip_channels=in_channels,  # Skip from backbone stage 3
                out_channels=channels // 2,
                use_gated_fusion=use_gated_fusion,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Stage 2: H/4 -> H/2 (channels//2 -> channels//4)
        self.stages.append(
            DecoderStage(
                in_channels=channels // 2,
                skip_channels=channels // 4,  # Skip from backbone stage 2
                out_channels=channels // 4,
                use_gated_fusion=use_gated_fusion,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Stage 3: H/2 -> H (channels//4 -> channels//8)
        self.stages.append(
            DecoderStage(
                in_channels=channels // 4,
                skip_channels=channels // 8,  # Skip from backbone stage 1
                out_channels=channels // 8,
                use_gated_fusion=use_gated_fusion,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        
        # Optional Stage 4: H -> H (no skip)
        if num_stages > 3:
            self.stages.append(
                DecoderStage(
                    in_channels=channels // 8,
                    skip_channels=0,
                    out_channels=channels // 8,
                    use_gated_fusion=False,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
    
    def forward(
        self, 
        x: Tensor, 
        skip_connections: Optional[List[Tensor]] = None
    ) -> Tensor:
        """
        Args:
            x: Input features (B, C, H/8, W/8)
            skip_connections: List of skip connections from encoder
                [stage1, stage2, stage3] at resolutions [H/4, H/2, H]
        
        Returns:
            Decoded features (B, C//8, H, W) or (B, C//8, H/2, W/2)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Progressive decoding
        for i, stage in enumerate(self.stages):
            skip = None
            if skip_connections is not None and i < len(skip_connections):
                skip = skip_connections[i]
            
            x = stage(x, skip)
        
        return x