"""
Custom implementation of components without mmcv/mmseg dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

# Type aliases
OptConfigType = Optional[Dict]
SampleList = List[Dict]

# BatchNorm types for isinstance checks
BATCH_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)

# All normalization types
NORM_TYPES = BATCH_NORM_TYPES + (
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


# ============================================================================
# BASE MODULE
# ============================================================================
class BaseModule(nn.Module):
    """Base module with initialization support."""
    
    def __init__(self, init_cfg: OptConfigType = None):
        super().__init__()
        self.init_cfg = init_cfg
        self._is_init = False
    
    def init_weights(self):
        """Initialize weights."""
        if self._is_init:
            return
        
        if self.init_cfg is not None:
            self._initialize_weights(self.init_cfg)
        
        self._is_init = True
    
    def _initialize_weights(self, init_cfg: Dict):
        """Apply initialization configuration."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, NORM_TYPES):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ============================================================================
# NORMALIZATION LAYERS
# ============================================================================
def build_norm_layer(cfg: Union[Dict, str], num_features: int) -> Tuple[str, nn.Module]:
    """Build normalization layer.
    
    Args:
        cfg: Config dict or string name ('BN', 'GN', 'LN', 'IN')
        num_features: Number of features
    
    Returns:
        Tuple of (name, layer)
    """
    if isinstance(cfg, str):
        cfg = {'type': cfg}
    
    norm_type = cfg.get('type', 'BN')
    
    if norm_type in ['BN', 'BatchNorm2d']:
        layer = nn.BatchNorm2d(num_features, **cfg.get('kwargs', {}))
        name = 'bn'
    elif norm_type in ['GN', 'GroupNorm']:
        num_groups = cfg.get('num_groups', 32)
        layer = nn.GroupNorm(num_groups, num_features, **cfg.get('kwargs', {}))
        name = 'gn'
    elif norm_type in ['LN', 'LayerNorm']:
        layer = nn.LayerNorm(num_features, **cfg.get('kwargs', {}))
        name = 'ln'
    elif norm_type in ['IN', 'InstanceNorm2d']:
        layer = nn.InstanceNorm2d(num_features, **cfg.get('kwargs', {}))
        name = 'in'
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_features, **cfg.get('kwargs', {}))
        name = 'sync_bn'
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')
    
    return name, layer


# ============================================================================
# ACTIVATION LAYERS
# ============================================================================
def build_activation_layer(cfg: Union[Dict, str, None]) -> Optional[nn.Module]:
    """Build activation layer.
    
    Args:
        cfg: Config dict or string name ('ReLU', 'LeakyReLU', 'PReLU', 'GELU', etc.)
    
    Returns:
        Activation layer or None
    """
    if cfg is None:
        return None
    
    if isinstance(cfg, str):
        cfg = {'type': cfg}
    
    act_type = cfg.get('type', 'ReLU')
    
    if act_type == 'ReLU':
        return nn.ReLU(inplace=cfg.get('inplace', True))
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=cfg.get('negative_slope', 0.01), 
                           inplace=cfg.get('inplace', True))
    elif act_type == 'PReLU':
        return nn.PReLU(num_parameters=cfg.get('num_parameters', 1))
    elif act_type == 'ReLU6':
        return nn.ReLU6(inplace=cfg.get('inplace', True))
    elif act_type == 'ELU':
        return nn.ELU(alpha=cfg.get('alpha', 1.0), inplace=cfg.get('inplace', True))
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    elif act_type == 'Hardswish':
        return nn.Hardswish(inplace=cfg.get('inplace', True))
    elif act_type == 'SiLU' or act_type == 'Swish':
        return nn.SiLU(inplace=cfg.get('inplace', True))
    else:
        raise ValueError(f'Unsupported activation type: {act_type}')


# ============================================================================
# CONVOLUTION MODULE
# ============================================================================
class ConvModule(nn.Module):
    """Convolution module with normalization and activation.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        dilation: Dilation
        groups: Groups
        bias: Whether to use bias
        conv_cfg: Convolution config
        norm_cfg: Normalization config
        act_cfg: Activation config
        order: Order of conv/norm/act
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        order: Tuple[str, ...] = ('conv', 'norm', 'act'),
    ):
        super().__init__()
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        
        # Determine bias
        if bias == 'auto':
            bias = norm_cfg is None
        
        # Build convolution
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        
        # Convolution layer
        if conv_cfg is None or conv_cfg.get('type') == 'Conv2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
        else:
            raise ValueError(f"Unsupported conv type: {conv_cfg.get('type')}")
        
        # Normalization layer
        if self.with_norm:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.add_module(norm_name, norm_layer)
            self.norm_name = norm_name
        
        # Activation layer
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)
    
    @property
    def norm(self):
        """Get normalization layer."""
        if self.with_norm:
            return getattr(self, self.norm_name)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer_name in self.order:
            if layer_name == 'conv':
                x = self.conv(x)
            elif layer_name == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer_name == 'act' and self.with_activation:
                x = self.activate(x)
        return x


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def resize(
    input: torch.Tensor,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = 'bilinear',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    """Resize tensor.
    
    Args:
        input: Input tensor (B, C, H, W)
        size: Target size (H, W)
        scale_factor: Scale factor
        mode: Interpolation mode
        align_corners: Whether to align corners
    
    Returns:
        Resized tensor
    """
    if size is None and scale_factor is None:
        raise ValueError('Either size or scale_factor must be specified')
    
    # Set default align_corners for different modes
    if align_corners is None:
        align_corners = False if mode in ['bilinear', 'bicubic'] else None
    
    # Use F.interpolate
    if align_corners is not None:
        return F.interpolate(
            input, 
            size=size, 
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners
        )
    else:
        return F.interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode
        )


# ============================================================================
# DAPPM MODULE
# ============================================================================
class DAPPM(BaseModule):
    """DAPPM module (Dual Attention Pyramid Pooling Module).
    
    Args:
        in_channels: Input channels
        branch_channels: Channels for each branch
        out_channels: Output channels
        num_scales: Number of scales
        kernel_sizes: Kernel sizes for each scale
        strides: Strides for each scale
        paddings: Paddings for each scale
        norm_cfg: Normalization config
        act_cfg: Activation config
    """
    
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        num_scales: int = 5,
        kernel_sizes: Tuple[int, ...] = (5, 9, 17, 33),
        strides: Tuple[int, ...] = (2, 4, 8, 16),
        paddings: Tuple[int, ...] = (2, 4, 8, 16),
        norm_cfg: OptConfigType = dict(type='BN'),
        act_cfg: OptConfigType = dict(type='ReLU'),
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        
        self.num_scales = num_scales
        
        # Initial convolution
        self.conv_init = ConvModule(
            in_channels,
            branch_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Multi-scale branches
        self.scales = nn.ModuleList()
        for i in range(num_scales - 1):
            self.scales.append(
                nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i]
                    ),
                    ConvModule(
                        branch_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                )
            )
        
        # Compression convolution
        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Shortcut
        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get input size
        input_size = x.shape[2:]
        
        # Initial convolution
        out = self.conv_init(x)
        
        # Multi-scale features
        multi_scale_features = [out]
        for scale in self.scales:
            scaled_out = scale(out)
            # Upsample to original size
            scaled_out = resize(scaled_out, size=input_size, mode='bilinear', align_corners=False)
            multi_scale_features.append(scaled_out)
        
        # Concatenate multi-scale features
        out = torch.cat(multi_scale_features, dim=1)
        
        # Compression
        out = self.compression(out)
        
        # Add shortcut
        shortcut = self.shortcut(x)
        out = out + shortcut
        
        return out


# ============================================================================
# BASE DECODE HEAD
# ============================================================================
class BaseDecodeHead(BaseModule):
    """Base class for decode heads.
    
    Args:
        in_channels: Input channels
        channels: Hidden channels
        num_classes: Number of classes
        dropout_ratio: Dropout ratio
        norm_cfg: Normalization config
        act_cfg: Activation config
        align_corners: Whether to align corners in resize
        loss_decode: Loss config (not implemented in this minimal version)
    """
    
    def __init__(
        self,
        in_channels: Union[int, List[int]],
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
        norm_cfg: OptConfigType = dict(type='BN'),
        act_cfg: OptConfigType = dict(type='ReLU'),
        align_corners: bool = False,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        
        # Dropout
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        
        # Final classifier
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
    
    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Forward function."""
        raise NotImplementedError
    
    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Test ConvModule
    conv = ConvModule(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        padding=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU')
    )
    x = torch.randn(2, 64, 32, 32)
    out = conv(x)
    print(f"ConvModule output shape: {out.shape}")
    
    # Test DAPPM
    dappm = DAPPM(
        in_channels=128,
        branch_channels=64,
        out_channels=128
    )
    x = torch.randn(2, 128, 32, 32)
    out = dappm(x)
    print(f"DAPPM output shape: {out.shape}")
    
    # Test resize
    x = torch.randn(2, 64, 32, 32)
    out = resize(x, size=(64, 64), mode='bilinear', align_corners=False)
    print(f"Resize output shape: {out.shape}")