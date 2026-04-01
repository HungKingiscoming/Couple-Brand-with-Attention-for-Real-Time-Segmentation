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
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# ============================================================================
# DAPPM MODULE
# ============================================================================
class DAPPM(BaseModule):
    """DAPPM module in `DDRNet <https://arxiv.org/abs/2101.06085>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


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


class PAPPM(DAPPM):
    """PAPPM module in `PIDNet <https://arxiv.org/abs/2206.02066>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.1).
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **self.conv_cfg)

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)


class RPPM(DAPPM):
    """RPPM.

        Args:
            in_channels (int): Input channels.
            branch_channels (int): Branch channels.
            out_channels (int): Output channels.
            num_scales (int): Number of scales.
            kernel_sizes (list[int]): Kernel sizes of each scale.
            strides (list[int]): Strides of each scale.
            paddings (list[int]): Paddings of each scale.
            norm_cfg (dict): Config dict for normalization layer.
                Default: dict(type='BN', momentum=0.1).
            act_cfg (dict): Config dict for activation layer in ConvModule.
                Default: dict(type='ReLU', inplace=True).
            conv_cfg (dict): Config dict for convolution layer in ConvModule.
                Default: dict(order=('norm', 'act', 'conv'), bias=False).
            upsample_mode (str): Upsample mode. Default: 'bilinear'.
            deploy (bool): Whether in deploy mode. Default: False.
        """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', requires_grad=True),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear',
                 deploy: bool = False):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.deploy = deploy

        self.processes = RepParallel(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            deploy=self.deploy,
        )

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepParallel):
                m.switch_to_deploy()


class RepParallel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 deploy: bool = False):
        super().__init__()

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.relu = build_activation_layer(act_cfg)

        if deploy:
            self.reparam_parallel = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            self.parallel1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.parallel2 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:

        inputs = self.relu(self.norm(inputs))

        if hasattr(self, 'reparam_parallel'):
            return self.reparam_parallel(inputs)

        return self.parallel1(inputs) + self.parallel2(inputs)

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel1, bias1 = self._fuse_bn_tensor(self.parallel1)
        kernel2, bias2 = self._fuse_bn_tensor(self.parallel2)

        return kernel1 + kernel2, bias1 + bias2

    def _fuse_bn_tensor(self, conv: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific conv layer.

        Args:
            conv (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel = conv.conv.weight
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'reparam_parallel'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_parallel = nn.Conv2d(
            in_channels=self.parallel1.conv.in_channels,
            out_channels=self.parallel1.conv.out_channels,
            kernel_size=self.parallel1.conv.kernel_size,
            stride=self.parallel1.conv.stride,
            padding=self.parallel1.conv.padding,
            groups=self.parallel1.conv.groups,
            bias=True)
        self.reparam_parallel.weight.data = kernel
        self.reparam_parallel.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('parallel1')
        self.__delattr__('parallel2')
        self.deploy = True
class BasicBlock(BaseModule):
    """Basic block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at the
            last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.downsample = downsample
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block from `ResNet <https://arxiv.org/abs/1512.03385>`_.

    Args:
        in_channels (int): Input channels.
        channels (int): Output channels.
        stride (int): Stride of the first block. Default: 1.
        downsample (nn.Module, optional): Downsample operation on identity.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU', inplace=True).
        act_cfg_out (dict, optional): Config dict for activation layer at
            the last of the block. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    expansion = 2

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act_cfg_out: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            channels,
            channels,
            3,
            stride,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            channels,
            channels * self.expansion,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        if act_cfg_out:
            self.act = MODELS.build(act_cfg_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual

        if hasattr(self, 'act'):
            out = self.act(out)

        return out
