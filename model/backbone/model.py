import math
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from components.components import (
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType,
)

def build_norm_layer(norm_cfg, num_features):
    """
    Build normalization layer from config
    
    Args:
        norm_cfg: dict with 'type' key ('BN', 'GN', 'SyncBN')
        num_features: number of channels
    
    Returns:
        tuple: (name, norm_layer)
    """
    if norm_cfg is None:
        return None, nn.Identity()
    
    norm_type = norm_cfg.get('type', 'BN')
    requires_grad = norm_cfg.get('requires_grad', True)
    
    if norm_type == 'BN':
        norm_layer = nn.BatchNorm2d(num_features)
    elif norm_type == 'SyncBN':
        norm_layer = nn.SyncBatchNorm(num_features)
    elif norm_type == 'GN':
        num_groups = norm_cfg.get('num_groups', 32)
        # Ensure num_groups divides num_features
        while num_features % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        norm_layer = nn.GroupNorm(num_groups, num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    
    # Set requires_grad
    for param in norm_layer.parameters():
        param.requires_grad = requires_grad
    
    return norm_type, norm_layer

def build_activation_layer(act_cfg):
    """
    Build activation layer from config
    
    Args:
        act_cfg: dict with 'type' key ('ReLU', 'LeakyReLU', etc.)
    
    Returns:
        nn.Module: activation layer
    """
    if act_cfg is None:
        return nn.Identity()
    
    act_type = act_cfg.get('type', 'ReLU')
    inplace = act_cfg.get('inplace', True)
    
    if act_type == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'LeakyReLU':
        negative_slope = act_cfg.get('negative_slope', 0.01)
        return nn.LeakyReLU(negative_slope, inplace=inplace)
    elif act_type == 'PReLU':
        return nn.PReLU()
    elif act_type == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class ConvModule(nn.Module):
    """
    Conv-Norm-Act module
    Compatible với norm_cfg và act_cfg
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        
        # Conv
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Norm
        if norm_cfg is not None:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
            self.add_module(norm_name.lower(), norm_layer)  # 'bn' or 'gn'
            self.norm_name = norm_name.lower()
        else:
            self.norm_name = None
        
        # Activation
        if act_cfg is not None:
            self.activate = build_activation_layer(act_cfg)
        else:
            self.activate = None
    
    def forward(self, x):
        x = self.conv(x)
        
        if self.norm_name is not None:
            norm = getattr(self, self.norm_name)
            x = norm(x)
        
        if self.activate is not None:
            x = self.activate(x)
        
        return x
# ===========================
# FIXED GCBlock classes - Support GroupNorm
# ===========================

class Block1x1(nn.Module):
    """1x1 conv block with double 1x1 structure"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=0,
                 bias=True,
                 norm_cfg=None,
                 deploy=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        
        if deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=padding, bias=True
            )
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
    
    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    def _fuse_bn_tensor(self, conv):
        """Fuse conv + bn"""
        kernel = conv.conv.weight
        bias = conv.conv.bias
        
        if hasattr(conv, 'bn'):
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            
            if bias is not None:
                return kernel * t, beta + (bias - running_mean) * gamma / std
            else:
                return kernel * t, beta - running_mean * gamma / std
        else:
            if bias is None:
                bias = torch.zeros(kernel.shape[0], device=kernel.device)
            return kernel, bias
    
    def switch_to_deploy(self):
        """Convert to deploy mode"""
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=1, stride=self.stride,
            padding=self.padding, bias=True
        )
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1
        )
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class Block3x3(nn.Module):
    """3x3 conv block with 3x3 → 1x1 structure"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=1,
                 bias=True,
                 norm_cfg=None,
                 deploy=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        
        if deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride,
                padding=padding, bias=True
            )
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
    
    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    def _fuse_bn_tensor(self, conv):
        """Fuse conv + bn"""
        kernel = conv.conv.weight
        bias = conv.conv.bias
        
        if hasattr(conv, 'bn'):
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            
            if bias is not None:
                return kernel * t, beta + (bias - running_mean) * gamma / std
            else:
                return kernel * t, beta - running_mean * gamma / std
        else:
            if bias is None:
                bias = torch.zeros(kernel.shape[0], device=kernel.device)
            return kernel, bias
    
    def switch_to_deploy(self):
        """Convert to deploy mode"""
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride,
            padding=self.padding, bias=True
        )
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1
        )
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCBlock(nn.Module):
    """
    ✅ FIXED VERSION - Accepts norm_cfg and act_cfg parameters
    
    GCBlock with reparameterization support (optional deploy mode)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: dict = None,
                 act_cfg: dict = None,
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        
        # Default configs if None
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        # Activation
        if act:
            self.relu = build_activation_layer(act_cfg)
        else:
            self.relu = nn.Identity()
        
        if deploy:
            # Deployed mode: single fused conv
            self.reparam_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                padding_mode=padding_mode
            )
        else:
            # Training mode: multi-path
            
            # Residual path (identity)
            if (out_channels == in_channels) and stride == 1:
                _, self.path_residual = build_norm_layer(norm_cfg, in_channels)
            else:
                self.path_residual = None
            
            # Path 1: 3x3 → 1x1
            self.path_3x3_1 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
            )
            
            # Path 2: 3x3 → 1x1
            self.path_3x3_2 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
            )
            
            # Path 3: 1x1 → 1x1
            self.path_1x1 = Block1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding_11,
                bias=False,
                norm_cfg=norm_cfg,
            )
    
    def forward(self, x):
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(x))
        
        # Multi-path forward
        if self.path_residual is None:
            id_out = 0
        else:
            id_out = self.path_residual(x)
        
        return self.relu(
            self.path_3x3_1(x) + 
            self.path_3x3_2(x) + 
            self.path_1x1(x) + 
            id_out
        )
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, conv):
        """Fuse conv + bn into single conv"""
        if conv is None:
            return 0, 0
        
        if isinstance(conv, ConvModule):
            kernel = conv.conv.weight
            
            # Check if has BatchNorm
            if hasattr(conv, 'bn'):
                running_mean = conv.bn.running_mean
                running_var = conv.bn.running_var
                gamma = conv.bn.weight
                beta = conv.bn.bias
                eps = conv.bn.eps
            elif hasattr(conv, 'gn'):
                # GroupNorm cannot be fused easily, return as-is
                bias = conv.conv.bias
                if bias is None:
                    bias = torch.zeros(kernel.shape[0], device=kernel.device)
                return kernel, bias
            else:
                # No norm layer
                bias = conv.conv.bias
                if bias is None:
                    bias = torch.zeros(kernel.shape[0], device=kernel.device)
                return kernel, bias
            
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        
        else:
            # Identity branch (just norm layer)
            if isinstance(conv, nn.GroupNorm):
                # Create identity kernel for GroupNorm
                if not hasattr(self, 'id_tensor'):
                    kernel_value = np.zeros(
                        (self.in_channels, self.in_channels, 3, 3),
                        dtype=np.float32
                    )
                    for i in range(self.in_channels):
                        kernel_value[i, i, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
                return self.id_tensor, torch.zeros(self.in_channels, device=conv.weight.device)
            
            # BatchNorm identity
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
            
            if not hasattr(self, 'id_tensor'):
                kernel_value = np.zeros(
                    (self.in_channels, self.in_channels, 3, 3),
                    dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
            
            kernel = self.id_tensor
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
    
    def get_equivalent_kernel_bias(self):
        """Get equivalent kernel and bias for deployment"""
        self.path_3x3_1.switch_to_deploy()
        kernel3x3_1 = self.path_3x3_1.conv.weight.data
        bias3x3_1 = self.path_3x3_1.conv.bias.data
        
        self.path_3x3_2.switch_to_deploy()
        kernel3x3_2 = self.path_3x3_2.conv.weight.data
        bias3x3_2 = self.path_3x3_2.conv.bias.data
        
        self.path_1x1.switch_to_deploy()
        kernel1x1 = self.path_1x1.conv.weight.data
        bias1x1 = self.path_1x1.conv.bias.data
        
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)
        
        kernel = (
            kernel3x3_1 + 
            kernel3x3_2 + 
            self._pad_1x1_to_3x3_tensor(kernel1x1) + 
            kernelid
        )
        bias = bias3x3_1 + bias3x3_2 + bias1x1 + biasid
        
        return kernel, bias
    
    def switch_to_deploy(self):
        """Convert to deploy mode (optional)"""
        if hasattr(self, 'reparam_3x3'):
            return
        
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True
        )
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        
        # Delete training components
        for p in self.parameters():
            p.detach_()
        if hasattr(self, 'path_3x3_1'):
            self.__delattr__('path_3x3_1')
        if hasattr(self, 'path_3x3_2'):
            self.__delattr__('path_3x3_2')
        if hasattr(self, 'path_1x1'):
            self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        
        self.deploy = True


# ===========================
# DWSA + MultiScale (giữ nguyên)
# ===========================

def _get_valid_groups(channels, desired_groups):
    """Tìm số group lớn nhất chia hết cho channels."""
    if desired_groups <= 1:
        return 1
    g = min(desired_groups, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


class EfficientAttention(nn.Module):
    def __init__(self, channels, num_heads=2, reduction=8, alpha=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        reduced = channels // reduction
        self.reduced = reduced
        
        # Single shared projection cho Q,K,V
        self.qkv_proj = nn.Conv2d(channels, reduced * 3, 1, bias=False)
        self.bn = nn.BatchNorm2d(reduced * 3)
        self.out_proj = nn.Conv2d(reduced, channels, 1, bias=False)
        
        self.scale = (reduced // num_heads) ** -0.5
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        identity = x
        
        # Single projection for Q,K,V
        qkv = self.bn(self.qkv_proj(x))
        qkv = qkv.reshape(B, 3, self.reduced, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Multi-head attention
        q = q.view(B, self.num_heads, -1, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, -1, N).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, -1, N).permute(0, 1, 3, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.clamp(-10, 10), dim=-1)
        
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, self.reduced, H, W)
        out = self.out_proj(out)
        
        return identity + self.alpha * out


class MultiScaleContextModule(nn.Module):
    def __init__(self, channels, scales=(1, 2), alpha=0.1):
        super().__init__()
        self.scales = scales
        
        # Giảm branch channels
        branch_ch = channels // (len(scales) * 2)
        self.branches = nn.ModuleList()
        
        for s in scales:
            if s == 1:
                self.branches.append(nn.Conv2d(channels, branch_ch, 1, bias=False))
            else:
                self.branches.append(nn.Sequential(
                    nn.AvgPool2d(s, s),
                    nn.Conv2d(channels, branch_ch, 1, bias=False)
                ))
        
        # Depthwise separable fusion
        fused_ch = branch_ch * len(scales)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, 3, padding=1, groups=fused_ch, bias=False),
            nn.BatchNorm2d(fused_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_ch, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x):
        B, C, H, W = x.shape
        outs = []
        
        for s, branch in zip(self.scales, self.branches):
            o = branch(x)
            if o.shape[-2:] != (H, W):
                o = F.interpolate(o, (H, W), mode='bilinear', align_corners=False)
            outs.append(o)
        
        fused = self.fusion(torch.cat(outs, dim=1))
        return x + self.alpha * fused


# ===========================
# GCNetCore + GCNetWithEnhance (giữ nguyên - copy từ document 4)
# ===========================

class GCNetCore(BaseModule):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List[int] = [4, 4, [5, 4], [5, 4], [2, 2]],
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.channels = channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy

        self.stem = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            *[GCBlock(
                in_channels=channels,
                out_channels=channels,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy
            ) for _ in range(num_blocks_per_stage[0])],
            GCBlock(
                in_channels=channels,
                out_channels=channels * 2,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy),
            *[GCBlock(
                in_channels=channels * 2,
                out_channels=channels * 2,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy
            ) for _ in range(num_blocks_per_stage[1] - 1)],
        )
        self.relu = build_activation_layer(act_cfg)

        self.semantic_branch_layers = nn.ModuleList()
        self.semantic_branch_layers.append(
            nn.Sequential(
                GCBlock(channels * 2, channels * 4, stride=2,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(channels * 4, channels * 4, stride=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[2][0] - 2)],
                GCBlock(channels * 4, channels * 4, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            )
        )
        self.semantic_branch_layers.append(
            nn.Sequential(
                GCBlock(channels * 4, channels * 8, stride=2,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(channels * 8, channels * 8, stride=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[3][0] - 2)],
                GCBlock(channels * 8, channels * 8, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            )
        )
        self.semantic_branch_layers.append(
            nn.Sequential(
                GCBlock(channels * 8, channels * 16, stride=2,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(channels * 16, channels * 16, stride=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[4][0] - 2)],
                GCBlock(channels * 16, channels * 16, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            )
        )

        self.detail_branch_layers = nn.ModuleList()
        self.detail_branch_layers.append(
            nn.Sequential(
                *[GCBlock(channels * 2, channels * 2, stride=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[2][1] - 1)],
                GCBlock(channels * 2, channels * 2, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            )
        )
        self.detail_branch_layers.append(
            nn.Sequential(
                *[GCBlock(channels * 2, channels * 2, stride=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[3][1] - 1)],
                GCBlock(channels * 2, channels * 2, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            )
        )
        self.detail_branch_layers.append(
            nn.Sequential(
                GCBlock(channels * 2, channels * 4, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(channels * 4, channels * 4, stride=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[4][1] - 2)],
                GCBlock(channels * 4, channels * 4, stride=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            )
        )

        self.compression_1 = ConvModule(
            channels * 4, channels * 2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2, channels * 4, kernel_size=3,
            stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8, channels * 2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2, channels * 4,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                channels * 4, channels * 8,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=None))

        self.spp = DAPPM(
            in_channels=channels * 16,
            branch_channels=ppm_channels,
            out_channels=channels * 4,
            num_scales=5,
            kernel_sizes=[5, 9, 17, 33],
            strides=[2, 4, 8, 16],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        c1 = None
        c2 = None
        feat = x
        for i, layer in enumerate(self.stem):
            feat = layer(feat)
            if i == 0:
                c1 = feat
            if i == 1:
                c2 = feat
        x = feat

        x_s4 = self.semantic_branch_layers[0](x)
        x_d4 = self.detail_branch_layers[0](x)
        comp_c4 = self.compression_1(self.relu(x_s4))
        x_s4 = x_s4 + self.down_1(self.relu(x_d4))
        x_d4 = x_d4 + resize(
            comp_c4, size=out_size,
            mode='bilinear', align_corners=self.align_corners)

        x_s5 = self.semantic_branch_layers[1](self.relu(x_s4))
        x_d5 = self.detail_branch_layers[1](self.relu(x_d4))
        comp_c5 = self.compression_2(self.relu(x_s5))
        x_s5 = x_s5 + self.down_2(self.relu(x_d5))
        x_d5 = x_d5 + resize(
            comp_c5, size=out_size,
            mode='bilinear', align_corners=self.align_corners)

        x_d6 = self.detail_branch_layers[2](self.relu(x_d5))
        x_s6 = self.semantic_branch_layers[2](self.relu(x_s5))

        x_spp = self.spp(x_s6)
        x_spp = resize(
            x_spp, size=out_size,
            mode='bilinear', align_corners=self.align_corners)

        out = x_d6 + x_spp

        return dict(
            c1=c1,
            c2=c2,
            c4=x_d4,
            s4=x_s4,
            s5=x_s5,
            s6=x_s6,
            x_d6=x_d6,
            spp=x_spp,
            out=out,
        )


class GCNetWithEnhance(BaseModule):
    """FIXED VERSION - Complete với GroupNorm support"""
    
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
                 dwsa_stages: List[str] = ('stage5', 'stage6'),
                 dwsa_num_heads: int = 4,
                 dwsa_reduction: int = 4,
                 dwsa_qk_sharing: bool = True,
                 dwsa_groups: int = 4,
                 dwsa_drop: float = 0.1,
                 dwsa_alpha: float = 0.1,
                 use_multi_scale_context: bool = True,
                 ms_scales: Tuple[int, ...] = (1, 2),
                 ms_branch_ratio: int = 8,
                 ms_alpha: float = 0.1,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        self.align_corners = align_corners
        self.channels = channels

        valid_stages = {'stage4', 'stage5', 'stage6'}
        invalid = set(dwsa_stages) - valid_stages
        if invalid:
            raise ValueError(f"Invalid dwsa_stages: {invalid}. Valid: {valid_stages}")

        self.backbone = GCNetCore(
            in_channels=in_channels,
            channels=channels,
            ppm_channels=ppm_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            align_corners=align_corners,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=None,
            deploy=deploy,
        )

        C = channels
        self.dwsa4 = None
        self.dwsa5 = None
        self.dwsa6 = None

        for stage in dwsa_stages:
            if stage == 'stage4':
                self.dwsa4 = EfficientAttention(
                    C * 4,
                    num_heads=dwsa_num_heads,
                    reduction=dwsa_reduction,
                    qk_sharing=dwsa_qk_sharing,
                    groups=dwsa_groups,
                    drop=dwsa_drop,
                    alpha=dwsa_alpha,
                )
            elif stage == 'stage5':
                self.dwsa5 = EfficientAttention(
                    C * 8,
                    num_heads=dwsa_num_heads,
                    reduction=dwsa_reduction,
                    qk_sharing=dwsa_qk_sharing,
                    groups=dwsa_groups,
                    drop=dwsa_drop,
                    alpha=dwsa_alpha,
                )
            elif stage == 'stage6':
                self.dwsa6 = EfficientAttention(
                    C * 16,
                    num_heads=dwsa_num_heads,
                    reduction=dwsa_reduction,
                    qk_sharing=dwsa_qk_sharing,
                    groups=dwsa_groups,
                    drop=dwsa_drop,
                    alpha=dwsa_alpha,
                )

        if use_multi_scale_context:
            self.ms_context = MultiScaleContextModule(
                C * 4, C * 4,
                scales=ms_scales,
                branch_ratio=ms_branch_ratio,
                alpha=ms_alpha,
            )
        else:
            self.ms_context = None

        self.final_proj = ConvModule(
            in_channels=C * 4,
            out_channels=C * 4,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        
        gcnet_feats = self.backbone(x)
        
        c1 = gcnet_feats['c1']
        c2 = gcnet_feats['c2']
        c4 = gcnet_feats['c4']
        s4 = gcnet_feats['s4']
        s5 = gcnet_feats['s5']
        s6 = gcnet_feats['s6']
        x_d6 = gcnet_feats['x_d6']
        x_spp = gcnet_feats['spp']
        
        if self.dwsa4 is not None:
            s4 = self.dwsa4(s4)
        if self.dwsa5 is not None:
            s5 = self.dwsa5(s5)
        if self.dwsa6 is not None:
            s6 = self.dwsa6(s6)
        
        if self.ms_context is not None:
            x_spp = self.ms_context(x_spp)
        
        final_feat = self.final_proj(x_spp + x_d6)
        
        return {
            'c1': c1,
            'c2': c2,
            'c4': c4,
            'c5': final_feat,
            's4': s4,
            's5': s5,
            's6': s6,
            'x_d6': x_d6,
        }
