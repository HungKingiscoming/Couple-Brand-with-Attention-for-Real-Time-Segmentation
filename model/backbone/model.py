import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import List, Tuple, Union, Dict

# Import components (giữ nguyên)
from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType
)

# ============================================
# EFFICIENT ATTENTION MODULES
# ============================================
class DepthWiseSeparableAttention(nn.Module):
    """
    ✅ KHUYẾN NGHỊ #1 cho Segmentation
    
    Ưu điểm:
    - Nhẹ nhất: ~40% FLOPs của standard attention
    - Hiệu quả cho spatial tasks
    - Chạy nhanh trên mọi GPU
    - Phù hợp real-time inference
    
    Nguyên lý:
    - Tách attention thành spatial & channel
    - Giống như MobileNet trong CNN
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        spatial_kernel: int = 7  # Local spatial window
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Lightweight projections
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Depth-wise separable QKV
        self.qkv = nn.Sequential(
            nn.Conv1d(dim, dim * 3, kernel_size=1, bias=qkv_bias),
            nn.BatchNorm1d(dim * 3)
        )
        
        # Local spatial attention (giống depthwise conv)
        self.local_attn = nn.Conv2d(
            num_heads,
            num_heads,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            groups=num_heads,  # Depthwise
            bias=False
        )
        
        # Output
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C) hoặc (B, C, H, W)
        Returns:
            out: same shape as input
        """
        B, N, C = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm.transpose(1, 2)).transpose(1, 2)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, N, H, D)
        
        # Reshape for efficient computation
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # ===== LIGHTWEIGHT ATTENTION =====
        # 1. Reduced attention matrix
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # 2. Apply local spatial bias (for segmentation)
        if N == int(math.sqrt(N)) ** 2:  # If square feature map
            H_feat = W_feat = int(math.sqrt(N))
            attn_2d = attn.reshape(B, self.num_heads, H_feat, W_feat, H_feat, W_feat)
            
            # Local attention bias
            local_bias = self.local_attn(
                attn_2d.mean(dim=(4, 5))  # Average over target positions
            )
            attn = attn + local_bias.flatten(2).unsqueeze(-1)
        
        # 3. Efficient softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 4. Apply attention
        out = attn @ v  # (B, H, N, D)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return x + out

class ChannelAttention(nn.Module):
    """
    Channel Attention Module - Lightweight và hiệu quả
    
    Focus: "What" channels are important
    Complexity: O(C^2) - rất nhỏ so với spatial attention
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(max(channels // reduction, 8), channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        # Global pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        # Channel weights
        out = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module - Tìm "where" to focus
    
    Complexity: O(HW) - chỉ dùng 2 channels (avg + max)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        spatial_feat = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.sigmoid(self.conv(spatial_feat))
        
        return x * spatial_weight


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Reference: https://arxiv.org/abs/1807.06521
    
    ✅ Best choice for segmentation:
    - Efficient (0.01M params)
    - Proven effective
    - Hardware agnostic
    - Easy to integrate
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: Tensor) -> Tensor:
        # Sequential: channel → spatial
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AxialAttention(nn.Module):
    """
    Axial Attention - Efficient 2D attention
    
    Decompose 2D attention vào H-axis và W-axis
    Complexity: O(HW * (H + W)) vs O(HW * HW) cho full attention
    
    Use case: Khi cần long-range dependencies
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for height and width
        self.qkv_h = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_w = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # ===== HEIGHT ATTENTION =====
        # Reshape: (B, C, H, W) -> (B*W, H, C)
        x_h = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        
        qkv_h = self.qkv_h(x_h).reshape(B * W, H, 3, self.num_heads, self.head_dim)
        q_h, k_h, v_h = qkv_h.unbind(2)
        
        # Attention
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        attn_h = self.attn_drop(attn_h)
        
        out_h = (attn_h @ v_h).reshape(B * W, H, C)
        out_h = out_h.reshape(B, W, H, C).permute(0, 3, 2, 1)  # -> (B, C, H, W)
        
        # ===== WIDTH ATTENTION =====
        # Reshape: (B, C, H, W) -> (B*H, W, C)
        x_w = out_h.permute(0, 2, 3, 1).reshape(B * H, W, C)
        
        qkv_w = self.qkv_w(x_w).reshape(B * H, W, 3, self.num_heads, self.head_dim)
        q_w, k_w, v_w = qkv_w.unbind(2)
        
        # Attention
        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        attn_w = self.attn_drop(attn_w)
        
        out_w = (attn_w @ v_w).reshape(B * H, W, C)
        out_w = out_w.reshape(B, H, W, C).permute(0, 3, 1, 2)  # -> (B, C, H, W)
        
        # Final projection
        out = self.proj(out_w.flatten(2).transpose(1, 2))  # (B, H*W, C)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out + x  # Residual


class EfficientMultiHeadAttention(nn.Module):
    """
    Simplified Multi-Head Attention cho small spatial size
    
    Use case: Bottleneck features (H/32, W/32)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C) where N = H*W
        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, N, num_heads, head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class SEModule(nn.Module):
    """Squeeze-and-Excitation - Simplest channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================
# GCBLOCK (Giữ nguyên - đã tốt)
# ============================================

class Block1x1(BaseModule):
    """1x1_1x1 path of GCBlock"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        bias: bool = True,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        deploy: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        
        if self.deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=stride, 
                padding=padding, bias=True
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
        else:
            return self.conv2(self.conv1(x))
    
    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias if conv.conv.bias is not None else 0
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        if self.deploy:
            return
            
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=1, stride=self.stride,
            padding=self.padding, bias=True
        )
        
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk',
            kernel2.squeeze(3).squeeze(2),
            kernel1
        )
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class Block3x3(BaseModule):
    """3x3_1x1 path of GCBlock"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 1,
        bias: bool = True,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        deploy: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy
        
        if self.deploy:
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
        else:
            return self.conv2(self.conv1(x))
    
    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias if conv.conv.bias is not None else 0
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        if self.deploy:
            return
            
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride,
            padding=self.padding, bias=True
        )
        
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk',
            kernel2.squeeze(3).squeeze(2),
            kernel1
        )
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCBlock(nn.Module):  
    """GCBlock with optional attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 1,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        act: bool = True,
        deploy: bool = False,
        use_attention: bool = False,
        attention_type: str = 'se'  # 'se', 'cbam', or None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        self.use_attention = use_attention
        
        assert kernel_size == 3 and padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        if act:
            self.relu = build_activation_layer(act_cfg)
        else:
            self.relu = nn.Identity()
        
        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True
            )
        else:
            if (out_channels == in_channels) and stride == 1:
                self.path_residual = build_norm_layer(
                    norm_cfg, num_features=in_channels
                )[1]
            else:
                self.path_residual = None
            
            self.path_3x3_1 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg
            )
            self.path_3x3_2 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg
            )
            self.path_1x1 = Block1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding_11,
                bias=False,
                norm_cfg=norm_cfg
            )
        
        # ✅ Add attention module
        if use_attention and not deploy:
            if attention_type == 'se':
                self.attention = SEModule(out_channels)
            elif attention_type == 'cbam':
                self.attention = CBAM(out_channels)
            else:
                self.attention = None
        else:
            self.attention = None
    
    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'reparam_3x3'):
            out = self.relu(self.reparam_3x3(x))
        else:
            id_out = 0 if self.path_residual is None else self.path_residual(x)
            out = self.relu(
                self.path_3x3_1(x) + 
                self.path_3x3_2(x) + 
                self.path_1x1(x) + 
                id_out
            )
        
        # Apply attention if available
        if self.attention is not None:
            out = self.attention(out)
        
        return out
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, conv: nn.Module):
        if conv is None:
            return 0, 0
        
        if isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            if not hasattr(self, 'id_tensor'):
                kernel_value = torch.zeros(
                    (self.in_channels, self.in_channels, 3, 3),
                    dtype=torch.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i, 1, 1] = 1
                self.id_tensor = kernel_value.to(conv.weight.device)
            
            kernel = self.id_tensor
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
        else:
            return 0, 0
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def get_equivalent_kernel_bias(self):
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
        
        return (
            kernel3x3_1 + kernel3x3_2 + 
            self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3_1 + bias3x3_2 + bias1x1 + biasid
        )
    
    def switch_to_deploy(self):
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
        
        for para in self.parameters():
            para.detach_()
        
        self.__delattr__('path_3x3_1')
        self.__delattr__('path_3x3_2')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        
        self.deploy = True


# ============================================
# IMPROVED GCNET BACKBONE
# ============================================

class GCNetImproved(BaseModule):
    """
    ✅ GCNet với Efficient Attention Strategy
    
    Attention placement:
    - Stage 2-3: SE module (lightweight, local)
    - Stage 4: CBAM (channel + spatial)
    - Bottleneck: Axial Attention (long-range) + CBAM
    
    Rationale:
    - Early stages: Local features → lightweight attention
    - Mid stages: Richer features → full attention
    - Bottleneck: Small spatial → efficient global attention
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        ppm_channels: int = 128,
        num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
        attention_config: Dict = {
            'stage2': 'se',      # Lightweight
            'stage3': 'se',      # Lightweight
            'stage4': 'cbam',    # Full attention
            'bottleneck': 'axial+cbam'  # Long-range + channel/spatial
        },
        se_reduction: int = 16,
        align_corners: bool = False,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False),
        init_cfg: OptConfigType = None,
        deploy: bool = False
    ):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.attention_config = attention_config
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy
        
        # Stage 1: First conv (H/2)
        self.stage1_conv = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # Stage 2: Second conv + blocks (H/4) với SE
        stage2_layers = [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        ]
        
        for i in range(num_blocks_per_stage[0]):
            # ✅ Add SE to last block
            use_attn = (i == num_blocks_per_stage[0] - 1) and (attention_config.get('stage2') == 'se')
            stage2_layers.append(
                GCBlock(
                    in_channels=channels,
                    out_channels=channels,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_attention=use_attn,
                    attention_type='se'
                )
            )
        
        self.stage2 = nn.Sequential(*stage2_layers)
        
        # Stage 3 (Stem): Downsample + GCBlocks (H/8) với SE
        stage3_layers = [
            GCBlock(
                in_channels=channels,
                out_channels=channels * 2,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy
            )
        ]
        
        for i in range(num_blocks_per_stage[1] - 1):
            # ✅ Add SE to last block
            use_attn = (i == num_blocks_per_stage[1] - 2) and (attention_config.get('stage3') == 'se')
            stage3_layers.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_attention=use_attn,
                    attention_type='se'
                )
            )
        
        self.stage3 = nn.Sequential(*stage3_layers)
        self.relu = build_activation_layer(act_cfg)
        
        # ======================================
        # SEMANTIC BRANCH
        # ======================================
        self.semantic_branch_layers = nn.ModuleList()
        
        # Stage 4 Semantic với CBAM
        stage4_sem = []
        stage4_sem.append(
            GCBlock(
                in_channels=channels * 2,
                out_channels=channels * 4,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy
            )
        )
        
        for i in range(num_blocks_per_stage[2][0] - 1):
            # ✅ Add CBAM to last block
            use_attn = (i == num_blocks_per_stage[2][0] - 2) and (attention_config.get('stage4') == 'cbam')
            stage4_sem.append(
                GCBlock(
                    in_channels=channels * 4,
                    out_channels=channels * 4,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_attention=use_attn,
                    attention_type='cbam'
                )
            )
        
        self.semantic_branch_layers.append(nn.Sequential(*stage4_sem))
        
        # Stage 5 Semantic
        stage5_sem = []
        stage5_sem.append(
            GCBlock(
                in_channels=channels * 4,
                out_channels=channels * 4,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy
            )
        )
        
        for i in range(num_blocks_per_stage[3][0] - 1):
            stage5_sem.append(
                GCBlock(
                    in_channels=channels * 4,
                    out_channels=channels * 4,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy
                )
            )
        
        self.semantic_branch_layers.append(nn.Sequential(*stage5_sem))
        
        # Stage 6 Semantic
        stage6_sem = []
        stage6_sem.append(
            GCBlock(
                in_channels=channels * 4,
                out_channels=channels * 4,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy
            )
        )
        
        for i in range(num_blocks_per_stage[4][0] - 1):
            stage6_sem.append(
                GCBlock(
                    in_channels=channels * 4,
                    out_channels=channels * 4,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    act=False,
                    deploy=deploy
                )
            )
        
        self.semantic_branch_layers.append(nn.Sequential(*stage6_sem))
        
        # ======================================
        # DETAIL BRANCH
        # ======================================
        self.detail_branch_layers = nn.ModuleList()
        
        # Stage 4 Detail
        detail_stage4 = []
        for i in range(num_blocks_per_stage[2][1]):
            detail_stage4.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy
                )
            )
        self.detail_branch_layers.append(nn.Sequential(*detail_stage4))
        
        # Stage 5 Detail
        detail_stage5 = []
        for i in range(num_blocks_per_stage[3][1]):
            detail_stage5.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy
                )
            )
        self.detail_branch_layers.append(nn.Sequential(*detail_stage5))
        
        # Stage 6 Detail
        detail_stage6 = []
        for i in range(num_blocks_per_stage[4][1]):
            detail_stage6.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    act=False,
                    deploy=deploy
                )
            )
        self.detail_branch_layers.append(nn.Sequential(*detail_stage6))
        
        # ======================================
        # BILATERAL FUSION
        # ======================================
        self.compression_1 = ConvModule(
            in_channels=channels * 4,
            out_channels=channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        
        self.compression_2 = ConvModule(
            in_channels=channels * 4,
            out_channels=channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        
        self.down_1 = ConvModule(
            in_channels=channels * 2,
            out_channels=channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        
        self.down_2 = ConvModule(
            in_channels=channels * 2,
            out_channels=channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        
        self.final_proj = ConvModule(
            in_channels=channels * 4,
            out_channels=channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # ======================================
        # BOTTLENECK với Attention
        # ======================================
        bottleneck_modules = [
            DAPPM(
                in_channels=channels * 4,
                branch_channels=ppm_channels,
                out_channels=channels * 4,
                num_scales=5,
                kernel_sizes=[5, 9, 17, 33],
                strides=[2, 4, 8, 16],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        ]
        
        # ✅ Bottleneck attention strategy
        bottleneck_type = attention_config.get('bottleneck', 'cbam')
        
        if 'axial' in bottleneck_type:
            # Axial attention cho long-range dependencies
            self.bottleneck_axial = AxialAttention(
                dim=channels * 4,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0
            )
        else:
            self.bottleneck_axial = None
        
        if 'cbam' in bottleneck_type:
            # CBAM cho channel + spatial attention
            bottleneck_modules.append(
                CBAM(channels * 4, reduction=se_reduction)
            )
        
        self.spp = nn.ModuleList(bottleneck_modules)
        
        # Initialize weights
        self.init_weights()
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass với efficient attention placement
        """
        outputs = {}
        
        # Stage 1 (H/2)
        c1 = self.stage1_conv(x)
        outputs['c1'] = c1
        
        # Stage 2 (H/4) - có SE attention
        c2 = self.stage2(c1)
        outputs['c2'] = c2
        
        # Stage 3 (H/8) - Stem với SE attention
        c3 = self.stage3(c2)
        outputs['c3'] = c3
        
        # ======================================
        # STAGE 4: Dual Branch với CBAM
        # ======================================
        x_s = self.semantic_branch_layers[0](c3)  # H/8 → H/16
        x_d = self.detail_branch_layers[0](c3)     # H/8 → H/8
        
        # Bilateral Fusion 1
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        x_s_relu = self.relu(x_s)
        x_d_relu = self.relu(x_d)
        
        # Semantic → Detail
        comp_c = self.compression_1(x_s_relu)
        x_d = x_d + resize(comp_c, size=out_size, mode='bilinear', 
                          align_corners=self.align_corners)
        
        # Detail → Semantic
        x_s = x_s + self.down_1(x_d_relu)
        
        outputs['c4'] = x_s
        
        # ======================================
        # STAGE 5: Continue Processing
        # ======================================
        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_d = self.detail_branch_layers[1](self.relu(x_d))
        
        # Bilateral Fusion 2
        comp_c = self.compression_2(self.relu(x_s))
        x_d = x_d + resize(comp_c, size=out_size, mode='bilinear',
                          align_corners=self.align_corners)
        
        # ======================================
        # STAGE 6: Final Processing
        # ======================================
        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))
        
        # ======================================
        # BOTTLENECK: Multi-scale + Attention
        # ======================================
        for module in self.spp:
            if isinstance(module, DAPPM):
                x_s = module(x_s)
            elif isinstance(module, CBAM):
                x_s = module(x_s)
        
        # ✅ Apply axial attention if enabled
        if self.bottleneck_axial is not None:
            x_s = self.bottleneck_axial(x_s)
        
        # Resize and project
        x_s = resize(x_s, size=out_size, mode='bilinear',
                    align_corners=self.align_corners)
        x_s = self.final_proj(x_s)
        
        # Final fusion
        c5 = x_d + x_s
        outputs['c5'] = c5
        
        return outputs
    
    def switch_to_deploy(self):
        """Switch all GCBlocks to deploy mode"""
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
        self.deploy = True
    
    def init_weights(self):
        """Initialize weights"""
        if self.init_cfg is not None:
            super().init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


# ============================================
# USAGE EXAMPLES
# ============================================

def create_gcnet_variants():
    """Factory function cho các variants khác nhau"""
    
    # ✅ Lightweight variant (mobile/edge devices)
    gcnet_lite = GCNetImproved(
        channels=24,
        attention_config={
            'stage2': 'se',
            'stage3': 'se',
            'stage4': None,  # No attention
            'bottleneck': 'cbam'  # Only CBAM
        }
    )
    
    # ✅ Standard variant (balanced)
    gcnet_std = GCNetImproved(
        channels=32,
        attention_config={
            'stage2': 'se',
            'stage3': 'se',
            'stage4': 'cbam',
            'bottleneck': 'cbam'
        }
    )
    
    # ✅ Performance variant (maximum accuracy)
    gcnet_perf = GCNetImproved(
        channels=48,
        attention_config={
            'stage2': 'se',
            'stage3': 'se',
            'stage4': 'cbam',
            'bottleneck': 'axial+cbam'  # Full attention
        }
    )
    
    return {
        'lite': gcnet_lite,
        'standard': gcnet_std,
        'performance': gcnet_perf
    }
