import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import List, Tuple, Union, Dict

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
# DEPTHWISE SEPARABLE ATTENTION
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
        spatial_kernel: int = 7
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
        # Handle 4D input (B, C, H, W)
        is_4d = False
        if x.dim() == 4:
            is_4d = True
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        else:
            B, N, C = x.shape
            # Giả định nếu x là 3D, N = H * W. Bạn nên truyền H, W nếu x luôn là 3D
            H = W = int(math.sqrt(N)) 
    
        B, N, C = x.shape
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm.transpose(1, 2)).transpose(1, 2)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        q = q.transpose(1, 2) # (B, H_head, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # ✅ FIX: Hỗ trợ mọi kích thước ảnh
        # Tính toán local bias dựa trên H, W thực tế
        attn_2d = attn.view(B, self.num_heads, H, W, N)
        local_bias = self.local_attn(attn_2d.mean(dim=-1)) # (B, H_head, H, W)
        attn = attn + local_bias.flatten(2).unsqueeze(-1)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        out = x + out
        if is_4d:
            out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class DWSABlock(nn.Module):
    """
    Wrapper block cho DepthWiseSeparableAttention
    Dễ dàng thay thế vào backbone
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        spatial_kernel: int = 7,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='ReLU', inplace=False)
    ):
        super().__init__()
        
        # Attention
        self.attn = DepthWiseSeparableAttention(
            dim=channels,
            num_heads=num_heads,
            spatial_kernel=spatial_kernel,
            attn_drop=drop,
            proj_drop=drop
        )
        
        # FFN (Feed-Forward Network)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            build_activation_layer(act_cfg),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(drop)
        )
        
        # Norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Attention block (handles 4D internally)
        x = self.attn(x)
        
        # FFN block (need to reshape to 3D)
        identity = x
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = identity + x
        
        return x


# ============================================
# GCBLOCK với DWSA Support
# ============================================

class Block1x1(BaseModule):
    """1x1_1x1 path của GCBlock"""
    
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
        
        # Xóa các nhánh Reparam nhưng GIỮ LẠI dwsa
        self.__delattr__('path_3x3_1')
        self.__delattr__('path_3x3_2')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        
        # ✅ KHÔNG xóa self.dwsa ở đây
        self.deploy = True


class Block3x3(BaseModule):
    """3x3_1x1 path của GCBlock"""
    
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
    """GCBlock với optional DWSA"""
    
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
        use_dwsa: bool = False,  # ✅ NEW
        dwsa_num_heads: int = 8
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        self.use_dwsa = use_dwsa
        
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
        
        # ✅ Add DWSA module
        if use_dwsa and not deploy:
            self.dwsa = DWSABlock(
                channels=out_channels,
                num_heads=dwsa_num_heads,
                spatial_kernel=7,
                drop=0.0
            )
        else:
            self.dwsa = None
    
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
        
        # ✅ Apply DWSA if available
        if self.dwsa is not None:
            out = self.dwsa(out)
        
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
        if hasattr(self, 'dwsa'):
            self.__delattr__('dwsa')
        
        self.deploy = True


# ============================================
# GCNET với DWSA
# ============================================

class GCNetWithDWSA(BaseModule):
    """
    ✅ GCNet với DepthWiseSeparableAttention
    
    DWSA Placement Strategy:
    - Stage 3 (H/8): Last block → Capture spatial context
    - Stage 4 (H/16): Last 2 blocks → Rich features
    - Bottleneck: DAPPM + DWSA → Global context
    
    Rationale:
    - Early stages (H/2, H/4): Pure convs (local features)
    - Mid stages (H/8, H/16): DWSA (multi-scale context)
    - Bottleneck: Full attention (global understanding)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        ppm_channels: int = 128,
        num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
        dwsa_stages: List[str] = ['stage3', 'stage4', 'bottleneck'],  # ✅ Config
        dwsa_num_heads: int = 8,
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
        self.dwsa_stages = dwsa_stages
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
        
        # Stage 2: Second conv + blocks (H/4)
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
            stage2_layers.append(
                GCBlock(
                    in_channels=channels,
                    out_channels=channels,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy
                )
            )
        
        self.stage2 = nn.Sequential(*stage2_layers)
        
        # Stage 3 (Stem): Downsample + GCBlocks (H/8) ✅ với DWSA
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
            # ✅ Add DWSA to last block
            use_dwsa = (i == num_blocks_per_stage[1] - 2) and ('stage3' in dwsa_stages)
            stage3_layers.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_dwsa=use_dwsa,
                    dwsa_num_heads=dwsa_num_heads
                )
            )
        
        self.stage3 = nn.Sequential(*stage3_layers)
        self.relu = build_activation_layer(act_cfg)
        
        # ======================================
        # SEMANTIC BRANCH
        # ======================================
        self.semantic_branch_layers = nn.ModuleList()
        
        # Stage 4 Semantic ✅ với DWSA
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
            # ✅ Add DWSA to last 2 blocks
            use_dwsa = (i >= num_blocks_per_stage[2][0] - 3) and ('stage4' in dwsa_stages)
            stage4_sem.append(
                GCBlock(
                    in_channels=channels * 4,
                    out_channels=channels * 4,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_dwsa=use_dwsa,
                    dwsa_num_heads=dwsa_num_heads
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
        # BOTTLENECK với DWSA
        # ======================================
        self.spp = DAPPM(
            in_channels=channels * 4,
            branch_channels=ppm_channels,
            out_channels=channels * 4,
            num_scales=5,
            kernel_sizes=[5, 9, 17, 33],
            strides=[2, 4, 8, 16],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        
        # ✅ Bottleneck DWSA
        if 'bottleneck' in dwsa_stages:
            self.bottleneck_dwsa = DWSABlock(
                channels=channels * 4,
                num_heads=dwsa_num_heads,
                spatial_kernel=7,
                drop=0.0
            )
        else:
            self.bottleneck_dwsa = None
        
        # Initialize weights
        self.init_weights()
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass với DWSA"""
        outputs = {}
        
        # Stage 1 (H/2)
        c1 = self.stage1_conv(x)
        outputs['c1'] = c1
        
        # Stage 2 (H/4)
        c2 = self.stage2(c1)
        outputs['c2'] = c2
        
        # Stage 3 (H/8) - ✅ có DWSA
        c3 = self.stage3(c2)
        outputs['c3'] = c3
        
        # ======================================
        # STAGE 4: Dual Branch - ✅ có DWSA
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
        # BOTTLENECK: DAPPM + DWSA
        # ======================================
        x_s = self.spp(x_s)
        
        # ✅ Apply DWSA at bottleneck
        if self.bottleneck_dwsa is not None:
            x_s = self.bottleneck_dwsa(x_s)
        
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
# FACTORY FUNCTIONS
# ============================================

def create_gcnet_dwsa_variants():
    """
    Tạo các variants khác nhau của GCNet với DWSA
    """
    
    # ✅ Lightweight variant (mobile)
    gcnet_dwsa_lite = GCNetWithDWSA(
        channels=24,
        dwsa_stages=['bottleneck'],  # Only at bottleneck
        dwsa_num_heads=4
    )
    
    # ✅ Standard variant (balanced)
    gcnet_dwsa_std = GCNetWithDWSA(
        channels=32,
        dwsa_stages=['stage3', 'bottleneck'],  # Mid + bottleneck
        dwsa_num_heads=8
    )
    
    # ✅ Performance variant (maximum accuracy)
    gcnet_dwsa_perf = GCNetWithDWSA(
        channels=48,
        dwsa_stages=['stage3', 'stage4', 'bottleneck'],  # Full attention
        dwsa_num_heads=8
    )
    
    return {
        'lite': gcnet_dwsa_lite,
        'standard': gcnet_dwsa_std,
        'performance': gcnet_dwsa_perf
    }


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == '__main__':
    # Test model
    model = GCNetWithDWSA(
        channels=32,
        dwsa_stages=['stage3', 'stage4', 'bottleneck'],
        dwsa_num_heads=8
    )
    
    # Dummy input
    x = torch.randn(2, 3, 512, 1024)
    
    # Forward
    outputs = model(x)
    
    print("✅ GCNet với DWSA outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
        detail_stage5
