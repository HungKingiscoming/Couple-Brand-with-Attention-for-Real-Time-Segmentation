import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import warnings
from typing import List, Tuple, Union, Dict


# Import components
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
from components.components import BATCH_NORM_TYPES, NORM_TYPES

# ============================================
# FLASH ATTENTION SETUP (FIXED)
# ============================================

FLASH_ATTN_AVAILABLE = False
flash_attn_func = None

try:
    # Thử import theo cách mới (v2.x)
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func as _flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    flash_attn_func = _flash_attn_func
    print("✓ FlashAttention v2 loaded successfully")
except ImportError:
    try:
        # Fallback: import từ interface
        from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
        FLASH_ATTN_AVAILABLE = True
        print("✓ FlashAttention loaded from interface")
    except ImportError:
        warnings.warn(
            "FlashAttention not available. Install with:\n"
            "pip install flash-attn --no-build-isolation\n"
            "Requirements: CUDA >= 12.3, H100/A100 GPU"
        )


def check_flash_attention_support():
    """Kiểm tra xem FlashAttention có thể chạy không"""
    if not FLASH_ATTN_AVAILABLE:
        return False
    
    if not torch.cuda.is_available():
        return False
    
    # Check GPU capability (cần compute capability >= 8.0 cho Ampere+)
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 8:
        warnings.warn(
            f"GPU compute capability {device_capability} < 8.0. "
            "FlashAttention requires Ampere (A100), Ada (RTX 4090), or Hopper (H100) GPUs."
        )
        return False
    
    return True


# ============================================
# ATTENTION MODULES (FIXED)
# ============================================

class FlashAttentionBlock(nn.Module):
    """FlashAttention block với xử lý đúng chuẩn"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # Pre-norm cho gradient stability
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Dropout cho attention
        self.attn_drop_prob = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Check FlashAttention support khi khởi tạo
        self.can_use_flash = check_flash_attention_support()
        if not self.can_use_flash:
            warnings.warn(
                f"FlashAttentionBlock initialized but FlashAttention not available. "
                f"Will use standard attention fallback."
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C) where N = H*W
        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # QKV projection: (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # ===== FLASHATTENTION (FIXED) =====
        if self.can_use_flash and x.is_cuda:
            try:
                # Đảm bảo dtype phù hợp (fp16 hoặc bf16)
                if qkv.dtype not in [torch.float16, torch.bfloat16]:
                    # FlashAttention chỉ support fp16/bf16
                    original_dtype = qkv.dtype
                    qkv = qkv.to(torch.float16)
                    use_fp16 = True
                else:
                    original_dtype = None
                    use_fp16 = False
                
                # FlashAttention expects: (batch, seqlen, 3, nheads, headdim)
                # Shape đã đúng rồi!
                out = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.attn_drop_prob if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=self.causal
                )
                # Output shape: (B, N, num_heads, head_dim)
                
                # Convert back nếu cần
                if use_fp16:
                    out = out.to(original_dtype)
                
                # Reshape: (B, N, num_heads, head_dim) -> (B, N, C)
                out = out.reshape(B, N, C)
                
            except Exception as e:
                warnings.warn(f"FlashAttention failed: {e}. Falling back to standard attention.")
                out = self._standard_attention(qkv, B, N, C)
        else:
            # Fallback to standard attention
            out = self._standard_attention(qkv, B, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Residual connection
        return x + out
    
    def _standard_attention(self, qkv: Tensor, B: int, N: int, C: int) -> Tensor:
        """Standard scaled dot-product attention fallback"""
        # qkv shape: (B, N, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)  # Each: (B, N, num_heads, head_dim)
        
        # Transpose to: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if self.causal:
            # Causal mask
            mask = torch.triu(
                torch.ones(N, N, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention: (B, num_heads, N, head_dim)
        out = attn @ v
        
        # Reshape: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        return out


class FlashAttentionStage(nn.Module):
    """Stacked FlashAttention blocks với FFN (như Transformer block)"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': FlashAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=drop
                ),
                'norm': nn.LayerNorm(dim, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(int(dim * mlp_ratio), dim),
                    nn.Dropout(drop)
                )
            })
            for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C) - flattened spatial features
        Returns:
            out: (B, N, C)
        """
        for layer in self.layers:
            # Attention (có residual bên trong)
            x = layer['attn'](x)
            
            # FFN với pre-norm và residual
            x = x + layer['mlp'](layer['norm'](x))
        
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation module"""
    
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
# GCBLOCK (Unchanged - đã đúng)
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
        deploy: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        
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
    
    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(x))
        
        id_out = 0 if self.path_residual is None else self.path_residual(x)
        
        return self.relu(
            self.path_3x3_1(x) + 
            self.path_3x3_2(x) + 
            self.path_1x1(x) + 
            id_out
        )
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, conv: nn.Module):
        if conv is None:
            return 0, 0
        
        if isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d, BATCH_NORM_TYPES)):
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
# GATED FUSION MODULE
# ============================================

class GatedFusion(nn.Module):
    """Gated fusion for adaptive feature selection"""
    
    def __init__(
        self,
        channels: int,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
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
        concat = torch.cat([enc_feat, dec_feat], dim=1)
        gate = self.sigmoid(self.gate_conv(concat))
        return gate * enc_feat + (1 - gate) * dec_feat


# ============================================
# IMPROVED GCNET BACKBONE (FIXED)
# ============================================

class GCNetImproved(BaseModule):
    """
    GCNet Improved Backbone với FlashAttention (FIXED)
    
    Improvements:
    - ✅ FlashAttention được sử dụng đúng cách với dtype checking
    - ✅ Automatic fallback nếu hardware không support
    - ✅ SE modules cho channel attention
    - ✅ Dual-branch architecture
    - ✅ Enhanced bottleneck với DAPPM
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        ppm_channels: int = 128,
        num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
        use_flash_attention: bool = True,
        flash_attn_stage: int = 4,
        flash_attn_layers: int = 2,
        flash_attn_heads: int = 8,
        flash_attn_drop: float = 0.0,
        use_se: bool = True,
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
        self.use_flash_attention = use_flash_attention and check_flash_attention_support()
        self.flash_attn_stage = flash_attn_stage
        self.use_se = use_se
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy
        
        if use_flash_attention and not self.use_flash_attention:
            warnings.warn(
                "FlashAttention requested but not available or GPU not supported. "
                "Using standard attention fallback."
            )
        
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
        
        for _ in range(num_blocks_per_stage[0]):
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
        
        # Stage 3 (Stem): Downsample + GCBlocks (H/8)
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
        
        for _ in range(num_blocks_per_stage[1] - 1):
            stage3_layers.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy
                )
            )
        
        self.stage3 = nn.Sequential(*stage3_layers)
        self.relu = build_activation_layer(act_cfg)
        
        # ======================================
        # SEMANTIC BRANCH
        # ======================================
        self.semantic_branch_layers = nn.ModuleList()
        
        # Stage 4 Semantic
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
        
        for _ in range(num_blocks_per_stage[2][0] - 1):
            stage4_sem.append(
                GCBlock(
                    in_channels=channels * 4,
                    out_channels=channels * 4,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy
                )
            )
        
        self.semantic_branch_layers.append(nn.Sequential(*stage4_sem))
        
        # Stage 4 FlashAttention
        if self.use_flash_attention and flash_attn_stage == 4:
            self.stage4_attention = FlashAttentionStage(
                dim=channels * 4,
                num_heads=flash_attn_heads,
                num_layers=flash_attn_layers,
                drop=flash_attn_drop,
                attn_drop=flash_attn_drop
            )
        else:
            self.stage4_attention = None
        
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
        
        for _ in range(num_blocks_per_stage[3][0] - 1):
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
        
        for _ in range(num_blocks_per_stage[4][0] - 1):
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
        for _ in range(num_blocks_per_stage[2][1]):
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
        for _ in range(num_blocks_per_stage[3][1]):
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
        for _ in range(num_blocks_per_stage[4][1]):
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
            in_channels=channels * 4,  # 128
            out_channels=channels * 2,  # 64
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        # ======================================
        # BOTTLENECK (FIXED)
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
        # Global FlashAttention in bottleneck
        if self.use_flash_attention:
            bottleneck_modules.append(
                FlashAttentionStage(
                    dim=channels * 4,
                    num_heads=flash_attn_heads,
                    num_layers=flash_attn_layers,
                    drop=flash_attn_drop,
                    attn_drop=flash_attn_drop
                )
            )
        
        # SE Module in bottleneck
        if self.use_se:
            bottleneck_modules.append(
                SEModule(channels * 4, reduction=se_reduction)
            )
        
        self.spp = nn.ModuleList(bottleneck_modules)
        
        # Initialize weights
        self.init_weights()
    
    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """(B, C, H, W) -> (B, H*W, C)"""
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)
    
    def _reshape_from_attention(self, x: Tensor, H: int, W: int) -> Tensor:
        """(B, H*W, C) -> (B, C, H, W)"""
        B, N, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        ✅ STABLE VERSION: Simplified bilateral fusion
        
        Changes:
        - Bilateral fusion 1: Bidirectional (semantic ↔ detail)
        - Bilateral fusion 2: One-way only (semantic → detail)
        
        This prevents shape mismatch while keeping performance
        """
        outputs = {}
        
        # Stage 1 (H/2)
        c1 = self.stage1_conv(x)
        outputs['c1'] = c1
        
        # Stage 2 (H/4)
        c2 = self.stage2(c1)
        outputs['c2'] = c2
        
        # Stage 3 (H/8) - Stem
        c3 = self.stage3(c2)
        outputs['c3'] = c3
        
        # ======================================
        # STAGE 4: Dual Branch Start
        # ======================================
        x_s = self.semantic_branch_layers[0](c3)  # Semantic: H/8 → H/16
        x_d = self.detail_branch_layers[0](c3)     # Detail: H/8 → H/8
        
        # FlashAttention at stage 4 (if enabled)
        if self.stage4_attention is not None:
            B, C, H, W = x_s.shape
            x_s_flat = self._reshape_for_attention(x_s)
            x_s_flat = self.stage4_attention(x_s_flat)
            x_s = self._reshape_from_attention(x_s_flat, H, W)
        
        # ✅ Bilateral Fusion 1 (Both directions)
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        x_s_relu = self.relu(x_s)
        x_d_relu = self.relu(x_d)
        # Semantic → Detail
        comp_c = self.compression_1(x_s_relu)  # 128 → 64
        x_d = x_d + resize(comp_c, size=out_size, mode='bilinear', 
                          align_corners=self.align_corners)
        
        # Detail → Semantic
        x_s = x_s + self.down_1(x_d_relu)  # Downsample H/8 → H/16
        
        # Save c4 for auxiliary head
        outputs['c4'] = x_s
        
        # ======================================
        # STAGE 5: Continue Processing
        # ======================================
        x_s = self.semantic_branch_layers[1](self.relu(x_s))  # H/16 → H/32
        x_d = self.detail_branch_layers[1](self.relu(x_d))     # H/8 → H/8
        
        # ✅ Bilateral Fusion 2 (One-way only: semantic → detail)
        # Reason: x_s is at H/32, too deep to receive info from H/8
        
        # Semantic → Detail (compress and upsample)
        comp_c = self.compression_2(self.relu(x_s))  # 128 → 64
        x_d = x_d + resize(comp_c, size=out_size, mode='bilinear',
                          align_corners=self.align_corners)
        
        # Semantic keeps its own path (no fusion from detail)
        # This avoids shape mismatch
        
        # ======================================
        # STAGE 6: Final Processing
        # ======================================
        x_d = self.detail_branch_layers[2](self.relu(x_d))  # H/8 → H/8
        x_s = self.semantic_branch_layers[2](self.relu(x_s))  # H/32 → H/32
        
        # ======================================
        # BOTTLENECK: Multi-scale + Attention
        # ======================================
        for module in self.spp:
            if isinstance(module, DAPPM):
                x_s = module(x_s)
            elif isinstance(module, FlashAttentionStage):
                B, C, H, W = x_s.shape
                x_s_flat = self._reshape_for_attention(x_s)
                x_s_flat = module(x_s_flat)
                x_s = self._reshape_from_attention(x_s_flat, H, W)
            elif isinstance(module, SEModule):
                x_s = module(x_s)
        
        # STEP 1: Resize spatial (H/32 → H/8)
        x_s = resize(x_s, size=out_size, mode='bilinear',
                    align_corners=self.align_corners)
        
        # STEP 2: Project channels (128 → 64)
        x_s = self.final_proj(x_s)
        
        # STEP 3: Final Fusion
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
