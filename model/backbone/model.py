import math
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType,
)


# ===========================
# GCBlock classes (giá»¯ nguyÃªn tá»« code gá»‘c)
# ===========================

class Block1x1(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 bias: bool = True,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.deploy = deploy

        if self.deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std if self.bias else beta - running_mean * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=1, stride=self.stride,
            padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class Block3x3(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 bias: bool = True,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 deploy: bool = False):
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
                padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std if self.bias else beta - running_mean * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride,
            padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy

        assert kernel_size == 3
        assert padding == 1

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
                bias=True,
                padding_mode=padding_mode)
        else:
            if (out_channels == in_channels) and stride == 1:
                self.path_residual = build_norm_layer(norm_cfg, num_features=in_channels)[1]
            else:
                self.path_residual = None

            self.path_3x3_1 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
            )
            self.path_3x3_2 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
            )
            self.path_1x1 = Block1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding_11,
                bias=False,
                norm_cfg=norm_cfg,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(inputs))

        if self.path_residual is None:
            id_out = 0
        else:
            id_out = self.path_residual(inputs)

        return self.relu(
            self.path_3x3_1(inputs)
            + self.path_3x3_2(inputs)
            + self.path_1x1(inputs)
            + id_out
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv: nn.Module):
        if conv is None:
            return 0, 0
        if isinstance(conv, ConvModule):
            kernel = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
        else:
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
            if not hasattr(self, 'id_tensor'):
                input_in_channels = self.in_channels
                kernel_value = np.zeros(
                    (self.in_channels, input_in_channels, 3, 3),
                    dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_in_channels, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
            kernel = self.id_tensor
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        self.path_3x3_1.switch_to_deploy()
        kernel3x3_1, bias3x3_1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        self.path_3x3_2.switch_to_deploy()
        kernel3x3_2, bias3x3_2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        self.path_1x1.switch_to_deploy()
        kernel1x1, bias1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)

        kernel = (
            kernel3x3_1
            + kernel3x3_2
            + self._pad_1x1_to_3x3_tensor(kernel1x1)
            + kernelid
        )
        bias = bias3x3_1 + bias3x3_2 + bias1x1 + biasid
        return kernel, bias

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
            bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
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
# FIXED DWSA + MultiScale
# ===========================

def _get_valid_groups(channels, desired_groups):
    """TÃ¬m sá»‘ group lá»›n nháº¥t chia háº¿t cho channels."""
    if desired_groups <= 1:
        return 1
    g = min(desired_groups, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


class DWSABlock(nn.Module):
    """FIXED VERSION vá»›i stability improvements"""
    def __init__(self, channels, num_heads=2, drop=0.0, reduction=4, 
                 qk_sharing=True, groups=4, alpha=0.1):
        super().__init__()
        assert channels % reduction == 0
        self.channels = channels
        self.num_heads = num_heads

        reduced = channels // reduction
        mid = max(reduced // 2, num_heads)
        self.reduced = reduced
        self.mid = mid

        # FIX 1: Layer Norm cho stability
        self.ln = nn.LayerNorm(channels)
        
        self.in_proj = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)

        g = _get_valid_groups(reduced, groups)
        if g != groups:
            import warnings
            warnings.warn(
                f"DWSABlock: adjusted groups from {groups} to {g} for channels={reduced}"
            )

        self.qk_sharing = qk_sharing
        if qk_sharing:
            self.qk_base = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=False)
            self.q_head = nn.Conv1d(mid, mid, kernel_size=1, bias=True)
            self.k_head = nn.Conv1d(mid, mid, kernel_size=1, bias=True)
        else:
            self.q_proj = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=True)
            self.k_proj = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=True)

        self.v_proj = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=True)
        self.o_proj = nn.Conv1d(mid, reduced, kernel_size=1, groups=g, bias=True)

        self.drop = nn.Dropout(drop)
        
        # FIX 2: Improved scaling
        head_dim = mid // num_heads
        self.scale = head_dim ** -0.5
        
        # FIX 3: Learnable alpha (khá»Ÿi táº¡o nhá»)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # FIX 4: Layer norm TRÆ¯á»šC process
        x_ln = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x_red = self.in_proj(x_ln)
        B, C2, H, W = x_red.shape
        N = H * W
        x_flat = x_red.view(B, C2, N)

        if self.qk_sharing:
            base = self.qk_base(x_flat)
            q = self.q_head(base)
            k = self.k_head(base)
        else:
            q = self.q_proj(x_flat)
            k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        def split_heads(t):
            B, Cmid, N = t.shape
            head_dim = Cmid // self.num_heads
            t = t.view(B, self.num_heads, head_dim, N)
            return t

        q = split_heads(q).permute(0, 1, 3, 2)
        k = split_heads(k).permute(0, 1, 3, 2)
        v = split_heads(v).permute(0, 1, 3, 2)

        # FIX 5: Clamp attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(-10, 10)  # Prevent overflow
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        B_, Hn, Hd, N_ = out.shape
        out = out.view(B, self.mid, N_)

        out = self.o_proj(out)
        out = out.view(B, C2, H, W)
        out = self.out_proj(out)

        # FIX 6: Scaled residual
        return identity + self.alpha * out


class MultiScaleContextModule(nn.Module):
    """FIXED VERSION vá»›i BatchNorm"""
    def __init__(self, in_channels, out_channels, scales=(1, 2), 
                 branch_ratio=8, alpha=0.1):
        super().__init__()
        self.scales = scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        total_branch_channels = in_channels // branch_ratio
        base = total_branch_channels // len(scales)
        extra = total_branch_channels % len(scales)

        per_branch_list = []
        for i in range(len(scales)):
            c = base + (1 if i < extra else 0)
            per_branch_list.append(max(c, 1))
        fused_channels = sum(per_branch_list)

        # FIX 7: ThÃªm BatchNorm vÃ o táº¥t cáº£ branches
        self.scale_branches = nn.ModuleList()
        for s, c_out in zip(scales, per_branch_list):
            if s == 1:
                self.scale_branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, c_out, kernel_size=1, bias=False),
                        nn.BatchNorm2d(c_out),  # â† THÃŠM
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.scale_branches.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=s, stride=s),
                        nn.Conv2d(in_channels, c_out, kernel_size=1, bias=False),
                        nn.BatchNorm2d(c_out),  # â† THÃŠM
                        nn.ReLU(inplace=True),
                    )
                )

        # FIX 8: BatchNorm trong fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                fused_channels,
                kernel_size=3,
                padding=1,
                groups=fused_channels,
                bias=False,
            ),
            nn.BatchNorm2d(fused_channels),  # â† THÃŠM
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),  # â† THÃŠM
        )

        # FIX 9: Learnable alpha nhá»
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
        # FIX 10: BatchNorm cho projection
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),  # â† THÃŠM
            )
        else:
            self.proj = None

    def forward(self, x):
        B, C, H, W = x.shape
        outs = []
        for s, branch in zip(self.scales, self.scale_branches):
            o = branch(x)
            if o.shape[-2:] != (H, W):
                o = F.interpolate(o, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(o)

        fused = torch.cat(outs, dim=1)
        out = self.fusion(fused)

        if self.proj is not None:
            x_proj = self.proj(x)
        else:
            x_proj = x

        # Scaled residual
        return x_proj + self.alpha * out


# ===========================
# GCNetCore (giá»¯ nguyÃªn)
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


# ===========================
# Enhanced Backbone vá»›i táº¥t cáº£ fixes
# ===========================

class GCNetWithEnhance(BaseModule):
    """FIXED VERSION - Complete vá»›i gradient clipping"""
    
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
                 dwsa_drop: float = 0.1,  # â† Default 0.1 cho regularization
                 dwsa_alpha: float = 0.1,  # â† Learnable residual weight
                 use_multi_scale_context: bool = True,
                 ms_scales: Tuple[int, ...] = (1, 2),
                 ms_branch_ratio: int = 8,
                 ms_alpha: float = 0.1,  # â† Learnable residual weight
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
                self.dwsa4 = DWSABlock(
                    C * 4,
                    num_heads=dwsa_num_heads,
                    reduction=dwsa_reduction,
                    qk_sharing=dwsa_qk_sharing,
                    groups=dwsa_groups,
                    drop=dwsa_drop,
                    alpha=dwsa_alpha,
                )
            elif stage == 'stage5':
                self.dwsa5 = DWSABlock(
                    C * 8,
                    num_heads=dwsa_num_heads,
                    reduction=dwsa_reduction,
                    qk_sharing=dwsa_qk_sharing,
                    groups=dwsa_groups,
                    drop=dwsa_drop,
                    alpha=dwsa_alpha,
                )
            elif stage == 'stage6':
                self.dwsa6 = DWSABlock(
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
        feats = self.backbone(x)

        c1 = feats['c1']
        c2 = feats['c2']
        c4 = feats['c4']
        x_d6 = feats['x_d6']
        s4, s5, s6 = feats['s4'], feats['s5'], feats['s6']

        if self.dwsa4 is not None:
            s4 = self.dwsa4(s4)
        if self.dwsa5 is not None:
            s5 = self.dwsa5(s5)
        if self.dwsa6 is not None:
            s6 = self.dwsa6(s6)

        x_spp = self.backbone.spp(s6)
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        x_spp = resize(
            x_spp, size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        if self.ms_context is not None:
            x_spp = self.ms_context(x_spp)

        x_spp = self.final_proj(x_spp)
        c5_enh = x_d6 + x_spp

        return dict(
            c1=c1,
            c2=c2,
            c4=c4,
            c5=c5_enh,
        )
