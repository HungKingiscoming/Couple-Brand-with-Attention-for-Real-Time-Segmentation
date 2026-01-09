import math
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from torchvision.ops import DeformConv2d   # <-- BỎ DÒNG NÀY

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
# GCBlock gốc (giữ nguyên)
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
            # BN only
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
# DWSA + MultiScale
# ===========================

class DWSABlock(nn.Module):
    def __init__(self, channels, num_heads=2, drop=0.0, reduction=4, qk_sharing=True, groups=4):
        super().__init__()
        assert channels % reduction == 0
        self.channels = channels
        self.num_heads = num_heads

        reduced = channels // reduction      # C'
        mid = max(reduced // 2, num_heads)   # bottleneck trong attention
        self.reduced = reduced
        self.mid = mid

        # C -> C'
        self.in_proj = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        # C' -> C
        self.out_proj = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)

        # Thay Linear bằng 1x1 conv trên C' với group
        g = min(groups, reduced) if groups > 1 else 1
        if reduced % g != 0:  # đảm bảo chia hết
            g = 1

        # base proj cho QK sharing
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
        self.scale = (mid // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        # Giảm chiều C -> C'
        x_red = self.in_proj(x)              # B, C', H, W
        B, C2, H, W = x_red.shape
        N = H * W

        # B, C', HW
        x_flat = x_red.view(B, C2, N)

        # Q, K, V (Conv1d + group, có thể chia sẻ)
        if self.qk_sharing:
            base = self.qk_base(x_flat)          # B, mid, N
            q = self.q_head(base)
            k = self.k_head(base)
        else:
            q = self.q_proj(x_flat)
            k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)                  # B, mid, N

        # B, heads, N, dim
        def reshape_heads(t):
            B, Cmid, N = t.shape
            head_dim = Cmid // self.num_heads
            t = t.view(B, self.num_heads, head_dim, N)
            return t.permute(0, 1, 3, 2)        # B, heads, N, head_dim

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # B, heads, N, N
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)             # B, heads, N, head_dim
        B_, Hn, N_, Hd = out.shape
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.mid, N)  # B, mid, N

        # O proj
        out = self.o_proj(out)                 # B, C', N
        out = out.view(B, C2, H, W)
        out = self.out_proj(out)               # B, C, H, W

        return out + x



class MultiScaleContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(1, 2), branch_ratio=8):
        super().__init__()
        self.scales = scales

        total_branch_channels = in_channels // branch_ratio
        per_branch = max(total_branch_channels // len(scales), 1)

        self.scale_branches = nn.ModuleList()
        for s in scales:
            if s == 1:
                self.scale_branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, per_branch, kernel_size=1, bias=False),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.scale_branches.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=s, stride=s),
                        nn.Conv2d(in_channels, per_branch, kernel_size=1, bias=False),
                        nn.ReLU(inplace=True),
                    )
                )

        fused_channels = per_branch * len(scales)
        # fusion depthwise + pointwise
        self.fusion = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                fused_channels,
                kernel_size=3,
                padding=1,
                groups=fused_channels,    # depthwise
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
        )

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
        return out + x




# ===========================
# GCNetCore (giữ nguyên)
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

        # stage1-3
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

        # semantic branch
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

        # detail branch
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

        # fusion
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

        # DAPPM
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
    
        # ===== STEM (stage1-3) =====
        c1 = None
        c2 = None
        feat = x
        for i, layer in enumerate(self.stem):
            feat = layer(feat)
            # stem = [conv(stride2), conv(stride2), GCBlockx4, GCBlock(stride2), GCBlockx3]
            if i == 0:
                # sau conv1, H/2
                c1 = feat
            if i == 1:
                # sau conv2, H/4
                c2 = feat
        x = feat  # output stem, H/8, channels*2
    
        # ===== stage4 =====
        x_s4 = self.semantic_branch_layers[0](x)
        x_d4 = self.detail_branch_layers[0](x)
        comp_c4 = self.compression_1(self.relu(x_s4))
        x_s4 = x_s4 + self.down_1(self.relu(x_d4))
        x_d4 = x_d4 + resize(
            comp_c4, size=out_size,
            mode='bilinear', align_corners=self.align_corners)
    
        # ===== stage5 =====
        x_s5 = self.semantic_branch_layers[1](self.relu(x_s4))
        x_d5 = self.detail_branch_layers[1](self.relu(x_d4))
        comp_c5 = self.compression_2(self.relu(x_s5))
        x_s5 = x_s5 + self.down_2(self.relu(x_d5))
        x_d5 = x_d5 + resize(
            comp_c5, size=out_size,
            mode='bilinear', align_corners=self.align_corners)
    
        # ===== stage6 =====
        x_d6 = self.detail_branch_layers[2](self.relu(x_d5))
        x_s6 = self.semantic_branch_layers[2](self.relu(x_s5))
    
        # ===== SPP =====
        x_spp = self.spp(x_s6)
        x_spp = resize(
            x_spp, size=out_size,
            mode='bilinear', align_corners=self.align_corners)
    
        out = x_d6 + x_spp
    
        return dict(
            c1=c1,        # H/2, 32 ch
            c2=c2,        # H/4, 64 ch
            c4=x_d4,      # H/8, 64 ch (detail)
            s4=x_s4,
            s5=x_s5,
            s6=x_s6,
            x_d6=x_d6,    # H/8, 128 ch
            spp=x_spp,    # H/8, 128 ch
            out=out,      # H/8, 128 ch (GCNet gốc)
        )


# ===========================
# Backbone có DWSA/MultiScale (không DCN)
# ===========================

class GCNetWithEnhance(BaseModule):
    """
    GCNet backbone + DWSA/MultiScale (không DCN):

    - Backbone: GCNetCore (giống GCNet-s gốc, load 90%+ weight).
    - DWSA: chỉ trên semantic deep (stage5: s5, stage6: s6).
    - MultiScaleContext: sau SPP(s6), trước khi fuse với detail.
    - Output cho head:
        c1: (B, C,   H/2,  W/2)
        c2: (B, 2C,  H/4,  W/4)
        c4: (B, 2C,  H/8,  W/8)  # detail H/8
        c5: (B, 4C,  H/8,  W/8)  # enhanced deep feature
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
                 dwsa_stages: List[str] = ('stage5', 'stage6'),
                 dwsa_num_heads: int = 4,
                 use_multi_scale_context: bool = True,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        self.align_corners = align_corners
        self.channels = channels

        # ===== GCNet core giữ nguyên để load weight GCNet-s =====
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

        C = channels  # C=32 cho GCNet-s

        # ===== DWSA: chỉ stage5, stage6 =====
        self.dwsa4 = None  # không dùng DWSA ở stage4
        self.dwsa5 = None
        self.dwsa6 = DWSABlock(C * 16, num_heads=2, reduction=4, qk_sharing=True, groups=4) 

        # ===== KHÔNG CÓ DCN =====
        # self.dcn5 = None
        # self.dcn6 = None

        # ===== MultiScaleContext: sau SPP(s6) =====
        # SPP output: 4C (128 kênh nếu C=32)
        self.ms_context = MultiScaleContextLite(C * 4, C * 4, scales=(1, 2), branch_ratio=8)


        # Projection cuối cho feature deep
        self.final_proj = ConvModule(
            in_channels=C * 4,
            out_channels=C * 4,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Returns:
            {
                'c1': (B, C,   H/2,  W/2),
                'c2': (B, 2C,  H/4,  W/4),
                'c4': (B, 2C,  H/8,  W/8),
                'c5': (B, 4C,  H/8,  W/8),  # enhanced
                'c5_gcnet': (B, 4C, H/8, W/8)  # optional, GCNet original out
            }
        """
        feats = self.backbone(x)

        # lấy feature chuẩn cho head
        c1 = feats['c1']      # H/2,  C
        c2 = feats['c2']      # H/4,  2C
        c4 = feats['c4']      # H/8,  2C (detail)
        x_d6 = feats['x_d6']  # H/8,  4C (detail stage6)
        s4, s5, s6 = feats['s4'], feats['s5'], feats['s6']

        # ===== DWSA chỉ ở s5, s6 =====
        if self.dwsa5 is not None:
            s5 = self.dwsa5(s5)
        if self.dwsa6 is not None:
            s6 = self.dwsa6(s6)

        # ===== KHÔNG DCN TRÊN s6 =====
        # (s6 giữ nguyên sau DWSA)

        # ===== SPP + MultiScale trên semantic deep =====
        # Dùng lại DAPPM của GCNetCore, nhưng input là s6 đã enhance
        x_spp = self.backbone.spp(s6)  # (B, 4C, H/8, W/8)
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        x_spp = resize(
            x_spp, size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        if self.ms_context is not None:
            x_spp = self.ms_context(x_spp)

        x_spp = self.final_proj(x_spp)     # (B, 4C, H/8, W/8)

        # ===== C5 enhanced =====
        # GCNet gốc: out = x_d6 + spp(s6_gốc)
        c5_gcnet = feats['out']            # (B, 4C, H/8, W/8)
        # Phiên bản enhanced: detail stage6 + spp(s6_enhanced)
        c5_enh = x_d6 + x_spp              # (B, 4C, H/8, W/8)

        # Cho head dùng c5_enh là 'c5'
        return dict(
            c1=c1,
            c2=c2,
            c4=c4,
            c5=c5_enh,
            c5_gcnet=c5_gcnet,
        )
