"""
GCNet Final — FoggyAwareNorm + MultiScaleContext, không DWSA
=============================================================
Backbone:
  - Stem 2 conv đầu dùng FoggyAwareNorm (IN + BN learnable gate)
  - GCNetCore bilateral branch giữ nguyên
  - MultiScaleContextModule trên x_spp (/8, C*4)
  - Bỏ hoàn toàn DWSA

Head:
  - EnhancedDecoder: c5(/8) + c4(/8) + c2(/4) + c1(/2)
  - GatedFusion tại mỗi skip connection
  - Aux head trên c4

Deploy:
  - switch_to_deploy() gộp GCBlock → single Conv3x3
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from components.components import (
    BaseModule,
    ConvModule,
    DAPPM,
    build_activation_layer,
    build_norm_layer,
    resize,
    OptConfigType,
)


# =============================================================================
# FoggyAwareNorm
# =============================================================================

class FoggyAwareNorm(nn.Module):
    """Learnable blend của InstanceNorm và BatchNorm.

    alpha → 1 : thiên về IN  (robust với foggy / domain shift)
    alpha → 0 : thiên về BN  (tốt cho clear images / in-domain)

    Khởi tạo alpha=0.5 — trung tính, không bias ban đầu.
    Đặt tại 2 conv đầu của stem (stage1) nơi ảnh foggy chưa được
    normalize lần nào.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5,
                 momentum: float = 0.1, requires_grad: bool = True):
        super().__init__()
        self.bn  = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum,
                                  affine=True, track_running_stats=True)
        self.in_ = nn.InstanceNorm2d(num_channels, eps=eps,
                                     affine=True, track_running_stats=False)
        # Per-channel learnable gate
        self.alpha = nn.Parameter(
            torch.ones(1, num_channels, 1, 1) * 0.5,
            requires_grad=requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        a = torch.sigmoid(self.alpha)
        return a * self.in_(x) + (1.0 - a) * self.bn(x)


# =============================================================================
# GCBlock primitives  (giữ nguyên từ bản gốc)
# =============================================================================

class Block1x1(BaseModule):
    def __init__(self, in_channels, out_channels, stride=1, padding=0,
                 bias=True, norm_cfg=dict(type='BN', requires_grad=True),
                 deploy=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride  = stride
        self.padding = padding
        self.bias    = bias
        self.deploy  = deploy

        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, 1,
                                  stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(in_channels, out_channels, 1,
                                    stride=stride, padding=padding,
                                    bias=bias, norm_cfg=norm_cfg, act_cfg=None)
            self.conv2 = ConvModule(out_channels, out_channels, 1,
                                    stride=1, padding=padding,
                                    bias=bias, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def _fuse_bn(self, conv):
        k, b = conv.conv.weight, conv.conv.bias
        rm, rv = conv.bn.running_mean, conv.bn.running_var
        g, beta, eps = conv.bn.weight, conv.bn.bias, conv.bn.eps
        std = (rv + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        fused_b = (beta + (b - rm) * g / std
                   if self.bias else beta - rm * g / std)
        return k * t, fused_b

    def switch_to_deploy(self):
        k1, b1 = self._fuse_bn(self.conv1)
        k2, b2 = self._fuse_bn(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1,
                              stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', k2.squeeze(3).squeeze(2), k1)
        self.conv.bias.data = (b2
                               + (b1.view(1, -1, 1, 1) * k2).sum(3).sum(2).sum(1))
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class Block3x3(BaseModule):
    def __init__(self, in_channels, out_channels, stride=1, padding=0,
                 bias=True, norm_cfg=dict(type='BN', requires_grad=True),
                 deploy=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride  = stride
        self.padding = padding
        self.bias    = bias
        self.deploy  = deploy

        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, 3,
                                  stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(in_channels, out_channels, 3,
                                    stride=stride, padding=padding,
                                    bias=bias, norm_cfg=norm_cfg, act_cfg=None)
            self.conv2 = ConvModule(out_channels, out_channels, 1,
                                    stride=1, padding=0,
                                    bias=bias, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def _fuse_bn(self, conv):
        k, b = conv.conv.weight, conv.conv.bias
        rm, rv = conv.bn.running_mean, conv.bn.running_var
        g, beta, eps = conv.bn.weight, conv.bn.bias, conv.bn.eps
        std = (rv + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        fused_b = (beta + (b - rm) * g / std
                   if self.bias else beta - rm * g / std)
        return k * t, fused_b

    def switch_to_deploy(self):
        k1, b1 = self._fuse_bn(self.conv1)
        k2, b2 = self._fuse_bn(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3,
                              stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', k2.squeeze(3).squeeze(2), k1)
        self.conv.bias.data = (b2
                               + (b1.view(1, -1, 1, 1) * k2).sum(3).sum(2).sum(1))
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCBlock(nn.Module):
    """Re-parameterizable block: 2×3x3 + 1x1 + BN-identity → single 3x3 at deploy."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, padding_mode='zeros',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 act=True, deploy=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride  = stride
        self.padding = padding
        self.deploy  = deploy

        assert kernel_size == 3 and padding == 1
        padding_11 = padding - kernel_size // 2

        self.relu = build_activation_layer(act_cfg) if act else nn.Identity()

        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=True,
                padding_mode=padding_mode)
        else:
            self.path_residual = (
                build_norm_layer(norm_cfg, in_channels)[1]
                if (out_channels == in_channels and stride == 1) else None)
            self.path_3x3_1 = Block3x3(in_channels, out_channels, stride=stride,
                                        padding=padding, bias=False, norm_cfg=norm_cfg)
            self.path_3x3_2 = Block3x3(in_channels, out_channels, stride=stride,
                                        padding=padding, bias=False, norm_cfg=norm_cfg)
            self.path_1x1   = Block1x1(in_channels, out_channels, stride=stride,
                                        padding=padding_11, bias=False, norm_cfg=norm_cfg)

    def forward(self, x):
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(x))
        id_out = 0 if self.path_residual is None else self.path_residual(x)
        return self.relu(
            self.path_3x3_1(x) + self.path_3x3_2(x)
            + self.path_1x1(x) + id_out)

    def _pad_1x1_to_3x3(self, k):
        return 0 if k is None else F.pad(k, [1, 1, 1, 1])

    def _fuse_bn(self, conv):
        if conv is None:
            return 0, 0
        if isinstance(conv, ConvModule):
            k = conv.conv.weight
            rm, rv = conv.bn.running_mean, conv.bn.running_var
            g, beta, eps = conv.bn.weight, conv.bn.bias, conv.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                kv = np.zeros(
                    (self.in_channels, self.in_channels, 3, 3), np.float32)
                for i in range(self.in_channels):
                    kv[i, i, 1, 1] = 1.0
                self.id_tensor = torch.from_numpy(kv).to(conv.weight.device)
            k = self.id_tensor
            rm, rv = conv.running_mean, conv.running_var
            g, beta, eps = conv.weight, conv.bias, conv.eps
        std = (rv + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        return k * t, beta - rm * g / std

    def get_equivalent_kernel_bias(self):
        self.path_3x3_1.switch_to_deploy()
        k1, b1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        self.path_3x3_2.switch_to_deploy()
        k2, b2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        self.path_1x1.switch_to_deploy()
        k3, b3 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        ki, bi = self._fuse_bn(self.path_residual)
        return (k1 + k2 + self._pad_1x1_to_3x3(k3) + ki,
                b1 + b2 + b3 + bi)

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_3x3'):
            return
        k, b = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, bias=True)
        self.reparam_3x3.weight.data = k
        self.reparam_3x3.bias.data   = b
        for p in self.parameters():
            p.detach_()
        for attr in ('path_3x3_1', 'path_3x3_2', 'path_1x1',
                     'path_residual', 'id_tensor'):
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.deploy = True


# =============================================================================
# MultiScaleContextModule
# =============================================================================

class MultiScaleContextModule(nn.Module):
    """Multi-scale context trên x_spp (/8, C*4).

    scales=(1,2): branch 1x1 conv + branch avgpool(2)→upsample.
    Nhẹ hơn attention, không redundant với DAPPM (DAPPM trên /64,
    MSC trên fused feature /8).
    Residual: x + clamp(alpha, 0, 0.5) * fused.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 scales: Tuple = (1, 2), branch_ratio: int = 8,
                 alpha: float = 0.1):
        super().__init__()
        self.scales = scales

        total = in_channels // branch_ratio
        base  = total // len(scales)
        extra = total % len(scales)
        per_branch = [max(base + (1 if i < extra else 0), 1)
                      for i in range(len(scales))]
        fused_ch = sum(per_branch)

        self.branches = nn.ModuleList()
        for s, c_out in zip(scales, per_branch):
            if s == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, c_out, 1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.AvgPool2d(s, stride=s),
                    nn.Conv2d(in_channels, c_out, 1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ))

        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, 3, padding=1,
                      groups=fused_ch, bias=False),
            nn.BatchNorm2d(fused_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_ch, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ) if in_channels != out_channels else None
        )

        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        outs = []
        for s, branch in zip(self.scales, self.branches):
            o = branch(x)
            if o.shape[-2:] != (H, W):
                o = F.interpolate(o, (H, W), mode='bilinear', align_corners=False)
            outs.append(o)
        fused  = self.fusion(torch.cat(outs, dim=1))
        x_proj = self.proj(x) if self.proj is not None else x
        alpha  = torch.clamp(self.alpha, 0.0, 0.5)
        return x_proj + alpha * fused


# =============================================================================
# GCNetBackbone
# =============================================================================

class GCNetBackbone(BaseModule):
    """Bilateral branch backbone với FoggyAwareNorm tại stem.

    Thay đổi so với GCNet gốc:
      1. stem_conv1 và stem_conv2 dùng FoggyAwareNorm thay BN thuần.
      2. MultiScaleContextModule trên x_spp trước final_proj.
      3. Không có DWSA ở bất kỳ stage nào.

    Outputs dict:
      c1 : /2,  C        — skip cho decoder
      c2 : /4,  C        — skip cho decoder
      c4 : /8,  C*2      — detail branch sau stage4 fusion (aux loss)
      c5 : /8,  C*4      — fused + MSC enhanced (input cho head)
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = None,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 fan_eps: float = 1e-5,
                 fan_momentum: float = 0.1,
                 ms_scales: Tuple = (1, 2),
                 ms_branch_ratio: int = 8,
                 ms_alpha: float = 0.1,
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        if num_blocks_per_stage is None:
            num_blocks_per_stage = [4, 4, [5, 4], [5, 4], [2, 2]]

        self.align_corners = align_corners
        self.channels      = channels
        self.deploy        = deploy

        C  = channels
        nb = num_blocks_per_stage

        # ------------------------------------------------------------------ #
        # Stem — stage1: FoggyAwareNorm ở 2 conv đầu                         #
        # ------------------------------------------------------------------ #
        self.stem_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, stride=2, padding=1, bias=False),
            FoggyAwareNorm(C, eps=fan_eps, momentum=fan_momentum),
            build_activation_layer(act_cfg),
        )
        self.stem_conv2 = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            FoggyAwareNorm(C, eps=fan_eps, momentum=fan_momentum),
            build_activation_layer(act_cfg),
        )

        # Stage 2
        self.stem_stage2 = nn.Sequential(
            *[GCBlock(C, C, stride=1, norm_cfg=norm_cfg,
                      act_cfg=act_cfg, deploy=deploy)
              for _ in range(nb[0])]
        )

        # Stage 3
        self.stem_stage3 = nn.Sequential(
            GCBlock(C, C * 2, stride=2, norm_cfg=norm_cfg,
                    act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C * 2, C * 2, stride=1, norm_cfg=norm_cfg,
                      act_cfg=act_cfg, deploy=deploy)
              for _ in range(nb[1] - 1)],
        )

        self.relu = build_activation_layer(act_cfg)

        # ------------------------------------------------------------------ #
        # Semantic branch (stage 4 → 6)                                       #
        # ------------------------------------------------------------------ #
        self.semantic_branch = nn.ModuleList([
            nn.Sequential(
                GCBlock(C*2, C*4, stride=2, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(nb[2][0] - 2)],
                GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                GCBlock(C*4, C*8, stride=2, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*8, C*8, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(nb[3][0] - 2)],
                GCBlock(C*8, C*8, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                GCBlock(C*8, C*16, stride=2, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*16, C*16, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(nb[4][0] - 2)],
                GCBlock(C*16, C*16, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
        ])

        # ------------------------------------------------------------------ #
        # Detail branch (stage 4 → 6)                                         #
        # ------------------------------------------------------------------ #
        self.detail_branch = nn.ModuleList([
            nn.Sequential(
                *[GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(nb[2][1] - 1)],
                GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                *[GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(nb[3][1] - 1)],
                GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                GCBlock(C*2, C*4, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(nb[4][1] - 2)],
                GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
        ])

        # ------------------------------------------------------------------ #
        # Bilateral fusion                                                      #
        # ------------------------------------------------------------------ #
        self.compression_1 = ConvModule(
            C*4, C*2, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.down_1 = ConvModule(
            C*2, C*4, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=None)

        self.compression_2 = ConvModule(
            C*8, C*2, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(C*2, C*4, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*4, C*8, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=None),
        )

        # ------------------------------------------------------------------ #
        # DAPPM + MSC + final_proj                                             #
        # ------------------------------------------------------------------ #
        self.spp = DAPPM(C*16, ppm_channels, C*4, num_scales=5,
                         norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.ms_context = MultiScaleContextModule(
            in_channels=C * 4,
            out_channels=C * 4,
            scales=ms_scales,
            branch_ratio=ms_branch_ratio,
            alpha=ms_alpha,
        )

        self.final_proj = ConvModule(
            C * 4, C * 4, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self._kaiming_init()

    def _kaiming_init(self):
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
        out_size = (math.ceil(x.shape[-2] / 8),
                    math.ceil(x.shape[-1] / 8))

        # ---- Stage 1 — FoggyAwareNorm ----
        c1   = self.stem_conv1(x)       # /2,  C
        c2   = self.stem_conv2(c1)      # /4,  C

        # ---- Stage 2 & 3 ----
        feat = self.stem_stage2(c2)     # /4,  C
        feat = self.stem_stage3(feat)   # /8,  C*2

        # ---- Stage 4 — bilateral ----
        x_s  = self.semantic_branch[0](feat)         # /16, C*4
        x_d  = self.detail_branch[0](feat)           # /8,  C*2
        comp = self.compression_1(self.relu(x_s))
        x_s  = x_s + self.down_1(self.relu(x_d))
        x_d  = x_d + resize(comp, size=out_size, mode='bilinear',
                             align_corners=self.align_corners)
        c4   = x_d                                   # /8,  C*2  → aux loss

        # ---- Stage 5 — bilateral ----
        x_s  = self.semantic_branch[1](self.relu(x_s))  # /32, C*8
        x_d  = self.detail_branch[1](self.relu(x_d))    # /8,  C*2
        comp = self.compression_2(self.relu(x_s))
        x_s  = x_s + self.down_2(self.relu(x_d))
        x_d  = x_d + resize(comp, size=out_size, mode='bilinear',
                             align_corners=self.align_corners)

        # ---- Stage 6 ----
        x_d6 = self.detail_branch[2](self.relu(x_d))    # /8,  C*4
        x_s6 = self.semantic_branch[2](self.relu(x_s))  # /64, C*16

        spp  = self.spp(x_s6)
        spp  = resize(spp, size=out_size, mode='bilinear',
                      align_corners=self.align_corners)

        # ---- MSC + fuse + proj ----
        spp  = self.ms_context(spp)
        c5   = self.final_proj(x_d6 + spp)              # /8,  C*4

        return dict(c1=c1, c2=c2, c4=c4, c5=c5)

    def switch_to_deploy(self):
        """Fuse tất cả GCBlock → single Conv3x3."""
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
        self.deploy = True


# =============================================================================
# Decoder components
# =============================================================================

class GatedFusion(nn.Module):
    """gate * skip + (1 - gate) * dec"""

    def __init__(self, channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.gate = nn.Sequential(
            ConvModule(channels * 2, channels, 1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1,
                       norm_cfg=None, act_cfg=dict(type='Sigmoid')),
        )

    def forward(self, skip: Tensor, dec: Tensor) -> Tensor:
        g = self.gate(torch.cat([skip, dec], dim=1))
        return g * skip + (1.0 - g) * dec


class DWConvModule(nn.Module):
    """Depthwise-separable conv."""

    def __init__(self, channels, kernel_size=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.dw = ConvModule(channels, channels, kernel_size,
                             padding=kernel_size // 2, groups=channels,
                             norm_cfg=norm_cfg, act_cfg=None)
        self.pw = ConvModule(channels, channels, 1,
                             norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


# =============================================================================
# EnhancedDecoder
# =============================================================================

class EnhancedDecoder(nn.Module):
    """4-skip decoder: c5(/8) + c4(/8) + c2(/4) + c1(/2).

    Pipeline (simplified):
      c5 ──GatedFusion(c4_proj)──> /8
           ResidualBlock → Conv → D
           ↓ up×2
      GatedFusion(c2_proj) ──> /4, D
           ResidualBlock → Conv → D//2
           ↓ up×2
      simple add(c1_proj) ──> /2, D//2   # bỏ GatedFusion tại /2
           ↓ up×2
      Conv1x1 → dropout ──> full, D//2   # bỏ DWConv×2 tại full res
    """

    def __init__(self,
                 c5_channels: int,
                 c4_channels: int,
                 c2_channels: int,
                 c1_channels: int,
                 decoder_channels: int = 96,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout_ratio: float = 0.1):
        super().__init__()
        D = decoder_channels

        # /8: fuse c4 vào c5
        self.c4_proj = ConvModule(c4_channels, c5_channels, 1,
                                  norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c4 = GatedFusion(c5_channels, norm_cfg, act_cfg)
        self.refine0 = nn.Sequential(
            ResidualBlock(c5_channels, norm_cfg, act_cfg),
            ConvModule(c5_channels, D, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /8 → /4: fuse c2 (giữ GatedFusion vì /4 vẫn reasonable)
        self.up1     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c2_proj = ConvModule(c2_channels, D, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c2 = GatedFusion(D, norm_cfg, act_cfg)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg, act_cfg),
            ConvModule(D, D // 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /4 → /2: c1 dùng simple add thay GatedFusion
        # GatedFusion tại /2 phải xử lý tensor 256×512 — quá tốn
        self.up2     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c1_proj = ConvModule(c1_channels, D // 2, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)  # có act để align range

        # /2 → full: bỏ DWConv×2, chỉ giữ 1 conv nhẹ
        # DWConv tại 512×1024 tốn activation dù params ít
        self.up3        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_proj = ConvModule(D // 2, D // 2, 3, padding=1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout    = (nn.Dropout2d(dropout_ratio)
                           if dropout_ratio > 0 else nn.Identity())
        self.out_channels = D // 2

    def forward(self, c5: Tensor, c4: Tensor,
                c2: Tensor, c1: Tensor) -> Tensor:
        # /8: fuse c4 → c5
        x = self.fuse_c4(self.c4_proj(c4), c5)
        x = self.refine0(x)

        # /8 → /4: fuse c2
        x   = self.up1(x)
        c2p = self.c2_proj(c2)
        if c2p.shape[-2:] != x.shape[-2:]:
            c2p = F.interpolate(c2p, x.shape[-2:],
                                mode='bilinear', align_corners=False)
        x = self.fuse_c2(c2p, x)
        x = self.refine1(x)

        # /4 → /2: simple add thay vì GatedFusion
        x   = self.up2(x)
        c1p = self.c1_proj(c1)
        if c1p.shape[-2:] != x.shape[-2:]:
            c1p = F.interpolate(c1p, x.shape[-2:],
                                mode='bilinear', align_corners=False)
        x = x + c1p  # add thay vì gate — /2 resolution quá lớn cho gate

        # /2 → full: 1 conv3x3 thay DWConv×2
        x = self.up3(x)
        x = self.final_proj(x)
        x = self.dropout(x)
        return x


# =============================================================================
# GCNetHead
# =============================================================================

class GCNetHead(nn.Module):
    """Main head: EnhancedDecoder + conv_seg + aux head.

    Training  → (aux_logit, main_logit)
    Eval      → main_logit
    """

    def __init__(self,
                 channels: int = 32,
                 num_classes: int = 19,
                 decoder_channels: int = 128,
                 dropout_ratio: float = 0.1,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners
        C = channels

        self.decoder = EnhancedDecoder(
            c5_channels=C * 4,
            c4_channels=C * 2,
            c2_channels=C,
            c1_channels=C,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=dropout_ratio,
        )

        out_ch = self.decoder.out_channels
        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(out_ch, num_classes, 1),
        )

        # Aux head trên c4 (/8, C*2)
        self.aux_head = nn.Sequential(
            ConvModule(C * 2, C * 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(C * 2, num_classes, 1),
        )

    def forward(self, inputs: Dict[str, Tensor]):
        c1 = inputs['c1']
        c2 = inputs['c2']
        c4 = inputs['c4']
        c5 = inputs['c5']

        feat       = self.decoder(c5, c4, c2, c1)
        main_logit = self.conv_seg(feat)

        if self.training:
            aux_logit = self.aux_head(c4)
            return aux_logit, main_logit
        return main_logit


# =============================================================================
# GCNetSegmentor — entry point
# =============================================================================

class GCNetSegmentor(nn.Module):
    """Full model = GCNetBackbone (FAN + MSC) + GCNetHead (EnhancedDecoder).

    Cách dùng:
        model = GCNetSegmentor(channels=32, num_classes=19)

        # Training
        model.train()
        aux_logit, main_logit = model(x)

        # Eval
        model.eval()
        logit = model(x)

        # Deploy — fuse GCBlocks → single Conv3x3, nhanh hơn ~20-30%
        model.switch_to_deploy()
        model.eval()
        logit = model(x)
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = None,
                 num_classes: int = 19,
                 decoder_channels: int = 128,
                 dropout_ratio: float = 0.1,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 # FoggyAwareNorm
                 fan_eps: float = 1e-5,
                 fan_momentum: float = 0.1,
                 # MultiScaleContext
                 ms_scales: Tuple = (1, 2),
                 ms_branch_ratio: int = 8,
                 ms_alpha: float = 0.1,
                 deploy: bool = False):
        super().__init__()

        self.backbone = GCNetBackbone(
            in_channels=in_channels,
            channels=channels,
            ppm_channels=ppm_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            align_corners=align_corners,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            fan_eps=fan_eps,
            fan_momentum=fan_momentum,
            ms_scales=ms_scales,
            ms_branch_ratio=ms_branch_ratio,
            ms_alpha=ms_alpha,
            deploy=deploy,
        )

        self.head = GCNetHead(
            channels=channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
        )

    def forward(self, x: Tensor):
        return self.head(self.backbone(x))

    def switch_to_deploy(self):
        """Fuse tất cả GCBlock → single Conv3x3 để inference nhanh hơn."""
        self.backbone.switch_to_deploy()

    def get_param_groups(self,
                         head_lr: float = 3e-4,
                         backbone_lr: float = 3e-5,
                         fan_lr: float = 3e-5,
                         msc_lr: float = 5e-5) -> List[Dict]:
        """Param groups cho AdamW discriminative LR.

        fan_lr = 3e-5  : FAN alpha học chậm cùng backbone để ổn định.
        msc_lr = 5e-5  : MSC alpha không cần lr cao, tránh oscillation.
        """
        fan_params  = (list(self.backbone.stem_conv1.parameters())
                       + list(self.backbone.stem_conv2.parameters()))
        fan_ids     = {id(p) for p in fan_params}

        msc_params  = (list(self.backbone.ms_context.parameters())
                       + list(self.backbone.final_proj.parameters()))
        msc_ids     = {id(p) for p in msc_params}

        head_params = list(self.head.parameters())
        head_ids    = {id(p) for p in head_params}

        bb_params = [p for p in self.backbone.parameters()
                     if id(p) not in fan_ids and id(p) not in msc_ids]

        return [
            {'params': head_params, 'lr': head_lr,     'name': 'head'},
            {'params': bb_params,   'lr': backbone_lr, 'name': 'backbone'},
            {'params': fan_params,  'lr': fan_lr,      'name': 'fan'},
            {'params': msc_params,  'lr': msc_lr,      'name': 'msc'},
        ]
