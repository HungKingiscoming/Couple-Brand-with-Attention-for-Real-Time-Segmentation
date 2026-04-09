"""
GCNet Final — tích hợp backbone mới + head mới
===============================================

Backbone (GCNetBackbone):
  - stem_conv1/2: FoggyAwareNorm (IN + BN learnable gate per-channel)
  - stage2~6: bilateral branch GCBlock giữ nguyên
  - MultiScaleContextModule trên x_spp (/8, C*4)
  - Không DWSA
  - Training  → (c4_feat, c6_feat, c1, c2)
  - Inference → (c6_feat, c1, c2)

Head (GCNetHead / GCNetHeadLite):
  - GCNetHead    : dùng đủ c1,c2,c4,c6 skip — cần backbone mới
  - GCNetHeadLite: chỉ dùng c4,c6 — tương thích backward với backbone cũ

Segmentor (GCNetSegmentor):
  - Kết nối backbone mới + GCNetHead
  - Training  → (aux_logit/2, main_logit/2)
  - Inference → main_logit/2  (resize trong loss/predict)
  - switch_to_deploy(): fuse GCBlock → single Conv3x3

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
    SampleList,
)


# =============================================================================
# Accuracy helper
# =============================================================================

def accuracy(pred: Tensor, target: Tensor, ignore_index: int = 255) -> Tensor:
    pred_label = pred.argmax(dim=1)
    mask    = target != ignore_index
    correct = (pred_label[mask] == target[mask]).sum().float()
    total   = mask.sum().float().clamp(min=1)
    return correct / total * 100.0


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, loss_weight: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight  = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * F.cross_entropy(
            pred, target, ignore_index=self.ignore_index)


# =============================================================================
# FoggyAwareNorm
# =============================================================================

class FoggyAwareNorm(nn.Module):
    """Learnable blend: alpha*IN(x) + (1-alpha)*BN(x).

    alpha → 1 : thiên về IN  (robust với foggy / domain shift)
    alpha → 0 : thiên về BN  (tốt cho clear / in-domain)
    Khởi tạo alpha=0.5, per-channel.
    Đặt tại 2 conv đầu stem — nơi ảnh foggy chưa được normalize lần nào.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5,
                 momentum: float = 0.1, requires_grad: bool = True):
        super().__init__()
        self.bn  = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum,
                                  affine=True, track_running_stats=True)
        self.in_ = nn.InstanceNorm2d(num_channels, eps=eps,
                                     affine=True, track_running_stats=False)
        self.alpha = nn.Parameter(
            torch.ones(1, num_channels, 1, 1) * 0.5,
            requires_grad=requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        a = torch.sigmoid(self.alpha)
        return a * self.in_(x) + (1.0 - a) * self.bn(x)


# =============================================================================
# GCBlock primitives
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
        return k * t, (beta + (b - rm) * g / std
                       if self.bias else beta - rm * g / std)

    def switch_to_deploy(self):
        k1, b1 = self._fuse_bn(self.conv1)
        k2, b2 = self._fuse_bn(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1,
                              stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', k2.squeeze(3).squeeze(2), k1)
        self.conv.bias.data = b2 + (b1.view(1, -1, 1, 1) * k2).sum(3).sum(2).sum(1)
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
        return k * t, (beta + (b - rm) * g / std
                       if self.bias else beta - rm * g / std)

    def switch_to_deploy(self):
        k1, b1 = self._fuse_bn(self.conv1)
        k2, b2 = self._fuse_bn(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3,
                              stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', k2.squeeze(3).squeeze(2), k1)
        self.conv.bias.data = b2 + (b1.view(1, -1, 1, 1) * k2).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCBlock(nn.Module):
    """Re-parameterizable: 2×3x3 + 1x1 + BN-identity → single Conv3x3 at deploy."""

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

    scales=(1,2): 1x1 conv + avgpool(2)→upsample.
    Nhẹ hơn attention, không redundant với DAPPM.
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

    Training  → (c4_feat, c6_feat, c1, c2)
    Inference → (c6_feat, c1, c2)

    Tương thích với GCNetHead (dùng đủ 4 skip) và GCNetHeadLite (chỉ c4+c6).
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

        # ---- Stem: FoggyAwareNorm ở 2 conv đầu ----
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

        self.stem_stage2 = nn.Sequential(
            *[GCBlock(C, C, stride=1, norm_cfg=norm_cfg,
                      act_cfg=act_cfg, deploy=deploy)
              for _ in range(nb[0])]
        )
        self.stem_stage3 = nn.Sequential(
            GCBlock(C, C*2, stride=2, norm_cfg=norm_cfg,
                    act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                      act_cfg=act_cfg, deploy=deploy)
              for _ in range(nb[1] - 1)],
        )

        self.relu = build_activation_layer(act_cfg)

        # ---- Semantic branch ----
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

        # ---- Detail branch ----
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

        # ---- Bilateral fusion ----
        self.compression_1 = ConvModule(C*4, C*2, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.down_1        = ConvModule(C*2, C*4, 3, stride=2, padding=1,
                                        norm_cfg=norm_cfg, act_cfg=None)
        self.compression_2 = ConvModule(C*8, C*2, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.down_2        = nn.Sequential(
            ConvModule(C*2, C*4, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*4, C*8, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=None),
        )

        # ---- DAPPM + MSC + final_proj ----
        self.spp = DAPPM(C*16, ppm_channels, C*4, num_scales=5,
                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ms_context = MultiScaleContextModule(
            C*4, C*4, scales=ms_scales,
            branch_ratio=ms_branch_ratio, alpha=ms_alpha)
        self.final_proj = ConvModule(C*4, C*4, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        """
        Training  → (c4_feat, c6_feat, c1, c2)
        Inference → (c6_feat, c1, c2)

        c4_feat : /8,  C*2  — detail branch sau stage4, dùng cho aux loss
        c6_feat : /8,  C*4  — fused + MSC enhanced, main input cho head
        c1      : /2,  C    — skip decoder
        c2      : /4,  C    — skip decoder
        """
        out_size = (math.ceil(x.shape[-2] / 8),
                    math.ceil(x.shape[-1] / 8))

        # Stage 1 — FoggyAwareNorm
        c1   = self.stem_conv1(x)      # /2,  C
        c2   = self.stem_conv2(c1)     # /4,  C

        # Stage 2 & 3
        feat = self.stem_stage2(c2)    # /4,  C
        feat = self.stem_stage3(feat)  # /8,  C*2

        # Stage 4 — bilateral
        x_s  = self.semantic_branch[0](feat)
        x_d  = self.detail_branch[0](feat)
        comp = self.compression_1(self.relu(x_s))
        x_s  = x_s + self.down_1(self.relu(x_d))
        x_d  = x_d + resize(comp, size=out_size, mode='bilinear',
                             align_corners=self.align_corners)
        c4_feat = x_d                  # /8,  C*2  → aux loss

        # Stage 5 — bilateral
        x_s  = self.semantic_branch[1](self.relu(x_s))
        x_d  = self.detail_branch[1](self.relu(x_d))
        comp = self.compression_2(self.relu(x_s))
        x_s  = x_s + self.down_2(self.relu(x_d))
        x_d  = x_d + resize(comp, size=out_size, mode='bilinear',
                             align_corners=self.align_corners)

        # Stage 6
        x_d6 = self.detail_branch[2](self.relu(x_d))   # /8,  C*4
        x_s6 = self.semantic_branch[2](self.relu(x_s)) # /64, C*16

        spp  = self.spp(x_s6)
        spp  = resize(spp, size=out_size, mode='bilinear',
                      align_corners=self.align_corners)
        spp  = self.ms_context(spp)
        c6_feat = self.final_proj(x_d6 + spp)          # /8,  C*4

        if self.training:
            return c4_feat, c6_feat, c1, c2
        else:
            return c6_feat, c1, c2

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
        self.deploy = True


# =============================================================================
# Head primitives (dùng chung cho GCNetHead và GCNetHeadLite)
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


class ResidualBlock(nn.Module):
    def __init__(self, channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=None)
        self.act   = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


# =============================================================================
# EnhancedDecoder — dừng tại /2
# =============================================================================

class EnhancedDecoder(nn.Module):
    """3-skip decoder: c6(/8) + c4_skip(/8) + c2(/4) + c1(/2).

    Dừng tại /2 — KHÔNG upsample lên full res.
    Loss trong GCNetHead resize logit /2 → full res khi tính CE.
    → Không có activation nào tại full resolution.

    Pipeline:
      c6 ──GatedFusion(c4_proj)──> /8
           ResidualBlock → Conv → D
           ↓ up×2
      GatedFusion(c2_proj) ──> /4, D
           ResidualBlock → Conv → D//2
           ↓ up×2
      simple add(c1_proj) ──> /2, D//2   ← OUTPUT
    """

    def __init__(self,
                 c6_channels: int,
                 c4_channels: int,
                 c2_channels: int,
                 c1_channels: int,
                 decoder_channels: int = 96,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        D = decoder_channels

        # /8: fuse c4 vào c6
        self.c4_proj = ConvModule(c4_channels, c6_channels, 1,
                                  norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c4 = GatedFusion(c6_channels, norm_cfg, act_cfg)
        self.refine0 = nn.Sequential(
            ResidualBlock(c6_channels, norm_cfg, act_cfg),
            ConvModule(c6_channels, D, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /8 → /4: fuse c2 (GatedFusion — /4 còn nhỏ, chi phí ok)
        self.up1     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c2_proj = ConvModule(c2_channels, D, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c2 = GatedFusion(D, norm_cfg, act_cfg)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg, act_cfg),
            ConvModule(D, D // 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /4 → /2: simple add c1 (/2 quá lớn cho GatedFusion)
        self.up2     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c1_proj = ConvModule(c1_channels, D // 2, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.out_channels = D // 2

    def forward(self, c6: Tensor, c4_skip: Tensor,
                c2: Tensor, c1: Tensor) -> Tensor:
        # /8: fuse c4 → c6
        x = self.fuse_c4(self.c4_proj(c4_skip), c6)
        x = self.refine0(x)

        # /8 → /4: fuse c2
        x   = self.up1(x)
        c2p = self.c2_proj(c2)
        if c2p.shape[-2:] != x.shape[-2:]:
            c2p = F.interpolate(c2p, x.shape[-2:],
                                mode='bilinear', align_corners=False)
        x = self.fuse_c2(c2p, x)
        x = self.refine1(x)

        # /4 → /2: simple add c1
        x   = self.up2(x)
        c1p = self.c1_proj(c1)
        if c1p.shape[-2:] != x.shape[-2:]:
            c1p = F.interpolate(c1p, x.shape[-2:],
                                mode='bilinear', align_corners=False)
        return x + c1p   # /2, D//2


# =============================================================================
# GCNetHead — full skip (c1, c2, c4, c6), dùng với GCNetBackbone mới
# =============================================================================

class GCNetHead(BaseModule):
    """Decode head với EnhancedDecoder.

    Nhận output từ GCNetBackbone:
      Training  : (c4_feat, c6_feat, c1, c2)
      Inference : (c6_feat, c1, c2)

    Training  → (aux_logit/2, main_logit/2)
    Inference → main_logit/2
    Loss tự resize /2 → full res trong loss().
    """

    def __init__(self,
                 in_channels: int,          # c6 channels = C*4
                 channels: int,             # backbone base C
                 num_classes: int,
                 decoder_channels: int = 96,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
                 dropout_ratio: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.align_corners   = align_corners
        self.ignore_index    = ignore_index
        self.loss_weight_aux = loss_weight_aux
        C = channels

        self.decoder = EnhancedDecoder(
            c6_channels=in_channels,        # C*4
            c4_channels=in_channels // 2,   # C*2
            c2_channels=C,
            c1_channels=C,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        out_ch = self.decoder.out_channels  # D//2

        self.dropout  = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(out_ch, num_classes, kernel_size=1)

        # Aux head trên c4_feat (/8, C*2)
        self.aux_head = nn.Sequential(
            build_norm_layer(norm_cfg, in_channels // 2)[1],
            build_activation_layer(act_cfg),
            ConvModule(in_channels // 2, in_channels // 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.aux_cls_seg = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

        self.loss_main = CrossEntropyLoss(ignore_index=ignore_index, loss_weight=1.0)
        self.loss_aux  = CrossEntropyLoss(ignore_index=ignore_index,
                                          loss_weight=loss_weight_aux)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Training  : inputs = (c4_feat, c6_feat, c1, c2)
                    returns (aux_logit/2, main_logit/2)
        Inference : inputs = (c6_feat, c1, c2)
                    returns main_logit/2
        """
        if self.training:
            c4_feat, c6_feat, c1, c2 = inputs
            feat       = self.decoder(c6_feat, c4_feat, c2, c1)
            main_logit = self.cls_seg(self.dropout(feat))
            aux_logit  = self.aux_cls_seg(self.aux_head(c4_feat))
            return aux_logit, main_logit
        else:
            c6_feat, c1, c2 = inputs
            feat = self.decoder(c6_feat, None, c2, c1)
            return self.cls_seg(self.dropout(feat))

    def loss(self, seg_logits: Tuple[Tensor, Tensor],
             seg_label: Tensor) -> Dict[str, Tensor]:
        """Resize /2 → full res NGOÀI backward graph decoder."""
        aux_logit, main_logit = seg_logits
        target_size = seg_label.shape[1:]

        main_logit = resize(main_logit, size=target_size,
                            mode='bilinear', align_corners=self.align_corners)
        aux_logit  = resize(aux_logit,  size=target_size,
                            mode='bilinear', align_corners=self.align_corners)

        return {
            'loss_main': self.loss_main(main_logit, seg_label),
            'loss_aux':  self.loss_aux(aux_logit,  seg_label),
            'acc_seg':   accuracy(main_logit, seg_label,
                                  ignore_index=self.ignore_index),
        }

    def predict(self, inputs,
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit


# =============================================================================
# GCNetHeadLite — fallback, tương thích backbone CŨ (c4, c6) only
# =============================================================================

class GCNetHeadLite(BaseModule):
    """Head nhẹ, tương thích 100% backbone cũ trả về (c4_feat, c6_feat).

    Không cần c1/c2 skip — pipeline:
      c6(/8) ──GatedFusion(c4_proj)──> /8
               ResidualBlock → Conv → D
               ↓ up×2
               ResidualBlock → Conv → D//2   (/4)
               ↓ up×2
               Conv → D//2                   (/2) ← DỪNG

    Training  → (aux_logit/2, main_logit/2)
    Inference → main_logit/2
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 decoder_channels: int = 96,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
                 dropout_ratio: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.align_corners   = align_corners
        self.ignore_index    = ignore_index
        self.loss_weight_aux = loss_weight_aux
        D   = decoder_channels
        C4  = in_channels        # C*4
        C4h = in_channels // 2   # C*2

        self.c4_proj = ConvModule(C4h, C4, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c4 = GatedFusion(C4, norm_cfg, act_cfg)
        self.refine0 = nn.Sequential(
            ResidualBlock(C4, norm_cfg, act_cfg),
            ConvModule(C4, D, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.up1     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg, act_cfg),
            ConvModule(D, D // 2, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.up2     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = ConvModule(D // 2, D // 2, 3, padding=1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dropout     = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg     = nn.Conv2d(D // 2, num_classes, kernel_size=1)

        self.aux_head = nn.Sequential(
            build_norm_layer(norm_cfg, C4h)[1],
            build_activation_layer(act_cfg),
            ConvModule(C4h, C4h, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.aux_cls_seg = nn.Conv2d(C4h, num_classes, kernel_size=1)

        self.loss_main = CrossEntropyLoss(ignore_index=ignore_index, loss_weight=1.0)
        self.loss_aux  = CrossEntropyLoss(ignore_index=ignore_index,
                                          loss_weight=loss_weight_aux)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _decode(self, c6: Tensor, c4: Tensor) -> Tensor:
        x = self.fuse_c4(self.c4_proj(c4), c6)
        x = self.refine1(self.up1(self.refine0(x)))
        x = self.refine2(self.up2(x))
        return x

    def forward(self, inputs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Training  : inputs = (c4_feat, c6_feat) → (aux_logit/2, main_logit/2)
        Inference : inputs = c6_feat            → main_logit/2
        """
        if self.training:
            c4_feat, c6_feat = inputs
            feat       = self._decode(c6_feat, c4_feat)
            main_logit = self.cls_seg(self.dropout(feat))
            aux_logit  = self.aux_cls_seg(self.aux_head(c4_feat))
            return aux_logit, main_logit
        else:
            c6_feat = inputs
            B, C, H, W = c6_feat.shape
            c4_dummy = torch.zeros(B, C // 2, H, W,
                                   dtype=c6_feat.dtype, device=c6_feat.device)
            return self.cls_seg(self.dropout(self._decode(c6_feat, c4_dummy)))

    def loss(self, seg_logits: Tuple[Tensor, Tensor],
             seg_label: Tensor) -> Dict[str, Tensor]:
        aux_logit, main_logit = seg_logits
        target_size = seg_label.shape[1:]
        main_logit = resize(main_logit, size=target_size,
                            mode='bilinear', align_corners=self.align_corners)
        aux_logit  = resize(aux_logit,  size=target_size,
                            mode='bilinear', align_corners=self.align_corners)
        return {
            'loss_main': self.loss_main(main_logit, seg_label),
            'loss_aux':  self.loss_aux(aux_logit,  seg_label),
            'acc_seg':   accuracy(main_logit, seg_label,
                                  ignore_index=self.ignore_index),
        }

    def predict(self, inputs,
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit


# =============================================================================
# GCNetSegmentor — full model entry point
# =============================================================================

class GCNetSegmentor(nn.Module):
    """Full model = GCNetBackbone (FAN + MSC) + GCNetHead (EnhancedDecoder).

    Training  → (aux_logit/2, main_logit/2)
    Inference → main_logit/2
    Deploy    → switch_to_deploy() fuse GCBlock → single Conv3x3

    Cách dùng:
        model = GCNetSegmentor(channels=32, num_classes=19)
        model.train()
        aux_logit, main_logit = model(x)

        model.eval()
        logit = model(x)

        model.switch_to_deploy(); model.eval()
        logit = model(x)   # nhanh hơn ~20-30%
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = None,
                 num_classes: int = 19,
                 decoder_channels: int = 96,
                 dropout_ratio: float = 0.1,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
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
            in_channels=channels * 4,   # c6 = C*4
            channels=channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            ignore_index=ignore_index,
            loss_weight_aux=loss_weight_aux,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, x: Tensor):
        return self.head(self.backbone(x))

    def switch_to_deploy(self):
        self.backbone.switch_to_deploy()

    def get_param_groups(self,
                         head_lr: float = 3e-4,
                         backbone_lr: float = 3e-5,
                         fan_lr: float = 3e-5,
                         msc_lr: float = 5e-5) -> List[Dict]:
        """Discriminative LR groups cho AdamW."""
        fan_params = (list(self.backbone.stem_conv1.parameters())
                      + list(self.backbone.stem_conv2.parameters()))
        fan_ids    = {id(p) for p in fan_params}

        msc_params = (list(self.backbone.ms_context.parameters())
                      + list(self.backbone.final_proj.parameters()))
        msc_ids    = {id(p) for p in msc_params}

        head_params = list(self.head.parameters())

        bb_params = [p for p in self.backbone.parameters()
                     if id(p) not in fan_ids and id(p) not in msc_ids]

        return [
            {'params': head_params, 'lr': head_lr,     'name': 'head'},
            {'params': bb_params,   'lr': backbone_lr, 'name': 'backbone'},
            {'params': fan_params,  'lr': fan_lr,      'name': 'fan'},
            {'params': msc_params,  'lr': msc_lr,      'name': 'msc'},
        ]
