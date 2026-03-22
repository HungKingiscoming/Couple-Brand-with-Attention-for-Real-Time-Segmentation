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
# GCBlock classes (giÃ¡Â»Â¯ nguyÃƒÂªn tÃ¡Â»Â« code gÃ¡Â»â€˜c Ã¢â‚¬â€ hÃ¡Â»â€” trÃ¡Â»Â£ deploy)
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
# DWSA + MultiScale
# ===========================

def _get_valid_groups(channels, desired_groups):
    if desired_groups <= 1:
        return 1
    g = min(desired_groups, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


def _partition_windows(x: Tensor, ws: int) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Chia feature map thÃƒ nh cÃƒÂ¡c windows khÃƒÂ´ng chÃ¡Â»â€œng lÃ¡ÂºÂ·p.
    Args:
        x  : (B, C, H, W)
        ws : window size
    Returns:
        windows : (B * nH * nW, C, ws, ws)
        (nH, nW): sÃ¡Â»â€˜ windows theo chiÃ¡Â»Âu H vÃƒ  W
    """
    B, C, H, W = x.shape
    nH, nW = H // ws, W // ws
    # (B, C, nH, ws, nW, ws) Ã¢â€ â€™ (B, nH, nW, C, ws, ws) Ã¢â€ â€™ (B*nH*nW, C, ws, ws)
    x = x.view(B, C, nH, ws, nW, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(B * nH * nW, C, ws, ws)
    return windows, (nH, nW)


def _merge_windows(windows: Tensor, nH: int, nW: int, B: int) -> Tensor:
    """
    GhÃƒÂ©p windows lÃ¡ÂºÂ¡i thÃƒ nh feature map.
    Args:
        windows: (B * nH * nW, C, ws, ws)
    Returns:
        x      : (B, C, nH*ws, nW*ws)
    """
    _, C, ws, _ = windows.shape
    x = windows.view(B, nH, nW, C, ws, ws)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, nH * ws, nW * ws)


class DWSABlock(nn.Module):
    def __init__(self, channels, num_heads=2, drop=0.0, reduction=4,
                 qk_sharing=True, groups=4, alpha=0.0,
                 window_size: int = 0):

        super().__init__()
        assert channels % reduction == 0
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size

        reduced = channels // reduction
        mid = max(reduced // 2, num_heads)
        self.reduced = reduced
        self.mid = mid

        # BN trÃ†Â°Ã¡Â»â€ºc in_proj
        self.bn_in = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        # BN sau out_proj
        self.bn_out = nn.BatchNorm2d(channels)

        g = _get_valid_groups(reduced, groups)

        self.qk_sharing = qk_sharing
        if qk_sharing:
            self.qk_base = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=False)
            self.q_head  = nn.Conv1d(mid, mid, kernel_size=1, bias=True)
            self.k_head  = nn.Conv1d(mid, mid, kernel_size=1, bias=True)
        else:
            self.q_proj = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=True)
            self.k_proj = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=True)

        self.v_proj = nn.Conv1d(reduced, mid, kernel_size=1, groups=g, bias=True)
        self.o_proj = nn.Conv1d(mid, reduced, kernel_size=1, groups=g, bias=True)

        self.drop  = nn.Dropout(drop)
        self.scale = (mid // num_heads) ** -0.5
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _attention(self, x_flat: Tensor) -> Tensor:
        """
        Core attention computation.
        Args:
            x_flat: (B', reduced, N)  Ã¢â‚¬â€ B' = B*nH*nW khi dÃƒÂ¹ng window attention
        Returns:
            out   : (B', reduced, N)
        """
        if self.qk_sharing:
            base = self.qk_base(x_flat)
            q = self.q_head(base)
            k = self.k_head(base)
        else:
            q = self.q_proj(x_flat)
            k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        def split_heads(t):
            B_, Cm, N = t.shape
            hd = Cm // self.num_heads
            return t.view(B_, self.num_heads, hd, N).permute(0, 1, 3, 2)
            # Ã¢â€ â€™ (B', heads, N, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.training:
            attn = attn.clamp(-10, 10)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)                          # (B', heads, N, hd)
        out = out.permute(0, 1, 3, 2).contiguous()          # (B', heads, hd, N)
        B_, Hn, Hd, N = out.shape
        return out.view(B_, self.mid, N)                     # (B', mid, N)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        identity = x

        # Normalize + project xuÃ¡Â»â€˜ng reduced dim
        x_norm = self.bn_in(x)
        x_red  = self.in_proj(x_norm)   # (B, reduced, H, W)

        if self.window_size > 0:
            # Ã¢â€â‚¬Ã¢â€â‚¬ Window attention (stage4) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
            ws = self.window_size

            # Pad nÃ¡ÂºÂ¿u H, W khÃƒÂ´ng chia hÃ¡ÂºÂ¿t cho ws
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x_red = F.pad(x_red, (0, pad_w, 0, pad_h))
            Hp, Wp = x_red.shape[2], x_red.shape[3]

            # Partition Ã¢â€ â€™ (B*nH*nW, reduced, ws, ws)
            windows, (nH, nW) = _partition_windows(x_red, ws)
            Bw, C2, _, _ = windows.shape
            x_flat = windows.view(Bw, C2, ws * ws)  # (B*nH*nW, reduced, wsÃ‚Â²)

            # Attention trong tÃ¡Â»Â«ng window Ã„â€˜Ã¡Â»â„¢c lÃ¡ÂºÂ­p
            out_flat = self._attention(x_flat)       # (B*nH*nW, mid, wsÃ‚Â²)

            # Project back
            out_flat = self.o_proj(out_flat)         # (B*nH*nW, reduced, wsÃ‚Â²)
            out_win  = out_flat.view(Bw, C2, ws, ws)

            # Merge windows Ã¢â€ â€™ (B, reduced, Hp, Wp)
            out_red = _merge_windows(out_win, nH, nW, B)

            # Crop padding nÃ¡ÂºÂ¿u cÃƒÂ³
            if pad_h > 0 or pad_w > 0:
                out_red = out_red[:, :, :H, :W]

        else:
            # Ã¢â€â‚¬Ã¢â€â‚¬ Full attention (stage5, stage6) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
            N = H * W
            x_flat   = x_red.view(B, self.reduced, N)
            out_flat = self._attention(x_flat)       # (B, mid, N)
            out_flat = self.o_proj(out_flat)         # (B, reduced, N)
            out_red  = out_flat.view(B, self.reduced, H, W)

        # Project back lÃƒÂªn channels + BN
        out = self.bn_out(self.out_proj(out_red))    # (B, C, H, W)

        alpha = torch.sigmoid(self.alpha)
        return identity + alpha * out


class MultiScaleContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(1, 2),
                 branch_ratio=8, alpha=0.0,align_corners=False):
        super().__init__()
        self.scales = scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        total_branch_channels = max(in_channels // branch_ratio, len(scales))
        base = total_branch_channels // len(scales)
        extra = total_branch_channels % len(scales)

        per_branch_list = []
        for i in range(len(scales)):
            c = base + (1 if i < extra else 0)
            per_branch_list.append(max(c, 1))
        fused_channels = sum(per_branch_list)

        self.scale_branches = nn.ModuleList()
        for s, c_out in zip(scales, per_branch_list):
            if s == 1:
                self.scale_branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, c_out, kernel_size=1, bias=False),
                        nn.BatchNorm2d(c_out),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.scale_branches.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=s, stride=s),
                        nn.Conv2d(in_channels, c_out, kernel_size=1, bias=False),
                        nn.BatchNorm2d(c_out),
                        nn.ReLU(inplace=True),
                    )
                )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                fused_channels, fused_channels,
                kernel_size=3, padding=1,
                groups=fused_channels, bias=False,
            ),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.alpha = nn.Parameter(torch.tensor(alpha))

        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = None

    def forward(self, x):
        B, C, H, W = x.shape
        outs = []
        for s, branch in zip(self.scales, self.scale_branches):
            o = branch(x)
            if o.shape[-2:] != (H, W):
                o = F.interpolate(o, size=(H, W), mode='bilinear', align_corners=self.align_corners)
            outs.append(o)

        fused = torch.cat(outs, dim=1)
        out = self.fusion(fused)

        x_proj = self.proj(x) if self.proj is not None else x
        alpha = self.alpha.clamp(0.0, 1.0)
        return x_proj + alpha * out



class GCNetCore(BaseModule):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
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
                kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            *[GCBlock(
                in_channels=channels, out_channels=channels, stride=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy
            ) for _ in range(num_blocks_per_stage[0])],
            GCBlock(
                in_channels=channels, out_channels=channels * 2, stride=2,
                norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(
                in_channels=channels * 2, out_channels=channels * 2, stride=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy
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

        self.down_c4 = nn.Sequential(
            ConvModule(C*2, C*4, kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*4, C*8, kernel_size=1,     # thêm dòng này
                       norm_cfg=norm_cfg, act_cfg=None)
        )
        
        self.down_c4 = ConvModule(
            channels * 2, channels * 4, kernel_size=3,
            stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=None
        )
        
        self.comp_c5 = ConvModule(
            channels * 8, channels * 2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None
        )
        
        self.down_c5 = nn.Sequential(
            ConvModule(C*2, C*4, kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*4, C*8, kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*8, C*16, kernel_size=1,    # thêm dòng này
                       norm_cfg=norm_cfg, act_cfg=None)
        )


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

    def forward_stem(self, x: Tensor):
        """Stem: trÃ¡ÂºÂ£ vÃ¡Â»Â (feat, c1, c2, out_size) Ã„â€˜Ã¡Â»Æ’ GCNetWithEnhance inject DWSA giÃ¡Â»Â¯a stages."""
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        c1 = c2 = None
        feat = x
        for i, layer in enumerate(self.stem):
            feat = layer(feat)
            if i == 0: c1 = feat   # H/2, C
            if i == 1: c2 = feat   # H/4, C
        return feat, c1, c2, out_size

    def forward_stage4(self, x: Tensor, out_size: Tuple) -> Tuple[Tensor, Tensor]:
        x_s4 = self.semantic_branch_layers[0](x)
        x_d4 = self.detail_branch_layers[0](x)
        comp_c4 = self.comp_c4(self.relu(x_s4))
        x_d4 = x_d4 + resize(
            comp_c4,
            size=x_d4.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return x_s4, x_d4

    def forward_stage5(self, x_s4: Tensor, x_d4: Tensor, out_size: Tuple):
        # Cache relu một lần
        relu_s4 = self.relu(x_s4)
    
        x_s5 = self.semantic_branch_layers[1](x_s4)
        x_d5 = self.detail_branch_layers[1](x_d4)
    
        # semantic → detail
        comp_c5 = self.comp_c5(self.relu(x_s5))
        x_d5 = x_d5 + F.interpolate(
            comp_c5,
            size=x_d5.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
    
        # detail → semantic — dùng relu_s4 đã cache
        comp_c4 = self.comp_c4(relu_s4)
        down_c4 = self.down_c4(comp_c4)
        if down_c4.shape[-2:] != x_s5.shape[-2:]:
            down_c4 = F.interpolate(
                down_c4,
                size=x_s5.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        x_s5 = x_s5 + down_c4
    
        return x_s5, x_d5

    def forward_stage6(self, x_s5: Tensor, x_d5: Tensor) -> Tuple[Tensor, Tensor]:
        # Cache relu một lần duy nhất
        relu_s5 = self.relu(x_s5)
        relu_d5 = self.relu(x_d5)

        x_d6 = self.detail_branch_layers[2](relu_d5)
        x_s6 = self.semantic_branch_layers[2](relu_s5)

        # comp_c5 tính MỘT LẦN, tái dùng cho cả hai nhánh
        comp_c5 = self.comp_c5(relu_s5)

        # semantic → detail
        x_d6 = x_d6 + F.interpolate(
            comp_c5,
            size=x_d6.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        # detail → semantic
        down_c5 = self.down_c5(comp_c5)
        if down_c5.shape[-2:] != x_s6.shape[-2:]:
            down_c5 = F.interpolate(
                down_c5,
                size=x_s6.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        x_s6 = x_s6 + down_c5

        return x_s6, x_d6

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Standard forward Ã¢â‚¬â€ dÃƒÂ¹ng khi standalone (khÃƒÂ´ng cÃƒÂ³ GCNetWithEnhance bÃ¡Â»Âc ngoÃƒ i).
        DWSA vÃƒ  SPP khÃƒÂ´ng Ã„â€˜Ã†Â°Ã¡Â»Â£c apply Ã¡Â»Å¸ Ã„â€˜ÃƒÂ¢y.
        """
        feat, c1, c2, out_size = self.forward_stem(x)
        x_s4, x_d4 = self.forward_stage4(feat, out_size)
        c4 = x_d4
        x_s5, x_d5 = self.forward_stage5(x_s4, x_d4, out_size)
        x_s6, x_d6 = self.forward_stage6(x_s5, x_d5)
        return dict(
            c1=c1, c2=c2, c4=c4,
            x_s4=x_s4, x_s5=x_s5, x_s6=x_s6,
            x_d6=x_d6,
        )

    def switch_to_deploy(self):
        # 1. Fuse tất cả GCBlock thành single conv
        self.backbone.switch_to_deploy()
    
        # 2. Tắt dropout trong tất cả DWSABlock
        for dwsa in [self.dwsa4, self.dwsa5, self.dwsa6]:
            if dwsa is not None:
                dwsa.drop = nn.Identity()
    
        # 3. Tắt dropout trong MultiScaleContextModule nếu có
        # (không có dropout ở đây nhưng để sẵn cho tương lai)
    
        self.deploy = True


# ===========================
# GCNetWithEnhance Ã¢â‚¬â€ FIXED:
#   - SPP chÃ¡Â»â€° tÃƒÂ­nh MÃ¡Â»ËœT LÃ¡ÂºÂ¦N, sau DWSA6
#   - switch_to_deploy() hoÃƒ n chÃ¡Â»â€°nh cho cÃ¡ÂºÂ£ DWSA + GCNetCore
#   - DWSA Ã„â€˜Ã†Â°Ã¡Â»Â£c bypass khi deploy (khÃƒÂ´ng cÃƒÂ³ tÃƒÂ¡c dÃ¡Â»Â¥ng trong inference
#     vÃƒÂ¬ alpha Ã„â€˜ÃƒÂ£ learned, cÃƒÂ³ thÃ¡Â»Æ’ fold vÃƒ o final_proj)
# ===========================

class GCNetWithEnhance(BaseModule):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
                 dwsa_stages: List[str] = ('stage4', 'stage5', 'stage6'),
                 dwsa_num_heads: int = 4,
                 dwsa_reduction: int = 4,
                 dwsa_qk_sharing: bool = True,
                 dwsa_groups: int = 4,
                 dwsa_drop: float = 0.1,
                 dwsa_alpha: float = 0.1,
                 # stage4 dÃƒÂ¹ng window attention Ã„â€˜Ã¡Â»Æ’ trÃƒÂ¡nh OOM (H/16 Ã¢â€ â€™ N=1024)
                 # window_size=8 Ã¢â€ â€™ N=64 per window, memory ~0.25MB vs 64MB full
                 dwsa4_window_size: int = 4,
                 use_multi_scale_context: bool = True,
                 ms_scales: Tuple[int, ...] = (1, 2),
                 ms_branch_ratio: int = 16,
                 ms_alpha: float = 0.1,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        self.align_corners = align_corners
        self.channels = channels
        self.deploy = deploy
        self.dwsa4_window_size = dwsa4_window_size

        valid_stages = {'stage4', 'stage5', 'stage6'}
        invalid = set(dwsa_stages) - valid_stages
        if invalid:
            raise ValueError(f"Invalid dwsa_stages: {invalid}. Valid: {valid_stages}")
        self.c5_gate = ConvModule(
            in_channels=C * 4 * 2,   # x_d6(128) + x_spp(128) = 256
            out_channels=C * 4,       # gate shape = 128
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None              # sigmoid apply thủ công trong forward
        )
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
                    window_size=dwsa4_window_size,  # window attention Ã¢â‚¬â€ trÃƒÂ¡nh OOM
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
                    window_size=0,  # full attention Ã¢â‚¬â€ N=256, safe
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
                    window_size=0,  # full attention Ã¢â‚¬â€ N=64, trivial
                )

        if use_multi_scale_context:
            self.ms_context = MultiScaleContextModule(
                C * 4, C * 4,
                scales=ms_scales,
                branch_ratio=ms_branch_ratio,
                alpha=ms_alpha,
                align_corners=align_corners,    # truyền xuống
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
        bb = self.backbone
        feat, c1, c2, out_size = bb.forward_stem(x)

       
        x_s4, x_d4 = bb.forward_stage4(feat, out_size)
        c4 = x_d4 
        if self.dwsa4 is not None:
            x_s4 = self.dwsa4(x_s4)

       
        x_s5, x_d5 = bb.forward_stage5(x_s4, x_d4, out_size)
        if self.dwsa5 is not None:
            x_s5 = self.dwsa5(x_s5)
        x_s6, x_d6 = bb.forward_stage6(x_s5, x_d5)

        if self.dwsa6 is not None:
            x_s6 = self.dwsa6(x_s6)

        x_spp = bb.spp(x_s6)
        x_spp = resize(x_spp, size=out_size, mode='bilinear', align_corners=True)

        if self.ms_context is not None:
            x_spp = self.ms_context(x_spp)

        x_spp = self.final_proj(x_spp)

        gate = torch.sigmoid(self.c5_gate(torch.cat([x_d6, x_spp], dim=1)))
        c5 = x_d6 + gate * x_spp

        return dict(
            c1=c1,   # H/2, C   = 32  Ã¢â‚¬â€ decoder skip (stem layer 0)
            c2=c2,   # H/4, C   = 32  Ã¢â‚¬â€ decoder skip (stem layer 1)
            c4=c4,   # H/8, C*2 = 64  Ã¢â‚¬â€ detail branch: aux head + decoder skip stage0
            c5=c5,   # H/8, C*4 = 128 Ã¢â‚¬â€ fused output: main decoder input
        )

    def switch_to_deploy(self):
        # Fuse GCNetCore
        self.backbone.switch_to_deploy()

        # Mark deploy
        self.deploy = True

        

    @torch.no_grad()
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        backbone_core = sum(p.numel() for p in self.backbone.parameters()
                           if not any(p is sp for sp in self.backbone.spp.parameters()))
        spp = sum(p.numel() for p in self.backbone.spp.parameters())
        dwsa = sum(
            p.numel() for m in [self.dwsa4, self.dwsa5, self.dwsa6]
            if m is not None for p in m.parameters()
        )
        ms = sum(p.numel() for p in self.ms_context.parameters()) if self.ms_context else 0
        proj = sum(p.numel() for p in self.final_proj.parameters())

       
        return total
