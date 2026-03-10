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

class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d, không activation — dùng bên trong GCBlock."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,   # bias=False vì BN absorb bias
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

    def fuse(self) -> Tuple[Tensor, Tensor]:
        """Trả về (fused_weight, fused_bias) để dùng trong switch_to_deploy."""
        return _fuse_conv_bn(self.conv.weight, None, self.bn)
# ===========================
# GCBlock classes (giữ nguyên từ code gốc — hỗ trợ deploy)
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
_PATHS = {
    'stem_same':       3,
    'stem_down':       3,
    'detail_same':     3,
    'detail_down':     3,
    'semantic_0_down': 3,
    'semantic_0_same': 4,
    'semantic_1_down': 4,
    'semantic_1_same': 4,
    'semantic_2_down': 4,
    'semantic_2_same': 5,
}

class GCBlock(nn.Module):
    """
    Multi-path Re-parameterizable Conv Block.

    Training mode:
        out = act(Σ_{i=1}^{N} BN_i(W_i^{3x3} * x)
                  + BN_{1x1}(W^{1x1} * x)
                  + BN_{id}(x)           # chỉ khi same channels & stride=1
                 )

    Deploy mode (sau switch_to_deploy()):
        out = act(W_fused * x + b_fused)  # 1 Conv3x3, no BN

    Args:
        num_3x3_paths: số lượng parallel 3x3 paths.
                       Gợi ý:
                         stem/detail_branch → 3
                         semantic_branch[0] → 4
                         semantic_branch[1] → 4
                         semantic_branch[2] → 5
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1,
                 act: bool = True,
                 num_3x3_paths: int = 4,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 act_cfg: dict = dict(type='ReLU', inplace=True),
                 deploy: bool = False):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.num_3x3_paths = num_3x3_paths
        self.deploy       = deploy

        # Activation
        if act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

        if deploy:
            # ── Deploy mode: single fused conv ──────────────────────────
            self.reparam_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=padding,
                bias=True,
            )
        else:
            # ── Training mode: multi-path ────────────────────────────────

            # N paths 3x3
            self.paths_3x3 = nn.ModuleList([
                ConvBN(in_channels, out_channels,
                       kernel_size=3, stride=stride, padding=padding)
                for _ in range(num_3x3_paths)
            ])

            # 1 path 1x1 (padding=0, fuse bằng cách pad kernel)
            self.path_1x1 = ConvBN(in_channels, out_channels,
                                   kernel_size=1, stride=stride, padding=0)

            # Identity path (BN only) — chỉ khi same channels & stride=1
            if in_channels == out_channels and stride == 1:
                self.path_identity = nn.BatchNorm2d(in_channels)
            else:
                self.path_identity = None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.act(self.reparam_conv(x))

        # Sum tất cả paths
        out = sum(path(x) for path in self.paths_3x3)
        out = out + self.path_1x1(x)

        if self.path_identity is not None:
            out = out + self.path_identity(x)

        return self.act(out)

    # ── Re-parameterization ───────────────────────────────────────────────────

    def _fuse_path_3x3(self, convbn: ConvBN) -> Tuple[Tensor, Tensor]:
        """Fuse 1 ConvBN(3x3) → (kernel_3x3, bias)."""
        return convbn.fuse()

    def _fuse_path_1x1(self) -> Tuple[Tensor, Tensor]:
        """Fuse ConvBN(1x1) → (kernel_3x3, bias) bằng cách pad 1x1 → 3x3."""
        k, b = self.path_1x1.fuse()
        return _pad_kernel_to_3x3(k), b

    def _fuse_path_identity(self) -> Tuple[Tensor, Tensor]:
        """
        Fuse identity path (BN only) → (kernel_3x3, bias).

        Identity = x, tương đương Conv với kernel = I (identity matrix).
        I ở dạng 3x3: pixel trung tâm = 1, còn lại = 0.
        """
        if self.path_identity is None:
            device = next(self.parameters()).device
            zero_k = torch.zeros(
                self.out_channels, self.in_channels, 3, 3,
                device=device
            )
            zero_b = torch.zeros(self.out_channels, device=device)
            return zero_k, zero_b

        bn = self.path_identity
        # Tạo identity kernel: (out_ch, in_ch, 3, 3)
        # với kernel[i, i, 1, 1] = 1
        input_ch = self.in_channels
        kernel_value = torch.zeros(
            self.in_channels, input_ch, 3, 3,
            device=bn.weight.device
        )
        for i in range(self.in_channels):
            kernel_value[i, i, 1, 1] = 1.0

        return _fuse_conv_bn(kernel_value, None, bn)

    @torch.no_grad()
    def switch_to_deploy(self):
        """
        Fuse tất cả paths về 1 Conv3x3 + bias duy nhất.

        Math:
            W_fused = Σ W_i^{3x3}_fused + pad(W^{1x1}_fused) + W^{id}_fused
            b_fused = Σ b_i^{3x3}_fused + b^{1x1}_fused      + b^{id}_fused
        """
        if self.deploy:
            return  # already deployed

        device = next(self.parameters()).device

        # Accumulate fused kernels and biases
        kernel_sum = torch.zeros(
            self.out_channels, self.in_channels, 3, 3, device=device
        )
        bias_sum = torch.zeros(self.out_channels, device=device)

        # Fuse N paths 3x3
        for convbn in self.paths_3x3:
            k, b = self._fuse_path_3x3(convbn)
            kernel_sum += k
            bias_sum   += b

        # Fuse path 1x1
        k1, b1 = self._fuse_path_1x1()
        kernel_sum += k1
        bias_sum   += b1

        # Fuse identity path
        kid, bid = self._fuse_path_identity()
        kernel_sum += kid
        bias_sum   += bid

        # Tạo conv fused
        self.reparam_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, stride=self.stride, padding=self.padding,
            bias=True,
        ).to(device)
        self.reparam_conv.weight.data = kernel_sum
        self.reparam_conv.bias.data   = bias_sum

        # Xóa training-only modules để giảm memory
        del self.paths_3x3
        del self.path_1x1
        if hasattr(self, 'path_identity'):
            del self.path_identity

        self.deploy = True

    def extra_repr(self) -> str:
        if self.deploy:
            return (f"in={self.in_channels}, out={self.out_channels}, "
                    f"stride={self.stride}, deploy=True")
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"stride={self.stride}, num_3x3_paths={self.num_3x3_paths}")


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
    Chia feature map thành các windows không chồng lặp.
    Args:
        x  : (B, C, H, W)
        ws : window size
    Returns:
        windows : (B * nH * nW, C, ws, ws)
        (nH, nW): số windows theo chiều H và W
    """
    B, C, H, W = x.shape
    nH, nW = H // ws, W // ws
    # (B, C, nH, ws, nW, ws) → (B, nH, nW, C, ws, ws) → (B*nH*nW, C, ws, ws)
    x = x.view(B, C, nH, ws, nW, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(B * nH * nW, C, ws, ws)
    return windows, (nH, nW)


def _merge_windows(windows: Tensor, nH: int, nW: int, B: int) -> Tensor:
    """
    Ghép windows lại thành feature map.
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
    """
    Depthwise Separable Attention with optional Window Attention.

    Fixes vs original:
      - alpha is nn.Parameter (was register_buffer → grad was always None)
      - o_proj runs in fp32 alongside attention (was running in fp16 → precision mismatch)
      - No manual clamp on attention logits (fp32 softmax is numerically stable)
    """
    def __init__(self, channels, num_heads=2, drop=0.0, reduction=4,
                 qk_sharing=True, groups=4, alpha=0.01, window_size=0):
        super().__init__()
        assert channels % reduction == 0
        self.channels    = channels
        self.num_heads   = num_heads
        self.window_size = window_size

        reduced     = channels // reduction
        mid         = max(reduced // 2, num_heads)
        self.reduced = reduced
        self.mid     = mid

        self.bn_in   = nn.BatchNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, reduced, 1, bias=False)
        self.out_proj = nn.Conv2d(reduced, channels, 1, bias=False)
        self.bn_out  = nn.BatchNorm2d(channels)

        g = _get_valid_groups(reduced, groups)

        self.qk_sharing = qk_sharing
        if qk_sharing:
            self.qk_base = nn.Conv1d(reduced, mid, 1, groups=g, bias=False)
            self.q_head  = nn.Conv1d(mid, mid, 1, bias=True)
            self.k_head  = nn.Conv1d(mid, mid, 1, bias=True)
        else:
            self.q_proj = nn.Conv1d(reduced, mid, 1, groups=g, bias=True)
            self.k_proj = nn.Conv1d(reduced, mid, 1, groups=g, bias=True)

        self.v_proj = nn.Conv1d(reduced, mid, 1, groups=g, bias=True)

        # FIX 2: o_proj is now called inside _attention in fp32
        # keeping it here as nn.Conv1d so parameters are registered normally
        self.o_proj = nn.Conv1d(mid, reduced, 1, groups=g, bias=True)

        self.drop  = nn.Dropout(drop)
        self.scale = (mid // num_heads) ** -0.5

        # FIX 1: was register_buffer → grad always None → module never learned
        # init small (0.01) so residual connection dominates early training
        self.alpha = nn.Parameter(torch.full((channels,), alpha))

    def _attention(self, x_flat: Tensor) -> Tensor:
        """
        Full attention computation in fp32.
        x_flat: (B, reduced, N)  — may be fp16 on entry
        returns: (B, reduced, N) — same dtype as input
        """
        orig_dtype = x_flat.dtype
        x_fp32 = x_flat.float()   # cast to fp32

        if self.qk_sharing:
            base = self.qk_base(x_fp32)
            q    = self.q_head(base)
            k    = self.k_head(base)
        else:
            q = self.q_proj(x_fp32)
            k = self.k_proj(x_fp32)
        v = self.v_proj(x_fp32)

        def split_heads(t):
            B_, Cm, N = t.shape
            hd = Cm // self.num_heads
            return t.view(B_, self.num_heads, hd, N).permute(0, 1, 3, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # fp32 softmax — numerically stable, no manual clamp needed
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)                        # (B, heads, N, hd)
        out = out.permute(0, 1, 3, 2).contiguous()
        B_, Hn, Hd, N = out.shape
        out = out.view(B_, self.mid, N)                    # (B, mid, N)

        # FIX 2: o_proj runs in fp32 here (not in fp16 after cast-back)
        out = self.o_proj(out)                             # (B, reduced, N) in fp32

        return out.to(orig_dtype)                          # cast back to input dtype

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        identity = x

        x_norm = self.bn_in(x)
        x_red  = self.in_proj(x_norm)   # (B, reduced, H, W)

        if self.window_size > 0:
            ws    = self.window_size
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x_red = F.pad(x_red, (0, pad_w, 0, pad_h))
            windows, (nH, nW) = _partition_windows(x_red, ws)
            Bw, C2, _, _ = windows.shape
            out_flat = self._attention(windows.view(Bw, C2, ws * ws))
            out_red  = _merge_windows(out_flat.view(Bw, C2, ws, ws), nH, nW, B)
            if pad_h > 0 or pad_w > 0:
                out_red = out_red[:, :, :H, :W]
        else:
            N       = H * W
            out_flat = self._attention(x_red.view(B, self.reduced, N))
            out_red  = out_flat.view(B, self.reduced, H, W)

        out = self.bn_out(self.out_proj(out_red))

        # alpha is a Parameter now — clamped [0,1] only for stability
        alpha = self.alpha.clamp(0.0, 1.0).view(1, -1, 1, 1)
        return identity + alpha * out

class MultiScaleContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(1, 2),
                 branch_ratio=16, alpha=0.1):
        super().__init__()
        self.scales      = scales
        self.in_channels  = in_channels
        self.out_channels = out_channels

        total = max(in_channels // branch_ratio, len(scales))
        base  = total // len(scales)
        extra = total % len(scales)
        per   = [max(base + (1 if i < extra else 0), 1) for i in range(len(scales))]
        fused = sum(per)

        self.scale_branches = nn.ModuleList()
        for s, c in zip(scales, per):
            if s == 1:
                self.scale_branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, c, 1, bias=False),
                    nn.BatchNorm2d(c), nn.ReLU(inplace=True)))
            else:
                self.scale_branches.append(nn.Sequential(
                    nn.AvgPool2d(s, stride=s),
                    nn.Conv2d(in_channels, c, 1, bias=False),
                    nn.BatchNorm2d(c), nn.ReLU(inplace=True)))

        self.fusion = nn.Sequential(
            nn.Conv2d(fused, fused, 3, padding=1, groups=fused, bias=False),
            nn.BatchNorm2d(fused), nn.ReLU(inplace=True),
            nn.Conv2d(fused, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        # init 0.1 instead of 1e-4 so module contributes from early epochs
        self.alpha = nn.Parameter(torch.full((out_channels,), 0.1))

        self.proj = (nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else None)

    def forward(self, x):
        B, C, H, W = x.shape
        outs = []
        for s, branch in zip(self.scales, self.scale_branches):
            o = branch(x)
            if o.shape[-2:] != (H, W):
                o = F.interpolate(o, (H, W), mode='bilinear', align_corners=False)
            outs.append(o)
        fused = torch.cat(outs, dim=1)
        out   = self.fusion(fused)
        x_proj = self.proj(x) if self.proj is not None else x
        alpha  = self.alpha.clamp(0.0, 1.0).view(1, -1, 1, 1)
        return x_proj + alpha * out


# ===========================
# GCNetCore — FIXED:
#   - KHÔNG tính SPP ở đây nữa, trả về s6 raw
#   - c4 dùng clone() tránh gradient corruption
#   - Giữ nguyên switch_to_deploy() từ gốc
# ===========================

class GCNetCore(BaseModule):
    """
    Bilateral segmentation backbone với multi-path re-parameterizable blocks.

    Channels layout (channels=32):
        stem output    : H/8,  C*2  = 64
        semantic[0]    : H/16, C*4  = 128
        semantic[1]    : H/32, C*8  = 256
        semantic[2]    : H/64, C*16 = 512
        detail[0,1]    : H/8,  C*2  = 64
        detail[2]      : H/8,  C*4  = 128
        spp output     : H/8,  C*4  = 128

    num_3x3_paths được cấu hình qua _PATHS dict ở trên.
    Có thể override bằng tham số paths_cfg.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False,
                 paths_cfg: Optional[Dict[str, int]] = None):
        super().__init__(init_cfg)

        self.in_channels  = in_channels
        self.channels     = channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg  = act_cfg
        self.deploy   = deploy

        # Merge với default paths
        p = dict(_PATHS)
        if paths_cfg:
            p.update(paths_cfg)

        C = channels   # alias ngắn gọn

        # ── Stem ──────────────────────────────────────────────────────────────
        # Cấu trúc: ConvModule(s2) → ConvModule(s2) → GCBlock×4 → GCBlock(s2) → GCBlock×3
        # 2 ConvModule đầu: stride conv, không phải GCBlock
        # GCBlock trong stem: spatial H/4–H/8 → dùng ít paths hơn
        self.stem = nn.Sequential(
            # i=0: stride=2, H → H/2
            ConvModule(
                in_channels=in_channels, out_channels=C,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            # i=1: stride=2, H/2 → H/4
            ConvModule(
                in_channels=C, out_channels=C,
                kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            # i=2..5: GCBlock same channels, stride=1 tại H/4
            *[GCBlock(
                in_channels=C, out_channels=C, stride=1,
                num_3x3_paths=p['stem_same'],
                norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy,
            ) for _ in range(num_blocks_per_stage[0])],
            # i=6: GCBlock stride=2, H/4 → H/8, channels C → C*2
            GCBlock(
                in_channels=C, out_channels=C * 2, stride=2,
                num_3x3_paths=p['stem_down'],
                norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            # i=7..9: GCBlock same channels, stride=1 tại H/8
            *[GCBlock(
                in_channels=C * 2, out_channels=C * 2, stride=1,
                num_3x3_paths=p['stem_same'],
                norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy,
            ) for _ in range(num_blocks_per_stage[1] - 1)],
        )
        self.relu = build_activation_layer(act_cfg)

        # ── Semantic Branch ───────────────────────────────────────────────────
        # [0]: H/8  → H/16, C*2 → C*4
        # [1]: H/16 → H/32, C*4 → C*8
        # [2]: H/32 → H/64, C*8 → C*16
        self.semantic_branch_layers = nn.ModuleList()

        # semantic_branch[0]: H/16, channels = C*4
        self.semantic_branch_layers.append(nn.Sequential(
            GCBlock(C*2, C*4, stride=2,
                    num_3x3_paths=p['semantic_0_down'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C*4, C*4, stride=1,
                      num_3x3_paths=p['semantic_0_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[2][0] - 2)],
            # Block cuối: act=False (bilateral fusion cần raw feature trước ReLU)
            GCBlock(C*4, C*4, stride=1,
                    num_3x3_paths=p['semantic_0_same'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # semantic_branch[1]: H/32, channels = C*8
        self.semantic_branch_layers.append(nn.Sequential(
            GCBlock(C*4, C*8, stride=2,
                    num_3x3_paths=p['semantic_1_down'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C*8, C*8, stride=1,
                      num_3x3_paths=p['semantic_1_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[3][0] - 2)],
            GCBlock(C*8, C*8, stride=1,
                    num_3x3_paths=p['semantic_1_same'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # semantic_branch[2]: H/64, channels = C*16 — spatial nhỏ nhất
        self.semantic_branch_layers.append(nn.Sequential(
            GCBlock(C*8, C*16, stride=2,
                    num_3x3_paths=p['semantic_2_down'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C*16, C*16, stride=1,
                      num_3x3_paths=p['semantic_2_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[4][0] - 2)],
            GCBlock(C*16, C*16, stride=1,
                    num_3x3_paths=p['semantic_2_same'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # ── Detail Branch ─────────────────────────────────────────────────────
        # [0]: H/8, C*2 → C*2  (refinement, same spatial và channels)
        # [1]: H/8, C*2 → C*2
        # [2]: H/8, C*2 → C*4  (channel expansion cuối cùng)
        self.detail_branch_layers = nn.ModuleList()

        # detail_branch[0]: H/8, channels = C*2
        self.detail_branch_layers.append(nn.Sequential(
            *[GCBlock(C*2, C*2, stride=1,
                      num_3x3_paths=p['detail_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[2][1] - 1)],
            GCBlock(C*2, C*2, stride=1,
                    num_3x3_paths=p['detail_same'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # detail_branch[1]: H/8, channels = C*2
        self.detail_branch_layers.append(nn.Sequential(
            *[GCBlock(C*2, C*2, stride=1,
                      num_3x3_paths=p['detail_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[3][1] - 1)],
            GCBlock(C*2, C*2, stride=1,
                    num_3x3_paths=p['detail_same'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # detail_branch[2]: H/8, C*2 → C*4 (channel expansion)
        self.detail_branch_layers.append(nn.Sequential(
            GCBlock(C*2, C*4, stride=1,
                    num_3x3_paths=p['detail_down'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C*4, C*4, stride=1,
                      num_3x3_paths=p['detail_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[4][1] - 2)],
            GCBlock(C*4, C*4, stride=1,
                    num_3x3_paths=p['detail_same'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # ── Bilateral Fusion Connectors ───────────────────────────────────────
        # compression: semantic → detail (downsample channels để match detail branch)
        self.compression_1 = ConvModule(
            C*4, C*2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        # down: detail → semantic (upsample spatial để match semantic branch)
        self.down_1 = ConvModule(
            C*2, C*4, kernel_size=3,
            stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=None)

        self.compression_2 = ConvModule(
            C*8, C*2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(C*2, C*4, kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*4, C*8, kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=None))

        # ── DAPPM (SPP) ───────────────────────────────────────────────────────
        # Input: H/64, C*16 = 512
        # Output: H/8, C*4 = 128 (sau resize)
        self.spp = DAPPM(
            in_channels=C * 16,
            branch_channels=ppm_channels,
            out_channels=C * 4,
            num_scales=5,
            kernel_sizes=[3, 5, 7, 9],
            strides=[1, 2, 2, 4],
            paddings=[1, 2, 3, 4],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.kaiming_init()

    # ── Initialization ────────────────────────────────────────────────────────

    def kaiming_init(self):
        """
        Kaiming normal cho Conv2d, constant cho BN.
        ConvBN.conv cũng là nn.Conv2d nên được init tự động.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward_stem(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
        """
        Stem forward với fp32 cast cho 2 ConvModule stride đầu.

        Lý do fp32 cast (giữ nguyên từ gốc):
            - i=0,1: spatial H×W và H/2×W/2 rất lớn
            - AMP fp16 tích lũy gradient dễ overflow ở spatial lớn
            - → Force fp32 cho 2 layer đầu, cast về fp16 sau đó
            - Các GCBlock sau: spatial đã nhỏ hơn, fp16 OK

        Returns:
            feat   : (B, C*2, H/8, W/8)
            c1     : (B, C,   H/2, W/2)  — skip cho decoder
            c2     : (B, C,   H/4, W/4)  — skip cho decoder
            out_size: (H/8, W/8)         — target size cho bilateral fusion resize
        """
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        c1 = c2 = None
        feat = x

        for i, layer in enumerate(self.stem):
            if self.training and i <= 1:
                # Force fp32 cho ConvModule stride tại spatial lớn
                with torch.autocast(device_type='cuda', enabled=False):
                    feat = layer(feat.float())
                feat = feat.to(x.dtype)
            else:
                feat = layer(feat)

            if i == 0:
                c1 = feat   # H/2, C
            if i == 1:
                c2 = feat   # H/4, C

        return feat, c1, c2, out_size

    def forward_stage4(self, x: Tensor, out_size: Tuple) -> Tuple[Tensor, Tensor]:
        """
        Stage 4 bilateral fusion.

        Spatial flow:
            x     : H/4,  C*2
            x_s4  : H/16, C*4   ← semantic_branch[0]
            x_d4  : H/8,  C*2   ← detail_branch[0]

        Bilateral fusion:
            x_s4 ← x_s4 + down_1(relu(x_d4))      detail → semantic
            x_d4 ← x_d4 + resize(comp1(relu(x_s4))) semantic → detail

        Nếu DWSA4 được apply trên x_s4 sau khi return:
            → stage5 nhận x_s4 có semantic context tốt hơn
            → x_s5 tốt hơn → compression_2 tốt hơn → x_d5 tốt hơn
        """
        x_s4 = self.semantic_branch_layers[0](x)
        x_d4 = self.detail_branch_layers[0](x)
        comp_c4 = self.compression_1(self.relu(x_s4))
        x_s4 = x_s4 + self.down_1(self.relu(x_d4))
        x_d4 = x_d4 + resize(
            comp_c4, size=out_size, mode='bilinear',
            align_corners=self.align_corners)
        return x_s4, x_d4

    def forward_stage5(self, x_s4: Tensor, x_d4: Tensor,
                       out_size: Tuple) -> Tuple[Tensor, Tensor]:
        """
        Stage 5 bilateral fusion.

        Spatial flow:
            x_s4  : H/16, C*4   (đã qua DWSA4 nếu có)
            x_d4  : H/8,  C*2
            x_s5  : H/32, C*8   ← semantic_branch[1]
            x_d5  : H/8,  C*2   ← detail_branch[1]
        """
        x_s5 = self.semantic_branch_layers[1](self.relu(x_s4))
        x_d5 = self.detail_branch_layers[1](self.relu(x_d4))
        comp_c5 = self.compression_2(self.relu(x_s5))
        x_s5 = x_s5 + self.down_2(self.relu(x_d5))
        x_d5 = x_d5 + resize(
            comp_c5, size=out_size, mode='bilinear',
            align_corners=self.align_corners)
        return x_s5, x_d5

    def forward_stage6(self, x_s5: Tensor,
                       x_d5: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Stage 6.

        Spatial flow:
            x_s5  : H/32, C*8   (đã qua DWSA5 nếu có)
            x_d5  : H/8,  C*2
            x_s6  : H/64, C*16  ← semantic_branch[2] → DWSA6 → SPP
            x_d6  : H/8,  C*4   ← detail_branch[2]
        """
        x_d6 = self.detail_branch_layers[2](self.relu(x_d5))
        x_s6 = self.semantic_branch_layers[2](self.relu(x_s5))
        return x_s6, x_d6

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Standard forward — dùng khi standalone.
        GCNetWithEnhance override flow này để inject DWSA + SPP.
        """
        feat, c1, c2, out_size = self.forward_stem(x)
        x_s4, x_d4 = self.forward_stage4(feat, out_size)
        c4 = x_d4.clone()   # clone trước khi x_d4 bị dùng tiếp ở stage5
        x_s5, x_d5 = self.forward_stage5(x_s4, x_d4, out_size)
        x_s6, x_d6 = self.forward_stage6(x_s5, x_d5)
        return dict(
            c1=c1, c2=c2, c4=c4,
            x_s4=x_s4, x_s5=x_s5, x_s6=x_s6,
            x_d6=x_d6,
        )

    # ── Deploy ────────────────────────────────────────────────────────────────

    def switch_to_deploy(self):
        """
        Fuse tất cả GCBlock về single Conv3x3.
        Sau khi gọi:
            - Mỗi GCBlock: N paths + 1x1 + identity → 1 conv3x3
            - Inference cost = model với 1 conv3x3 thuần túy
            - Memory giảm ~(N+1)x so với training mode
        """
        count = 0
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
                count += 1
        self.deploy = True
        print(f"✅ Fused {count} GCBlock → single Conv3x3 each")


# ===========================
# GCNetWithEnhance — FIXED:
#   - SPP chỉ tính MỘT LẦN, sau DWSA6
#   - switch_to_deploy() hoàn chỉnh cho cả DWSA + GCNetCore
#   - DWSA được bypass khi deploy (không có tác dụng trong inference
#     vì alpha đã learned, có thể fold vào final_proj)
# ===========================

class GCNetWithEnhance(BaseModule):
    """
    Backbone wrapper: GCNetCore + DWSA stages + MultiScaleContext + final_proj.

    Output dict: {c1, c2, c4, c5}
      c1 : H/2,  C    = 32
      c2 : H/4,  C    = 32
      c4 : H/8,  C*2  = 64  (detail branch, aux head input)
      c5 : H/8,  C*4  = 128 (fused detail+semantic, main decoder input)
    """
    def __init__(self,
                 in_channels=3, channels=32, ppm_channels=128,
                 num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
                 dwsa_stages=('stage4', 'stage5', 'stage6'),
                 dwsa_num_heads=4, dwsa_reduction=4,
                 dwsa_qk_sharing=True, dwsa_groups=4,
                 dwsa_drop=0.1, dwsa_alpha=0.01,
                 dwsa4_window_size=8,
                 use_multi_scale_context=True,
                 ms_scales=(1, 2), ms_branch_ratio=16, ms_alpha=0.1,
                 align_corners=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None, deploy=False):
        super().__init__(init_cfg)

        self.align_corners = align_corners
        self.channels      = channels
        self.deploy        = deploy

        valid = {'stage4', 'stage5', 'stage6'}
        bad   = set(dwsa_stages) - valid
        if bad:
            raise ValueError(f"Invalid dwsa_stages: {bad}")

        self.backbone = GCNetCore(
            in_channels=in_channels, channels=channels,
            ppm_channels=ppm_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            align_corners=align_corners,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
            init_cfg=None, deploy=deploy)

        C = channels
        self.dwsa4 = self.dwsa5 = self.dwsa6 = None

        for stage in dwsa_stages:
            kw = dict(num_heads=dwsa_num_heads, reduction=dwsa_reduction,
                      qk_sharing=dwsa_qk_sharing, groups=dwsa_groups,
                      drop=dwsa_drop, alpha=dwsa_alpha)
            if stage == 'stage4':
                self.dwsa4 = DWSABlock(C*4,  window_size=dwsa4_window_size, **kw)
            elif stage == 'stage5':
                self.dwsa5 = DWSABlock(C*8,  window_size=0, **kw)
            elif stage == 'stage6':
                self.dwsa6 = DWSABlock(C*16, window_size=0, **kw)

        self.ms_context = (
            MultiScaleContextModule(C*4, C*4, scales=ms_scales,
                                    branch_ratio=ms_branch_ratio, alpha=ms_alpha)
            if use_multi_scale_context else None)

        self.final_proj = ConvModule(C*4, C*4, 1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        bb = self.backbone
        feat, c1, c2, out_size = bb.forward_stem(x)

        x_s4, x_d4 = bb.forward_stage4(feat, out_size)
        c4 = x_d4.clone()
        if self.dwsa4 is not None:
            x_s4 = self.dwsa4(x_s4)

        x_s5, x_d5 = bb.forward_stage5(x_s4, x_d4, out_size)
        if self.dwsa5 is not None:
            x_s5 = self.dwsa5(x_s5)

        x_s6, x_d6 = bb.forward_stage6(x_s5, x_d5)
        if self.dwsa6 is not None:
            x_s6 = self.dwsa6(x_s6)

        x_spp = bb.spp(x_s6)
        x_spp = resize(x_spp, size=out_size, mode='bilinear',
                       align_corners=self.align_corners)

        if self.ms_context is not None:
            x_spp = self.ms_context(x_spp)

        x_spp = self.final_proj(x_spp)
        c5    = x_d6 + x_spp

        return dict(c1=c1, c2=c2, c4=c4, c5=c5)

    def switch_to_deploy(self):
        self.backbone.switch_to_deploy()
        self.deploy = True
        print("Switched to deploy mode")
        print("  GCBlock: all paths fused -> single 3x3 conv")
        print("  DWSA: kept as-is (attention not fuseable)")

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

        print(f"\n{'='*50}")
        print(f"GCNetWithEnhance Parameter Count")
        print(f"{'='*50}")
        print(f"  GCNetCore (excl SPP): {backbone_core/1e6:.2f}M")
        print(f"  DAPPM (SPP):          {spp/1e6:.2f}M")
        print(f"  DWSA blocks:          {dwsa/1e6:.2f}M")
        print(f"  MultiScaleContext:    {ms/1e6:.2f}M")
        print(f"  final_proj:           {proj/1e6:.2f}M")
        print(f"{'='*50}")
        print(f"  TOTAL:                {total/1e6:.2f}M")
        print(f"{'='*50}\n")
        return total
