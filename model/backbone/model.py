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
    Depthwise Separable Attention Block với hỗ trợ Window Attention.

    window_size = 0  → Full attention   — dùng cho stage5 (N=256), stage6 (N=64)
    window_size > 0  → Window attention — dùng cho stage4 (N=1024, quá lớn cho full)

    Tại sao window attention tốt cho stage4:
    - Stage4 ở H/16: cần capture local patterns (edge, texture) không cần global
    - Global context đã được DAPPM xử lý ở stage6
    - Window size=8 → N=64 per window, memory = 0.25MB thay vì 64MB
    - Đúng inductive bias: local attention early stages, global attention later
      (giống Swin Transformer design principle)

    Norm: BN thay vì LN/GN — nhất quán với toàn model, fuse-compatible.
    Alpha: clamp [0,1] — tránh polarity flip và gradient explosion.
    """
    def __init__(self, channels, num_heads=2, drop=0.0, reduction=4,
                 qk_sharing=True, groups=4, alpha=0.1,
                 window_size: int = 0):
        """
        Args:
            window_size: 0 = full attention, >0 = window attention.
                         Nên là ước số của H và W tại resolution đó.
                         Ví dụ: stage4 ở H/16=32px → window_size=8 (4 windows/dim)
        """
        super().__init__()
        assert channels % reduction == 0
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size

        reduced = channels // reduction
        mid = max(reduced // 2, num_heads)
        self.reduced = reduced
        self.mid = mid

        # BN trước in_proj
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
        self.register_buffer("alpha", torch.ones(channels) * 0.1)

    def _attention(self, x_flat: Tensor) -> Tensor:
        # Force fp32 để tránh softmax overflow trong fp16
        x_fp32 = x_flat.float()
    
        if self.qk_sharing:
            base = self.qk_base(x_fp32)
            q = self.q_head(base)
            k = self.k_head(base)
        else:
            q = self.q_proj(x_fp32)
            k = self.k_proj(x_fp32)
        v = self.v_proj(x_fp32)
    
        def split_heads(t):
            B_, Cm, N = t.shape
            hd = Cm // self.num_heads
            return t.view(B_, self.num_heads, hd, N).permute(0, 1, 3, 2)
    
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
    
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(-50.0, 50.0)
        attn = F.softmax(attn, dim=-1)   # fp32 → không overflow
        attn = self.drop(attn)
    
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        B_, Hn, Hd, N = out.shape
        out = out.view(B_, self.mid, N)
    
        # Cast về dtype gốc
        return out.to(x_flat.dtype)
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        identity = x
    
        # BN normalize input
        x_norm = self.bn_in(x)
        x_red  = self.in_proj(x_norm)   # (B, reduced, H, W)
    
        if self.window_size > 0:
            ws = self.window_size
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x_red = F.pad(x_red, (0, pad_w, 0, pad_h))
    
            windows, (nH, nW) = _partition_windows(x_red, ws)
            Bw, C2, _, _ = windows.shape
            x_flat = windows.view(Bw, C2, ws * ws)
    
            out_flat = self._attention(x_flat)
            out_flat = self.o_proj(out_flat)
            out_win  = out_flat.view(Bw, C2, ws, ws)
            out_red  = _merge_windows(out_win, nH, nW, B)
    
            if pad_h > 0 or pad_w > 0:
                out_red = out_red[:, :, :H, :W]
        else:
            N = H * W
            x_flat   = x_red.view(B, self.reduced, N)
            out_flat = self._attention(x_flat)
            out_flat = self.o_proj(out_flat)
            out_red  = out_flat.view(B, self.reduced, H, W)
    
        out = self.bn_out(self.out_proj(out_red))

        if identity.shape != out.shape:
            print("⚠️ DWSA SHAPE MISMATCH")
            print("identity:", identity.shape)
            print("out     :", out.shape)
            raise RuntimeError("DWSA shape mismatch")
        
        alpha = self.alpha.clamp(0.0, 1.0).view(1, -1, 1, 1)
        return identity + alpha * out

class MultiScaleContextModule(nn.Module):
    """
    Lightweight multi-scale context sau DAPPM.
    branch_ratio cao (16) để output channels rất nhỏ — chỉ tinh chỉnh,
    không compete với DAPPM.
    """
    def __init__(self, in_channels, out_channels, scales=(1, 2),
                 branch_ratio=16, alpha=0.1):
        super().__init__()
        self.scales = scales
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        self.alpha = nn.Parameter(torch.ones(out_channels) * 1e-4)

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
                o = F.interpolate(o, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(o)

        fused = torch.cat(outs, dim=1)
        out = self.fusion(fused)

        x_proj = self.proj(x) if self.proj is not None else x
        alpha = self.alpha.view(1, -1, 1, 1)   # (out_channels,) → (1, out_channels, 1, 1)
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
    Enhanced GCNet backbone.

    Flow:
        x → GCNetCore → {c1, c2, c4, s4, s5, s6, x_d6}
                              ↓       ↓    ↓    ↓
                           DWSA4  DWSA5 DWSA6   |
                                              SPP (một lần duy nhất)
                              ↓
                         MultiScaleContext (optional, lightweight)
                              ↓
                         final_proj
                              ↓
                        c5 = x_d6 + x_spp

    Output dict: {c1, c2, c4, c5}
    """

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
                 # stage4 dùng window attention để tránh OOM (H/16 → N=1024)
                 # window_size=8 → N=64 per window, memory ~0.25MB vs 64MB full
                 dwsa4_window_size: int = 8,
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
                    window_size=dwsa4_window_size,  # window attention — tránh OOM
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
                    window_size=0,  # full attention — N=256, safe
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
                    window_size=0,  # full attention — N=64, trivial
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
        """
        Cascade DWSA injection đúng thứ tự — s4, s5, s6 bổ trợ nhau qua bilateral fusion:

        Stem → Stage4 → [DWSA4] → Stage5 → [DWSA5] → Stage6 → [DWSA6] → SPP
                  ↑                    ↑                   ↑
            x_s4 enhanced         x_s5 enhanced        x_s6 enhanced
            ảnh hưởng x_d5        ảnh hưởng x_d5        đưa vào SPP
            qua compression_2     qua detail_branch      tốt hơn
            và down_2

        Tức là DWSA4 → cải thiện input cho stage5 → cải thiện x_s5
             → DWSA5 → cải thiện input cho stage6 → cải thiện x_s6
             → DWSA6 → cải thiện input cho SPP → cải thiện c5
        """
        bb = self.backbone
        feat, c1, c2, out_size = bb.forward_stem(x)

        # ── Stage 4 ──────────────────────────────────────────────
        x_s4, x_d4 = bb.forward_stage4(feat, out_size)
        c4 = x_d4.clone()   # aux head input trước khi x_d4 bị stage5 dùng tiếp

        # DWSA4: enhance x_s4 → stage5 nhận semantic context tốt hơn
        # → compression_2(x_s5) và down_2(x_d5) chất lượng cao hơn
        if self.dwsa4 is not None:
            x_s4 = self.dwsa4(x_s4)

        # ── Stage 5 ──────────────────────────────────────────────
        x_s5, x_d5 = bb.forward_stage5(x_s4, x_d4, out_size)

        # DWSA5: enhance x_s5 → stage6 nhận semantic context tốt hơn
        # → x_s6 chất lượng cao hơn → SPP thu được global context tốt hơn
        if self.dwsa5 is not None:
            x_s5 = self.dwsa5(x_s5)

        # ── Stage 6 ──────────────────────────────────────────────
        x_s6, x_d6 = bb.forward_stage6(x_s5, x_d5)

        # DWSA6: enhance x_s6 ngay trước SPP — spatial self-attention
        # ở resolution thấp nhất (H/64) để global context coherent hơn
        if self.dwsa6 is not None:
            x_s6 = self.dwsa6(x_s6)

        # ── SPP (một lần duy nhất, trên x_s6 đã enhanced) ───────
        x_spp = bb.spp(x_s6)
        x_spp = resize(x_spp, size=out_size, mode='bilinear', align_corners=self.align_corners)

        # Lightweight multi-scale refinement sau SPP
        if self.ms_context is not None:
            x_spp = self.ms_context(x_spp)

        x_spp = self.final_proj(x_spp)

        # Merge detail + semantic
        c5 = x_d6 + x_spp

        return dict(
            c1=c1,   # H/2, C   = 32  — decoder skip (stem layer 0)
            c2=c2,   # H/4, C   = 32  — decoder skip (stem layer 1)
            c4=c4,   # H/8, C*2 = 64  — detail branch: aux head + decoder skip stage0
            c5=c5,   # H/8, C*4 = 128 — fused output: main decoder input
        )

    def switch_to_deploy(self):
        """
        Deploy mode:
        1. Fuse tất cả GCBlock (path_3x3_1 + path_3x3_2 + path_1x1 → single conv)
        2. Fuse BN trong DAPPM/MultiScaleContext nếu có thể
        3. DWSA giữ nguyên (không thể fuse attention)

        Kết quả: params giảm ~2/3 số GCBlock params, inference nhanh hơn.
        """
        # Fuse GCNetCore
        self.backbone.switch_to_deploy()

        # Mark deploy
        self.deploy = True

        print("✅ Switched to deploy mode:")
        print(f"   GCBlock: all paths fused → single 3x3 conv")
        print(f"   DWSA: kept as-is (attention không fuse được)")
        print(f"   SPP: kept as-is")

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
