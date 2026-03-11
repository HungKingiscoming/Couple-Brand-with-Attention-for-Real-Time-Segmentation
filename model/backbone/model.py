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


# =============================================================================
# Block3x3 — từ file 6
# Conv3x3→BN → Conv1x1→BN (double-conv, richer representation)
# =============================================================================

class Block3x3(BaseModule):
    """
    Double-conv path: Conv3x3→BN → Conv1x1→BN.

    Tại sao double-conv tốt hơn single ConvBN:
      Conv3x3: học spatial patterns (edges, textures)
      Conv1x1: học channel interactions trên spatial features đó
      → biểu diễn phong phú hơn với cùng receptive field

    Khi deploy: 2 conv fuse thành 1 Conv3x3 bằng torch.einsum
    → inference cost giống single-conv, expressiveness cao hơn lúc training.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 deploy: bool = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.deploy       = deploy

        if deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=padding, bias=True)
        else:
            # Conv3x3→BN: học spatial features
            self.conv1 = ConvModule(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=stride, padding=padding,
                bias=bias, norm_cfg=norm_cfg, act_cfg=None)
            # Conv1x1→BN: học channel mixing trên spatial features đã học
            self.conv2 = ConvModule(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0,
                bias=bias, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def _fuse_bn_tensor(self, conv_module: nn.Module):
        """Fuse ConvModule (Conv + BN) thành kernel và bias."""
        kernel       = conv_module.conv.weight
        running_mean = conv_module.bn.running_mean
        running_var  = conv_module.bn.running_var
        gamma        = conv_module.bn.weight
        beta         = conv_module.bn.bias
        eps          = conv_module.bn.eps
        std          = (running_var + eps).sqrt()
        t            = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        Fuse Conv3x3→BN + Conv1x1→BN thành 1 Conv3x3 duy nhất.

        Toán học:
          y = W2_fused * (W1_fused * x + b1) + b2
            = (W2 @ W1) * x + (W2 * b1 + b2)

        torch.einsum('oi,ichw->ochw', W2_sq, W1):
          W2_sq: (out, out) — squeeze spatial dims
          W1   : (out, in, 3, 3)
          result: (out, in, 3, 3)
        """
        W1, b1 = self._fuse_bn_tensor(self.conv1)   # (out, in, 3, 3), (out,)
        W2, b2 = self._fuse_bn_tensor(self.conv2)   # (out, out, 1, 1), (out,)

        W_fused = torch.einsum(
            'oi,ichw->ochw',
            W2.squeeze(3).squeeze(2),   # (out, out)
            W1                          # (out, in, 3, 3)
        )
        b_fused = b2 + (b1.view(1, -1, 1, 1) * W2).sum(3).sum(2).sum(1)

        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3,
            stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = W_fused
        self.conv.bias.data   = b_fused
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


# =============================================================================
# Block1x1 — từ file 6
# Conv1x1→BN → Conv1x1→BN (double-conv cho 1x1 path)
# =============================================================================

class Block1x1(BaseModule):
    """
    Double-conv 1x1: Conv1x1→BN → Conv1x1→BN.
    Khi deploy: fuse thành 1 Conv1x1 → pad thành 3x3 để cộng với kernel chính.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 deploy: bool = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.deploy       = deploy

        if deploy:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, bias=True)
        else:
            self.conv1 = ConvModule(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=stride, padding=0,
                bias=bias, norm_cfg=norm_cfg, act_cfg=None)
            self.conv2 = ConvModule(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0,
                bias=bias, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def _fuse_bn_tensor(self, conv_module: nn.Module):
        kernel       = conv_module.conv.weight
        running_mean = conv_module.bn.running_mean
        running_var  = conv_module.bn.running_var
        gamma        = conv_module.bn.weight
        beta         = conv_module.bn.bias
        eps          = conv_module.bn.eps
        std          = (running_var + eps).sqrt()
        t            = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Fuse 2 Conv1x1 thành 1 Conv1x1."""
        W1, b1 = self._fuse_bn_tensor(self.conv1)   # (out, in, 1, 1)
        W2, b2 = self._fuse_bn_tensor(self.conv2)   # (out, out, 1, 1)

        W_fused = torch.einsum(
            'oi,ichw->ochw',
            W2.squeeze(3).squeeze(2),
            W1
        )
        b_fused = b2 + (b1.view(1, -1, 1, 1) * W2).sum(3).sum(2).sum(1)

        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1,
            stride=self.stride, padding=0, bias=True)
        self.conv.weight.data = W_fused
        self.conv.bias.data   = b_fused
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


# =============================================================================
# GCBlock — kết hợp:
#   - File 6: 2 paths cố định (Block3x3 double-conv + Block1x1 double-conv)
#   - File 6: deploy dùng torch.einsum bên trong từng Block
#   - File cũ: identity path fuse logic
# =============================================================================

class GCBlock(nn.Module):
    """
    Re-parameterizable Block — 2 paths cố định.

    Training:
        path_3x3_1   : Block3x3 (Conv3x3→BN → Conv1x1→BN)
        path_3x3_2   : Block3x3 (Conv3x3→BN → Conv1x1→BN)
        path_1x1     : Block1x1 (Conv1x1→BN → Conv1x1→BN)
        path_identity: BN only  (chỉ khi in==out, stride==1)

    Deploy:
        reparam_conv : single Conv3x3 + bias  (tất cả paths fused)
        → inference cost = 1 Conv3x3, expressiveness cao hơn lúc training
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.deploy       = deploy

        assert kernel_size == 3
        assert padding == 1

        self.act_fn = build_activation_layer(act_cfg) if act else nn.Identity()

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=True,
                padding_mode=padding_mode)
        else:
            # Identity path: chỉ khi in==out và stride==1
            if in_channels == out_channels and stride == 1:
                self.path_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.path_identity = None

            self.path_3x3_1 = Block3x3(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, padding=padding, bias=False, norm_cfg=norm_cfg)
            self.path_3x3_2 = Block3x3(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, padding=padding, bias=False, norm_cfg=norm_cfg)
            self.path_1x1 = Block1x1(
                in_channels=in_channels, out_channels=out_channels,
                stride=stride, bias=False, norm_cfg=norm_cfg)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'reparam_conv'):
            return self.act_fn(self.reparam_conv(x))

        id_out = self.path_identity(x) if self.path_identity is not None else 0

        return self.act_fn(
            self.path_3x3_1(x)
            + self.path_3x3_2(x)
            + self.path_1x1(x)
            + id_out
        )

    def _fuse_identity_to_kernel_bias(self):
        """Fuse BN-only identity path thành kernel 3x3 tương đương."""
        device = next(self.parameters()).device
        if self.path_identity is None:
            return (
                torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device),
                torch.zeros(self.out_channels, device=device)
            )
        bn    = self.path_identity
        std   = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std

        W_id = torch.zeros(self.in_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.in_channels):
            W_id[i, i, 1, 1] = scale[i]

        b_id = bn.bias - bn.weight * bn.running_mean / std
        return W_id, b_id

    @torch.no_grad()
    def switch_to_deploy(self):
        """
        Fuse tất cả paths → 1 Conv3x3.
        torch.einsum xảy ra bên trong switch_to_deploy() của Block3x3/Block1x1.
        """
        if hasattr(self, 'reparam_conv'):
            return

        # Fuse từng path (einsum bên trong)
        self.path_3x3_1.switch_to_deploy()
        W1, b1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data

        self.path_3x3_2.switch_to_deploy()
        W2, b2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data

        self.path_1x1.switch_to_deploy()
        W_1x1   = self.path_1x1.conv.weight.data
        b_1x1   = self.path_1x1.conv.bias.data
        W_1x1_p = F.pad(W_1x1, [1, 1, 1, 1])   # (out,in,1,1) → (out,in,3,3)

        W_id, b_id = self._fuse_identity_to_kernel_bias()

        W_fused = W1 + W2 + W_1x1_p + W_id
        b_fused = b1 + b2 + b_1x1   + b_id

        device = W_fused.device
        self.reparam_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, bias=True
        ).to(device)
        self.reparam_conv.weight.data = W_fused
        self.reparam_conv.bias.data   = b_fused

        for p in self.parameters():
            p.detach_()
        for attr in ['path_3x3_1', 'path_3x3_2', 'path_1x1',
                     'path_identity', 'id_tensor']:
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.deploy = True

    def extra_repr(self):
        if self.deploy:
            return f"in={self.in_channels}, out={self.out_channels}, [DEPLOY]"
        has_id = self.path_identity is not None
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"stride={self.stride}, "
                f"2×Block3x3(double-conv) + Block1x1(double-conv) "
                f"+ {'identity' if has_id else 'no_identity'}")


# =============================================================================
# DWSA helpers
# =============================================================================

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
    B, C, H, W = x.shape
    nH, nW = H // ws, W // ws
    x = x.view(B, C, nH, ws, nW, ws)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B * nH * nW, C, ws, ws), (nH, nW)


def _merge_windows(windows: Tensor, nH: int, nW: int, B: int) -> Tensor:
    _, C, ws, _ = windows.shape
    x = windows.view(B, nH, nW, C, ws, ws)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, nH * ws, nW * ws)


# =============================================================================
# DWSABlock — Fixes từ Bước 2:
#   1. alpha: nn.Parameter (không phải register_buffer)
#   2. o_proj trong fp32 bên trong _attention
#   3. sigmoid(alpha) thay vì clamp
# =============================================================================

class DWSABlock(nn.Module):
    """
    Depthwise Separable Attention với Window Attention tùy chọn.

    window_size = 0  → Full attention   (stage5 N=256, stage6 N=64)
    window_size > 0  → Window attention (stage4 N=1024 → OOM nếu full)
    """
    def __init__(self, channels, num_heads=2, drop=0.0, reduction=4,
                 qk_sharing=True, groups=4, alpha=0.01, window_size=0):
        super().__init__()
        assert channels % reduction == 0
        self.channels    = channels
        self.num_heads   = num_heads
        self.window_size = window_size

        reduced = channels // reduction
        mid     = max(reduced // 2, num_heads)
        self.reduced = reduced
        self.mid     = mid

        self.bn_in   = nn.BatchNorm2d(channels)
        self.in_proj  = nn.Conv2d(channels, reduced, 1, bias=False)
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
        self.o_proj = nn.Conv1d(mid, reduced, 1, groups=g, bias=True)

        self.drop  = nn.Dropout(drop)
        self.scale = (mid // num_heads) ** -0.5

        # FIX 1: nn.Parameter → grad luôn tồn tại
        # FIX 3: lưu pre-sigmoid, sigmoid(-4.60) ≈ 0.01
        alpha_init = math.log(alpha / (1.0 - alpha))
        self.alpha = nn.Parameter(torch.full((channels,), alpha_init))

    def _attention(self, x_flat: Tensor) -> Tensor:
        """
        Attention + o_proj trong fp32.
        FIX 2: o_proj chạy fp32 trước khi cast về dtype gốc.
        """
        orig_dtype = x_flat.dtype
        x_fp32    = x_flat.float()

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

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous()
        B_, Hn, Hd, N = out.shape
        out = out.view(B_, self.mid, N)

        # FIX 2: o_proj trong fp32
        out = self.o_proj(out)

        return out.to(orig_dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        identity = x

        x_norm = self.bn_in(x)
        x_red  = self.in_proj(x_norm)

        if self.window_size > 0:
            ws    = self.window_size
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x_red = F.pad(x_red, (0, pad_w, 0, pad_h))
            windows, (nH, nW) = _partition_windows(x_red, ws)
            Bw, C2, _, _      = windows.shape
            out_flat = self._attention(windows.view(Bw, C2, ws * ws))
            out_red  = _merge_windows(out_flat.view(Bw, C2, ws, ws), nH, nW, B)
            if pad_h > 0 or pad_w > 0:
                out_red = out_red[:, :, :H, :W]
        else:
            N        = H * W
            out_flat = self._attention(x_red.view(B, self.reduced, N))
            out_red  = out_flat.view(B, self.reduced, H, W)

        out = self.bn_out(self.out_proj(out_red))

        # FIX 3: sigmoid → gradient luôn > 0
        alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return identity + alpha * out


# =============================================================================
# MultiScaleContextModule — Fixes từ Bước 2:
#   - alpha init pre-sigmoid (sigmoid(-2.2) ≈ 0.1)
#   - sigmoid bound: alpha ∈ (0,1)
# =============================================================================

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

        # FIX: pre-sigmoid init, sigmoid(-2.2) ≈ 0.1
        alpha_init = math.log(alpha / (1.0 - alpha))
        self.alpha = nn.Parameter(torch.full((out_channels,), alpha_init))

        self.proj = (
            nn.Sequential(
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
        fused  = torch.cat(outs, dim=1)
        out    = self.fusion(fused)
        x_proj = self.proj(x) if self.proj is not None else x
        alpha  = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return x_proj + alpha * out


# =============================================================================
# GCNetCore — giữ nguyên từ file cũ
# GCBlock bên trong tự động dùng architecture mới
# =============================================================================

class GCNetCore(BaseModule):
    def __init__(self,
                 in_channels=3, channels=32, ppm_channels=128,
                 num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
                 align_corners=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None, deploy=False):
        super().__init__(init_cfg)
        self.in_channels   = in_channels
        self.channels      = channels
        self.ppm_channels  = ppm_channels
        self.align_corners = align_corners
        self.norm_cfg      = norm_cfg
        self.act_cfg       = act_cfg
        self.deploy        = deploy
        C = channels

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvModule(in_channels, C, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C, C, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            *[GCBlock(C, C, stride=1, norm_cfg=norm_cfg,
                      act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[0])],
            GCBlock(C, C*2, stride=2, norm_cfg=norm_cfg,
                    act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                      act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[1] - 1)],
        )
        self.relu = build_activation_layer(act_cfg)

        # ── Semantic Branch ───────────────────────────────────────────────────
        self.semantic_branch_layers = nn.ModuleList([
            nn.Sequential(
                GCBlock(C*2, C*4, stride=2, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[2][0] - 2)],
                GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                GCBlock(C*4, C*8, stride=2, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*8, C*8, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[3][0] - 2)],
                GCBlock(C*8, C*8, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                GCBlock(C*8, C*16, stride=2, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*16, C*16, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[4][0] - 2)],
                GCBlock(C*16, C*16, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
        ])

        # ── Detail Branch ─────────────────────────────────────────────────────
        self.detail_branch_layers = nn.ModuleList([
            nn.Sequential(
                *[GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[2][1] - 1)],
                GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                *[GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[3][1] - 1)],
                GCBlock(C*2, C*2, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
            nn.Sequential(
                GCBlock(C*2, C*4, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                          act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[4][1] - 2)],
                GCBlock(C*4, C*4, stride=1, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, act=False, deploy=deploy),
            ),
        ])

        # ── Bilateral Fusion ──────────────────────────────────────────────────
        self.compression_1 = ConvModule(C*4, C*2, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.down_1        = ConvModule(C*2, C*4, 3, stride=2, padding=1,
                                         norm_cfg=norm_cfg, act_cfg=None)
        self.compression_2 = ConvModule(C*8, C*2, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.down_2        = nn.Sequential(
            ConvModule(C*2, C*4, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(C*4, C*8, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=None))

        # ── DAPPM ─────────────────────────────────────────────────────────────
        self.spp = DAPPM(
            in_channels=C*16, branch_channels=ppm_channels,
            out_channels=C*4, num_scales=5,
            kernel_sizes=[3, 5, 7, 9], strides=[1, 2, 2, 4],
            paddings=[1, 2, 3, 4], norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.kaiming_init()

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_stem(self, x: Tensor):
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))
        c1 = c2 = None
        feat = x
        for i, layer in enumerate(self.stem):
            if self.training and i <= 1:
                with torch.autocast(device_type='cuda', enabled=False):
                    feat = layer(feat.float())
                feat = feat.to(x.dtype)
            else:
                feat = layer(feat)
            if i == 0: c1 = feat
            if i == 1: c2 = feat
        return feat, c1, c2, out_size

    def forward_stage4(self, x: Tensor, out_size: Tuple) -> Tuple[Tensor, Tensor]:
        """
        Stage 4 bilateral fusion.
        Out: x_s4 (H/16,C*4) → [DWSA4] → stage5 semantic tốt hơn
             x_d4 (H/8, C*2) → cascade detail
        """
        x_s4 = self.semantic_branch_layers[0](x)
        x_d4 = self.detail_branch_layers[0](x)
        comp = self.compression_1(self.relu(x_s4))
        x_s4 = x_s4 + self.down_1(self.relu(x_d4))
        x_d4 = x_d4 + resize(comp, size=out_size, mode='bilinear',
                              align_corners=self.align_corners)
        return x_s4, x_d4

    def forward_stage5(self, x_s4: Tensor, x_d4: Tensor,
                       out_size: Tuple) -> Tuple[Tensor, Tensor]:
        """
        Stage 5 bilateral fusion.
        In:  x_s4 đã qua DWSA4 → semantic context tốt hơn
        Out: x_s5 (H/32,C*8) → [DWSA5]
             x_d5 (H/8, C*2)
        """
        x_s5 = self.semantic_branch_layers[1](self.relu(x_s4))
        x_d5 = self.detail_branch_layers[1](self.relu(x_d4))
        comp = self.compression_2(self.relu(x_s5))
        x_s5 = x_s5 + self.down_2(self.relu(x_d5))
        x_d5 = x_d5 + resize(comp, size=out_size, mode='bilinear',
                              align_corners=self.align_corners)
        return x_s5, x_d5

    def forward_stage6(self, x_s5: Tensor,
                       x_d5: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Stage 6.
        In:  x_s5 đã qua DWSA5
        Out: x_s6 (H/64,C*16) → [DWSA6] → SPP
             x_d6 (H/8, C*4)  → GatedFusion với x_spp → c5
        """
        x_d6 = self.detail_branch_layers[2](self.relu(x_d5))
        x_s6 = self.semantic_branch_layers[2](self.relu(x_s5))
        return x_s6, x_d6

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feat, c1, c2, out_size = self.forward_stem(x)
        x_s4, x_d4 = self.forward_stage4(feat, out_size)
        c4          = x_d4.clone()
        x_s5, x_d5 = self.forward_stage5(x_s4, x_d4, out_size)
        x_s6, x_d6 = self.forward_stage6(x_s5, x_d5)
        return dict(c1=c1, c2=c2, c4=c4,
                    x_s4=x_s4, x_s5=x_s5, x_s6=x_s6, x_d6=x_d6)

    def switch_to_deploy(self):
        count = 0
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
                count += 1
        self.deploy = True
        print(f"Fused {count} GCBlock → single Conv3x3 (torch.einsum)")


# =============================================================================
# GCNetWithEnhance — file cũ + GatedFusion (Bước 3)
# =============================================================================

class GCNetWithEnhance(BaseModule):
    """
    Enhanced GCNet backbone.

    Output: {c1, c2, c4, c5}
      c1: H/2,  C    = 32  — decoder skip
      c2: H/4,  C    = 32  — decoder skip
      c4: H/8,  C*2  = 64  — aux head input
      c5: H/8,  C*4  = 128 — main decoder input

    Cascade DWSA:
      Stage4→[DWSA4]→Stage5→[DWSA5]→Stage6→[DWSA6]→SPP
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

        # ── GatedFusion (Bước 3) ─────────────────────────────────────────────
        # gate = sigmoid(Conv(concat(x_d6, x_spp))) ∈ (0,1)
        # c5   = gate * x_d6 + (1-gate) * x_spp
        # Init: weight=0, bias=0 → sigmoid(0)=0.5 → tin đều lúc đầu
        self.fusion_gate = nn.Sequential(
            ConvModule(C*4 * 2, C*4, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(C*4, C*4, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.fusion_gate[1].weight)
        nn.init.zeros_(self.fusion_gate[1].bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
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

        # GatedFusion thay vì x_d6 + x_spp
        gate = self.fusion_gate(torch.cat([x_d6, x_spp], dim=1))
        c5   = gate * x_d6 + (1.0 - gate) * x_spp

        return dict(c1=c1, c2=c2, c4=c4, c5=c5)

    def switch_to_deploy(self):
        self.backbone.switch_to_deploy()
        self.deploy = True
        print("Switched to deploy mode:")
        print("  GCBlock: 2×Block3x3(double) + Block1x1(double) → 1 Conv3x3")
        print("  Deploy: torch.einsum (chính xác)")
        print("  DWSA/fusion_gate: kept as-is")

    @torch.no_grad()
    def count_params(self):
        total   = sum(p.numel() for p in self.parameters())
        spp_set = set(self.backbone.spp.parameters())
        bb_core = sum(p.numel() for p in self.backbone.parameters()
                      if p not in spp_set)
        spp     = sum(p.numel() for p in self.backbone.spp.parameters())
        dwsa    = sum(p.numel() for m in [self.dwsa4, self.dwsa5, self.dwsa6]
                      if m is not None for p in m.parameters())
        ms      = (sum(p.numel() for p in self.ms_context.parameters())
                   if self.ms_context else 0)
        proj    = sum(p.numel() for p in self.final_proj.parameters())
        gate    = sum(p.numel() for p in self.fusion_gate.parameters())

        print(f"\n{'='*50}")
        print("GCNetWithEnhance Parameters")
        print(f"{'='*50}")
        print(f"  GCNetCore (excl SPP): {bb_core/1e6:.2f}M")
        print(f"  DAPPM (SPP):          {spp/1e6:.2f}M")
        print(f"  DWSA blocks:          {dwsa/1e6:.2f}M")
        print(f"  MultiScaleContext:    {ms/1e6:.2f}M")
        print(f"  final_proj:           {proj/1e6:.2f}M")
        print(f"  fusion_gate:          {gate/1e6:.2f}M")
        print(f"{'='*50}")
        print(f"  TOTAL:                {total/1e6:.2f}M")
        print(f"{'='*50}\n")
        return total
