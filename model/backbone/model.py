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
# ConvBN
# =============================================================================

class ConvBN(nn.Module):
    """Conv2d + BN, no activation. Used inside GCBlock parallel paths."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse_to_kernel_bias(self):
        W     = self.conv.weight
        gamma = self.bn.weight
        beta  = self.bn.bias
        mean  = self.bn.running_mean
        var   = self.bn.running_var
        std   = (var + self.bn.eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return W * scale, beta - gamma * mean / std


# =============================================================================
# GCBlock — Multi-path Re-parameterizable Block
# =============================================================================

_PATHS = {
    # Stem + detail: spatial lớn (H/4-H/8), 2 paths đủ
    'stem_same':       2,
    'stem_down':       2,
    'detail_same':     2,
    'detail_down':     2,
    # Semantic[0]: H/16, C*4=128
    'semantic_0_down': 2,
    'semantic_0_same': 2,
    # Semantic[1]: H/32, C*8=256
    'semantic_1_down': 3,
    'semantic_1_same': 3,
    # Semantic[2]: H/64, C*16=512 — không tăng paths vì channel đã rất đắt
    'semantic_2_down': 3,
    'semantic_2_same': 3,
}


class GCBlock(nn.Module):
    """
    Training : N x ConvBN(3x3) + ConvBN(1x1) + BN(identity if same ch & stride=1)
    Deploy   : single Conv3x3 + bias  (all paths fused)
    """
    def __init__(self, in_channels, out_channels, stride=1, padding=1,
                 act=True, num_3x3_paths=4,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 deploy=False):
        super().__init__()
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.stride        = stride
        self.padding       = padding
        self.num_3x3_paths = num_3x3_paths
        self.deploy        = deploy
        self.act_fn        = nn.ReLU(inplace=True) if act else nn.Identity()

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels, out_channels, 3, stride=stride,
                padding=padding, bias=True)
        else:
            self.paths_3x3 = nn.ModuleList([
                ConvBN(in_channels, out_channels, 3, stride=stride, padding=padding)
                for _ in range(num_3x3_paths)
            ])
            self.path_1x1 = ConvBN(in_channels, out_channels, 1, stride=stride)
            self.path_identity = (
                nn.BatchNorm2d(in_channels)
                if in_channels == out_channels and stride == 1 else None
            )

    def forward(self, x):
        if self.deploy:
            return self.act_fn(self.reparam_conv(x))
        out = sum(p(x) for p in self.paths_3x3)
        out = out + self.path_1x1(x)
        if self.path_identity is not None:
            out = out + self.path_identity(x)
        return self.act_fn(out)

    def _fuse_identity(self):
        device = next(self.parameters()).device
        zk = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        zb = torch.zeros(self.out_channels, device=device)
        if self.path_identity is None:
            return zk, zb
        bn = self.path_identity
        std   = (bn.running_var + bn.eps).sqrt()
        ik    = torch.zeros_like(zk)
        for i in range(self.in_channels):
            ik[i, i, 1, 1] = 1.0
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        return ik * scale, bn.bias - bn.weight * bn.running_mean / std

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        device = next(self.parameters()).device
        K = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        b = torch.zeros(self.out_channels, device=device)
        for p in self.paths_3x3:
            k, bi = p.fuse_to_kernel_bias(); K += k; b += bi
        k1, b1 = self.path_1x1.fuse_to_kernel_bias()
        K += F.pad(k1, [1, 1, 1, 1]); b += b1
        kid, bid = self._fuse_identity()
        K += kid; b += bid
        self.reparam_conv = nn.Conv2d(
            self.in_channels, self.out_channels, 3,
            stride=self.stride, padding=self.padding, bias=True).to(device)
        self.reparam_conv.weight.data = K
        self.reparam_conv.bias.data   = b
        del self.paths_3x3, self.path_1x1
        if hasattr(self, 'path_identity'):
            del self.path_identity
        self.deploy = True

    def extra_repr(self):
        if self.deploy:
            return f"in={self.in_channels}, out={self.out_channels}, [DEPLOY]"
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"stride={self.stride}, paths={self.num_3x3_paths}x3x3+1x1"
                f"+{'id' if self.path_identity is not None else '0'}")


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


def _partition_windows(x, ws):
    B, C, H, W = x.shape
    nH, nW = H // ws, W // ws
    x = x.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B * nH * nW, C, ws, ws), (nH, nW)


def _merge_windows(windows, nH, nW, B):
    _, C, ws, _ = windows.shape
    x = windows.view(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, nH * ws, nW * ws)


# =============================================================================
# DWSABlock — FIXED:
#   1. alpha: register_buffer → nn.Parameter (can now be trained)
#   2. o_proj moved inside _attention so everything runs in fp32
#   3. attn.clamp removed — handled by fp32 softmax naturally
# =============================================================================

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


# =============================================================================
# MultiScaleContextModule
# alpha init 0.1 (was 1e-4 — too small to contribute early in training)
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


# =============================================================================
# GCNetCore
# =============================================================================

class GCNetCore(BaseModule):
    def __init__(self,
                 in_channels=3, channels=32, ppm_channels=128,
                 num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
                 align_corners=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None, deploy=False,
                 paths_cfg=None):
        super().__init__(init_cfg)
        self.in_channels  = in_channels
        self.channels     = channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg  = act_cfg
        self.deploy   = deploy

        p = dict(_PATHS)
        if paths_cfg:
            p.update(paths_cfg)
        C = channels

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvModule(in_channels, C, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),           # i=0
            ConvModule(C, C, 3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),           # i=1
            *[GCBlock(C, C, stride=1, num_3x3_paths=p['stem_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[0])],               # i=2..5
            GCBlock(C, C*2, stride=2, num_3x3_paths=p['stem_down'],
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy), # i=6
            *[GCBlock(C*2, C*2, stride=1, num_3x3_paths=p['stem_same'],
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[1] - 1)],           # i=7..9
        )
        self.relu = build_activation_layer(act_cfg)

        # ── Semantic Branch ───────────────────────────────────────────────────
        self.semantic_branch_layers = nn.ModuleList([

            nn.Sequential(                                             # [0] H/16, C*4
                GCBlock(C*2, C*4, stride=2, num_3x3_paths=p['semantic_0_down'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*4, C*4, stride=1, num_3x3_paths=p['semantic_0_same'],
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[2][0] - 2)],
                GCBlock(C*4, C*4, stride=1, num_3x3_paths=p['semantic_0_same'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            ),

            nn.Sequential(                                             # [1] H/32, C*8
                GCBlock(C*4, C*8, stride=2, num_3x3_paths=p['semantic_1_down'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*8, C*8, stride=1, num_3x3_paths=p['semantic_1_same'],
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[3][0] - 2)],
                GCBlock(C*8, C*8, stride=1, num_3x3_paths=p['semantic_1_same'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            ),

            nn.Sequential(                                             # [2] H/64, C*16
                GCBlock(C*8, C*16, stride=2, num_3x3_paths=p['semantic_2_down'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*16, C*16, stride=1, num_3x3_paths=p['semantic_2_same'],
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[4][0] - 2)],
                GCBlock(C*16, C*16, stride=1, num_3x3_paths=p['semantic_2_same'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            ),
        ])

        # ── Detail Branch ─────────────────────────────────────────────────────
        self.detail_branch_layers = nn.ModuleList([

            nn.Sequential(                                             # [0] H/8, C*2
                *[GCBlock(C*2, C*2, stride=1, num_3x3_paths=p['detail_same'],
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[2][1] - 1)],
                GCBlock(C*2, C*2, stride=1, num_3x3_paths=p['detail_same'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            ),

            nn.Sequential(                                             # [1] H/8, C*2
                *[GCBlock(C*2, C*2, stride=1, num_3x3_paths=p['detail_same'],
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[3][1] - 1)],
                GCBlock(C*2, C*2, stride=1, num_3x3_paths=p['detail_same'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            ),

            nn.Sequential(                                             # [2] H/8, C*2→C*4
                GCBlock(C*2, C*4, stride=1, num_3x3_paths=p['detail_down'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
                *[GCBlock(C*4, C*4, stride=1, num_3x3_paths=p['detail_same'],
                          norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
                  for _ in range(num_blocks_per_stage[4][1] - 2)],
                GCBlock(C*4, C*4, stride=1, num_3x3_paths=p['detail_same'],
                        norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
            ),
        ])

        # ── Bilateral Fusion Connectors ───────────────────────────────────────
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

    def forward_stem(self, x):
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

    def forward_stage4(self, x, out_size):
        x_s4 = self.semantic_branch_layers[0](x)
        x_d4 = self.detail_branch_layers[0](x)
        comp  = self.compression_1(self.relu(x_s4))
        x_s4  = x_s4 + self.down_1(self.relu(x_d4))
        x_d4  = x_d4 + resize(comp, size=out_size, mode='bilinear',
                               align_corners=self.align_corners)
        return x_s4, x_d4

    def forward_stage5(self, x_s4, x_d4, out_size):
        x_s5 = self.semantic_branch_layers[1](self.relu(x_s4))
        x_d5 = self.detail_branch_layers[1](self.relu(x_d4))
        comp  = self.compression_2(self.relu(x_s5))
        x_s5  = x_s5 + self.down_2(self.relu(x_d5))
        x_d5  = x_d5 + resize(comp, size=out_size, mode='bilinear',
                               align_corners=self.align_corners)
        return x_s5, x_d5

    def forward_stage6(self, x_s5, x_d5):
        x_d6 = self.detail_branch_layers[2](self.relu(x_d5))
        x_s6 = self.semantic_branch_layers[2](self.relu(x_s5))
        return x_s6, x_d6

    def forward(self, x):
        feat, c1, c2, out_size = self.forward_stem(x)
        x_s4, x_d4 = self.forward_stage4(feat, out_size)
        c4 = x_d4.clone()
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
        print(f"Fused {count} GCBlock -> single Conv3x3 each")

    def count_paths_summary(self):
        print(f"\n{'='*55}")
        print("GCBlock paths (training mode):")
        print(f"{'='*55}")
        sections = {
            'stem':              self.stem,
            'semantic_branch[0]': self.semantic_branch_layers[0],
            'semantic_branch[1]': self.semantic_branch_layers[1],
            'semantic_branch[2]': self.semantic_branch_layers[2],
            'detail_branch[0]':   self.detail_branch_layers[0],
            'detail_branch[1]':   self.detail_branch_layers[1],
            'detail_branch[2]':   self.detail_branch_layers[2],
        }
        for name, mod in sections.items():
            blocks = [m for m in mod.modules() if isinstance(m, GCBlock)]
            if not blocks:
                continue
            uniq = sorted({b.num_3x3_paths for b in blocks})
            print(f"  {name:<25}: {len(blocks)} blocks, paths={uniq}")
        total = sum(1 for m in self.modules() if isinstance(m, GCBlock))
        print(f"{'='*55}")
        print(f"  Total GCBlocks: {total}")
        print(f"{'='*55}\n")


# =============================================================================
# GCNetWithEnhance
# =============================================================================

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
        total    = sum(p.numel() for p in self.parameters())
        spp_set  = set(self.backbone.spp.parameters())
        bb_core  = sum(p.numel() for p in self.backbone.parameters()
                       if p not in spp_set)
        spp      = sum(p.numel() for p in self.backbone.spp.parameters())
        dwsa     = sum(p.numel() for m in [self.dwsa4, self.dwsa5, self.dwsa6]
                       if m is not None for p in m.parameters())
        ms       = sum(p.numel() for p in self.ms_context.parameters()) \
                   if self.ms_context else 0
        proj     = sum(p.numel() for p in self.final_proj.parameters())
        print(f"\n{'='*45}")
        print("GCNetWithEnhance Parameters")
        print(f"{'='*45}")
        print(f"  GCNetCore (excl SPP): {bb_core/1e6:.2f}M")
        print(f"  DAPPM (SPP):          {spp/1e6:.2f}M")
        print(f"  DWSA blocks:          {dwsa/1e6:.2f}M")
        print(f"  MultiScaleContext:    {ms/1e6:.2f}M")
        print(f"  final_proj:           {proj/1e6:.2f}M")
        print(f"{'='*45}")
        print(f"  TOTAL:                {total/1e6:.2f}M")
        print(f"{'='*45}\n")
        return total
