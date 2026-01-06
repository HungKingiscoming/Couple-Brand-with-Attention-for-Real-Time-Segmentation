import math
from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import DeformConv2d

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    DAPPM,
    OptConfigType,
)

# ============================================================
# MODERN LIGHT ATTENTION: ConvNeXt-style + ECA
# ============================================================

class ConvNeXtStyleBlock(nn.Module):
    """ConvNeXt-style block: large-kernel depthwise conv + LN + MLP.
    Rất hợp cho segmentation backbone hiện đại. [web:143][web:150]
    """
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        shortcut = x
        x = self.dwconv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x + shortcut


class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA) - kênh attention rất nhẹ. [web:156]"""
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class ModernAttentionBlock(nn.Module):
    """ConvNeXt-style depthwise + ECA attention.
    Thay thế cho DWSA để nhẹ hơn nhưng vẫn mạnh. [web:143][web:155][web:156]
    """
    def __init__(self, channels: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.convnext_block = ConvNeXtStyleBlock(channels, mlp_ratio=mlp_ratio, drop=drop)
        self.eca = ECALayer(channels, k_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convnext_block(x)
        x = self.eca(x)
        return x


# ============================================================
# MULTI-SCALE CONTEXT MODULE (Inception-style, nhẹ hơn)
# ============================================================

class MultiScaleContextModule(nn.Module):
    """Multi-scale context kiểu Inception/ConvNeXt: 1x1, 3x3, 1xk, kx1. [web:144][web:157]"""
    def __init__(self, in_channels: int, out_channels: int, k: int = 7):
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4 for MSC module."
        branch_channels = in_channels // 4

        self.path_1x1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.path_3x3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.path_1xk = nn.Conv2d(
            in_channels,
            branch_channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
        )
        self.path_kx1 = nn.Conv2d(
            in_channels,
            branch_channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
        )

        self.fusion = nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        p1 = self.path_1x1(x)
        p2 = self.path_3x3(x)
        p3 = self.path_1xk(x)
        p4 = self.path_kx1(x)
        out = torch.cat([p1, p2, p3, p4], dim=1)
        out = self.fusion(out)
        return out


# ============================================================
# GCBLOCK COMPONENTS (RepVGG-style với fix reparam)
# ============================================================

class Block1x1(BaseModule):
    """1x1_1x1 path of GCBlock (RepVGG-style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        bias: bool = True,
        norm_cfg: OptConfigType = dict(type="BN", requires_grad=True),
        deploy: bool = False,
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
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=True,
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
                act_cfg=None,
            )
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )

    def forward(self, x: Tensor) -> Tensor:
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

        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)  # (C_mid,in,1,1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)  # (C_out,C_mid,1,1)

        k1 = kernel1.squeeze()  # (C_mid, in)
        k2 = kernel2.squeeze()  # (C_out, C_mid)

        # W_eq = W2 @ W1
        kernel = torch.einsum("oi,ic->oc", k2, k1)  # (C_out, in)

        # b_eq = W2 @ b1 + b2
        bias = bias2 + k2 @ bias1  # (C_out,)

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.conv.weight.data = kernel.view(self.out_channels, self.in_channels, 1, 1)
        self.conv.bias.data = bias

        del self.conv1, self.conv2
        self.deploy = True


class Block3x3(BaseModule):
    """3x3_1x1 path of GCBlock (RepVGG-style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 1,
        bias: bool = True,
        norm_cfg: OptConfigType = dict(type="BN", requires_grad=True),
        deploy: bool = False,
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
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=True,
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
                act_cfg=None,
            )
            self.conv2 = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )

    def forward(self, x: Tensor) -> Tensor:
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

        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)  # (C_mid,in,3,3)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)  # (C_out,C_mid,1,1)

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )

        # W_eq = W2 (1x1) @ W1 (3x3)  → (C_out, in, 3, 3)
        self.conv.weight.data = torch.einsum(
            "oi,icjk->ocjk",
            kernel2.squeeze(3).squeeze(2),
            kernel1,
        )
        # b_eq = W2 @ b1 + b2
        # Đây là dạng tương đương với công thức trong RepVGG [web:166][web:169]
        self.conv.bias.data = bias2 + (
            bias1.view(1, -1, 1, 1) * kernel2
        ).sum(3).sum(2).sum(1)

        del self.conv1, self.conv2
        self.deploy = True


class GCBlock(nn.Module):
    """GCBlock với RepVGG-style + optional DCN + optional ModernAttention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 1,
        norm_cfg: OptConfigType = dict(type="BN", requires_grad=True),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=False),
        act: bool = True,
        deploy: bool = False,
        use_dwsa: bool = False,  # sẽ dùng ModernAttentionBlock thay vì DWSA cũ
        dwsa_num_heads: int = 8,  # không dùng nhưng giữ API
        use_dcn: bool = False,
        use_large_kernel_branch: bool = False,
        large_kernel_size: int = 7,
    ):
        super().__init__()

        assert kernel_size == 3 and padding == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        self.use_dcn = use_dcn
        self.use_large_kernel_branch = use_large_kernel_branch

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
            )
            self.path_residual = None
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

        # DCN
        if use_dcn and not deploy:
            self.offset_conv = nn.Conv2d(
                out_channels,
                18,
                kernel_size=3,
                padding=1,
            )
            self.dcn = DeformConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.dcn = None
            self.offset_conv = None

        # Optional large-kernel depthwise branch (train-only, không reparam)
        if use_large_kernel_branch and not deploy:
            self.large_dw = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=large_kernel_size,
                padding=large_kernel_size // 2,
                groups=out_channels,
                bias=False,
            )
            self.large_dw_bn = nn.BatchNorm2d(out_channels)
        else:
            self.large_dw = None
            self.large_dw_bn = None

        # Modern attention block (ConvNeXt-style + ECA)
        if use_dwsa and not deploy:
            self.attn_block = ModernAttentionBlock(out_channels, mlp_ratio=2.0, drop=0.0)
        else:
            self.attn_block = None

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "reparam_3x3"):
            out = self.relu(self.reparam_3x3(x))
        else:
            id_out = 0 if self.path_residual is None else self.path_residual(x)
            base = self.path_3x3_1(x) + self.path_3x3_2(x) + self.path_1x1(x) + id_out

            # Large-kernel depthwise branch (chỉ train-time)
            if self.large_dw is not None:
                large = self.large_dw_bn(self.large_dw(base))
                base = base + large

            out = self.relu(base)

        # DCN
        if self.dcn is not None:
            offset = self.offset_conv(out)
            out = self.dcn(out, offset)

        # Modern attention
        if self.attn_block is not None:
            out = self.attn_block(out)

        return out

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: Tensor):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv: nn.Module):
        if conv is None:
            return 0, 0

        if isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            if not hasattr(self, "id_tensor"):
                kernel_value = torch.zeros(
                    (self.in_channels, self.in_channels, 3, 3),
                    dtype=torch.float32,
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
            kernel3x3_1
            + kernel3x3_2
            + self._pad_1x1_to_3x3_tensor(kernel1x1)
            + kernelid,
            bias3x3_1 + bias3x3_2 + bias1x1 + biasid,
        )

    def switch_to_deploy(self):
        if hasattr(self, "reparam_3x3"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.reparam_3x3 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )

        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias

        for para in self.parameters():
            para.detach_()

        # Xoá các nhánh train-only
        for name in ["path_3x3_1", "path_3x3_2", "path_1x1", "path_residual", "id_tensor",
                     "dcn", "offset_conv", "attn_block", "large_dw", "large_dw_bn"]:
            if hasattr(self, name):
                delattr(self, name)

        self.deploy = True


# ============================================================
# GCNET WITH MODERN FEATURES (KAGGLE-READY)
# ============================================================

class GCNetWithDWSA(BaseModule):
    """
    GCNet với:
    - RepVGG-style GCBlocks (fixed reparam) [web:166][web:169]
    - ModernAttentionBlock (ConvNeXt-style + ECA)
    - DCN ở stage4
    - DAPPM + Multi-Scale Context Inception-style

    Tối ưu cho train from scratch + inference nhanh trên Kaggle.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 48,
        ppm_channels: int = 128,
        num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
        dwsa_stages: List[str] = ["stage3", "stage4", "bottleneck"],
        dwsa_num_heads: int = 8,  # giữ API
        use_dcn_in_stage4: bool = True,
        use_multi_scale_context: bool = True,
        align_corners: bool = False,
        norm_cfg: OptConfigType = dict(type="BN", requires_grad=True),
        act_cfg: OptConfigType = dict(type="GELU", inplace=False),
        init_cfg: OptConfigType = None,
        deploy: bool = False,
    ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.channels = channels
        self.ppm_channels = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dwsa_stages = dwsa_stages
        self.use_dcn_in_stage4 = use_dcn_in_stage4
        self.use_multi_scale_context = use_multi_scale_context
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy

        # Stage 1: H/2
        self.stage1_conv = ConvModule(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Stage 2: H/4
        stage2_layers = [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
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
                    deploy=deploy,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
                )
            )
        self.stage2 = nn.Sequential(*stage2_layers)

        # Stage 3: H/8
        stage3_layers = [
            GCBlock(
                in_channels=channels,
                out_channels=channels * 2,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy,
                use_large_kernel_branch=True,
                large_kernel_size=7,
            )
        ]
        for i in range(num_blocks_per_stage[1] - 1):
            use_attn = (i == num_blocks_per_stage[1] - 2) and ("stage3" in dwsa_stages)
            stage3_layers.append(
                GCBlock(
                    in_channels=channels * 2,
                    out_channels=channels * 2,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_dwsa=use_attn,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
                )
            )
        self.stage3 = nn.Sequential(*stage3_layers)
        self.relu = build_activation_layer(act_cfg)

        # ===================== SEMANTIC BRANCH =====================

        self.semantic_branch_layers = nn.ModuleList()

        # Stage 4 Semantic (H/16)
        stage4_sem = []
        stage4_sem.append(
            GCBlock(
                in_channels=channels * 2,
                out_channels=channels * 4,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy,
                use_large_kernel_branch=True,
                large_kernel_size=7,
            )
        )
        for i in range(num_blocks_per_stage[2][0] - 1):
            use_attn = (i >= num_blocks_per_stage[2][0] - 3) and ("stage4" in dwsa_stages)
            use_dcn = use_dcn_in_stage4 and (i >= num_blocks_per_stage[2][0] - 2)
            stage4_sem.append(
                GCBlock(
                    in_channels=channels * 4,
                    out_channels=channels * 4,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    deploy=deploy,
                    use_dwsa=use_attn,
                    use_dcn=use_dcn,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
                )
            )
        self.semantic_branch_layers.append(nn.Sequential(*stage4_sem))

        # Stage 5 Semantic (H/32)
        stage5_sem = []
        stage5_sem.append(
            GCBlock(
                in_channels=channels * 4,
                out_channels=channels * 4,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy,
                use_large_kernel_branch=True,
                large_kernel_size=7,
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
                    deploy=deploy,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
                )
            )
        self.semantic_branch_layers.append(nn.Sequential(*stage5_sem))

        # Stage 6 Semantic (H/64)
        stage6_sem = []
        stage6_sem.append(
            GCBlock(
                in_channels=channels * 4,
                out_channels=channels * 4,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                deploy=deploy,
                use_large_kernel_branch=True,
                large_kernel_size=7,
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
                    deploy=deploy,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
                )
            )
        self.semantic_branch_layers.append(nn.Sequential(*stage6_sem))

        # ===================== DETAIL BRANCH =====================

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
                    deploy=deploy,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
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
                    deploy=deploy,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
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
                    deploy=deploy,
                    use_large_kernel_branch=True,
                    large_kernel_size=7,
                )
            )
        self.detail_branch_layers.append(nn.Sequential(*detail_stage6))

        # ===================== BILATERAL FUSION =====================

        self.compression_1 = ConvModule(
            in_channels=channels * 4,
            out_channels=channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.compression_2 = ConvModule(
            in_channels=channels * 4,
            out_channels=channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.down_1 = ConvModule(
            in_channels=channels * 2,
            out_channels=channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.down_2 = ConvModule(
            in_channels=channels * 2,
            out_channels=channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.final_proj = ConvModule(
            in_channels=channels * 4,
            out_channels=channels * 2,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # ===================== BOTTLENECK =====================

        self.spp = DAPPM(
            in_channels=channels * 4,
            branch_channels=ppm_channels,
            out_channels=channels * 4,
            num_scales=5,
            kernel_sizes=[5, 9, 17, 33],
            strides=[2, 4, 8, 16],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Bottleneck attention
        if "bottleneck" in dwsa_stages:
            self.bottleneck_attn = ModernAttentionBlock(
                channels=channels * 4,
                mlp_ratio=2.0,
                drop=0.0,
            )
        else:
            self.bottleneck_attn = None

        # Multi-scale context module
        if use_multi_scale_context:
            self.multi_scale_context = MultiScaleContextModule(
                in_channels=channels * 4,
                out_channels=channels * 4,
                k=7,
            )
        else:
            self.multi_scale_context = None

        self.init_weights()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        outputs: Dict[str, Tensor] = {}

        # Stage 1 (H/2)
        c1 = self.stage1_conv(x)
        outputs["c1"] = c1

        # Stage 2 (H/4)
        c2 = self.stage2(c1)
        outputs["c2"] = c2

        # Stage 3 (H/8)
        c3 = self.stage3(c2)
        outputs["c3"] = c3

        # Stage 4 dual branch
        x_s = self.semantic_branch_layers[0](c3)  # H/8 → H/16
        x_d = self.detail_branch_layers[0](c3)    # H/8 → H/8

        out_size = (
            math.ceil(x.shape[-2] / 8),
            math.ceil(x.shape[-1] / 8),
        )
        x_s_relu = self.relu(x_s)
        x_d_relu = self.relu(x_d)

        # Semantic → Detail
        comp_c = self.compression_1(x_s_relu)
        x_d = x_d + resize(
            comp_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        # Detail → Semantic
        x_s = x_s + self.down_1(x_d_relu)
        outputs["c4"] = x_s

        # Stage 5
        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_d = self.detail_branch_layers[1](self.relu(x_d))

        # Bilateral Fusion 2
        comp_c = self.compression_2(self.relu(x_s))
        x_d = x_d + resize(
            comp_c,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        # Stage 6
        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))

        # Bottleneck: DAPPM + attention + context
        x_s = self.spp(x_s)

        if self.bottleneck_attn is not None:
            x_s = self.bottleneck_attn(x_s)

        if self.multi_scale_context is not None:
            x_s = self.multi_scale_context(x_s)

        # Resize + project
        x_s = resize(
            x_s,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        x_s = self.final_proj(x_s)

        # Final fusion
        c5 = x_d + x_s
        outputs["c5"] = c5

        return outputs

    def switch_to_deploy(self):
        """Reparam toàn bộ GCBlocks để inference nhanh trên Kaggle."""
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
        self.deploy = True

    def init_weights(self):
        if self.init_cfg is not None:
            super().init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode="fan_out",
                        nonlinearity="relu",
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


# ============================================================
# FACTORY
# ============================================================

def create_gcnet_dwsa_variants():
    """Factory tạo variant enhanced/modern."""
    gcnet_dwsa_enhanced = GCNetWithDWSA(
        channels=48,
        dwsa_stages=["stage3", "stage4", "bottleneck"],
        dwsa_num_heads=8,
        use_dcn_in_stage4=True,
        use_multi_scale_context=True,
    )
    return {"enhanced": gcnet_dwsa_enhanced}


if __name__ == "__main__":
    model = create_gcnet_dwsa_variants()["enhanced"]
    x = torch.randn(1, 3, 512, 1024)
    outputs = model(x)
    print("Model outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
