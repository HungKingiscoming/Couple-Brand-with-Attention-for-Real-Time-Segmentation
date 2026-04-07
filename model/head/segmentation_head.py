import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from components.components import (
    BaseModule,
    ConvModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    OptConfigType,
    SampleList,
)


# =============================================================================
# Accuracy helper
# =============================================================================

def accuracy(pred: Tensor,
             target: Tensor,
             ignore_index: int = 255) -> Tensor:
    """Tính pixel accuracy, bỏ qua các pixel có nhãn = ignore_index.

    Args:
        pred (Tensor): Logits shape (B, C, H, W).
        target (Tensor): Ground truth shape (B, H, W).
        ignore_index (int): Label value to ignore. Default: 255.

    Returns:
        Tensor: Scalar accuracy in [0, 100].
    """
    pred_label = pred.argmax(dim=1)             # (B, H, W)
    mask       = target != ignore_index
    correct    = (pred_label[mask] == target[mask]).sum().float()
    total      = mask.sum().float().clamp(min=1)
    return correct / total * 100.0


# =============================================================================
# Cross-entropy loss wrapper
# =============================================================================

class CrossEntropyLoss(nn.Module):
    """Wrapper nhỏ quanh F.cross_entropy để tương thích với loss_decode API.

    Args:
        ignore_index (int): Label value to ignore. Default: 255.
        loss_weight (float): Scalar weight applied to this loss. Default: 1.0.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight  = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * F.cross_entropy(
            pred, target, ignore_index=self.ignore_index)


# =============================================================================
# GCNetHead
# =============================================================================

class GCNetHead(BaseModule):

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 align_corners=False,
                 ignore_index=255,
                 loss_weight_aux=0.4,
                 dropout_ratio=0.1,
                 init_cfg=None):

        super().__init__(init_cfg)

        self.align_corners = align_corners
        self.ignore_index = ignore_index

        # =========================
        # 🔥 MAIN DECODER (REFINE ONLY)
        # =========================

        self.conv1 = ConvModule(
            in_channels, channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        self.conv2 = ConvModule(
            channels, channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        # 🔥 lightweight attention
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # residual refinement
        self.res = ConvModule(
            channels, channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=None
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

        # =========================
        # AUX HEAD (giữ nguyên)
        # =========================

        self.aux_head = nn.Sequential(
            ConvModule(in_channels // 2, channels, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        self.aux_cls = nn.Conv2d(channels, num_classes, 1)

        # =========================
        # LOSS
        # =========================

        self.loss_c4 = CrossEntropyLoss(ignore_index, loss_weight_aux)
        self.loss_c6 = CrossEntropyLoss(ignore_index, 1.0)

    # =====================================================

    def forward(self, inputs):

        if self.training:
            c4, fused = inputs   # fused đã là 1/8

            x = self.conv1(fused)
            x = self.conv2(x)

            # attention
            w = self.attn(x)
            x = x * w

            # residual
            x = self.res(x) + x

            out = self.cls_seg(self.dropout(x))

            aux = self.aux_cls(self.aux_head(c4))

            return aux, out

        else:
            fused = inputs

            x = self.conv1(fused)
            x = self.conv2(x)

            w = self.attn(x)
            x = x * w

            x = self.res(x) + x

            return self.cls_seg(self.dropout(x))

    # =====================================================

    def loss(self, seg_logits, seg_label):

        aux, main = seg_logits
        target_size = seg_label.shape[1:]

        aux = resize(aux, size=target_size,
                     mode='bilinear', align_corners=self.align_corners)

        main = resize(main, size=target_size,
                      mode='bilinear', align_corners=self.align_corners)

        return {
            'loss_c4': self.loss_c4(aux, seg_label),
            'loss_c6': self.loss_c6(main, seg_label),
            'acc_seg': accuracy(main, seg_label, self.ignore_index)
        }
