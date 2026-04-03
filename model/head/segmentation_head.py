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
    """Decode head for GCNet.

    Nhận output từ GCNet backbone:
      - Training  : (c4_feat, c6_feat) — c4 cho auxiliary loss, c6 cho main loss
      - Inference : c6_feat only

    Loss:
      loss_c4 = CE(upsample(c4_logit), gt)   weight = loss_weight_aux
      loss_c6 = CE(upsample(c6_logit), gt)   weight = 1.0
      acc_seg  = pixel accuracy trên c6_logit

    Args:
        in_channels (int): Channels của c6_feat (backbone output chính).
            Với GCNet-S/M channels=32: in_channels = channels*4 = 128.
        channels (int): Hidden channels bên trong head.
        num_classes (int): Số lớp phân đoạn (bao gồm background).
        norm_cfg (dict): Norm config. Default: BN.
        act_cfg (dict): Activation config. Default: ReLU.
        align_corners (bool): F.interpolate align_corners. Default: False.
        ignore_index (int): Label ignored trong loss và accuracy. Default: 255.
        loss_weight_aux (float): Weight của auxiliary loss (c4). Default: 0.4.
        dropout_ratio (float): Dropout trước cls_seg. Default: 0.1.
        init_cfg (dict, optional): Init config. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
                 dropout_ratio: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels    = in_channels
        self.channels       = channels
        self.num_classes    = num_classes
        self.norm_cfg       = norm_cfg
        self.act_cfg        = act_cfg
        self.align_corners  = align_corners
        self.ignore_index   = ignore_index
        self.loss_weight_aux = loss_weight_aux

        # ---- Main head (c6) ---------------------------------------------- #
        # c6_feat: in_channels = channels*4 (backbone output)
        self.head = self._make_base_head(in_channels, channels)

        # ---- Auxiliary head (c4) ----------------------------------------- #
        # c4_feat: in_channels // 2 vì detail branch ở stage 4 = channels*2
        self.aux_head_c4    = self._make_base_head(in_channels // 2, channels)
        self.aux_cls_seg_c4 = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Final classifiers ------------------------------------------- #
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Loss functions ---------------------------------------------- #
        self.loss_c4 = CrossEntropyLoss(ignore_index=ignore_index,
                                         loss_weight=loss_weight_aux)
        self.loss_c6 = CrossEntropyLoss(ignore_index=ignore_index,
                                         loss_weight=1.0)

        self.init_weights()

    # ---------------------------------------------------------------------- #
    # Weight init                                                              #
    # ---------------------------------------------------------------------- #

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.cls_seg or m is self.aux_cls_seg_c4:
                    # Final classifier layer → init nhẹ hơn rất nhiều
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self,
                inputs: Union[Tensor, Tuple[Tensor, Tensor]]
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Training  : inputs = (c4_feat, c6_feat) → returns (c4_logit, c6_logit)
        Inference : inputs = c6_feat            → returns c6_logit
        """
        if self.training:
            c4_feat, c6_feat = inputs

            c4_logit = self.aux_cls_seg_c4(self.aux_head_c4(c4_feat))
            c6_logit = self.cls_seg(self.dropout(self.head(c6_feat)))

            return c4_logit, c6_logit

        else:
            c6_logit = self.cls_seg(self.dropout(self.head(inputs)))
            return c6_logit

    # ---------------------------------------------------------------------- #
    # Loss                                                                     #
    # ---------------------------------------------------------------------- #

    def loss(self,
             seg_logits: Tuple[Tensor, Tensor],
             seg_label: Tensor) -> Dict[str, Tensor]:
        """Tính loss từ logits và ground-truth label.

        Args:
            seg_logits: (c4_logit, c6_logit) — output của forward() khi training.
                c4_logit: (B, num_classes, H4, W4)
                c6_logit: (B, num_classes, H6, W6)
            seg_label: Ground-truth shape (B, H, W), dtype=torch.long.
                Pixels cần ignore mang giá trị ignore_index.

        Returns:
            dict với keys: 'loss_c4', 'loss_c6', 'acc_seg'.
        """
        c4_logit, c6_logit = seg_logits
        target_size = seg_label.shape[1:]   # (H, W)

        # Upsample logits về kích thước gt
        c4_logit = resize(c4_logit, size=target_size,
                          mode='bilinear', align_corners=self.align_corners)
        c6_logit = resize(c6_logit, size=target_size,
                          mode='bilinear', align_corners=self.align_corners)

        losses = {
            'loss_c4': self.loss_c4(c4_logit, seg_label),
            'loss_c6': self.loss_c6(c6_logit, seg_label),
            'acc_seg': accuracy(c6_logit, seg_label,
                                ignore_index=self.ignore_index),
        }
        return losses

    # ---------------------------------------------------------------------- #
    # Helper                                                                   #
    # ---------------------------------------------------------------------- #

    def _make_base_head(self, in_channels: int, channels: int) -> nn.Sequential:
        """BN → ReLU → Conv3×3(BN, ReLU)."""
        return nn.Sequential(
            build_norm_layer(self.norm_cfg, in_channels)[1],
            build_activation_layer(self.act_cfg),
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('conv', 'norm', 'act')),
        )

    # ---------------------------------------------------------------------- #
    # Inference helper                                                         #
    # ---------------------------------------------------------------------- #

    def predict(self,
                inputs: Union[Tensor, Tuple[Tensor, Tensor]],
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """Inference: forward + upsample về img_size nếu cần.

        Args:
            inputs: c6_feat hoặc (c4_feat, c6_feat).
            img_size: (H, W) của ảnh gốc. Nếu None, không upsample.

        Returns:
            Tensor: Segmentation map (B, num_classes, H, W).
        """
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit
