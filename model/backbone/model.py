import math
import numpy as np
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
# OHEM Cross-entropy loss
# =============================================================================

class OHEMCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining Cross-Entropy Loss.

    [CITYSCAPES FOGGY - ĐIỀU CHỈNH 3]
    Thay CrossEntropyLoss thuần bằng OHEM cho loss_c6 (main loss).

    Lý do: Cityscapes Foggy có class imbalance nặng hơn Cityscapes thường —
    fog che khuất nhiều pixel của các class nhỏ (người đi bộ, xe đạp, biển báo),
    khiến model dễ dominated bởi background và road (class lớn, loss thấp).
    OHEM giải quyết bằng cách chỉ backprop qua các pixel "khó" (loss cao),
    buộc model focus vào các vùng bị fog che phủ.

    Cơ chế:
      1. Tính per-pixel CE loss với reduction='none'.
      2. Loại bỏ pixel ignore_index.
      3. Sort loss giảm dần.
      4. Giữ lại top-K pixel: K = max(n_pixel_with_loss > thresh, min_kept).
      5. Backprop qua mean loss của K pixel đó.

    Args:
        ignore_index (int): Label value to ignore. Default: 255.
        loss_weight (float): Scalar weight applied to this loss. Default: 1.0.
        thresh (float): Loss threshold để xác định "hard example".
            Pixel có loss > thresh đều được giữ. Default: 0.7.
            Tăng thresh → strict hơn (chỉ lấy pixel rất khó).
            Giảm thresh → lấy nhiều pixel hơn (gần với CE thường).
        min_kept (int): Số pixel tối thiểu được giữ dù loss < thresh.
            Đảm bảo gradient không quá thưa khi fog nhẹ. Default: 100_000.
            Với Cityscapes (1024×2048 → ~2M pixel/image), 100k ≈ 5% pixel/image.

    Lưu ý training:
        - Với batch_size nhỏ (< 4), có thể tăng min_kept lên 200_000.
        - Nếu model diverge sớm, thử tăng thresh lên 0.9 (chỉ lấy pixel
          rất khó, giảm gradient noise).
        - OHEM không áp dụng cho loss_c4 (auxiliary) — CE thường là đủ.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0,
                 thresh: float = 0.7,
                 min_kept: int = 100_000):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight  = loss_weight
        self.thresh       = thresh
        self.min_kept     = min_kept

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred   (Tensor): Logits shape (B, C, H, W).
            target (Tensor): Ground truth shape (B, H, W), dtype=torch.long.

        Returns:
            Tensor: Scalar OHEM loss.
        """
        # Per-pixel CE, không reduce
        losses = F.cross_entropy(
            pred, target,
            ignore_index=self.ignore_index,
            reduction='none'
        ).view(-1)                                   # (B*H*W,)

        # Loại pixel ignore
        valid_mask = target.view(-1) != self.ignore_index
        losses     = losses[valid_mask]              # (N_valid,)

        if losses.numel() == 0:
            return pred.sum() * 0.0                 # safe zero-grad

        # Sort giảm dần, giữ top-K
        losses_sorted, _ = losses.sort(descending=True)

        # K = max(số pixel có loss > thresh, min_kept)
        n_above_thresh = (losses_sorted > self.thresh).sum().item()
        n_keep         = int(max(n_above_thresh, self.min_kept))
        n_keep         = min(n_keep, losses_sorted.numel())   # clamp to available

        return self.loss_weight * losses_sorted[:n_keep].mean()


# =============================================================================
# Fog Consistency Loss
# =============================================================================

class FogConsistencyLoss(nn.Module):
    """Fog Consistency Loss — KL divergence giữa predictions của cùng scene
    ở hai mức fog khác nhau.

    [CITYSCAPES FOGGY - ĐIỀU CHỈNH 4]
    Cityscapes Foggy cung cấp 3 mức fog (beta = 0.005, 0.01, 0.02) cho cùng
    scene. Loss này khuyến khích model cho prediction nhất quán giữa các mức
    fog: p(y | foggy_light) ≈ p(y | foggy_heavy).

    Ý tưởng: Nhãn ground-truth là như nhau cho cả 3 mức → model không nên
    thay đổi prediction đột ngột khi fog thay đổi mức độ.

    Cơ chế:
      KL(softmax(logit_a/T) || softmax(logit_b/T)) * T²
      Temperature T làm mềm distribution, tập trung vào "dark knowledge".

    Args:
        temperature (float): Softmax temperature. Default: 4.0.
            T cao → soft distribution, focus vào class-level similarity.
            T thấp → hard distribution, gần với CE thường.
        loss_weight (float): Scalar weight. Default: 0.1.
            Giữ nhỏ để không át loss_c6 chính. Tăng lên 0.2 nếu
            train multi-beta với batch cân bằng giữa các mức fog.

    Cách dùng trong training loop:
        # Giả sử mỗi batch có ảnh foggy ở 2 mức beta khác nhau:
        logit_light = head(backbone(img_light))   # beta=0.005
        logit_heavy = head(backbone(img_heavy))   # beta=0.02
        loss_fog = fog_consistency(logit_light, logit_heavy)

    Lưu ý:
        - Chỉ có ý nghĩa khi train với NHIỀU MỨC fog trong cùng batch.
          Nếu chỉ dùng một mức beta, bỏ loss này — không có tác dụng.
        - logit_a và logit_b phải cùng shape (B, C, H, W).
        - Nên detach() một trong hai nếu muốn one-way KL (thường detach logit_b
          để logit_a học về phía logit_b — heavy fog học từ light fog).
    """

    def __init__(self,
                 temperature: float = 4.0,
                 loss_weight: float = 0.1):
        super().__init__()
        self.T           = temperature
        self.loss_weight = loss_weight

    def forward(self, logit_a: Tensor, logit_b: Tensor) -> Tensor:
        """
        Args:
            logit_a (Tensor): Logits shape (B, C, H, W) — ảnh foggy mức A.
            logit_b (Tensor): Logits shape (B, C, H, W) — ảnh foggy mức B.
                Nên .detach() logit_b nếu muốn one-way KL
                (logit_a học về phía logit_b).

        Returns:
            Tensor: Scalar consistency loss.
        """
        assert logit_a.shape == logit_b.shape, (
            f"FogConsistencyLoss: shape mismatch "
            f"{logit_a.shape} vs {logit_b.shape}"
        )

        log_p = F.log_softmax(logit_a / self.T, dim=1)   # log Q
        q     = F.softmax(logit_b / self.T, dim=1)        # P (target)

        # KL(P || Q) * T² — scale back bởi T² để gradient magnitude
        # không phụ thuộc vào lựa chọn temperature
        kl = F.kl_div(log_p, q, reduction='batchmean') * (self.T ** 2)

        return self.loss_weight * kl


# =============================================================================
# GCNetHead
# =============================================================================

class GCNetHead(BaseModule):
    """Decode head for GCNet.

    Nhận output từ GCNet backbone:
      - Training  : (c4_feat, c6_feat) — c4 cho auxiliary loss, c6 cho main loss
      - Inference : c6_feat only

    Loss:
      loss_c4 = CE(upsample(c4_logit), gt)           weight = loss_weight_aux
      loss_c6 = OHEM_CE(upsample(c6_logit), gt)      weight = 1.0   [ĐIỀU CHỈNH 3]
      loss_fog = FogConsistency(logit_a, logit_b)     weight = 0.1   [ĐIỀU CHỈNH 4]
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
        ohem_thresh (float): OHEM threshold cho loss_c6. Default: 0.7.
        ohem_min_kept (int): OHEM min pixel kept cho loss_c6. Default: 100_000.
        fog_consistency_weight (float): Weight của FogConsistencyLoss.
            Set 0.0 để disable nếu không train multi-beta. Default: 0.1.
        fog_temperature (float): Temperature cho FogConsistencyLoss. Default: 4.0.
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
                 # [CITYSCAPES FOGGY - ĐIỀU CHỈNH 3] OHEM params
                 ohem_thresh: float = 0.7,
                 ohem_min_kept: int = 100_000,
                 # [CITYSCAPES FOGGY - ĐIỀU CHỈNH 4] Fog consistency params
                 fog_consistency_weight: float = 0.1,
                 fog_temperature: float = 4.0,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels         = in_channels
        self.channels            = channels
        self.num_classes         = num_classes
        self.norm_cfg            = norm_cfg
        self.act_cfg             = act_cfg
        self.align_corners       = align_corners
        self.ignore_index        = ignore_index
        self.loss_weight_aux     = loss_weight_aux

        # ---- Main head (c6) ---------------------------------------------- #
        self.head = self._make_base_head(in_channels, channels)

        # ---- Auxiliary head (c4) ----------------------------------------- #
        self.aux_head_c4    = self._make_base_head(in_channels // 2, channels)
        self.aux_cls_seg_c4 = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Final classifiers ------------------------------------------- #
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Loss functions ---------------------------------------------- #
        # Auxiliary loss (c4): CE thường — đủ cho supervision ở feature level
        self.loss_c4 = CrossEntropyLoss(ignore_index=ignore_index,
                                         loss_weight=loss_weight_aux)

        # [CITYSCAPES FOGGY - ĐIỀU CHỈNH 3]
        # Main loss (c6): OHEM-CE thay CE thường.
        # Fog che khuất class nhỏ → OHEM focus backprop vào pixel khó.
        self.loss_c6 = OHEMCrossEntropyLoss(
            ignore_index=ignore_index,
            loss_weight=1.0,
            thresh=ohem_thresh,
            min_kept=ohem_min_kept,
        )

        # [CITYSCAPES FOGGY - ĐIỀU CHỈNH 4]
        # Fog consistency loss: chỉ active khi train multi-beta.
        # Set fog_consistency_weight=0.0 để disable hoàn toàn.
        self.fog_consistency_weight = fog_consistency_weight
        if fog_consistency_weight > 0.0:
            self.loss_fog = FogConsistencyLoss(
                temperature=fog_temperature,
                loss_weight=fog_consistency_weight,
            )
        else:
            self.loss_fog = None

        self.init_weights()

    # ---------------------------------------------------------------------- #
    # Weight init                                                              #
    # ---------------------------------------------------------------------- #

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
            Nếu loss_fog được enable, dict thêm key 'loss_fog_consistency'.

        Lưu ý [ĐIỀU CHỈNH 4]:
            FogConsistencyLoss KHÔNG được tính ở đây — nó cần logits của
            HAI ảnh khác mức beta, trong khi loss() chỉ nhận một batch.
            Gọi compute_fog_consistency() riêng từ training loop khi có
            cặp (logit_light, logit_heavy).
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
            # [ĐIỀU CHỈNH 3] OHEM-CE thay CE thường cho main loss
            'loss_c6': self.loss_c6(c6_logit, seg_label),
            'acc_seg': accuracy(c6_logit, seg_label,
                                ignore_index=self.ignore_index),
        }
        return losses

    def compute_fog_consistency(self,
                                 logit_a: Tensor,
                                 logit_b: Tensor) -> Optional[Tensor]:
        """[CITYSCAPES FOGGY - ĐIỀU CHỈNH 4]
        Tính FogConsistencyLoss giữa logits của cùng scene ở 2 mức fog.

        Gọi từ training loop khi có cặp ảnh multi-beta trong cùng batch.
        Trả về None nếu loss_fog bị disable (fog_consistency_weight=0.0).

        Args:
            logit_a (Tensor): Logits của ảnh foggy nhẹ, shape (B, C, H, W).
            logit_b (Tensor): Logits của ảnh foggy nặng, shape (B, C, H, W).
                .detach() logit_b để one-way KL (heavy fog học từ light fog):
                    logit_b = logit_b.detach()

        Returns:
            Tensor hoặc None: Scalar fog consistency loss.

        Ví dụ sử dụng trong training loop:
            feat_light = backbone(img_light)    # beta=0.005
            feat_heavy = backbone(img_heavy)    # beta=0.02

            logit_light = head(feat_light)
            logit_heavy = head(feat_heavy)

            losses = head.loss((c4, logit_light), label)

            fog_loss = head.compute_fog_consistency(
                logit_light,
                logit_heavy.detach()   # one-way: light học từ heavy distribution
            )
            if fog_loss is not None:
                losses['loss_fog_consistency'] = fog_loss
                total_loss = sum(losses.values())
        """
        if self.loss_fog is None:
            return None
        return self.loss_fog(logit_a, logit_b)

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
