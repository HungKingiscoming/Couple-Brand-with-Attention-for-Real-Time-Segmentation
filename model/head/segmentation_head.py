import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Tuple, Union

from components.components import (
    BaseModule,
    ConvModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    OptConfigType,
)


# =============================================================================
# GCNetHead
#
# NOTE (cleanup): Các thành phần sau đã bị loại bỏ khỏi bản gốc vì không được
# gọi ở bất kỳ đâu trong training loop thực tế (Trainer trong train.py tự định
# nghĩa và dùng OHEMLoss/DiceLoss riêng, không đi qua GCNetHead.loss()):
#   - accuracy()                     (chỉ được gọi trong loss(), nay đã xóa)
#   - CrossEntropyLoss               (chỉ dùng cho self.loss_c4, dead)
#   - OHEMCrossEntropyLoss           (chỉ dùng cho self.loss_c6, dead — khác với
#                                      OHEMLoss trong train.py, cái đó MỚI là cái
#                                      thực sự chạy khi train)
#   - FogConsistencyLoss             (được khai báo nhưng chưa từng gọi forward())
#   - GCNetHead.loss()                (method không được Trainer sử dụng)
#   - GCNetHead.compute_fog_consistency() (không được gọi ở đâu)
# Theo đó, các tham số constructor chỉ tồn tại để phục vụ các thành phần trên
# (ignore_index, loss_weight_aux, ohem_thresh, ohem_min_kept,
#  fog_consistency_weight, fog_temperature) cũng đã được loại bỏ.
# Nếu sau này cần tính loss ngay trong head (thay vì trong Trainer), hãy viết
# lại loss() dựa trên chính OHEMLoss/DiceLoss của train.py để tránh 2 bản loss
# OHEM khác nhau tồn tại song song như trước.
# =============================================================================

class GCNetHead(BaseModule):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 dropout_ratio: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels         = in_channels
        self.channels            = channels
        self.num_classes         = num_classes
        self.norm_cfg            = norm_cfg
        self.act_cfg             = act_cfg
        self.align_corners       = align_corners

        # ---- Main head (c6) ---------------------------------------------- #
        self.head = self._make_base_head(in_channels, channels)

        # ---- Auxiliary head (c4) ----------------------------------------- #
        # in_channels // 2: c4_feat là channels*2 = in_channels // 2
        self.aux_head_c4    = self._make_base_head(in_channels // 2, channels)
        self.aux_cls_seg_c4 = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Final classifiers ------------------------------------------- #
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(channels, num_classes, kernel_size=1)

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
        if self.training:
            if isinstance(inputs, (tuple, list)):
                assert len(inputs) == 2
                c4_feat, c6_feat = inputs
                c4_logit = self.aux_cls_seg_c4(self.aux_head_c4(c4_feat))
                c6_logit = self.cls_seg(self.dropout(self.head(c6_feat)))
                return c4_logit, c6_logit

            # Fast proxy mode explicitly asks the backbone for only the
            # fused feature, avoiding the auxiliary head entirely.
            return self.cls_seg(self.dropout(self.head(inputs)))

        else:
            # Inference: inputs có thể là tuple (nếu backbone.return_aux=True)
            # hoặc Tensor đơn — handle cả hai
            if isinstance(inputs, (tuple, list)):
                # Lấy c6_feat (index 1), bỏ c4_feat
                c6_feat = inputs[1]
            else:
                c6_feat = inputs
            return self.cls_seg(self.dropout(self.head(c6_feat)))

    # ---------------------------------------------------------------------- #
    # Helper                                                                   #
    # ---------------------------------------------------------------------- #

    def _make_base_head(self, in_channels: int, channels: int) -> nn.Sequential:
        return nn.Sequential(
            build_norm_layer(self.norm_cfg, in_channels)[1],   # [0] BN standalone
            build_activation_layer(self.act_cfg),              # [1] ReLU
            ConvModule(                                        # [2] Conv→BN→ReLU
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('conv', 'norm', 'act'),
            ),
        )

    # ---------------------------------------------------------------------- #
    # Inference helper                                                         #
    # ---------------------------------------------------------------------- #

    def predict(self,
                inputs: Union[Tensor, Tuple[Tensor, Tensor]],
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit
