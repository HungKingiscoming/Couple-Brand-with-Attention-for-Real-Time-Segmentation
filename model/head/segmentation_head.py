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
    pred_label = pred.argmax(dim=1)
    mask       = target != ignore_index
    correct    = (pred_label[mask] == target[mask]).sum().float()
    total      = mask.sum().float().clamp(min=1)
    return correct / total * 100.0


# =============================================================================
# Cross-entropy loss wrapper
# =============================================================================

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, loss_weight: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight  = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * F.cross_entropy(
            pred, target, ignore_index=self.ignore_index)


# =============================================================================
# Decoder primitives
# =============================================================================

class GatedFusion(nn.Module):
    """Learnable gate: g*skip + (1-g)*dec.

    Chỉ dùng tại /8 và /4 — resolution nhỏ, chi phí chấp nhận được.
    Tại /2 dùng simple add để tránh tốn activation.
    """

    def __init__(self, channels: int,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.gate = nn.Sequential(
            ConvModule(channels * 2, channels, 1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1,
                       norm_cfg=None, act_cfg=dict(type='Sigmoid')),
        )

    def forward(self, skip: Tensor, dec: Tensor) -> Tensor:
        g = self.gate(torch.cat([skip, dec], dim=1))
        return g * skip + (1.0 - g) * dec


class ResidualBlock(nn.Module):
    def __init__(self, channels: int,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(channels, channels, 3, padding=1,
                                norm_cfg=norm_cfg, act_cfg=None)
        self.act   = build_activation_layer(act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


# =============================================================================
# EnhancedDecoder — dừng tại /2, KHÔNG upsample lên full res
# =============================================================================

class EnhancedDecoder(nn.Module):
    """3-skip decoder: c6(/8) + c4_skip(/8) + c2(/4) + c1(/2).

    Pipeline:
      c6 ──GatedFusion(c4_proj)──> /8, C*4
           ResidualBlock → Conv → D
           ↓ up×2
      GatedFusion(c2_proj) ──> /4, D
           ResidualBlock → Conv → D//2
           ↓ up×2
      simple add(c1_proj) ──> /2, D//2   ← DỪNG TẠI ĐÂY

    Output tại /2 (resolution 256×512 với input 512×1024).
    Loss trong GCNetHead sẽ resize logit /2 → full res khi tính CE.
    Không có activation nào tại full resolution → tiết kiệm ~1.5 GB với bs=20.

    Args:
        c6_channels (int):  Channels của c6 (backbone output). = channels*4.
        c4_channels (int):  Channels của c4_feat (detail /8). = channels*2.
        c2_channels (int):  Channels của c2 (stem_conv2 output /4). = channels.
        c1_channels (int):  Channels của c1 (stem_conv1 output /2). = channels.
        decoder_channels (int): D. Default: 96.
    """

    def __init__(self,
                 c6_channels: int,
                 c4_channels: int,
                 c2_channels: int,
                 c1_channels: int,
                 decoder_channels: int = 96,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        D = decoder_channels

        # /8: fuse c4 vào c6
        self.c4_proj = ConvModule(c4_channels, c6_channels, 1,
                                  norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c4 = GatedFusion(c6_channels, norm_cfg, act_cfg)
        self.refine0 = nn.Sequential(
            ResidualBlock(c6_channels, norm_cfg, act_cfg),
            ConvModule(c6_channels, D, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /8 → /4: fuse c2
        self.up1     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c2_proj = ConvModule(c2_channels, D, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c2 = GatedFusion(D, norm_cfg, act_cfg)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg, act_cfg),
            ConvModule(D, D // 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /4 → /2: simple add c1 (không GatedFusion — /2 quá lớn cho gate)
        self.up2     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c1_proj = ConvModule(c1_channels, D // 2, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Dừng tại /2 — không up lên full res
        self.out_channels = D // 2

    def forward(self,
                c6: Tensor,
                c4_skip: Tensor,
                c2: Tensor,
                c1: Tensor) -> Tensor:
        # /8: fuse c4 → c6
        x = self.fuse_c4(self.c4_proj(c4_skip), c6)
        x = self.refine0(x)

        # /8 → /4: fuse c2
        x   = self.up1(x)
        c2p = self.c2_proj(c2)
        if c2p.shape[-2:] != x.shape[-2:]:
            c2p = F.interpolate(c2p, x.shape[-2:],
                                mode='bilinear', align_corners=False)
        x = self.fuse_c2(c2p, x)
        x = self.refine1(x)

        # /4 → /2: simple add c1
        x   = self.up2(x)
        c1p = self.c1_proj(c1)
        if c1p.shape[-2:] != x.shape[-2:]:
            c1p = F.interpolate(c1p, x.shape[-2:],
                                mode='bilinear', align_corners=False)
        x = x + c1p

        return x   # /2, D//2 — resize lên full res trong loss(), ngoài backward graph decoder


# =============================================================================
# GCNetHead v2 — EnhancedDecoder + aux head
# =============================================================================

class GCNetHead(BaseModule):
    """Decode head cho GCNet với EnhancedDecoder nhẹ.

    Backbone (model__13_.py) trả về:
      - Training  : (c4_feat, c6_feat)
          c4_feat : /8, channels*2   — auxiliary supervision
          c6_feat : /8, channels*4   — main feature
      - Inference : c6_feat only

    Head cần thêm c1, c2 từ backbone để dùng skip connections.
    Vì backbone hiện tại không expose c1/c2, có 2 lựa chọn:
      A) Sửa backbone.forward() để return thêm c1, c2 (khuyến nghị).
      B) Dùng head không có c1/c2 skip — chỉ c4 + c6 (fallback, xem bên dưới).

    Class này implement lựa chọn A.
    Nếu dùng lựa chọn B, xem GCNetHeadLite ở cuối file.

    Args:
        in_channels (int): Channels của c6_feat = channels*4.
        channels (int): Base channels của backbone (= 32 với GCNet-S).
        num_classes (int): Số class.
        decoder_channels (int): D trong EnhancedDecoder. Default: 96.
        norm_cfg, act_cfg, align_corners, ignore_index: như cũ.
        loss_weight_aux (float): Weight của aux loss (c4). Default: 0.4.
        dropout_ratio (float): Dropout trước cls_seg. Default: 0.1.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 decoder_channels: int = 96,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
                 dropout_ratio: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.align_corners   = align_corners
        self.ignore_index    = ignore_index
        self.loss_weight_aux = loss_weight_aux
        C = channels

        # EnhancedDecoder: c6(/8,C*4) + c4_skip(/8,C*2) + c2(/4,C) + c1(/2,C)
        self.decoder = EnhancedDecoder(
            c6_channels=in_channels,      # C*4
            c4_channels=in_channels // 2, # C*2
            c2_channels=C,
            c1_channels=C,
            decoder_channels=decoder_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        out_ch = self.decoder.out_channels  # D//2 = 48 với D=96

        # Main classifier
        self.dropout  = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(out_ch, num_classes, kernel_size=1)

        # Aux head trên c4_feat (/8, C*2) — giữ nguyên style head cũ
        self.aux_head = nn.Sequential(
            build_norm_layer(norm_cfg, in_channels // 2)[1],
            build_activation_layer(act_cfg),
            ConvModule(in_channels // 2, in_channels // 2, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.aux_cls_seg = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

        # Loss
        self.loss_main = CrossEntropyLoss(ignore_index=ignore_index, loss_weight=1.0)
        self.loss_aux  = CrossEntropyLoss(ignore_index=ignore_index,
                                          loss_weight=loss_weight_aux)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                inputs: Union[Tensor, Tuple]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Training:
            inputs = (c4_feat, c6_feat, c1, c2)
            returns (aux_logit, main_logit)  — cả 2 tại /2, loss tự resize

        Inference:
            inputs = (c6_feat, c1, c2)  hoặc chỉ c6_feat (xem GCNetHeadLite)
            returns main_logit tại /2
        """
        if self.training:
            c4_feat, c6_feat, c1, c2 = inputs

            feat      = self.decoder(c6_feat, c4_feat, c2, c1)
            main_logit = self.cls_seg(self.dropout(feat))

            aux_logit  = self.aux_cls_seg(self.aux_head(c4_feat))

            return aux_logit, main_logit

        else:
            c6_feat, c1, c2 = inputs

            feat = self.decoder(c6_feat, None, c2, c1)
            # Khi inference không có c4_feat — c4_proj sẽ không được gọi
            # Xem forward_inference() bên dưới để handle đúng
            return self.cls_seg(self.dropout(feat))

    def loss(self,
             seg_logits: Tuple[Tensor, Tensor],
             seg_label: Tensor) -> Dict[str, Tensor]:
        """Tính loss.

        Logits tại /2 (256×512), resize lên full res ở đây.
        Resize xảy ra NGOÀI backward graph của decoder → không tốn
        activation tại full resolution trong memory.
        """
        aux_logit, main_logit = seg_logits
        target_size = seg_label.shape[1:]  # (H, W) full res

        # Resize /2 → full res (chỉ tốn memory tạm thời khi tính loss)
        main_logit = resize(main_logit, size=target_size,
                            mode='bilinear', align_corners=self.align_corners)
        aux_logit  = resize(aux_logit,  size=target_size,
                            mode='bilinear', align_corners=self.align_corners)

        return {
            'loss_main': self.loss_main(main_logit, seg_label),
            'loss_aux':  self.loss_aux(aux_logit,  seg_label),
            'acc_seg':   accuracy(main_logit, seg_label,
                                  ignore_index=self.ignore_index),
        }

    def predict(self,
                inputs,
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit


# =============================================================================
# GCNetHeadLite — fallback không cần sửa backbone
# =============================================================================

class GCNetHeadLite(BaseModule):
    """Head nhẹ, tương thích 100% với backbone hiện tại (không sửa gì).

    Backbone trả về (c4_feat, c6_feat) khi training, c6_feat khi eval.
    Head này chỉ dùng c4 + c6, không có c1/c2 skip.

    Pipeline:
      c6(/8,C*4) ──GatedFusion(c4_proj)──> /8
                   ResidualBlock → Conv → D
                   ↓ up×2
                   Conv(D→D//2) ──> /4
                   ↓ up×2
                   Conv(D//2→D//2) ──> /2   ← DỪNG
      logit /2 → resize trong loss()

    Đây là lựa chọn ít xâm lấn nhất — chỉ cần thay class head,
    không đụng backbone, không đụng training loop.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 decoder_channels: int = 96,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
                 dropout_ratio: float = 0.1,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.align_corners   = align_corners
        self.ignore_index    = ignore_index
        self.loss_weight_aux = loss_weight_aux
        D = decoder_channels
        C4 = in_channels       # channels*4
        C4h = in_channels // 2 # channels*2

        # /8: fuse c4 vào c6
        self.c4_proj = ConvModule(C4h, C4, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.fuse_c4 = GatedFusion(C4, norm_cfg, act_cfg)
        self.refine0 = nn.Sequential(
            ResidualBlock(C4, norm_cfg, act_cfg),
            ConvModule(C4, D, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /8 → /4
        self.up1     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = nn.Sequential(
            ResidualBlock(D, norm_cfg, act_cfg),
            ConvModule(D, D // 2, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        # /4 → /2 — dừng tại đây
        self.up2     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine2 = ConvModule(D // 2, D // 2, 3, padding=1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Classifiers
        self.dropout  = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(D // 2, num_classes, kernel_size=1)

        # Aux head
        self.aux_head = nn.Sequential(
            build_norm_layer(norm_cfg, C4h)[1],
            build_activation_layer(act_cfg),
            ConvModule(C4h, C4h, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.aux_cls_seg = nn.Conv2d(C4h, num_classes, kernel_size=1)

        # Loss
        self.loss_main = CrossEntropyLoss(ignore_index=ignore_index, loss_weight=1.0)
        self.loss_aux  = CrossEntropyLoss(ignore_index=ignore_index,
                                          loss_weight=loss_weight_aux)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _decode(self, c6: Tensor, c4: Tensor) -> Tensor:
        x = self.fuse_c4(self.c4_proj(c4), c6)
        x = self.refine0(x)
        x = self.refine1(self.up1(x))
        x = self.refine2(self.up2(x))
        return x

    def forward(self,
                inputs: Union[Tensor, Tuple[Tensor, Tensor]]
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Training  : inputs = (c4_feat, c6_feat) → (aux_logit/2, main_logit/2)
        Inference : inputs = c6_feat            → main_logit/2

        Tương thích 100% với backbone hiện tại — không cần sửa gì.
        """
        if self.training:
            c4_feat, c6_feat = inputs

            feat       = self._decode(c6_feat, c4_feat)
            main_logit = self.cls_seg(self.dropout(feat))
            aux_logit  = self.aux_cls_seg(self.aux_head(c4_feat))

            return aux_logit, main_logit

        else:
            # Inference: backbone trả về c6_feat only
            # Không có c4 → dùng zeros làm dummy để fuse_c4 vẫn chạy
            # hoặc bypass fuse nếu c4=None
            c6_feat = inputs
            B, C, H, W = c6_feat.shape
            c4_dummy = torch.zeros(B, C // 2, H, W,
                                   dtype=c6_feat.dtype,
                                   device=c6_feat.device)
            feat = self._decode(c6_feat, c4_dummy)
            return self.cls_seg(self.dropout(feat))

    def loss(self,
             seg_logits: Tuple[Tensor, Tensor],
             seg_label: Tensor) -> Dict[str, Tensor]:
        aux_logit, main_logit = seg_logits
        target_size = seg_label.shape[1:]

        # Resize /2 → full res NẰM NGOÀI backward graph decoder
        main_logit = resize(main_logit, size=target_size,
                            mode='bilinear', align_corners=self.align_corners)
        aux_logit  = resize(aux_logit,  size=target_size,
                            mode='bilinear', align_corners=self.align_corners)

        return {
            'loss_main': self.loss_main(main_logit, seg_label),
            'loss_aux':  self.loss_aux(aux_logit,  seg_label),
            'acc_seg':   accuracy(main_logit, seg_label,
                                  ignore_index=self.ignore_index),
        }

    def predict(self,
                inputs,
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit
