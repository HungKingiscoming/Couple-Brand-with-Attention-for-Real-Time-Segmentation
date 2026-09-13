import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.losses import accuracy

@MODELS.register_module()
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Better for:
    - Extremely imbalanced datasets
    - Hard negative mining
    - One-stage detectors (but also good for segmentation)
    
    Args:
        use_sigmoid (bool): Whether to use sigmoid activation
        gamma (float): Focusing parameter (typically 2.0)
        alpha (float or list): Balancing parameter (typically 0.25)
        reduction (str): Reduction method
        loss_weight (float): Global loss weight
        ignore_index (int): Index to ignore
    """
    
    def __init__(
        self,
        use_sigmoid: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        ignore_index: int = 255,
        avg_non_ignore: bool = True
    ):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None
    ) -> Tensor:
        """
        Forward function
        
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, H, W)
        
        Returns:
            loss: Focal loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        if self.use_sigmoid:
            # Binary focal loss
            loss = self._sigmoid_focal_loss(pred, target)
        else:
            # Multi-class focal loss
            loss = self._softmax_focal_loss(pred, target)
        
        # Apply sample weights
        if weight is not None:
            loss = loss * weight
        
        # Reduction
        if reduction == 'mean':
            if avg_factor is None:
                if self.avg_non_ignore:
                    valid_mask = (target != self.ignore_index).float()
                    avg_factor = valid_mask.sum() + 1e-6
                else:
                    avg_factor = loss.numel()
            loss = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight
    
    def _sigmoid_focal_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Sigmoid focal loss for binary classification"""
        pred_sigmoid = pred.sigmoid()
        target = target.float()
        
        # Calculate pt
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        
        # Calculate focal weight
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        ) * focal_weight
        
        return loss
    
    def _softmax_focal_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Softmax focal loss for multi-class classification"""
        num_classes = pred.size(1)
        
        # Get probabilities
        pred_softmax = pred.softmax(dim=1)
        
        # One-hot encoding
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Get pt (probability of true class)
        pt = (pred_softmax * target_one_hot).sum(dim=1)
        
        # Calculate focal weight
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(
            pred, target,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply alpha
        if isinstance(self.alpha, (float, int)):
            loss = self.alpha * loss
        
        return loss