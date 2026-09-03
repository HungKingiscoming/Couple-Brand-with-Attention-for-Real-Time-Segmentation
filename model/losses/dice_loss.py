import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.losses import accuracy

@MODELS.register_module()
class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Better for:
    - Imbalanced datasets
    - Small objects
    - Binary/multi-class segmentation
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        exponent (int): Exponent for denominator (1 or 2)
        reduction (str): Reduction method
        class_weight (list): Weight for each class
        loss_weight (float): Global loss weight
        ignore_index (int): Index to ignore
        use_sigmoid (bool): Whether to use sigmoid activation
        naive_dice (bool): Use naive dice (simple implementation)
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        exponent: int = 2,
        reduction: str = 'mean',
        class_weight: Optional[List[float]] = None,
        loss_weight: float = 1.0,
        ignore_index: int = 255,
        use_sigmoid: bool = False,
        naive_dice: bool = False
    ):
        super().__init__()
        
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.use_sigmoid = use_sigmoid
        self.naive_dice = naive_dice
        
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
        else:
            self.class_weight = None
    
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
            loss: Dice loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        # Apply activation
        if self.use_sigmoid:
            pred = pred.sigmoid()
        else:
            pred = pred.softmax(dim=1)
        
        # Get number of classes
        num_classes = pred.size(1)
        
        # One-hot encoding for target
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Ignore index mask
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * valid_mask
            target_one_hot = target_one_hot * valid_mask
        
        # Calculate dice loss per class
        if self.naive_dice:
            loss = self._naive_dice_loss(pred, target_one_hot)
        else:
            loss = self._dice_loss(pred, target_one_hot)
        
        # Apply class weights
        if self.class_weight is not None:
            class_weight = self.class_weight.to(pred.device)
            loss = loss * class_weight
        
        # Apply sample weights
        if weight is not None:
            loss = loss * weight
        
        # Reduction
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight
    
    def _dice_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculate dice loss"""
        # Flatten predictions and targets
        pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
        target_flat = target.reshape(target.size(0), target.size(1), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        
        if self.exponent == 1:
            denominator = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        else:
            denominator = (pred_flat ** 2).sum(dim=2) + (target_flat ** 2).sum(dim=2)
        
        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Dice loss (1 - dice)
        loss = 1 - dice
        
        return loss.mean(dim=0)  # Average over batch
    
    def _naive_dice_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Naive dice loss (simpler but less stable)"""
        intersection = (pred * target).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        
        return loss