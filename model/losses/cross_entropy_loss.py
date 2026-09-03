import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.losses import accuracy


# ============================================
# CROSS ENTROPY LOSS (Enhanced)
# ============================================

@MODELS.register_module()
class EnhancedCrossEntropyLoss(nn.Module):
    """
    Enhanced Cross Entropy Loss with:
    - Class weighting
    - Label smoothing
    - Ignore index
    - Online hard example mining (OHEM)
    
    Args:
        use_sigmoid (bool): Whether to use sigmoid activation
        use_mask (bool): Whether to use mask for loss calculation
        reduction (str): Reduction method ('mean', 'sum', 'none')
        class_weight (list): Weight for each class
        loss_weight (float): Global loss weight
        ignore_index (int): Index to ignore in loss calculation
        label_smoothing (float): Label smoothing factor [0, 1]
        ohem_thresh (float): Threshold for OHEM (if > 0, enable OHEM)
        ohem_keep_ratio (float): Ratio of pixels to keep in OHEM
    """
    
    def __init__(
        self,
        use_sigmoid: bool = False,
        use_mask: bool = False,
        reduction: str = 'mean',
        class_weight: Optional[List[float]] = None,
        loss_weight: float = 1.0,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        ohem_thresh: float = 0.0,
        ohem_keep_ratio: float = 0.7,
        avg_non_ignore: bool = True
    ):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.ohem_thresh = ohem_thresh
        self.ohem_keep_ratio = ohem_keep_ratio
        self.avg_non_ignore = avg_non_ignore
        
        # Class weights
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
            pred: Predictions (B, C, H, W) or (B, H, W) if use_sigmoid
            target: Ground truth (B, H, W)
            weight: Sample weights (B, H, W)
            avg_factor: Average factor for loss
            reduction_override: Override reduction method
        
        Returns:
            loss: Computed loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        
        # Move class weight to same device as pred
        if self.class_weight is not None:
            class_weight = self.class_weight.to(pred.device)
        else:
            class_weight = None
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            target = self._label_smoothing(pred, target)
        
        # Calculate loss
        if self.use_sigmoid:
            # Binary segmentation
            loss = F.binary_cross_entropy_with_logits(
                pred,
                target.float(),
                weight=class_weight,
                reduction='none'
            )
        else:
            # Multi-class segmentation
            loss = F.cross_entropy(
                pred,
                target,
                weight=class_weight,
                ignore_index=self.ignore_index,
                reduction='none'
            )
        
        # Apply OHEM if enabled
        if self.ohem_thresh > 0:
            loss = self._ohem_loss(loss, target)
        
        # Apply sample weights
        if weight is not None:
            loss = loss * weight
        
        # Reduction
        if reduction == 'mean':
            if avg_factor is None:
                if self.avg_non_ignore:
                    # Average over non-ignore pixels
                    valid_mask = (target != self.ignore_index).float()
                    avg_factor = valid_mask.sum() + 1e-6
                else:
                    avg_factor = loss.numel()
            loss = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()
        
        return loss * self.loss_weight
    
    def _label_smoothing(self, pred: Tensor, target: Tensor) -> Tensor:
        """Apply label smoothing"""
        num_classes = pred.size(1)
        
        # Create one-hot encoding
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1),
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Apply smoothing
        target_smooth = target_one_hot * (1 - self.label_smoothing) + \
                       self.label_smoothing / num_classes
        
        return target_smooth
    
    def _ohem_loss(self, loss: Tensor, target: Tensor) -> Tensor:
        """Online Hard Example Mining"""
        # Flatten loss
        loss_flat = loss.view(-1)
        target_flat = target.view(-1)
        
        # Filter out ignore index
        valid_mask = (target_flat != self.ignore_index)
        loss_valid = loss_flat[valid_mask]
        
        if loss_valid.numel() == 0:
            return loss
        
        # Sort losses
        loss_sorted, _ = torch.sort(loss_valid, descending=True)
        
        # Keep top-k hard examples
        keep_num = max(1, int(loss_valid.numel() * self.ohem_keep_ratio))
        thresh = loss_sorted[keep_num - 1]
        
        # Apply threshold
        hard_mask = (loss >= thresh) & valid_mask.view_as(loss)
        loss = torch.where(hard_mask, loss, torch.zeros_like(loss))
        
        return loss