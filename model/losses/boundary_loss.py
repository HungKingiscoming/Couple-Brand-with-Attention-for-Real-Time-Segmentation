import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.losses import accuracy

class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better edge segmentation
    
    Focuses on pixels near class boundaries
    
    Args:
        loss_weight (float): Global loss weight
        ignore_index (int): Index to ignore
        kernel_size (int): Kernel size for boundary detection
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        ignore_index: int = 255,
        kernel_size: int = 5
    ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.kernel_size = kernel_size
        
        # Laplacian kernel for edge detection
        self.laplacian_kernel = self._get_laplacian_kernel()
    
    def _get_laplacian_kernel(self) -> Tensor:
        """Get Laplacian kernel for edge detection"""
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        kernel[0, 0, self.kernel_size // 2, self.kernel_size // 2] = \
            -(self.kernel_size ** 2 - 1)
        return kernel
    
    def _get_boundary_mask(self, target: Tensor) -> Tensor:
        """Extract boundary mask from target"""
        # Move kernel to same device
        kernel = self.laplacian_kernel.to(target.device)
        
        # Apply Laplacian filter
        target_float = target.unsqueeze(1).float()
        
        # Pad target
        padding = self.kernel_size // 2
        target_padded = F.pad(target_float, (padding,) * 4, mode='replicate')
        
        # Apply convolution
        boundary = F.conv2d(target_padded, kernel, padding=0)
        boundary = boundary.squeeze(1)
        
        # Binarize boundary
        boundary_mask = (boundary.abs() > 0).float()
        
        return boundary_mask
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward function
        
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, H, W)
        
        Returns:
            loss: Boundary loss
        """
        # Get boundary mask
        boundary_mask = self._get_boundary_mask(target)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(
            pred, target,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Apply boundary mask (focus on boundaries)
        loss = loss * (1 + boundary_mask)
        
        # Apply sample weights
        if weight is not None:
            loss = loss * weight
        
        # Average loss
        valid_mask = (target != self.ignore_index).float()
        avg_factor = valid_mask.sum() + 1e-6
        loss = loss.sum() / avg_factor
        
        return loss * self.loss_weight

