import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ============================================================
# 1️⃣ LOVASZ SOFTMAX (IoU-Driven)
# ============================================================

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, ignore_index):
    """
    Multi-class Lovasz-Softmax loss
    """
    C = probs.size(1)
    losses = []
    
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        
        pc = probs[:, c]
        errors = (fg - pc).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.).to(probs.device)


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss
    Paper: "The Lovász-Softmax loss: A tractable surrogate for the 
           optimization of the intersection-over-union measure in neural networks"
    """
    
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W) - raw logits
            labels: (B, H, W) - ground truth
        """
        logits = F.softmax(logits, dim=1)
        labels = labels.view(-1)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        
        # Filter ignore index
        valid = labels != self.ignore_index
        logits = logits[valid]
        labels = labels[valid]
        
        if logits.numel() == 0:
            return torch.tensor(0.).to(logits.device)
        
        return lovasz_softmax_flat(logits, labels, self.ignore_index)


# ============================================================
# 2️⃣ BOUNDARY LOSS (NHẸ – CỰC KỲ QUAN TRỌNG)
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss using Laplacian edge detection
    
    Helps with sharp object boundaries (poles, signs, etc.)
    """
    
    def __init__(self):
        super().__init__()
        
        # Laplacian kernel for edge detection
        kernel = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        self.register_buffer("kernel", kernel)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - predictions
            targets: (B, H, W) - ground truth
        """
        # Get predicted segmentation
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1, keepdim=True).float()
        
        # Ground truth as float
        gt = targets.unsqueeze(1).float()
        
        # Detect edges using Laplacian
        pred_edge = F.conv2d(pred, self.kernel, padding=1)
        gt_edge = F.conv2d(gt, self.kernel, padding=1)
        
        # L1 loss on edges
        return F.l1_loss(pred_edge, gt_edge)


# ============================================================
# 3️⃣ FINAL COMPOSITE LOSS
# ============================================================

class CompositeSegLoss(nn.Module):
    """
    ✅ Best loss combination for Cityscapes
    
    Components:
    1. Cross-Entropy: Class-weighted base loss
    2. Lovasz-Softmax: Direct IoU optimization
    3. Boundary Loss: Sharp edge prediction
    
    Default weights:
    - CE: 1.0 (base supervision)
    - Lovasz: 1.0 (IoU optimization)
    - Boundary: 0.5 (edge refinement)
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
        w_ce: float = 1.0,
        w_lovasz: float = 1.0,
        w_boundary: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Loss weights
        self.w_ce = w_ce
        self.w_lovasz = w_lovasz
        self.w_boundary = w_boundary
        
        # Cross-Entropy Loss (class-weighted)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        # Lovasz-Softmax Loss
        self.lovasz = LovaszSoftmaxLoss(ignore_index)
        
        # Boundary Loss
        self.boundary = BoundaryLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - model predictions
            targets: (B, H, W) - ground truth labels
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Compute individual losses
        loss_ce = self.ce(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)
        loss_boundary = self.boundary(logits, targets)
        
        # Weighted combination
        total = (
            self.w_ce * loss_ce
            + self.w_lovasz * loss_lovasz
            + self.w_boundary * loss_boundary
        )
        
        return {
            "loss": total,
            "loss_ce": loss_ce.detach(),
            "loss_lovasz": loss_lovasz.detach(),
            "loss_boundary": loss_boundary.detach(),
        }


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Test the composite loss
    
    num_classes = 19
    batch_size = 2
    H, W = 512, 1024
    
    # Dummy data
    logits = torch.randn(batch_size, num_classes, H, W)
    targets = torch.randint(0, num_classes, (batch_size, H, W))
    
    # Create loss
    criterion = CompositeSegLoss(
        num_classes=num_classes,
        ignore_index=255,
        w_ce=1.0,
        w_lovasz=1.0,
        w_boundary=0.5
    )
    
    # Compute loss
    loss_dict = criterion(logits, targets)
    
    print("✅ Composite Loss Test:")
    print(f"   Total Loss:    {loss_dict['loss']:.4f}")
    print(f"   CE Loss:       {loss_dict['loss_ce']:.4f}")
    print(f"   Lovasz Loss:   {loss_dict['loss_lovasz']:.4f}")
    print(f"   Boundary Loss: {loss_dict['loss_boundary']:.4f}")
    
    # Test backward
    loss_dict['loss'].backward()
    print("\n✅ Backward pass successful!")
