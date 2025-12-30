import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ============================================================
# 1️⃣ LOVASZ SOFTMAX (IoU-Driven)
# ============================================================

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, ignore_index):
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
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        logits = F.softmax(logits, dim=1)
        labels = labels.view(-1)
        logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])

        valid = labels != self.ignore_index
        logits = logits[valid]
        labels = labels[valid]

        return lovasz_softmax_flat(logits, labels, self.ignore_index)


# ============================================================
# 2️⃣ BOUNDARY LOSS (NHẸ – CỰC KỲ QUAN TRỌNG)
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss using Laplacian edge
    """

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1, keepdim=True).float()
        gt = targets.unsqueeze(1).float()

        pred_edge = F.conv2d(pred, self.kernel, padding=1)
        gt_edge = F.conv2d(gt, self.kernel, padding=1)

        return F.l1_loss(pred_edge, gt_edge)


# ============================================================
# 3️⃣ FINAL COMPOSITE LOSS
# ============================================================

class CompositeSegLoss(nn.Module):
    """
    ✅ Best loss for Cityscapes
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

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.lovasz = LovaszSoftmaxLoss(ignore_index)
        self.boundary = BoundaryLoss()

        self.w_ce = w_ce
        self.w_lovasz = w_lovasz
        self.w_boundary = w_boundary

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)
        loss_boundary = self.boundary(logits, targets)

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
