import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from components.framework.engine.model import BaseModule
from components.framework.utils.typing import OptConfigType


# ============================================
# DISTILLATION LOSSES
# ============================================

class KLDivLoss(nn.Module):
    """KL Divergence Loss for logit distillation"""
    
    def __init__(self, temperature: float = 4.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """
        Args:
            student_logits: (B, C, H, W)
            teacher_logits: (B, C, H, W)
        """
        # Soften probabilities
        student_log_softmax = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_softmax = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence
        loss = F.kl_div(
            student_log_softmax,
            teacher_softmax,
            reduction='none'
        )
        
        # Scale by temperature^2
        loss = loss * (self.temperature ** 2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FeatureDistillationLoss(nn.Module):
    """Feature-based distillation with adaptive channel alignment"""
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        loss_type: str = 'mse',  # 'mse', 'cosine', 'attention'
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True)
    ):
        super().__init__()
        self.loss_type = loss_type
        
        # Channel alignment if dimensions differ
        if student_channels != teacher_channels:
            self.align = nn.Sequential(
                nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
                nn.BatchNorm2d(teacher_channels) if norm_cfg['type'] == 'BN' else nn.Identity()
            )
        else:
            self.align = nn.Identity()
    
    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        """
        Args:
            student_feat: (B, C_s, H, W)
            teacher_feat: (B, C_t, H, W)
        """
        # Align channels
        student_feat = self.align(student_feat)
        
        # Normalize features
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_feat, teacher_feat)
        
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            similarity = F.cosine_similarity(student_feat, teacher_feat, dim=1)
            loss = (1 - similarity).mean()
        
        elif self.loss_type == 'attention':
            # Attention-based distillation
            # Use spatial attention maps
            student_attn = self._compute_attention_map(student_feat)
            teacher_attn = self._compute_attention_map(teacher_feat)
            loss = F.mse_loss(student_attn, teacher_attn)
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss
    
    def _compute_attention_map(self, feat: Tensor) -> Tensor:
        """Compute spatial attention map"""
        B, C, H, W = feat.shape
        # Channel-wise sum
        attn = feat.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        # Normalize
        attn = F.normalize(attn.view(B, -1), dim=1).view(B, 1, H, W)
        return attn


class StructuralDistillationLoss(nn.Module):
    """Structural similarity distillation (pixel affinity)"""
    
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        """
        Args:
            student_feat: (B, C, H, W)
            teacher_feat: (B, C, H, W)
        """
        # Compute pixel affinity matrices
        student_affinity = self._compute_affinity(student_feat)
        teacher_affinity = self._compute_affinity(teacher_feat)
        
        # MSE on affinity matrices
        loss = F.mse_loss(student_affinity, teacher_affinity)
        return loss
    
    def _compute_affinity(self, feat: Tensor) -> Tensor:
        """Compute pixel-wise affinity within local window"""
        B, C, H, W = feat.shape
        
        # Normalize
        feat = F.normalize(feat, dim=1)
        
        # Unfold to get local windows
        unfold_feat = F.unfold(
            feat,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2
        )  # (B, C*k*k, H*W)
        
        # Reshape
        unfold_feat = unfold_feat.view(B, C, -1, H * W)  # (B, C, k*k, H*W)
        
        # Compute affinity (dot product)
        center_feat = feat.view(B, C, 1, H * W)  # Center pixel
        affinity = (center_feat * unfold_feat).sum(dim=1)  # (B, k*k, H*W)
        
        return affinity


class BoundaryDistillationLoss(nn.Module):
    """Boundary-aware distillation for segmentation"""
    
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """
        Args:
            student_logits: (B, C, H, W)
            teacher_logits: (B, C, H, W)
        """
        # Get probability maps
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        
        # Extract boundaries
        student_boundary = self._extract_boundary(student_prob)
        teacher_boundary = self._extract_boundary(teacher_prob)
        
        # MSE on boundaries
        loss = F.mse_loss(student_boundary, teacher_boundary)
        return loss
    
    def _extract_boundary(self, prob: Tensor) -> Tensor:
        """Extract boundary using Sobel filter"""
        B, C, H, W = prob.shape
        
        # Process each class separately
        boundaries = []
        for c in range(C):
            prob_c = prob[:, c:c+1, :, :]  # (B, 1, H, W)
            
            # Apply Sobel filters
            grad_x = F.conv2d(prob_c, self.sobel_x, padding=1)
            grad_y = F.conv2d(prob_c, self.sobel_y, padding=1)
            
            # Gradient magnitude
            boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            boundaries.append(boundary)
        
        boundaries = torch.cat(boundaries, dim=1)  # (B, C, H, W)
        return boundaries


# ============================================
# DISTILLATION WRAPPER
# ============================================

class DistillationWrapper(BaseModule):
    """
    Comprehensive distillation wrapper for semantic segmentation
    
    Features:
    - Logit distillation (soft labels)
    - Multi-stage feature distillation
    - Structural similarity distillation
    - Boundary-aware distillation
    - Adaptive loss weighting
    
    Args:
        student_model: Student network
        teacher_model: Teacher network (will be frozen)
        distill_cfg: Distillation configuration
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        distill_cfg: Dict
    ):
        super().__init__()
        
        self.student = student_model
        self.teacher = teacher_model
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Distillation config
        cfg = distill_cfg
        self.temperature = cfg.get('temperature', 4.0)
        self.alpha = cfg.get('alpha', 0.5)  # Balance CE and KD loss
        
        # Loss weights
        self.logit_weight = cfg.get('logit_weight', 1.0)
        self.feature_weight = cfg.get('feature_weight', 0.5)
        self.structural_weight = cfg.get('structural_weight', 0.2)
        self.boundary_weight = cfg.get('boundary_weight', 0.3)
        
        # Feature distillation stages
        self.feature_stages = cfg.get('feature_stages', ['c3', 'c4', 'c5'])
        self.feature_loss_type = cfg.get('feature_loss_type', 'mse')
        
        # Initialize losses
        self.logit_loss = KLDivLoss(temperature=self.temperature)
        
        # Feature distillation modules
        self.feature_distill_modules = nn.ModuleDict()
        
        # Get channel dimensions from config
        student_channels = cfg.get('student_channels', {
            'c3': 64, 'c4': 128, 'c5': 64
        })
        teacher_channels = cfg.get('teacher_channels', {
            'c3': 64, 'c4': 128, 'c5': 64
        })
        
        for stage in self.feature_stages:
            self.feature_distill_modules[stage] = FeatureDistillationLoss(
                student_channels=student_channels.get(stage, 64),
                teacher_channels=teacher_channels.get(stage, 64),
                loss_type=self.feature_loss_type
            )
        
        # Structural distillation
        if self.structural_weight > 0:
            self.structural_loss = StructuralDistillationLoss(kernel_size=3)
        
        # Boundary distillation
        if self.boundary_weight > 0:
            self.boundary_loss = BoundaryDistillationLoss(kernel_size=5)
    
    def forward_train(
        self,
        img: Tensor,
        gt_semantic_seg: Tensor
    ) -> Dict[str, Tensor]:
        """
        Training forward with distillation
        
        Args:
            img: (B, 3, H, W)
            gt_semantic_seg: (B, 1, H, W) or (B, H, W)
        
        Returns:
            Dict containing:
                - 'loss_ce': Cross-entropy loss
                - 'loss_kd': KL divergence loss
                - 'loss_feat': Feature distillation loss
                - 'loss_struct': Structural distillation loss
                - 'loss_bound': Boundary distillation loss
                - 'loss_total': Total loss
        """
        losses = {}
        
        # Get student predictions and features
        student_backbone_out = self.student.backbone(img)
        student_logits = self.student.decode_head(student_backbone_out)
        
        # Get teacher predictions and features
        with torch.no_grad():
            teacher_backbone_out = self.teacher.backbone(img)
            teacher_logits = self.teacher.decode_head(teacher_backbone_out)
        
        # Resize to same size if needed
        if student_logits.shape[-2:] != teacher_logits.shape[-2:]:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 1. Cross-Entropy Loss (ground truth supervision)
        loss_ce = F.cross_entropy(
            student_logits,
            gt_semantic_seg.squeeze(1).long(),
            ignore_index=255
        )
        losses['loss_ce'] = loss_ce
        
        # 2. Logit Distillation (KL divergence)
        loss_kd = self.logit_loss(student_logits, teacher_logits)
        losses['loss_kd'] = loss_kd * self.logit_weight
        
        # 3. Feature Distillation (multi-stage)
        loss_feat_total = 0.0
        for stage in self.feature_stages:
            if stage in student_backbone_out and stage in teacher_backbone_out:
                student_feat = student_backbone_out[stage]
                teacher_feat = teacher_backbone_out[stage]
                
                # Resize if needed
                if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                    teacher_feat = F.interpolate(
                        teacher_feat,
                        size=student_feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                loss_feat = self.feature_distill_modules[stage](
                    student_feat,
                    teacher_feat
                )
                loss_feat_total += loss_feat
        
        if len(self.feature_stages) > 0:
            loss_feat_total /= len(self.feature_stages)
        losses['loss_feat'] = loss_feat_total * self.feature_weight
        
        # 4. Structural Distillation
        if self.structural_weight > 0:
            # Use c5 features for structural distillation
            loss_struct = self.structural_loss(
                student_backbone_out['c5'],
                teacher_backbone_out['c5']
            )
            losses['loss_struct'] = loss_struct * self.structural_weight
        
        # 5. Boundary Distillation
        if self.boundary_weight > 0:
            loss_bound = self.boundary_loss(student_logits, teacher_logits)
            losses['loss_bound'] = loss_bound * self.boundary_weight
        
        # Total loss: weighted combination
        loss_total = (
            (1 - self.alpha) * losses['loss_ce'] +
            self.alpha * losses['loss_kd'] +
            losses.get('loss_feat', 0.0) +
            losses.get('loss_struct', 0.0) +
            losses.get('loss_bound', 0.0)
        )
        losses['loss_total'] = loss_total
        
        return losses
    
    def forward_test(self, img: Tensor) -> Tensor:
        """
        Inference forward (only student)
        
        Args:
            img: (B, 3, H, W)
        
        Returns:
            Predictions: (B, num_classes, H, W)
        """
        backbone_out = self.student.backbone(img)
        logits = self.student.decode_head(backbone_out)
        return logits
    
    @torch.no_grad()
    def validate(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Validation: compare student vs teacher
        
        Returns:
            (student_logits, teacher_logits)
        """
        student_backbone_out = self.student.backbone(img)
        student_logits = self.student.decode_head(student_backbone_out)
        
        teacher_backbone_out = self.teacher.backbone(img)
        teacher_logits = self.teacher.decode_head(teacher_backbone_out)
        
        return student_logits, teacher_logits


# ============================================
# SELF-DISTILLATION MODULE
# ============================================

class SelfDistillation(BaseModule):
    """
    Self-distillation using auxiliary heads
    
    Train multiple prediction heads and distill knowledge
    from deeper layers to shallower layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        aux_stages: List[str] = ['c3', 'c4'],  # Stages with aux heads
        temperature: float = 4.0,
        aux_weight: float = 0.4
    ):
        super().__init__()
        
        self.model = model
        self.aux_stages = aux_stages
        self.temperature = temperature
        self.aux_weight = aux_weight
        
        self.kd_loss = KLDivLoss(temperature=temperature)
    
    def forward_train(
        self,
        img: Tensor,
        gt_semantic_seg: Tensor
    ) -> Dict[str, Tensor]:
        """
        Training with self-distillation
        
        Returns:
            Dict of losses
        """
        losses = {}
        
        # Forward pass
        backbone_out = self.model.backbone(img)
        main_logits = self.model.decode_head(backbone_out)
        
        # Main loss
        loss_main = F.cross_entropy(
            main_logits,
            gt_semantic_seg.squeeze(1).long(),
            ignore_index=255
        )
        losses['loss_main'] = loss_main
        
        # Auxiliary losses + self-distillation
        if hasattr(self.model, 'auxiliary_head'):
            aux_logits = self.model.auxiliary_head(backbone_out)
            
            # Aux CE loss
            aux_logits_resized = F.interpolate(
                aux_logits,
                size=gt_semantic_seg.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            loss_aux = F.cross_entropy(
                aux_logits_resized,
                gt_semantic_seg.squeeze(1).long(),
                ignore_index=255
            )
            losses['loss_aux'] = loss_aux * self.aux_weight
            
            # Self-distillation: aux learns from main
            main_logits_resized = F.interpolate(
                main_logits,
                size=aux_logits.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            loss_self_distill = self.kd_loss(aux_logits, main_logits_resized)
            losses['loss_self_distill'] = loss_self_distill * 0.2
        
        # Total loss
        losses['loss_total'] = sum(losses.values())
        
        return losses


