# ============================================
# COMPLETE DISTILLATION IMPLEMENTATION
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
import warnings

from components.components import (
    ConvModule,
    BaseModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    OptConfigType
)

# ============================================
# 1. DISTILLATION LOSSES
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
        T = self.temperature
        
        # Soften probabilities
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / T, dim=1)
        
        # KL divergence
        loss = F.kl_div(
            student_log_softmax,
            teacher_softmax,
            reduction='batchmean' if self.reduction == 'mean' else 'sum'
        )
        
        # Scale by temperature^2 (standard practice)
        loss = loss * (T ** 2)
        
        return loss


class FeatureDistillationLoss(nn.Module):
    """Feature-based distillation with channel alignment"""
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        loss_type: str = 'mse',  # 'mse', 'cosine', 'l1'
        align_channels: bool = True
    ):
        super().__init__()
        self.loss_type = loss_type
        
        # Channel alignment if needed
        if align_channels and student_channels != teacher_channels:
            self.align = nn.Sequential(
                nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
                nn.BatchNorm2d(teacher_channels)
            )
        else:
            self.align = nn.Identity()
    
    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        """
        Args:
            student_feat: (B, C_s, H, W)
            teacher_feat: (B, C_t, H, W)
        """
        # Align spatial dimensions if needed
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Align channels
        student_feat = self.align(student_feat)
        
        # Normalize features for better stability
        student_feat = F.normalize(student_feat, dim=1, p=2)
        teacher_feat = F.normalize(teacher_feat, dim=1, p=2)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(student_feat, teacher_feat)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(student_feat, teacher_feat)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            similarity = F.cosine_similarity(student_feat, teacher_feat, dim=1)
            loss = (1 - similarity).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss


class AttentionDistillationLoss(nn.Module):
    """Attention map distillation for spatial knowledge transfer"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        """
        Args:
            student_feat: (B, C, H, W)
            teacher_feat: (B, C, H, W)
        """
        # Compute attention maps (spatial importance)
        student_attn = self._compute_attention(student_feat)
        teacher_attn = self._compute_attention(teacher_feat)
        
        # Soften with temperature
        student_attn = student_attn / self.temperature
        teacher_attn = teacher_attn / self.temperature
        
        # MSE loss on attention maps
        loss = F.mse_loss(student_attn, teacher_attn)
        return loss
    
    def _compute_attention(self, feat: Tensor) -> Tensor:
        """Compute spatial attention map"""
        B, C, H, W = feat.shape
        
        # Sum over channels and normalize
        attn = feat.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        attn = attn.view(B, -1)  # (B, H*W)
        attn = F.softmax(attn, dim=1)
        attn = attn.view(B, 1, H, W)
        
        return attn


# ============================================
# 2. SEGMENTATION MODEL WITH DISTILLATION
# ============================================

class GCNetWithDistillation(BaseModule):
    """
    Complete GCNet model with distillation support
    
    Features:
    - Integrated teacher-student architecture
    - Multi-stage feature distillation
    - Logit distillation
    - Attention distillation
    - Easy to switch between training and inference mode
    
    Usage:
        # Training with distillation
        model = GCNetWithDistillation(
            backbone=student_backbone,
            decode_head=student_head,
            teacher_backbone=teacher_backbone,
            teacher_head=teacher_head,
            distillation_cfg={...}
        )
        
        # Inference (only student)
        model.eval()
        output = model(img)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        auxiliary_head: Optional[nn.Module] = None,
        teacher_backbone: Optional[nn.Module] = None,
        teacher_head: Optional[nn.Module] = None,
        distillation_cfg: Optional[Dict] = None,
        num_classes: int = 19
    ):
        super().__init__()
        
        # Student model
        self.backbone = backbone
        self.decode_head = decode_head
        self.auxiliary_head = auxiliary_head
        self.num_classes = num_classes
        
        # Teacher model (optional, for distillation)
        self.use_distillation = teacher_backbone is not None and teacher_head is not None
        
        if self.use_distillation:
            self.teacher_backbone = teacher_backbone
            self.teacher_head = teacher_head
            
            # Freeze teacher
            for param in self.teacher_backbone.parameters():
                param.requires_grad = False
            for param in self.teacher_head.parameters():
                param.requires_grad = False
            
            self.teacher_backbone.eval()
            self.teacher_head.eval()
            
            # Setup distillation
            self._setup_distillation(distillation_cfg or {})
        else:
            self.teacher_backbone = None
            self.teacher_head = None
    
    def _setup_distillation(self, cfg: Dict):
        """Setup distillation losses and configs"""
        
        # Temperature and weighting
        self.temperature = cfg.get('temperature', 4.0)
        self.alpha = cfg.get('alpha', 0.5)  # Balance between CE and KD
        
        # Loss weights
        self.logit_weight = cfg.get('logit_weight', 1.0)
        self.feature_weight = cfg.get('feature_weight', 0.5)
        self.attention_weight = cfg.get('attention_weight', 0.3)
        
        # Feature distillation stages
        self.feature_stages = cfg.get('feature_stages', ['c3', 'c4', 'c5'])
        
        # Initialize losses
        self.logit_loss = KLDivLoss(temperature=self.temperature)
        self.attention_loss = AttentionDistillationLoss(temperature=1.0)
        
        # Feature distillation modules for each stage
        self.feature_distill = nn.ModuleDict()
        
        # Get channel configs (you may need to adjust these)
        student_channels = cfg.get('student_channels', {
            'c3': 64,   # channels * 2
            'c4': 128,  # channels * 4
            'c5': 64    # channels * 2
        })
        
        teacher_channels = cfg.get('teacher_channels', {
            'c3': 64,
            'c4': 128,
            'c5': 64
        })
        
        for stage in self.feature_stages:
            self.feature_distill[stage] = FeatureDistillationLoss(
                student_channels=student_channels.get(stage, 64),
                teacher_channels=teacher_channels.get(stage, 64),
                loss_type='mse',
                align_channels=True
            )
        
        print(f"✓ Distillation enabled: T={self.temperature}, α={self.alpha}")
        print(f"  Feature stages: {self.feature_stages}")
        print(f"  Weights - Logit: {self.logit_weight}, Feature: {self.feature_weight}, Attention: {self.attention_weight}")
    
    def forward(
        self,
        img: Tensor,
        gt_semantic_seg: Optional[Tensor] = None,
        return_loss: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with optional distillation
        
        Args:
            img: (B, 3, H, W)
            gt_semantic_seg: (B, 1, H, W) - ground truth, optional
            return_loss: If True, compute and return losses (training mode)
        
        Returns:
            If return_loss=False: predictions (B, num_classes, H, W)
            If return_loss=True: dict of losses
        """
        if return_loss:
            return self.forward_train(img, gt_semantic_seg)
        else:
            return self.forward_test(img)
    
    def forward_train(
        self,
        img: Tensor,
        gt_semantic_seg: Tensor
    ) -> Dict[str, Tensor]:
        """
        Training forward with distillation
        
        Returns:
            Dict containing all losses
        """
        losses = {}
        
        # ========================================
        # STUDENT FORWARD
        # ========================================
        student_feats = self.backbone(img)
        student_logits = self.decode_head(student_feats)
        
        # Resize logits to match GT
        if student_logits.shape[-2:] != gt_semantic_seg.shape[-2:]:
            student_logits_resized = F.interpolate(
                student_logits,
                size=gt_semantic_seg.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            student_logits_resized = student_logits
        
        # ========================================
        # 1. CROSS-ENTROPY LOSS (Ground Truth)
        # ========================================
        loss_ce = F.cross_entropy(
            student_logits_resized,
            gt_semantic_seg.squeeze(1).long(),
            ignore_index=255,
            reduction='mean'
        )
        losses['loss_ce'] = loss_ce
        
        # ========================================
        # 2. AUXILIARY LOSS (if exists)
        # ========================================
        if self.auxiliary_head is not None:
            aux_logits = self.auxiliary_head(student_feats)
            
            # Resize to match GT
            if aux_logits.shape[-2:] != gt_semantic_seg.shape[-2:]:
                aux_logits = F.interpolate(
                    aux_logits,
                    size=gt_semantic_seg.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            loss_aux = F.cross_entropy(
                aux_logits,
                gt_semantic_seg.squeeze(1).long(),
                ignore_index=255,
                reduction='mean'
            )
            losses['loss_aux'] = loss_aux * 0.4
        
        # ========================================
        # 3. DISTILLATION LOSSES (if teacher exists)
        # ========================================
        if self.use_distillation and self.training:
            with torch.no_grad():
                # Teacher forward
                teacher_feats = self.teacher_backbone(img)
                teacher_logits = self.teacher_head(teacher_feats)
                
                # Resize teacher logits to match student
                if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
                    teacher_logits = F.interpolate(
                        teacher_logits,
                        size=student_logits.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
            
            # 3.1 Logit Distillation (KL Divergence)
            loss_kd = self.logit_loss(student_logits, teacher_logits)
            losses['loss_kd'] = loss_kd * self.logit_weight
            
            # 3.2 Feature Distillation (Multi-stage)
            loss_feat_total = 0.0
            num_stages = 0
            
            for stage in self.feature_stages:
                if stage in student_feats and stage in teacher_feats:
                    loss_feat = self.feature_distill[stage](
                        student_feats[stage],
                        teacher_feats[stage]
                    )
                    loss_feat_total += loss_feat
                    num_stages += 1
            
            if num_stages > 0:
                loss_feat_total /= num_stages
                losses['loss_feat'] = loss_feat_total * self.feature_weight
            
            # 3.3 Attention Distillation (on c5)
            if 'c5' in student_feats and 'c5' in teacher_feats:
                loss_attn = self.attention_loss(
                    student_feats['c5'],
                    teacher_feats['c5']
                )
                losses['loss_attn'] = loss_attn * self.attention_weight
        
        # ========================================
        # TOTAL LOSS
        # ========================================
        if self.use_distillation and 'loss_kd' in losses:
            # With distillation: balance CE and KD
            loss_total = (
                (1 - self.alpha) * losses['loss_ce'] +
                self.alpha * losses['loss_kd'] +
                losses.get('loss_feat', 0.0) +
                losses.get('loss_attn', 0.0) +
                losses.get('loss_aux', 0.0)
            )
        else:
            # Without distillation: standard training
            loss_total = (
                losses['loss_ce'] +
                losses.get('loss_aux', 0.0)
            )
        
        losses['loss'] = loss_total
        
        return losses
    
    def forward_test(self, img: Tensor) -> Tensor:
        """
        Inference forward (student only)
        
        Args:
            img: (B, 3, H, W)
        
        Returns:
            Predictions: (B, num_classes, H, W)
        """
        # Only use student network
        feats = self.backbone(img)
        logits = self.decode_head(feats)
        return logits
    
    def predict(
        self,
        img: Tensor,
        return_probs: bool = False,
        return_features: bool = False
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Convenient prediction method
        
        Args:
            img: (B, 3, H, W)
            return_probs: Return probabilities instead of logits
            return_features: Also return backbone features
        
        Returns:
            predictions: (B, num_classes, H, W) or (B, H, W)
            probs (optional): (B, num_classes, H, W)
            features (optional): Dict of features
        """
        self.eval()
        
        with torch.no_grad():
            feats = self.backbone(img)
            logits = self.decode_head(feats)
            
            if return_probs:
                probs = F.softmax(logits, dim=1)
                
                if return_features:
                    return logits, probs, feats
                else:
                    return logits, probs
            else:
                if return_features:
                    return logits, feats
                else:
                    return logits
    
    def switch_to_deploy(self):
        """Switch backbone to deploy mode (fuse BN layers)"""
        if hasattr(self.backbone, 'switch_to_deploy'):
            self.backbone.switch_to_deploy()
            print("✓ Switched to deploy mode (BN layers fused)")


# ============================================
# 3. TRAINING UTILITIES
# ============================================

def create_distillation_model(
    student_config: Dict,
    teacher_config: Optional[Dict] = None,
    distillation_config: Optional[Dict] = None,
    num_classes: int = 19
) -> GCNetWithDistillation:
    """
    Factory function to create model with distillation
    
    Args:
        student_config: Config for student model
        teacher_config: Config for teacher model (None for no distillation)
        distillation_config: Distillation settings
        num_classes: Number of segmentation classes
    
    Returns:
        Complete model with distillation support
    """
    from model.gcnet_improved import GCNetImproved
    from model.head import GCNetHead, GCNetAuxHead
    
    # Create student
    student_backbone = GCNetImproved(**student_config['backbone'])
    
    student_head = GCNetHead(
        in_channels=student_config['head']['in_channels'],
        channels=student_config['head']['channels'],
        num_classes=num_classes,
        **student_config['head'].get('extra_args', {})
    )
    
    # Auxiliary head (optional)
    if 'aux_head' in student_config:
        aux_head = GCNetAuxHead(
            in_channels=student_config['aux_head']['in_channels'],
            channels=student_config['aux_head']['channels'],
            num_classes=num_classes
        )
    else:
        aux_head = None
    
    # Create teacher (if distillation enabled)
    if teacher_config is not None:
        teacher_backbone = GCNetImproved(**teacher_config['backbone'])
        teacher_head = GCNetHead(
            in_channels=teacher_config['head']['in_channels'],
            channels=teacher_config['head']['channels'],
            num_classes=num_classes,
            **teacher_config['head'].get('extra_args', {})
        )
        
        # Load teacher weights if provided
        if 'pretrained' in teacher_config:
            print(f"Loading teacher weights from {teacher_config['pretrained']}")
            checkpoint = torch.load(teacher_config['pretrained'], map_location='cpu')
            teacher_backbone.load_state_dict(checkpoint['backbone'])
            teacher_head.load_state_dict(checkpoint['head'])
    else:
        teacher_backbone = None
        teacher_head = None
    
    # Create complete model
    model = GCNetWithDistillation(
        backbone=student_backbone,
        decode_head=student_head,
        auxiliary_head=aux_head,
        teacher_backbone=teacher_backbone,
        teacher_head=teacher_head,
        distillation_cfg=distillation_config,
        num_classes=num_classes
    )
    
    return model


# ============================================
# 4. EXAMPLE CONFIGURATIONS
# ============================================

def get_default_configs():
    """Get default configurations for distillation"""
    
    # Student config (smaller, faster)
    student_config = {
        'backbone': {
            'in_channels': 3,
            'channels': 16,  # Smaller than teacher
            'ppm_channels': 64,
            'num_blocks_per_stage': [2, 2, [3, 2], [3, 2], [2, 1]],
            'use_flash_attention': False,
            'use_se': True,
        },
        'head': {
            'in_channels': 32,  # channels * 2
            'channels': 64,
            'extra_args': {
                'decode_enabled': True,
                'decoder_channels': 64
            }
        },
        'aux_head': {
            'in_channels': 64,  # channels * 4
            'channels': 32
        }
    }
    
    # Teacher config (larger, more accurate)
    teacher_config = {
        'backbone': {
            'in_channels': 3,
            'channels': 32,  # Standard size
            'ppm_channels': 128,
            'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
            'use_flash_attention': True,
            'use_se': True,
        },
        'head': {
            'in_channels': 64,  # channels * 2
            'channels': 128,
            'extra_args': {
                'decode_enabled': True,
                'decoder_channels': 128
            }
        },
        'pretrained': 'checkpoints/teacher_best.pth'  # Path to pretrained teacher
    }
    
    # Distillation config
    distillation_config = {
        'temperature': 4.0,
        'alpha': 0.5,  # 50% CE, 50% KD
        'logit_weight': 1.0,
        'feature_weight': 0.5,
        'attention_weight': 0.3,
        'feature_stages': ['c3', 'c4', 'c5'],
        'student_channels': {
            'c3': 32,   # student channels * 2
            'c4': 64,   # student channels * 4
            'c5': 32    # student channels * 2
        },
        'teacher_channels': {
            'c3': 64,   # teacher channels * 2
            'c4': 128,  # teacher channels * 4
            'c5': 64    # teacher channels * 2
        }
    }
    
    return student_config, teacher_config, distillation_config


# ============================================
# 5. USAGE EXAMPLE
# ============================================

if __name__ == '__main__':
    """
    Example usage of distillation
    """
    
    # Get default configs
    student_cfg, teacher_cfg, distill_cfg = get_default_configs()
    
    # Create model with distillation
    model = create_distillation_model(
        student_config=student_cfg,
        teacher_config=teacher_cfg,
        distillation_config=distill_cfg,
        num_classes=19  # Cityscapes
    )
    
    # Training example
    model.train()
    
    # Dummy data
    img = torch.randn(2, 3, 512, 1024)
    gt = torch.randint(0, 19, (2, 1, 512, 1024))
    
    # Forward with distillation
    losses = model.forward_train(img, gt)
    
    print("Training losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Inference example
    model.eval()
    logits = model.forward_test(img)
    print(f"\nInference output shape: {logits.shape}")
    
    # Prediction with probabilities
    logits, probs = model.predict(img, return_probs=True)
    print(f"Probabilities shape: {probs.shape}")
