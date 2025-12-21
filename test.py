import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your modules
from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead, GCNetAuxHead


# ============================================
# CITYSCAPES COLOR PALETTE
# ============================================

CITYSCAPES_PALETTE = [
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
]

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


# ============================================
# MODEL WRAPPER
# ============================================

class GCNetSegmentor(nn.Module):
    """Complete segmentation model with backbone + head"""
    
    def __init__(
        self,
        num_classes: int,
        backbone_cfg: Dict,
        head_cfg: Dict,
        aux_head_cfg: Optional[Dict] = None
    ):
        super().__init__()
        
        # Backbone
        self.backbone = GCNetImproved(**backbone_cfg)
        
        # Main decode head
        self.decode_head = GCNetHead(
            in_channels=head_cfg['in_channels'],
            channels=head_cfg['channels'],
            num_classes=num_classes,
            decode_enabled=head_cfg.get('decode_enabled', True),
            decoder_channels=head_cfg.get('decoder_channels', 128),
            skip_channels=head_cfg.get('skip_channels', [64, 32, 32]),
            use_gated_fusion=head_cfg.get('use_gated_fusion', True),
            dropout_ratio=head_cfg.get('dropout_ratio', 0.1),
            align_corners=head_cfg.get('align_corners', False)
        )
        
        # Auxiliary head (optional)
        self.auxiliary_head = None
        if aux_head_cfg is not None:
            self.auxiliary_head = GCNetAuxHead(
                in_channels=aux_head_cfg['in_channels'],
                channels=aux_head_cfg['channels'],
                num_classes=num_classes,
                dropout_ratio=aux_head_cfg.get('dropout_ratio', 0.1),
                align_corners=aux_head_cfg.get('align_corners', False)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward"""
        features = self.backbone(x)
        logits = self.decode_head(features)
        return logits


# ============================================
# INFERENCE ENGINE
# ============================================

class Inferencer:
    """Inference engine for semantic segmentation"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        img_size: tuple = (1024, 2048),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        use_amp: bool = True,
        use_sliding_window: bool = False,
        window_size: tuple = (512, 1024),
        window_stride: tuple = (256, 512)
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.use_amp = use_amp
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.window_stride = window_stride
        
        # Preprocessing transforms
        self.transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
        print(f"✓ Inferencer initialized")
        print(f"  Device: {device}")
        print(f"  Image size: {img_size}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Sliding window: {use_sliding_window}")
    
    def preprocess(self, image_path: str) -> tuple:
        """
        Load and preprocess image
        
        Returns:
            (tensor, original_size, original_image)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        image_np = np.array(image)
        
        # Transform
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)  # (1, 3, H, W)
        
        return image_tensor, original_size, image_np
    
    @torch.no_grad()
    def predict_single(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict single image
        
        Args:
            image_tensor: (1, 3, H, W)
        
        Returns:
            pred: (1, H, W) class indices
        """
        image_tensor = image_tensor.to(self.device)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(image_tensor)
        else:
            logits = self.model(image_tensor)
        
        pred = logits.argmax(dim=1)  # (1, H, W)
        return pred
    
    @torch.no_grad()
    def predict_sliding_window(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict with sliding window for large images
        
        More accurate but slower than direct prediction
        """
        B, C, H, W = image_tensor.shape
        assert B == 1, "Sliding window only supports batch_size=1"
        
        window_h, window_w = self.window_size
        stride_h, stride_w = self.window_stride
        
        # Initialize output
        num_classes = 19
        logits_sum = torch.zeros((1, num_classes, H, W), device=self.device)
        count_map = torch.zeros((1, 1, H, W), device=self.device)
        
        # Sliding window
        for y in range(0, H - window_h + 1, stride_h):
            for x in range(0, W - window_w + 1, stride_w):
                # Extract window
                window = image_tensor[:, :, y:y+window_h, x:x+window_w]
                
                # Predict
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        window_logits = self.model(window.to(self.device))
                else:
                    window_logits = self.model(window.to(self.device))
                
                # Accumulate
                logits_sum[:, :, y:y+window_h, x:x+window_w] += window_logits
                count_map[:, :, y:y+window_h, x:x+window_w] += 1
        
        # Average overlapping predictions
        logits_avg = logits_sum / count_map.clamp(min=1)
        pred = logits_avg.argmax(dim=1)
        
        return pred
    
    def predict(self, image_path: str, return_prob: bool = False) -> Dict:
        """
        Complete prediction pipeline
        
        Args:
            image_path: Path to input image
            return_prob: Whether to return probability map
        
        Returns:
            Dict containing:
                - 'pred': (H, W) numpy array of class indices
                - 'prob': (num_classes, H, W) probability map (if return_prob=True)
                - 'original_size': (W, H) original image size
        """
        # Preprocess
        image_tensor, original_size, original_image = self.preprocess(image_path)
        
        # Predict
        if self.use_sliding_window:
            pred = self.predict_sliding_window(image_tensor)
        else:
            pred = self.predict_single(image_tensor)
        
        # Resize to original size
        pred_resized = F.interpolate(
            pred.float().unsqueeze(1),
            size=original_size[::-1],  # (H, W)
            mode='nearest'
        ).squeeze().cpu().numpy().astype(np.uint8)
        
        result = {
            'pred': pred_resized,
            'original_size': original_size
        }
        
        # Return probability if requested
        if return_prob:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(image_tensor.to(self.device))
            else:
                logits = self.model(image_tensor.to(self.device))
            
            prob = F.softmax(logits, dim=1)
            prob_resized = F.interpolate(
                prob,
                size=original_size[::-1],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).cpu().numpy()
            
            result['prob'] = prob_resized
        
        return result
    
    def predict_batch(self, image_paths: List[str], save_dir: str):
        """
        Batch prediction
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for image_path in tqdm(image_paths, desc='Inference'):
            # Predict
            result = self.predict(image_path)
            pred = result['pred']
            
            # Visualize
            vis = self.visualize(pred)
            
            # Save
            filename = Path(image_path).stem
            vis_path = save_dir / f'{filename}_pred.png'
            Image.fromarray(vis).save(vis_path)
            
            # Save raw prediction (optional)
            raw_path = save_dir / f'{filename}_pred_raw.png'
            Image.fromarray(pred).save(raw_path)
    
    @staticmethod
    def visualize(pred: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Visualize prediction with color palette
        
        Args:
            pred: (H, W) class indices
            alpha: Transparency (not used here, for overlay)
        
        Returns:
            vis: (H, W, 3) RGB visualization
        """
        H, W = pred.shape
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(CITYSCAPES_PALETTE):
            mask = pred == class_id
            vis[mask] = color
        
        return vis
    
    @staticmethod
    def overlay(image: np.ndarray, pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay prediction on original image
        
        Args:
            image: (H, W, 3) original image
            pred: (H, W) class indices
            alpha: Overlay transparency
        
        Returns:
            overlay: (H, W, 3) overlayed image
        """
        vis = Inferencer.visualize(pred)
        overlay = cv2.addWeighted(image, 1-alpha, vis, alpha, 0)
        return overlay


# ============================================
# MAIN INFERENCE SCRIPT
# ============================================

def load_model(checkpoint_path: str, device: str = 'cuda') -> GCNetSegmentor:
    """Load trained model from checkpoint"""
    
    num_classes = 19
    
    # Model config (must match training)
    backbone_cfg = {
        'in_channels': 3,
        'channels': 32,
        'ppm_channels': 128,
        'num_blocks_per_stage': [4, 4, [5, 4], [5, 4], [2, 2]],
        'use_flash_attention': True,
        'flash_attn_stage': 4,
        'flash_attn_layers': 2,
        'flash_attn_heads': 8,
        'use_se': True,
        'deploy': False  # Set True for deployment speedup
    }
    
    head_cfg = {
        'in_channels': 64,
        'channels': 128,
        'decode_enabled': True,
        'decoder_channels': 128,
        'skip_channels': [64, 32, 32],
        'use_gated_fusion': True,
        'dropout_ratio': 0.1,
        'align_corners': False
    }
    
    # Create model
    model = GCNetSegmentor(
        num_classes=num_classes,
        backbone_cfg=backbone_cfg,
        head_cfg=head_cfg,
        aux_head_cfg=None  # Not needed for inference
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best mIoU: {checkpoint.get('best_miou', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Switch to deploy mode for speed (optional)
    # model.backbone.switch_to_deploy()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Inference with GCNet')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='./inference_results',
                        help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--img_size', type=int, nargs=2, default=[1024, 2048],
                        help='Target image size (H W)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision')
    parser.add_argument('--sliding_window', action='store_true',
                        help='Use sliding window for large images')
    parser.add_argument('--window_size', type=int, nargs=2, default=[512, 1024],
                        help='Sliding window size (H W)')
    parser.add_argument('--overlay', action='store_true',
                        help='Save overlay visualization')
    parser.add_argument('--save_raw', action='store_true',
                        help='Save raw prediction masks')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device=device)
    
    # Create inferencer
    inferencer = Inferencer(
        model=model,
        device=device,
        img_size=tuple(args.img_size),
        use_amp=args.use_amp,
        use_sliding_window=args.sliding_window,
        window_size=tuple(args.window_size)
    )
    
    # Get image paths
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        # Directory of images
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_exts:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        image_paths = [str(p) for p in image_paths]
    else:
        raise ValueError(f"Input path not found: {input_path}")
    
    print(f"\nFound {len(image_paths)} images")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inference
    for image_path in tqdm(image_paths, desc='Processing'):
        # Predict
        result = inferencer.predict(image_path)
        pred = result['pred']
        
        # Filename
        filename = Path(image_path).stem
        
        # Visualize
        vis = inferencer.visualize(pred)
        vis_path = output_dir / f'{filename}_seg.png'
        Image.fromarray(vis).save(vis_path)
        
        # Overlay (optional)
        if args.overlay:
            original_image = np.array(Image.open(image_path).convert('RGB'))
            # Resize original to match prediction
            original_resized = cv2.resize(
                original_image,
                (pred.shape[1], pred.shape[0])
            )
            overlay = inferencer.overlay(original_resized, pred, alpha=0.5)
            overlay_path = output_dir / f'{filename}_overlay.png'
            Image.fromarray(overlay).save(overlay_path)
        
        # Save raw prediction (optional)
        if args.save_raw:
            raw_path = output_dir / f'{filename}_raw.png'
            Image.fromarray(pred).save(raw_path)
    
    print(f"\n✓ Inference completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
