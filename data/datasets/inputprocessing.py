import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


# ============================================
# Core Transforms for Semantic Segmentation
# ============================================

class SegmentationTransform:
    """
    Base class for segmentation transforms
    Ensures image and mask are transformed together
    """
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ============================================
# 1. RESIZE & CROP
# ============================================

class RandomScale(SegmentationTransform):
    """
    Random scaling - CRITICAL for semantic segmentation
    
    Scaling helps model learn scale-invariant features
    More important than rotation/flip for segmentation!
    
    Args:
        scale_range: (min_scale, max_scale) e.g., (0.5, 2.0)
        p: Probability of applying
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0), p: float = 1.0):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return image, mask
        
        scale = random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image with bilinear interpolation
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize mask with nearest neighbor (preserve label values)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return image, mask


class RandomCrop(SegmentationTransform):
    """
    Random crop to fixed size
    
    Args:
        crop_size: (height, width)
        pad_if_needed: Pad image if smaller than crop_size
        fill_value: Value to fill padding for image
        ignore_index: Value to fill padding for mask
    """
    
    def __init__(
        self, 
        crop_size: Tuple[int, int],
        pad_if_needed: bool = True,
        fill_value: int = 0,
        ignore_index: int = 255
    ):
        self.crop_h, self.crop_w = crop_size
        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.ignore_index = ignore_index
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        
        # Pad if needed
        if self.pad_if_needed:
            pad_h = max(self.crop_h - h, 0)
            pad_w = max(self.crop_w - w, 0)
            
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(
                    image, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=self.fill_value
                )
                mask = cv2.copyMakeBorder(
                    mask, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=self.ignore_index
                )
                h, w = image.shape[:2]
        
        # Random crop
        if h > self.crop_h:
            top = random.randint(0, h - self.crop_h)
        else:
            top = 0
        
        if w > self.crop_w:
            left = random.randint(0, w - self.crop_w)
        else:
            left = 0
        
        image = image[top:top + self.crop_h, left:left + self.crop_w]
        mask = mask[top:top + self.crop_h, left:left + self.crop_w]
        
        return image, mask


class ResizeAndPad(SegmentationTransform):
    """
    Resize while keeping aspect ratio, then pad to target size
    Useful for validation/inference
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int],
        fill_value: int = 0,
        ignore_index: int = 255
    ):
        self.target_h, self.target_w = target_size
        self.fill_value = fill_value
        self.ignore_index = ignore_index
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(self.target_h / h, self.target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Calculate padding
        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Pad
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.fill_value
        )
        mask = cv2.copyMakeBorder(
            mask, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.ignore_index
        )
        
        return image, mask


# ============================================
# 2. NORMALIZATION
# ============================================

class Normalize:
    """
    Normalize image with mean and std
    
    Standard ImageNet normalization:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    Cityscapes statistics (optional):
        mean = [0.485, 0.456, 0.406]  # Similar to ImageNet
        std = [0.229, 0.224, 0.225]
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB image in range [0, 255]
        Returns:
            Normalized image
        """
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image


# ============================================
# 3. DATA AUGMENTATION
# ============================================

class RandomHorizontalFlip:
    """Horizontal flip - simple but effective"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        return image, mask


class RandomRotation:
    """
    Random rotation
    Note: Less commonly used in segmentation than in classification
    Can distort spatial relationships
    """
    
    def __init__(self, degrees: float = 10, p: float = 0.5, ignore_index: int = 255):
        self.degrees = degrees
        self.p = p
        self.ignore_index = ignore_index
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() > self.p:
            return image, mask
        
        angle = random.uniform(-self.degrees, self.degrees)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        mask = cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.ignore_index
        )
        
        return image, mask


class ColorJitter:
    """
    Random color jittering
    Helps model be robust to lighting/weather changes
    """
    
    def __init__(
        self,
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.25,
        p: float = 0.5
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() > self.p:
            return image, mask
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(image)
        
        # Brightness
        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)
        
        # Contrast
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)
        
        # Saturation
        if self.saturation > 0:
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(factor)
        
        image = np.array(pil_image)
        return image, mask


class GaussianBlur:
    """
    Gaussian blur
    Helps model be robust to image quality
    """
    
    def __init__(self, kernel_size: int = 5, p: float = 0.3):
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        return image, mask


class RandomGrayscale:
    """Convert to grayscale with probability p"""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return image, mask


# ============================================
# 4. ADVANCED AUGMENTATION
# ============================================

class CutOut:
    """
    Random cutout/erasing
    Encourages model to use all parts of image
    """
    
    def __init__(
        self,
        num_holes: int = 3,
        max_h_size: int = 50,
        max_w_size: int = 50,
        fill_value: int = 0,
        p: float = 0.3
    ):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() > self.p:
            return image, mask
        
        h, w = image.shape[:2]
        
        for _ in range(self.num_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            
            y1 = max(0, y - self.max_h_size // 2)
            y2 = min(h, y + self.max_h_size // 2)
            x1 = max(0, x - self.max_w_size // 2)
            x2 = min(w, x + self.max_w_size // 2)
            
            image[y1:y2, x1:x2] = self.fill_value
        
        return image, mask


class GridMask:
    """
    Grid mask augmentation
    Effective for semantic segmentation
    """
    
    def __init__(
        self,
        ratio: float = 0.6,
        rotate: int = 45,
        p: float = 0.3
    ):
        self.ratio = ratio
        self.rotate = rotate
        self.p = p
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() > self.p:
            return image, mask
        
        h, w = image.shape[:2]
        d = random.randint(2, min(h, w) // 4)
        
        # Create grid
        grid = np.ones((h, w), dtype=np.uint8)
        for i in range(0, h, d * 2):
            grid[i:i+d, :] = 0
        for j in range(0, w, d * 2):
            grid[:, j:j+d] = 0
        
        # Apply grid
        if len(image.shape) == 3:
            grid = np.expand_dims(grid, axis=-1)
        image = image * grid
        
        return image, mask


class RandomMosaic:
    """
    Mosaic augmentation (from YOLOv4)
    Mix 4 images together - advanced technique
    """
    
    def __init__(self, p: float = 0.3, ignore_index: int = 255):
        self.p = p
        self.ignore_index = ignore_index
    
    def __call__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            images: List of 4 images
            masks: List of 4 masks
        """
        if random.random() > self.p or len(images) < 4:
            return images[0], masks[0]
        
        # Get size
        h, w = images[0].shape[:2]
        mosaic_h, mosaic_w = h * 2, w * 2
        
        # Create mosaic
        mosaic_image = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_mask = np.full((mosaic_h, mosaic_w), self.ignore_index, dtype=np.uint8)
        
        # Place 4 images
        positions = [
            (0, 0), (0, w), (h, 0), (h, w)
        ]
        
        for idx, (top, left) in enumerate(positions):
            if idx < len(images):
                img = cv2.resize(images[idx], (w, h))
                msk = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)
                
                mosaic_image[top:top+h, left:left+w] = img
                mosaic_mask[top:top+h, left:left+w] = msk
        
        # Random crop back to original size
        top = random.randint(0, h)
        left = random.randint(0, w)
        
        mosaic_image = mosaic_image[top:top+h, left:left+w]
        mosaic_mask = mosaic_mask[top:top+h, left:left+w]
        
        return mosaic_image, mosaic_mask


# ============================================
# 5. PIPELINE COMPOSITION
# ============================================

class SegmentationPipeline:
    """
    Compose multiple transforms into a pipeline
    """
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        for transform in self.transforms:
            if isinstance(transform, Normalize):
                # Normalize only applies to image
                image = transform(image)
            else:
                # Other transforms apply to both
                image, mask = transform(image, mask)
        
        return image, mask


# ============================================
# 6. PRE-DEFINED PIPELINES
# ============================================

def get_training_pipeline(
    crop_size: Tuple[int, int] = (512, 1024),
    scale_range: Tuple[float, float] = (0.5, 2.0),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    use_advanced_aug: bool = True
) -> SegmentationPipeline:
    """
    Standard training pipeline for GCNet
    
    Pipeline:
    1. RandomScale (MOST IMPORTANT!)
    2. RandomCrop
    3. RandomHorizontalFlip
    4. ColorJitter
    5. [Optional] Advanced augmentations
    6. Normalize
    """
    
    transforms = [
        # 1. Scale augmentation - CRITICAL!
        RandomScale(scale_range=scale_range, p=1.0),
        
        # 2. Crop to target size
        RandomCrop(crop_size=crop_size, pad_if_needed=True),
        
        # 3. Flip
        RandomHorizontalFlip(p=0.5),
        
        # 4. Color augmentation
        ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.25,
            p=0.5
        ),
    ]
    
    # Advanced augmentation
    if use_advanced_aug:
        transforms.extend([
            GaussianBlur(kernel_size=5, p=0.3),
            RandomGrayscale(p=0.1),
            CutOut(num_holes=3, max_h_size=50, max_w_size=50, p=0.3),
            GridMask(ratio=0.6, p=0.2),
        ])
    
    # Normalize (always last before tensor conversion)
    transforms.append(Normalize(mean=mean, std=std))
    
    return SegmentationPipeline(transforms)


def get_validation_pipeline(
    target_size: Tuple[int, int] = (512, 1024),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    resize_mode: str = 'resize'  # 'resize' or 'resize_and_pad'
) -> SegmentationPipeline:
    """
    Validation/Test pipeline - NO augmentation
    
    Args:
        resize_mode: 
            - 'resize': Direct resize (may distort)
            - 'resize_and_pad': Keep aspect ratio, pad to size
    """
    
    if resize_mode == 'resize_and_pad':
        transforms = [
            ResizeAndPad(target_size=target_size),
            Normalize(mean=mean, std=std)
        ]
    else:
        # Simple resize
        transforms = [
            lambda img, msk: (
                cv2.resize(img, (target_size[1], target_size[0]), 
                          interpolation=cv2.INTER_LINEAR),
                cv2.resize(msk, (target_size[1], target_size[0]), 
                          interpolation=cv2.INTER_NEAREST)
            ),
            Normalize(mean=mean, std=std)
        ]
    
    return SegmentationPipeline(transforms)


# ============================================
# 7. TENSOR CONVERSION
# ============================================

def to_tensor(image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors
    
    Args:
        image: (H, W, C) float32 array, normalized
        mask: (H, W) uint8 array
    
    Returns:
        image_tensor: (C, H, W) float32 tensor
        mask_tensor: (H, W) long tensor
    """
    # Image: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    
    # Mask: (H, W) -> (H, W)
    mask_tensor = torch.from_numpy(mask).long()
    
    return image_tensor, mask_tensor


# ============================================
# 8. COMPLETE TRANSFORM CLASS
# ============================================

class GCNetTransform:
    """
    Complete transform for GCNet training/validation
    
    Usage:
        # Training
        train_transform = GCNetTransform(
            mode='train',
            crop_size=(512, 1024),
            scale_range=(0.5, 2.0)
        )
        
        # Validation
        val_transform = GCNetTransform(
            mode='val',
            target_size=(512, 1024)
        )
        
        # Apply
        image, mask = train_transform(image, mask)
    """
    
    def __init__(
        self,
        mode: str = 'train',
        crop_size: Tuple[int, int] = (512, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        use_advanced_aug: bool = True,
        ignore_index: int = 255
    ):
        self.mode = mode
        self.ignore_index = ignore_index
        
        if mode == 'train':
            self.pipeline = get_training_pipeline(
                crop_size=crop_size,
                scale_range=scale_range,
                mean=mean,
                std=std,
                use_advanced_aug=use_advanced_aug
            )
        else:
            target_size = target_size or crop_size
            self.pipeline = get_validation_pipeline(
                target_size=target_size,
                mean=mean,
                std=std
            )
    
    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms and convert to tensors
        
        Args:
            image: RGB image (H, W, 3) or PIL Image
            mask: Label mask (H, W) or PIL Image
        
        Returns:
            image_tensor: (C, H, W) float tensor
            mask_tensor: (H, W) long tensor
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Apply pipeline
        image, mask = self.pipeline(image, mask)
        
        # Convert to tensor
        image_tensor, mask_tensor = to_tensor(image, mask)
        
        return image_tensor, mask_tensor


# ============================================
# 9. USAGE EXAMPLES
# ============================================

def example_basic_usage():
    """Example 1: Basic usage"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create transform
    transform = GCNetTransform(
        mode='train',
        crop_size=(512, 1024),
        scale_range=(0.5, 2.0)
    )
    
    # Dummy data
    image = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (1024, 2048), dtype=np.uint8)
    
    # Apply transform
    image_tensor, mask_tensor = transform(image, mask)
    
    print(f"Input:  image {image.shape}, mask {mask.shape}")
    print(f"Output: image {image_tensor.shape}, mask {mask_tensor.shape}")
    print(f"Image range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    print(f"Mask classes: {mask_tensor.unique().tolist()}")


def example_comparison():
    """Example 2: Compare train vs val transforms"""
    print("\n" + "=" * 60)
    print("Example 2: Train vs Val Transforms")
    print("=" * 60)
    
    # Same input
    image = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (1024, 2048), dtype=np.uint8)
    
    # Train transform
    train_transform = GCNetTransform(mode='train', crop_size=(512, 1024))
    train_img, train_mask = train_transform(image.copy(), mask.copy())
    
    # Val transform
    val_transform = GCNetTransform(mode='val', target_size=(512, 1024))
    val_img, val_mask = val_transform(image.copy(), mask.copy())
    
    print(f"Train output: {train_img.shape}, {train_mask.shape}")
    print(f"Val output:   {val_img.shape}, {val_mask.shape}")
    print("\nTrain uses: RandomScale, RandomCrop, Flip, ColorJitter, etc.")
    print("Val uses:   Direct resize + Normalize only")


def example_custom_pipeline():
    """Example 3: Custom pipeline"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Pipeline")
    print("=" * 60)
    
    # Build custom pipeline
    custom_pipeline = SegmentationPipeline([
        RandomScale(scale_range=(0.75, 1.25), p=1.0),
        RandomCrop(crop_size=(512, 512)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15, p=0.3),
        ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
        GaussianBlur(kernel_size=3, p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Apply
    image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    mask = np.random.randint(0, 21, (1024, 1024), dtype=np.uint8)
    
    image_aug, mask_aug = custom_pipeline(image, mask)
    image_tensor, mask_tensor = to_tensor(image_aug, mask_aug)
    
    print(f"Custom pipeline applied!")
    print(f"Output: {image_tensor.shape}, {mask_tensor.shape}")