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
from torch.cuda.amp import autocast

# ===================== MODEL =====================
from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead

# ===================== PALETTE =====================
CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

# ===================== MODEL WRAPPER =====================
class GCNetSegmentor(nn.Module):
    def __init__(self, num_classes, backbone_cfg, head_cfg):
        super().__init__()
        self.backbone = GCNetImproved(**backbone_cfg)
        self.decode_head = GCNetHead(
            in_channels=head_cfg['in_channels'],
            channels=head_cfg['channels'],
            num_classes=num_classes,
            decode_enabled=True,
            decoder_channels=head_cfg['decoder_channels'],
            skip_channels=head_cfg['skip_channels'],
            use_gated_fusion=True,
            dropout_ratio=0.1,
            align_corners=False
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)

# ===================== INFERENCER =====================
class Inferencer:
    def __init__(
        self,
        model,
        device,
        img_size=(512, 1024),
        use_amp=True,
        sliding_window=False,
        window_size=(512, 1024),
        stride=(256, 512)
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.use_amp = use_amp
        self.sliding_window = sliding_window
        self.window_size = window_size
        self.stride = stride

        self.transform = A.Compose([
            A.Resize(*img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def preprocess(self, path):
        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        tensor = self.transform(image=img)['image'].unsqueeze(0)
        return tensor, (h, w), img

    @torch.no_grad()
    def predict(self, path):
        x, orig_size, img_np = self.preprocess(path)
        x = x.to(self.device)

        with autocast(enabled=self.use_amp):
            if not self.sliding_window:
                logits = self.model(x)
            else:
                logits = self._sliding_window(x)

        pred = logits.argmax(1)
        pred = F.interpolate(
            pred.float().unsqueeze(1),
            size=orig_size,
            mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8)

        return pred, img_np

    def _sliding_window(self, x):
        B, C, H, W = x.shape
        wh, ww = self.window_size
        sh, sw = self.stride
        num_classes = self.model.decode_head.num_classes

        logits_sum = torch.zeros((1, num_classes, H, W), device=self.device)
        count = torch.zeros((1, 1, H, W), device=self.device)

        for y in range(0, H - wh + 1, sh):
            for x0 in range(0, W - ww + 1, sw):
                patch = x[:, :, y:y+wh, x0:x0+ww]
                with autocast(enabled=self.use_amp):
                    out = self.model(patch)
                logits_sum[:, :, y:y+wh, x0:x0+ww] += out
                count[:, :, y:y+wh, x0:x0+ww] += 1

        return logits_sum / count.clamp(min=1)

    @staticmethod
    def visualize(mask):
        h, w = mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for i, c in enumerate(CITYSCAPES_PALETTE):
            out[mask == i] = c
        return out

# ===================== LOAD MODEL =====================
def load_model(ckpt, device):
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
        'deploy': True
    }

    head_cfg = {
        'in_channels': 64,
        'channels': 128,
        'decoder_channels': 128,
        'skip_channels': [64, 32, 32]
    }

    model = GCNetSegmentor(19, backbone_cfg, head_cfg)
    ckpt = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='outputs')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--sliding', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)

    infer = Inferencer(
        model,
        device,
        use_amp=args.amp,
        sliding_window=args.sliding
    )

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    paths = [args.input] if Path(args.input).is_file() else list(Path(args.input).glob('*'))

    for p in tqdm(paths):
        mask, img = infer.predict(str(p))
        vis = infer.visualize(mask)
        Image.fromarray(vis).save(out_dir / f'{p.stem}_seg.png')

    print("✓ Done")

if __name__ == "__main__":
    main()
