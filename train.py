# ============================================
# CLEAN GCNET TRAINING PIPELINE (NO DISTILLATION)
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict

# ============================================
# IMPORTS
# ============================================

from model.backbone.model import GCNetImproved
from model.head.segmentation_head import GCNetHead, GCNetAuxHead
from data.custom import create_dataloaders

# ============================================
# MODEL CONFIG
# ============================================

class ModelConfig:
    @staticmethod
    def get_config():
        base_channels = 32
        return {
            "backbone": {
                "in_channels": 3,
                "channels": base_channels,
                "ppm_channels": 128,
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
                "use_flash_attention": False,
                "use_se": True,
                "deploy": False
            },
            "head": {
                "in_channels": base_channels * 2,   # c5 = 64
                "channels": 128,
                "decode_enabled": False,
                "skip_channels": [64, 32, 32],
                "dropout_ratio": 0.1,
                "align_corners": False
            },
            "aux_head": {
                "in_channels": base_channels * 4,   # c4 = 128
                "channels": 64,
                "dropout_ratio": 0.1,
                "align_corners": False
            }
        }

# ============================================
# SIMPLE SEGMENTOR
# ============================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head
        self.aux_head = aux_head

    def forward(self, x):
        feats = self.backbone(x)
        return self.decode_head(feats)

    def forward_train(self, x):
        feats = self.backbone(x)
        outputs = {"main": self.decode_head(feats)}
        if self.aux_head is not None:
            outputs["aux"] = self.aux_head(feats)
        return outputs

# ============================================
# LOSS (CE + DICE)
# ============================================

class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def dice_loss(self, logits, targets, eps=1e-6):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_oh = torch.nn.functional.one_hot(
            targets, num_classes
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        union = torch.sum(probs + targets_oh, dims)
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice_loss(logits, targets)

# ============================================
# EMA
# ============================================

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v * (1 - self.decay))

    def apply(self, model):
        self.backup = model.state_dict()
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=False)

# ============================================
# METRICS
# ============================================

class SegMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]
        for t, p in zip(target, pred):
            self.cm[t, p] += 1

    def miou(self):
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        return np.nanmean(inter / (union + 1e-10))

# ============================================
# TRAINER
# ============================================

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        aux_weight=0.4,
        use_amp=True,
        use_ema=True,
        save_dir="./checkpoints"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.aux_weight = aux_weight
        self.criterion = SegmentationLoss()
        self.scaler = GradScaler(enabled=use_amp)
        self.ema = EMAModel(model) if use_ema else None

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_miou = 0.0
        self.start_epoch = 0

    def train_epoch(self, loader):
        self.model.train()
        metrics = SegMetrics(num_classes=args.num_classes)
        total_loss = 0.0

        for imgs, masks in tqdm(loader, desc="Train"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).squeeze(1).long()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                outputs = self.model.forward_train(imgs)
                logits = outputs["main"]
                logits = nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = self.criterion(logits, masks)

                if "aux" in outputs:
                    aux = nn.functional.interpolate(
                        outputs["aux"], size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                    loss += self.aux_weight * self.criterion(aux, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            if self.ema:
                self.ema.update(self.model)

            pred = logits.argmax(1)
            metrics.update(pred, masks)
            total_loss += loss.item()

        return total_loss / len(loader), metrics.miou()

    @torch.no_grad()
    def validate(self, loader):
        if self.ema:
            self.ema.apply(self.model)

        self.model.eval()
        metrics = SegMetrics(num_classes=args.num_classes)

        for imgs, masks in tqdm(loader, desc="Val"):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).squeeze(1).long()
            logits = self.model(imgs)
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            pred = logits.argmax(1)
            metrics.update(pred, masks)

        miou = metrics.miou()

        if self.ema:
            self.ema.restore(self.model)

        return miou

    def save(self, epoch, miou):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "best_miou": miou
            },
            self.save_dir / "best.pth"
        )

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="./checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _ = create_dataloaders(
        args.train_txt,
        args.val_txt,
        args.batch_size,
        num_workers=4
    )

    cfg = ModelConfig.get_config()
    model = Segmentor(
        GCNetImproved(**cfg["backbone"]),
        GCNetHead(num_classes=args.num_classes, **cfg["head"]),
        GCNetAuxHead(num_classes=args.num_classes, **cfg["aux_head"])
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    trainer = Trainer(model, optimizer, scheduler, device, save_dir=args.save_dir)

    for epoch in range(args.epochs):
        loss, train_miou = trainer.train_epoch(train_loader)
        val_miou = trainer.validate(val_loader)

        print(f"[{epoch}] Loss={loss:.4f} Train mIoU={train_miou:.4f} Val mIoU={val_miou:.4f}")

        if val_miou > trainer.best_miou:
            trainer.best_miou = val_miou
            trainer.save(epoch, val_miou)

    print("✓ Training completed!")

if __name__ == "__main__":
    main()
