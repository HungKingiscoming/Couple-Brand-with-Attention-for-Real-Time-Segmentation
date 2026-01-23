"""
====================================================================
GIẢI PHÁP HOÀN CHỈNH: FIX MISSING BN RUNNING STATS
====================================================================

NGUYÊN NHÂN:
- Train code convert BN → GN
- Nhưng checkpoint lại có BN keys
- BN keys KHÔNG có running_mean/running_var
- Eval mode dùng default stats (0/1) → mIoU = 0

GIẢI PHÁP:
1. TRAIN: Đảm bảo lưu đúng norm type
2. EVAL: Warmup BN stats trước khi eval
3. INFERENCE: Dùng đúng norm type như training
====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict


# ============================================================
# SOLUTION 1: FIX TRAINING - LƯU CHECKPOINT ĐÚNG
# ============================================================

def save_checkpoint_with_correct_norm(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    metrics: dict,
    save_path: str,
    global_step: int = 0,
    best_miou: float = 0.0
):
    """
    ✅ FIXED: Save checkpoint với đúng norm type + running stats
    
    Thay thế hàm save_checkpoint cũ trong Trainer
    """
    
    # Detect norm type
    has_bn = any(isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) 
                for m in model.modules())
    has_gn = any(isinstance(m, nn.GroupNorm) 
                for m in model.modules())
    
    print(f"\n📦 Saving checkpoint...")
    print(f"   Norm type: {'BatchNorm' if has_bn else 'GroupNorm' if has_gn else 'Unknown'}")
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Verify BN running stats if using BN
    if has_bn:
        bn_stats_count = sum(1 for k in state_dict.keys() 
                            if 'running_mean' in k or 'running_var' in k)
        print(f"   BN running stats: {bn_stats_count} tensors")
        
        if bn_stats_count == 0:
            print(f"   ⚠️  WARNING: BatchNorm detected but NO running stats!")
            print(f"   This checkpoint will NOT work in eval mode!")
    
    checkpoint = {
        'epoch': epoch,
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict(),
        'best_miou': best_miou,
        'metrics': metrics,
        'global_step': global_step,
        # ✅ THÊM metadata để debug
        'norm_type': 'BatchNorm' if has_bn else 'GroupNorm' if has_gn else 'Mixed',
        'has_running_stats': bn_stats_count > 0 if has_bn else False,
    }
    
    torch.save(checkpoint, save_path)
    print(f"   ✅ Saved: {save_path}")


# ============================================================
# SOLUTION 2: WARMUP BN RUNNING STATS
# ============================================================

@torch.no_grad()
def warmup_bn_stats(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    num_batches: int = 100,
    reset_stats: bool = True
):
    """
    ✅ Warmup BatchNorm running statistics
    
    KHI NÀO CẦN:
    - Checkpoint không có running_mean/running_var
    - Hoặc chuyển từ GN checkpoint sang BN model
    - Hoặc model train với BN nhưng chưa có stats
    
    Args:
        model: Model cần warmup
        dataloader: Train/Val dataloader
        device: 'cuda' or 'cpu'
        num_batches: Số batch để collect stats (100-200 là đủ)
        reset_stats: Reset stats về 0 trước khi warmup
    """
    print("=" * 70)
    print("🔥 WARMING UP BATCHNORM RUNNING STATISTICS")
    print("=" * 70)
    
    # Check if model has BN
    bn_layers = [m for m in model.modules() 
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))]
    
    if not bn_layers:
        print("⚠️  No BatchNorm layers found! Skipping warmup.")
        return
    
    print(f"📊 Found {len(bn_layers)} BatchNorm layers")
    print(f"🔄 Processing {num_batches} batches...")
    
    # Set to train mode (CRITICAL!)
    model.train()
    
    # Reset stats if needed
    if reset_stats:
        for m in bn_layers:
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()
            # Use cumulative moving average (no momentum)
            m.momentum = None
    
    # Collect stats
    num_processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches)):
        if batch_idx >= num_batches:
            break
        
        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        images = images.to(device)
        
        # Forward pass (no gradient needed)
        _ = model(images)
        num_processed += 1
    
    print(f"\n✅ Warmup complete! Processed {num_processed} batches")
    
    # Verify stats were updated
    sample_bn = bn_layers[0]
    if hasattr(sample_bn, 'running_mean'):
        mean_val = sample_bn.running_mean.abs().mean().item()
        var_val = sample_bn.running_var.mean().item()
        print(f"   Sample stats: mean={mean_val:.4f}, var={var_val:.4f}")
        
        if mean_val < 1e-6 and abs(var_val - 1.0) < 1e-6:
            print(f"   ⚠️  WARNING: Stats look like defaults! May need more batches.")
    
    # Switch back to eval
    model.eval()
    print("=" * 70)


# ============================================================
# SOLUTION 3: COMPLETE EVALUATION WITH BN WARMUP
# ============================================================

@torch.no_grad()
def evaluate_with_bn_warmup(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = 'cuda',
    num_classes: int = 19,
    ignore_index: int = 255,
    warmup_batches: int = 100,
    use_warmup: bool = True
):
    """
    ✅ COMPLETE: Eval với BN warmup tự động
    
    Usage:
        metrics = evaluate_with_bn_warmup(
            model, val_loader, device='cuda',
            warmup_batches=100  # ← Tùy chỉnh
        )
    """
    
    # Step 1: Warmup BN if needed
    if use_warmup:
        warmup_bn_stats(
            model, val_loader, device, 
            num_batches=warmup_batches,
            reset_stats=True
        )
    
    # Step 2: Standard evaluation
    print("\n" + "=" * 70)
    print("📊 EVALUATING MODEL")
    print("=" * 70)
    
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for batch in tqdm(val_loader, desc="Validation"):
        if isinstance(batch, (tuple, list)):
            images, masks = batch[0], batch[1]
        else:
            images, masks = batch, None
        
        images = images.to(device)
        
        # Forward
        if hasattr(model, 'forward_train'):
            outputs = model.forward_train(images)
            logits = outputs.get('main', outputs)
        else:
            logits = model(images)
        
        # Handle dict output
        if isinstance(logits, dict):
            logits = logits.get('c5', logits.get('out', list(logits.values())[0]))
        
        # Resize to match target
        if masks is not None:
            masks = masks.to(device).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            
            H, W = masks.shape[-2:]
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(
                    logits, size=(H, W), 
                    mode='bilinear', align_corners=False
                )
            
            # Predictions
            preds = logits.argmax(1).cpu().numpy()
            targets = masks.cpu().numpy()
            
            # Update confusion matrix
            mask = (targets >= 0) & (targets < num_classes)
            label = num_classes * targets[mask].astype('int') + preds[mask]
            count = np.bincount(label, minlength=num_classes**2)
            confusion_matrix += count.reshape(num_classes, num_classes)
    
    # Compute metrics
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
    iou = intersection / (union + 1e-10)
    
    miou = np.nanmean(iou)
    acc = intersection.sum() / (confusion_matrix.sum() + 1e-10)
    
    print(f"\n✅ Evaluation Results:")
    print(f"   mIoU: {miou:.4f}")
    print(f"   Acc:  {acc:.4f}")
    print("=" * 70)
    
    return {
        'miou': miou,
        'accuracy': acc,
        'per_class_iou': iou,
        'confusion_matrix': confusion_matrix
    }


# ============================================================
# SOLUTION 4: FIX EXISTING CHECKPOINT (EMERGENCY)
# ============================================================

def fix_checkpoint_add_bn_stats(
    checkpoint_path: str,
    output_path: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    warmup_batches: int = 200
):
    """
    ✅ EMERGENCY: Sửa checkpoint cũ - Thêm BN running stats
    
    KHI NÀO DÙNG:
    - Đã train xong nhưng checkpoint thiếu BN stats
    - Không muốn train lại
    - Có dataloader để collect stats
    
    Args:
        checkpoint_path: Checkpoint cũ (thiếu stats)
        output_path: Checkpoint mới (có stats)
        model: Model architecture (chưa load weight)
        dataloader: Dataloader để collect stats
        warmup_batches: Số batch để collect
    """
    print("=" * 70)
    print("🔧 FIXING CHECKPOINT: ADDING BN RUNNING STATS")
    print("=" * 70)
    
    # Load old checkpoint
    print(f"\n📥 Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load weights to model
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device)
    
    # Warmup BN
    print(f"\n🔥 Collecting BN stats from {warmup_batches} batches...")
    warmup_bn_stats(model, dataloader, device, warmup_batches, reset_stats=True)
    
    # Get new state dict with stats
    new_state_dict = model.state_dict()
    
    # Count stats
    stats_count = sum(1 for k in new_state_dict.keys() 
                     if 'running_mean' in k or 'running_var' in k)
    
    print(f"\n📊 New checkpoint will have:")
    print(f"   Total keys: {len(new_state_dict)}")
    print(f"   BN running stats: {stats_count} tensors")
    
    # Update checkpoint
    ckpt['model'] = new_state_dict
    ckpt['fixed'] = True
    ckpt['fixed_info'] = {
        'original_path': checkpoint_path,
        'warmup_batches': warmup_batches,
        'has_running_stats': stats_count > 0
    }
    
    # Save
    torch.save(ckpt, output_path)
    print(f"\n✅ Fixed checkpoint saved: {output_path}")
    print("=" * 70)


# ============================================================
# SOLUTION 5: RECOMMENDED TRAINING WORKFLOW
# ============================================================

def recommended_training_modifications():
    """
    Hướng dẫn sửa train.py để tránh vấn đề này
    """
    
    print("=" * 70)
    print("📘 RECOMMENDED MODIFICATIONS TO train.py")
    print("=" * 70)
    
    print("""
OPTION 1: TRAIN VỚI GROUPNORM (ĐƠN GIẢN NHẤT)
════════════════════════════════════════════════════════════════════
✅ Ưu điểm:
   - Không cần running stats
   - Stable với batch size nhỏ
   - Inference đơn giản

📝 Code changes:

# Line ~790 trong train.py - GIỮ NGUYÊN
model = replace_bn_with_gn(model)  # ← GIỮ DÒNG NÀY

# Line ~950 trong save_checkpoint - THÊM VERIFY
save_checkpoint_with_correct_norm(...)  # ← DÙNG HÀM MỚI

# Inference code
model = replace_bn_with_gn(model)
model.load_state_dict(checkpoint['model'])
model.eval()
output = model(input)  # ← XONG!

════════════════════════════════════════════════════════════════════


OPTION 2: TRAIN VỚI BATCHNORM (NẾU BATCH SIZE LỚN ≥16)
════════════════════════════════════════════════════════════════════
⚠️  Lưu ý:
   - Cần đảm bảo BN ở train mode khi train
   - Checkpoint TỰ ĐỘNG có running stats
   - Eval mode hoạt động bình thường

📝 Code changes:

# Line ~790 trong train.py - XÓA HOẶC COMMENT
# model = replace_bn_with_gn(model)  # ← XÓA DÒNG NÀY!

# Hoặc thêm flag
if args.use_groupnorm:  # ← THÊM ARG MỚI
    model = replace_bn_with_gn(model)

# Line ~950 - VERIFY STATS
save_checkpoint_with_correct_norm(...)

# Inference code - ĐƠN GIẢN
model.load_state_dict(checkpoint['model'])
model.eval()  # ← BN tự động dùng running stats
output = model(input)

════════════════════════════════════════════════════════════════════


OPTION 3: HYBRID - TRAIN BN, EVAL GN (ADVANCED)
════════════════════════════════════════════════════════════════════
🎯 Khi nào dùng:
   - Train batch lớn (BN tốt hơn)
   - Inference batch nhỏ/variable (GN ổn định hơn)

📝 Code:

# Training - KHÔNG convert
# (model giữ nguyên BN)

# Inference - Convert checkpoint
from convert_utils import convert_bn_to_gn_checkpoint

# Load BN checkpoint
ckpt = torch.load('checkpoint_bn.pth')

# Convert to GN
gn_state_dict = convert_bn_to_gn_checkpoint(ckpt['model'])

# Load vào GN model
model_gn = replace_bn_with_gn(model)
model_gn.load_state_dict(gn_state_dict)
model_gn.eval()

════════════════════════════════════════════════════════════════════
""")


# ============================================================
# EMERGENCY USAGE FOR YOUR CURRENT CHECKPOINT
# ============================================================

def emergency_fix_your_checkpoint():
    """
    Hướng dẫn fix checkpoint hiện tại của bạn
    """
    
    print("\n" + "=" * 70)
    print("🚨 EMERGENCY FIX FOR YOUR CURRENT CHECKPOINT")
    print("=" * 70)
    
    print("""
Checkpoint của bạn: /kaggle/input/test-data12/weight_test.pth
Vấn đề: Có BN keys nhưng KHÔNG có running_mean/running_var

GIẢI PHÁP NHANH (Không cần train lại):
════════════════════════════════════════════════════════════════════

import torch
from your_model import build_model
from data.custom import create_dataloaders

# 1. Build model (match training config)
model = build_model(...)

# 2. Load checkpoint
ckpt = torch.load('weight_test.pth')
model.load_state_dict(ckpt['model'], strict=False)

# 3. Get dataloader
_, val_loader, _ = create_dataloaders(...)

# 4. Fix checkpoint - thêm BN stats
fix_checkpoint_add_bn_stats(
    checkpoint_path='weight_test.pth',
    output_path='weight_test_fixed.pth',
    model=model,
    dataloader=val_loader,
    device='cuda',
    warmup_batches=200  # ← 200 batches = ~1600 images
)

# 5. Eval với checkpoint mới
model_new = build_model(...)
ckpt_new = torch.load('weight_test_fixed.pth')
model_new.load_state_dict(ckpt_new['model'])

metrics = evaluate_with_bn_warmup(
    model_new, val_loader,
    use_warmup=False  # ← Đã có stats rồi
)

print(f"mIoU: {metrics['miou']:.4f}")

════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📚 COMPLETE SOLUTION GUIDE")
    print("=" * 70)
    
    # Show recommended workflow
    recommended_training_modifications()
    
    # Show emergency fix
    emergency_fix_your_checkpoint()
    
    print("\n" + "=" * 70)
    print("💡 WHAT TO DO NOW:")
    print("=" * 70)
    print("""
1️⃣ NGAY LẬP TỨC (Fix checkpoint hiện tại):
   → Dùng fix_checkpoint_add_bn_stats()
   → Hoặc warmup BN khi eval: evaluate_with_bn_warmup()

2️⃣ DÀI HẠN (Training mới):
   → Option 1: Train với GN (recommended cho batch nhỏ)
   → Option 2: Train với BN (nếu batch ≥16)
   → Verify checkpoint trước khi lưu

3️⃣ INFERENCE (Production):
   → Đảm bảo model + checkpoint cùng norm type
   → BN: Cần .eval() mode
   → GN: Không cần eval/train mode
""")
