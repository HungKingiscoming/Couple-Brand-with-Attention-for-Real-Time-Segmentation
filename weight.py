"""
make_foggy_weights.py — chạy thẳng, không cần argument:
    python make_foggy_weights.py
"""
import torch
from pathlib import Path

OUTPUT = '/kaggle/working/class_weights_foggy.pt'

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# Computed từ scan thực tế 8925 files (label_id format, median_freq)
# blend 35% scan + 65% official → ratio 2.77x, differentiation tốt
# motorcycle(0.099%) → 1.631, rider(0.135%) → 1.476, road(36.87%) → 0.590
WEIGHTS = torch.tensor([
    0.5898,  #  0 road         (36.870%)
    0.7090,  #  1 sidewalk     ( 6.085%)
    0.6209,  #  2 building     (22.824%)
    1.0145,  #  3 wall         ( 0.655%)
    0.9564,  #  4 fence        ( 0.877%)
    0.8979,  #  5 pole         ( 1.227%)
    1.2415,  #  6 traffic_light( 0.208%)
    1.0547,  #  7 traffic_sign ( 0.551%)
    0.6405,  #  8 vegetation   (15.929%)
    0.9088,  #  9 terrain      ( 1.158%)
    0.7582,  # 10 sky          ( 4.019%)
    0.8906,  # 11 person       ( 1.219%)
    1.4759,  # 12 rider        ( 0.135%) ← boosted
    0.6921,  # 13 car          ( 6.995%)
    1.2417,  # 14 truck        ( 0.267%) ← boosted
    1.2831,  # 15 bus          ( 0.235%) ← boosted
    1.2800,  # 16 train        ( 0.233%) ← boosted
    1.6310,  # 17 motorcycle   ( 0.099%) ← boosted most
    1.1134,  # 18 bicycle      ( 0.414%)
], dtype=torch.float32)

if __name__ == '__main__':
    w = WEIGHTS / WEIGHTS.mean()

    print(f"\n  {'Class':<16} {'Weight':>8}  Bar")
    print(f"  {'-'*55}")
    for cls, wv in zip(CLASSES, w):
        bar  = '█' * int(wv.item() * 14)
        mark = ' ← boosted' if wv.item() > 1.15 else ''
        print(f"  {cls:<16} {wv.item():>8.4f}  {bar}{mark}")
    print(f"\n  min={w.min():.4f}  max={w.max():.4f}  "
          f"mean={w.mean():.4f}  ratio={w.max()/w.min():.2f}x")

    out = Path(OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, out)
    print(f"\n  ✅ Saved → {out}\n")
