"""
Tạo class weights từ config gốc GCNet-S để dùng trong training.
Chạy trên Kaggle:
    python make_official_weights.py
"""
import torch

# Class weights từ config gốc GCNet-S (Cityscapes clear, 19 classes)
# Range 0.84-1.15, ratio 1.38x — nhẹ nhàng và stable
official_weights = torch.tensor([
    0.8373, 0.918,  0.866,  1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
], dtype=torch.float32)

classes = ['road','sidewalk','building','wall','fence','pole',
           'traffic_light','traffic_sign','vegetation','terrain','sky',
           'person','rider','car','truck','bus','train','motorcycle','bicycle']

print("Official GCNet-S class weights:")
for c, w in zip(classes, official_weights):
    bar = '█' * int(w.item() * 10)
    print(f"  {c:<16} {w:.4f}  {bar}")

print(f"\nmin={official_weights.min():.4f}  max={official_weights.max():.4f}")
print(f"mean={official_weights.mean():.4f}  ratio={official_weights.max()/official_weights.min():.2f}x")

torch.save(official_weights, '/kaggle/working/class_weights_official.pt')
print("\nSaved → /kaggle/working/class_weights_official.pt")
