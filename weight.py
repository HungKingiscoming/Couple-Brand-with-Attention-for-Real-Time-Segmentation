import torch

# IoU cuối run 1 (epoch 10 best):
# pole=0.476, fence=0.487, motorcycle=0.454, rider=0.490 → cần boost mạnh
# wall=0.549, traffic_light=0.514 → boost vừa
# road/building/veg/sky/car → đã tốt, không cần tăng

class_names = ['road','sidewalk','building','wall','fence','pole',
               'traffic_light','traffic_sign','vegetation','terrain',
               'sky','person','rider','car','truck','bus',
               'train','motorcycle','bicycle']

# Base: official weights, sau đó boost thêm cho nhóm yếu
official = torch.tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969,
                          0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843,
                          1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
                          1.0507])

# Boost factor dựa trên IoU gap (target 0.6 cho tất cả)
iou_run1 = torch.tensor([0.9735, 0.7948, 0.8788, 0.5475, 0.4864, 0.4750,
                          0.5128, 0.6263, 0.8778, 0.5932, 0.8756, 0.6886,
                          0.4880, 0.9089, 0.6001, 0.7525, 0.6940, 0.4566,
                          0.6361])

target_iou = 0.65
boost = torch.clamp((target_iou / iou_run1.clamp(min=0.3)) ** 0.5, 1.0, 2.5)
weights_v2 = (official * boost)
# Normalize về mean=1.0
weights_v2 = weights_v2 / weights_v2.mean()

print("Class weights v2:")
for n, w, b in zip(class_names, weights_v2, boost):
    bar = '█' * int(w * 8)
    print(f"  {n:<16} {w:.4f}  (boost x{b:.2f})  {bar}")

print(f"\nmin={weights_v2.min():.3f}  max={weights_v2.max():.3f}  "
      f"ratio={weights_v2.max()/weights_v2.min():.1f}x")

torch.save(weights_v2, '/kaggle/working/class_weights_v2.pt')
print("Saved → /kaggle/working/class_weights_v2.pt")
