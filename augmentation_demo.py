"""
augmentation_demo.py
Day 10: 数据增强可视化
"""

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# 生成一张模拟图（3x32x32）
img = torch.zeros(3, 32, 32)
# 画一个红色的圆
for i in range(32):
    for j in range(32):
        if 5 < ((i-16)**2 + (j-16)**2)**0.5 < 10:
            img[0, i, j] = 0.9  # R
            img[1, i, j] = 0.2  # G
            img[2, i, j] = 0.2  # B

# 定义 5 种增强
transforms = {
    "Original": T.Compose([]),
    "Crop": T.RandomCrop(28),
    "Flip": T.RandomHorizontalFlip(p=1.0),  # p=1 强制翻转
    "Rotate": T.RandomRotation(30),
    "ColorJitter": T.ColorJitter(brightness=0.5, contrast=0.5),
    "Combined": T.Compose([
        T.RandomCrop(28),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
    ])
}

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for idx, (name, trans) in enumerate(transforms.items()):
    # 每次增强都是随机的，所以同一张图每次结果不同
    aug = trans(img)
    axes[idx].imshow(aug.permute(1, 2, 0).numpy())
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.suptitle("Data Augmentation: Same Image, Different Views", fontsize=14)
plt.tight_layout()
plt.savefig('augmentation_demo.png', dpi=150)
plt.close()
print("图已保存: augmentation_demo.png")
print("观察：Crop/Flip/Rotate/Color 后，圆还是圆，但像素完全不同")
print("模型必须学会'不管怎么变，都是同一个类别'")