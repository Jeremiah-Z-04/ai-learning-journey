"""
yolo_toy.py
Day 11: 目标检测入门 —— 简化版 YOLO
作者: Jeremiah
日期: 2026-08-30
说明: 在 32x32 图上检测单个矩形框，理解 YOLO 核心思想
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ==================== 1. 生成模拟检测数据 ====================
"""
每张图：32x32，黑色背景上有一个彩色矩形。
标签：矩形中心 (cx, cy)、宽度 w、高度 h（全部归一化到 0~1）
"""
def generate_detection_data(n=2000):
    images = []
    labels = []  # [cx, cy, w, h]
    
    for _ in range(n):
        img = torch.zeros(3, 32, 32)
        # 随机矩形参数
        w = np.random.uniform(0.15, 0.6)  # 相对宽度
        h = np.random.uniform(0.15, 0.6)  # 相对高度
        cx = np.random.uniform(w/2, 1 - w/2)  # 中心x
        cy = np.random.uniform(h/2, 1 - h/2)  # 中心y
        
        # 转成像素坐标
        px = int(cx * 32)
        py = int(cy * 32)
        pw = int(w * 32)
        ph = int(h * 32)
        
        # 画矩形（随机颜色）
        color = torch.rand(3)
        x1 = max(0, px - pw // 2)
        y1 = max(0, py - ph // 2)
        x2 = min(32, px + pw // 2)
        y2 = min(32, py + ph // 2)
        
        img[0, y1:y2, x1:x2] = color[0]
        img[1, y1:y2, x1:x2] = color[1]
        img[2, y1:y2, x1:x2] = color[2]
        
        # 加噪声
        img = img + torch.randn(3, 32, 32) * 0.05
        img = torch.clamp(img, 0, 1)
        
        images.append(img)
        labels.append(torch.tensor([cx, cy, w, h], dtype=torch.float32))
    
    return torch.stack(images), torch.stack(labels)

X, Y = generate_detection_data(2000)
# 划分
idx = torch.randperm(2000)
X_train, Y_train = X[idx[:1600]], Y[idx[:1600]]
X_test, Y_test = X[idx[1600:]], Y[idx[1600:]]

print(f"训练集: {len(X_train)} 张")
print(f"测试集: {len(X_test)} 张")

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=100, shuffle=False)

# 显示样本
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for idx, ax in enumerate(axes.flatten()):
    img = X_train[idx].permute(1, 2, 0).numpy()
    cx, cy, w, h = Y_train[idx].numpy()
    ax.imshow(img)
    # 画框
    x1 = (cx - w/2) * 32
    y1 = (cy - h/2) * 32
    rect = patches.Rectangle((x1, y1), w*32, h*32, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"cx={cx:.2f}, cy={cy:.2f}\nw={w:.2f}, h={h:.2f}", fontsize=8)
    ax.axis('off')
plt.suptitle("Training Samples: Random Rectangles")
plt.tight_layout()
plt.savefig('yolo_samples.png', dpi=150)
plt.close()
print("样本图已保存: yolo_samples.png")

# ==================== 2. 简化版 YOLO 模型 ====================
"""
把 32x32 图分成 4x4 网格（每个格子 8x8 像素）。
每个格子预测：1个边界框 (cx, cy, w, h) + 置信度。
简化：假设每张图只有一个物体，用全局平均池化代替网格。
"""
class TinyYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: 提取特征
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 8->4
        )
        # Head: 预测框
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),  # 输出 [cx, cy, w, h]
            nn.Sigmoid(),        # 压到 0~1
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# ==================== 3. 训练 ====================
model = TinyYOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # 坐标回归用均方误差

print("\n开始训练...")
for epoch in range(1, 21):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # 测试
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += criterion(pred, target).item()
    test_loss /= len(test_loader)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

print("\n训练完成！")

# ==================== 4. 可视化预测 ====================
model.eval()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

with torch.no_grad():
    for i in range(8):
        img = X_test[i].unsqueeze(0).to(device)
        pred = model(img)[0].cpu().numpy()  # [cx, cy, w, h]
        true = Y_test[i].numpy()
        
        ax = axes[i // 4, i % 4]
        ax.imshow(X_test[i].permute(1, 2, 0).numpy())
        
        # 画真实框（绿色）
        tx1 = (true[0] - true[2]/2) * 32
        ty1 = (true[1] - true[3]/2) * 32
        rect_true = patches.Rectangle((tx1, ty1), true[2]*32, true[3]*32,
                                       linewidth=2, edgecolor='lime', facecolor='none', label='True')
        ax.add_patch(rect_true)
        
        # 画预测框（红色）
        px1 = (pred[0] - pred[2]/2) * 32
        py1 = (pred[1] - pred[3]/2) * 32
        rect_pred = patches.Rectangle((px1, py1), pred[2]*32, pred[3]*32,
                                       linewidth=2, edgecolor='red', facecolor='none', label='Pred')
        ax.add_patch(rect_pred)
        
        ax.set_title(f"Pred: ({pred[0]:.2f},{pred[1]:.2f})\n({pred[2]:.2f},{pred[3]:.2f})", fontsize=8)
        ax.axis('off')

plt.suptitle("Green=True Box, Red=Predicted Box", fontsize=14)
plt.tight_layout()
plt.savefig('yolo_predictions.png', dpi=150)
plt.close()
print("预测图已保存: yolo_predictions.png")

# ==================== 5. IoU 计算 ====================
def compute_iou(box1, box2):
    """
    box = [cx, cy, w, h]（归一化）
    返回交并比 0~1
    """
    # 转成左上角右下角
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # 交集
    xi1 = max(b1_x1, b2_x1)
    yi1 = max(b1_y1, b2_y1)
    xi2 = min(b1_x2, b2_x2)
    yi2 = min(b1_y2, b2_y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # 并集
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

# 计算测试集平均 IoU
model.eval()
ious = []
with torch.no_grad():
    for i in range(len(X_test)):
        img = X_test[i].unsqueeze(0).to(device)
        pred = model(img)[0].cpu().numpy()
        true = Y_test[i].numpy()
        ious.append(compute_iou(pred, true))

print(f"\n测试集平均 IoU: {np.mean(ious):.4f}")
print("IoU > 0.5 表示预测框和真实框重叠一半以上，通常认为'检测成功'")
print("\n今日任务完成！")
print("1. 观察：红框（预测）和绿框（真实）是否接近？")
print("2. 思考：如果图里有2个矩形，这个模型能检测吗？→ 不能，需要每个格子预测多个框")
print("3. Obsidian: [[YOLO]] [[目标检测]] [[边界框]] [[IoU]]")