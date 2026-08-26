"""
cnn_cifar10.py
Day 7: CNN 进阶 —— CIFAR-10 彩色图像 + BatchNorm + Dropout
作者: Jeremiah
日期: 2026-08-26
环境: PyTorch 2.3.0+cu121, RTX 4060
说明: 使用模拟彩色数据（网络不通），结构完全对标真实 CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ==================== 1. 设备 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== 2. 生成模拟 CIFAR-10 数据 ====================
"""
真实 CIFAR-10: 32x32 彩色图, 3 通道, 10 类 (飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车)
这里用几何图形+噪声生成模拟数据，零下载。
"""
print("\n生成模拟 CIFAR-10 数据 (3x32x32, 10类)...")

def generate_color_image(label, size=32):
    """生成一张 3x32x32 的模拟彩色图"""
    img = torch.zeros(3, size, size)
    
    # 每类一种颜色和形状组合
    colors = [
        [1.0, 0.2, 0.2],  # 0: 红 (飞机-三角)
        [0.2, 1.0, 0.2],  # 1: 绿 (汽车-方块)
        [0.2, 0.2, 1.0],  # 2: 蓝 (鸟-圆)
        [1.0, 1.0, 0.2],  # 3: 黄 (猫-叉)
        [1.0, 0.2, 1.0],  # 4: 紫 (鹿-菱形)
        [0.2, 1.0, 1.0],  # 5: 青 (狗-横线)
        [1.0, 0.5, 0.2],  # 6: 橙 (青蛙-竖线)
        [0.5, 0.2, 1.0],  # 7: 靛 (马-斜线)
        [0.8, 0.8, 0.8],  # 8: 灰 (船-网格)
        [0.5, 0.5, 0.2],  # 9: 棕 (卡车-十字)
    ]
    
    c = torch.tensor(colors[label])
    cx, cy = size // 2 + np.random.randint(-3, 4), size // 2 + np.random.randint(-3, 4)
    scale = np.random.uniform(0.7, 1.3)
    
    for i in range(size):
        for j in range(size):
            dx, dy = i - cx, j - cy
            
            if label == 0:      # 三角
                if abs(dx) < (8 - dy*0.5) * scale and dy < 8 * scale and dy > -8 * scale:
                    img[:, i, j] = c
            elif label == 1:    # 方块
                if abs(dx) < 6 * scale and abs(dy) < 6 * scale:
                    img[:, i, j] = c
            elif label == 2:    # 圆
                if (dx**2 + dy**2) ** 0.5 < 7 * scale:
                    img[:, i, j] = c
            elif label == 3:    # 叉
                if abs(dx - dy) < 3 * scale or abs(dx + dy) < 3 * scale:
                    img[:, i, j] = c
            elif label == 4:    # 菱形
                if abs(dx) + abs(dy) < 8 * scale:
                    img[:, i, j] = c
            elif label == 5:    # 横线
                if abs(dy) < 2 * scale:
                    img[:, i, j] = c
            elif label == 6:    # 竖线
                if abs(dx) < 2 * scale:
                    img[:, i, j] = c
            elif label == 7:    # 斜线
                if abs(dx - dy) < 2 * scale:
                    img[:, i, j] = c
            elif label == 8:    # 网格
                if (i % 4 == 0) or (j % 4 == 0):
                    img[:, i, j] = c
            elif label == 9:    # 十字
                if abs(dx) < 2 * scale or abs(dy) < 2 * scale:
                    img[:, i, j] = c
    
    # 加噪声和模糊
    img = img + torch.randn(3, size, size) * 0.08
    img = torch.clamp(img, 0, 1)
    # 简单模糊
    img = nn.functional.avg_pool2d(img.unsqueeze(0), kernel_size=2, stride=1, padding=1).squeeze(0)
    img = img[:, :size, :size]
    return img

# 生成数据：每类 400 张，共 4000 张
all_imgs, all_labels = [], []
for label in range(10):
    for _ in range(400):
        all_imgs.append(generate_color_image(label))
        all_labels.append(label)

X = torch.stack(all_imgs)  # [4000, 3, 32, 32]
y = torch.tensor(all_labels, dtype=torch.long)

# 标准化（模拟真实 CIFAR-10 的预处理）
mean = X.mean(dim=[0,2,3], keepdim=True)
std = X.std(dim=[0,2,3], keepdim=True) + 1e-6
X = (X - mean) / std

# 划分训练/测试
indices = torch.randperm(len(X))
train_size = int(0.8 * len(X))
X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
X_test, y_test = X[indices[train_size:]], y[indices[train_size:]]

print(f"训练集: {len(X_train)} 张")
print(f"测试集: {len(X_test)} 张")

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# 显示样本
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    idx = (y == i).nonzero(as_tuple=True)[0][0].item()
    img_show = X[idx] * std.squeeze().view(3,1,1) + mean.squeeze().view(3,1,1)  # 反标准化
    img_show = torch.clamp(img_show, 0, 1)
    axes[i//5, i%5].imshow(img_show.permute(1,2,0).numpy())
    axes[i//5, i%5].set_title(class_names[i])
    axes[i//5, i%5].axis('off')
plt.suptitle("Simulated CIFAR-10 Samples")
plt.tight_layout()
plt.savefig('cifar_samples.png', dpi=150)
plt.close()

# ==================== 3. 模型定义 ====================

class CNN(nn.Module):
    """
    VGG-like 小网络。
    关键：Conv -> BN -> ReLU 的顺序，以及 Dropout 的位置。
    """
    def __init__(self, use_bn=True, use_dropout=True):
        super(CNN, self).__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        
        # Block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 8x8 -> 4x4
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Block 1
        x = self.pool1(torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))))
        # Block 2
        x = self.pool2(torch.relu(self.bn4(self.conv4(torch.relu(self.bn3(self.conv3(x)))))))
        # Block 3
        x = self.pool3(torch.relu(self.bn6(self.conv6(torch.relu(self.bn5(self.conv5(x)))))))
        # Classifier
        x = self.flatten(x)
        x = torch.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== 4. 训练函数 ====================
def train_and_eval(model, train_loader, test_loader, epochs=15, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Test
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, pred = output.max(1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.1f}%")
    
    return train_losses, train_accs, test_losses, test_accs

# ==================== 5. 对比实验 ====================
print("\n" + "="*60)
print("实验 1: Baseline (有 BN + 有 Dropout)")
print("="*60)
model_baseline = CNN(use_bn=True, use_dropout=True)
baseline = train_and_eval(model_baseline, train_loader, test_loader, epochs=15)

print("\n" + "="*60)
print("实验 2: 无 BN (No BatchNorm)")
print("="*60)
model_nobn = CNN(use_bn=False, use_dropout=True)
nobn = train_and_eval(model_nobn, train_loader, test_loader, epochs=15)

print("\n" + "="*60)
print("实验 3: 无 Dropout")
print("="*60)
model_nodrop = CNN(use_bn=True, use_dropout=False)
nodrop = train_and_eval(model_nodrop, train_loader, test_loader, epochs=15)

# ==================== 6. 可视化对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 对比
axes[0].plot(baseline[0], 'b-o', label='Baseline (BN+Dropout)', markersize=4)
axes[0].plot(nobn[0], 'r-s', label='No BN', markersize=4)
axes[0].plot(nodrop[0], 'g-^', label='No Dropout', markersize=4)
axes[0].set_title('Train Loss Comparison')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test Acc 对比
axes[1].plot(baseline[3], 'b-o', label='Baseline', markersize=4)
axes[1].plot(nobn[3], 'r-s', label='No BN', markersize=4)
axes[1].plot(nodrop[3], 'g-^', label='No Dropout', markersize=4)
axes[1].set_title('Test Accuracy Comparison')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bn_dropout_comparison.png', dpi=150)
print("\n对比图已保存: bn_dropout_comparison.png")
plt.close()

# ==================== 7. 过拟合分析 ====================
print("\n" + "="*60)
print("过拟合分析 (最后 epoch 的差距)")
print("="*60)
print(f"Baseline      训练-测试差距: {baseline[1][-1] - baseline[3][-1]:.1f}%")
print(f"No BN         训练-测试差距: {nobn[1][-1] - nobn[3][-1]:.1f}%")
print(f"No Dropout    训练-测试差距: {nodrop[1][-1] - nodrop[3][-1]:.1f}%")
print("\n差距越大 = 过拟合越严重。Dropout 应该让差距变小。")

print("\n今日任务完成！")
print("1. 观察：No BN 的 loss 曲线是否更震荡？收敛更慢？")
print("2. 观察：No Dropout 的训练集是否远高于测试集（过拟合）？")
print("3. Obsidian: [[BatchNorm]] [[Dropout]] [[CIFAR-10]]")
print("4. GitHub: git add . && git commit -m 'Day7: CIFAR-10 + BN + Dropout' && git push")