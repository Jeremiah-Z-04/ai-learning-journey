"""
cnn_mnist.py
Day 6: CNN 入门 —— 手写数字识别（纯 PyTorch 零依赖版）
作者: Jeremiah
日期: 2026-08-25
环境: PyTorch 2.3.0+cu121, RTX 4060
说明: 用 PyTorch 生成模拟 MNIST 数据，无需下载任何数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ==================== 1. 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== 2. 生成模拟 MNIST 数据（无需下载）====================
"""
用 PyTorch 生成模拟手写数字数据：
- 每个数字类别生成 200 张 28x28 的图
- 用简单的几何图形模拟数字特征（竖线、横线、圆圈等）
- 加随机噪声让它更像真实数据
"""
print("\n正在生成模拟 MNIST 数据（无需下载）...")

def generate_digit_28x28(digit, num_samples=200):
    """
    用简单几何图形模拟手写数字，28x28。
    这不是真实 MNIST，但足够让 CNN 学到卷积核。
    """
    images = []
    labels = []
    
    for _ in range(num_samples):
        img = torch.zeros(28, 28)
        
        # 随机偏移和缩放，模拟手写变化
        offset_x = np.random.randint(-2, 3)
        offset_y = np.random.randint(-2, 3)
        scale = np.random.uniform(0.8, 1.2)
        
        center_x, center_y = 14 + offset_x, 14 + offset_y
        
        if digit == 0:
            # 画圆圈
            for i in range(28):
                for j in range(28):
                    dist = ((i - center_x)**2 + (j - center_y)**2) ** 0.5
                    if 6 * scale < dist < 10 * scale:
                        img[i, j] = 1.0
        elif digit == 1:
            # 竖线
            x = center_y
            for i in range(6, 22):
                img[i, x] = 1.0
                if np.random.rand() > 0.7:
                    img[i, x + np.random.randint(-1, 2)] = 0.5
        elif digit == 2:
            # 横线 + 弧线
            for j in range(6, 22):
                img[6, j] = 1.0
                img[14, j] = 1.0
                img[22, j] = 1.0
            for i in range(6, 15):
                img[i, 22 - (i - 6)] = 1.0
        elif digit == 3:
            # 两个半圆
            for i in range(6, 14):
                img[i, 20] = 1.0
                img[i, 20 - (i - 6)//2] = 1.0
            for i in range(14, 22):
                img[i, 20] = 1.0
                img[i, 20 - (22 - i)//2] = 1.0
        elif digit == 4:
            # 竖线 + 横线
            for i in range(6, 22):
                img[i, 8] = 1.0
            for j in range(8, 20):
                img[14, j] = 1.0
        elif digit == 5:
            # S 形
            for j in range(6, 20):
                img[6, j] = 1.0
                img[14, j] = 1.0
                img[22, j] = 1.0
            for i in range(6, 15):
                img[i, 6] = 1.0
            for i in range(14, 23):
                img[i, 20] = 1.0
        elif digit == 6:
            # 圆圈 + 竖线
            for i in range(28):
                for j in range(28):
                    dist = ((i - center_x)**2 + (j - center_y + 2)**2) ** 0.5
                    if 5 * scale < dist < 8 * scale:
                        img[i, j] = 1.0
            for i in range(6, 22):
                img[i, 6] = 1.0
        elif digit == 7:
            # 横线 + 斜线
            for j in range(6, 22):
                img[6, j] = 1.0
            for i in range(6, 22):
                img[i, 22 - (i - 6)] = 1.0
        elif digit == 8:
            # 两个圆圈
            for i in range(28):
                for j in range(28):
                    dist1 = ((i - center_x + 4)**2 + (j - center_y)**2) ** 0.5
                    dist2 = ((i - center_x - 4)**2 + (j - center_y)**2) ** 0.5
                    if 4 * scale < dist1 < 7 * scale or 4 * scale < dist2 < 7 * scale:
                        img[i, j] = 1.0
        elif digit == 9:
            # 圆圈 + 竖线
            for i in range(28):
                for j in range(28):
                    dist = ((i - center_x)**2 + (j - center_y - 2)**2) ** 0.5
                    if 5 * scale < dist < 8 * scale:
                        img[i, j] = 1.0
            for i in range(6, 22):
                img[i, 20] = 1.0
        
        # 加噪声和模糊，模拟真实手写
        noise = torch.randn(28, 28) * 0.1
        img = img + noise
        img = torch.clamp(img, 0, 1)
        
        # 轻微模糊
        img = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=1, padding=1).squeeze()
        img = img[:28, :28]
        
        images.append(img)
        labels.append(digit)
    
    return torch.stack(images), torch.tensor(labels)

# 生成数据：每类 200 张，共 2000 张
all_images = []
all_labels = []

for d in range(10):
    imgs, lbls = generate_digit_28x28(d, num_samples=200)
    all_images.append(imgs)
    all_labels.append(lbls)

X = torch.cat(all_images, dim=0).unsqueeze(1)  # (2000, 1, 28, 28)
y = torch.cat(all_labels, dim=0)               # (2000,)

# 划分训练集(80%)和测试集(20%)
indices = torch.randperm(len(X))
train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

print(f"总样本数: {len(y)}")
print(f"训练集: {len(y_train)} 张")
print(f"测试集: {len(y_test)} 张")

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 显示几张示例
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flatten()
for i in range(10):
    # 找第一个该类别的样本
    idx = (y == i).nonzero(as_tuple=True)[0][0].item()
    axes[i].imshow(X[idx, 0].numpy(), cmap='gray')
    axes[i].set_title(f'Label: {i}')
    axes[i].axis('off')
plt.suptitle('Generated Simulated MNIST Samples', fontsize=14)
plt.tight_layout()
plt.savefig('generated_samples.png', dpi=150)
print("\n模拟数据示例已保存: generated_samples.png")
plt.show()

# ==================== 3. 模型定义 ====================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = LeNet().to(device)
print("\n模型结构:")
print(model)

# ==================== 4. 参数统计 ====================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"\n总参数量: {total_params:,}")
print(f"约 {total_params/1e3:.1f} K")

mlp_params = 784*256 + 256 + 256*10 + 10
print(f"\n对比：简单 MLP（784→256→10）参数量: {mlp_params:,}")
print(f"CNN 参数量只有 MLP 的 {total_params/mlp_params*100:.1f}%")
print("这就是参数共享的威力：卷积核扫遍全图，而不是每个像素配一个权重。")

# ==================== 5. 损失函数和优化器 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==================== 6. 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==================== 7. 主训练 ====================
epochs = 20

train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("\n开始训练...")
for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"  训练 — Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  测试 — Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

print("\n训练完成！")

# ==================== 8. 可视化 1：训练曲线 ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, epochs+1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, epochs+1), test_losses, 'r-s', label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, epochs+1), train_accs, 'b-o', label='Train Acc')
axes[1].plot(range(1, epochs+1), test_accs, 'r-s', label='Test Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\n训练曲线已保存: training_curves.png")
plt.show()

# ==================== 9. 可视化 2：卷积核 ====================
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
axes = axes.flatten()

conv1_weights = model.features[0].weight.detach().cpu().numpy()

for i in range(6):
    ax = axes[i]
    kernel = conv1_weights[i, 0, :, :]
    im = ax.imshow(kernel, cmap='RdBu_r', interpolation='nearest')
    ax.set_title(f'Kernel {i+1}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('First Layer Conv Kernels (5x5)', fontsize=14)
plt.tight_layout()
plt.savefig('conv_kernels.png', dpi=150)
print("卷积核可视化已保存: conv_kernels.png")
plt.show()

# ==================== 10. 可视化 3：特征图 ====================
model.eval()

sample_img, sample_label = test_dataset[0]
sample_img = sample_img.unsqueeze(0).to(device)

with torch.no_grad():
    conv1_out = model.features[0](sample_img)
    conv1_out = model.features[1](conv1_out)
    pool1_out = model.features[2](conv1_out)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

axes[0, 0].imshow(sample_img[0, 0].cpu().numpy(), cmap='gray')
axes[0, 0].set_title(f'Original (Label: {sample_label.item()})')
axes[0, 0].axis('off')

for i in range(6):
    ax = axes[(i+1)//4, (i+1)%4]
    feature_map = pool1_out[0, i].cpu().numpy()
    ax.imshow(feature_map, cmap='viridis')
    ax.set_title(f'Feature Map {i+1}')
    ax.axis('off')

plt.suptitle('First Layer Feature Maps (after Pooling)', fontsize=14)
plt.tight_layout()
plt.savefig('feature_maps.png', dpi=150)
print("特征图可视化已保存: feature_maps.png")
plt.show()

# ==================== 11. 预测展示 ====================
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

indices = np.random.choice(len(test_dataset), 10, replace=False)

with torch.no_grad():
    for idx, ax in zip(indices, axes):
        img, label = test_dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        output = model(img_batch)
        pred = output.argmax(dim=1).item()
        
        ax.imshow(img[0].cpu().numpy(), cmap='gray')
        color = 'green' if pred == label.item() else 'red'
        ax.set_title(f'True: {label.item()} | Pred: {pred}', color=color)
        ax.axis('off')

plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=14)
plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
print("预测结果已保存: predictions.png")
plt.show()

print("\n所有可视化文件已生成！")
print("\n今日任务完成。记得：")
print("1. 在 Obsidian 创建概念卡片: [[卷积]] [[池化]] [[MNIST]] [[特征图]]")
print("2. GitHub 提交: git add . && git commit -m 'Day6: CNN MNIST' && git push")