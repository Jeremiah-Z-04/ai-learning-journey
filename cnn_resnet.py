"""
resnet_demo.py
Day 8: ResNet 核心原理 —— 50 行看懂残差连接
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ========== 1. 用 Day 6 的模拟 MNIST 数据（你已经会了）==========
def generate_mnist_like(n_per_class=200):
    images, labels = [], []
    for digit in range(10):
        for _ in range(n_per_class):
            img = torch.zeros(1, 28, 28)
            cx, cy = 14 + np.random.randint(-2, 3), 14 + np.random.randint(-2, 3)
            if digit == 0:  # 圆
                for i in range(28):
                    for j in range(28):
                        if 5 < ((i-cx)**2 + (j-cy)**2)**0.5 < 9:
                            img[0, i, j] = 1.0
            elif digit == 1:  # 竖线
                img[0, :, cy] = 1.0
            # ... 其他数字类似，为了简洁只生成 0 和 1 两类做对比
            images.append(img)
            labels.append(digit)
    return torch.stack(images), torch.tensor(labels)

# 为了快，只生成 0 和 1 两类，每类 200 张
X = torch.zeros(400, 1, 28, 28)
y = torch.zeros(400, dtype=torch.long)
for i in range(200):
    # 0: 圆
    cx, cy = 14, 14
    for a in range(28):
        for b in range(28):
            if 5 < ((a-cx)**2 + (b-cy)**2)**0.5 < 9:
                X[i, 0, a, b] = 1.0
    y[i] = 0
    # 1: 竖线
    X[i+200, 0, :, 14] = 1.0
    y[i+200] = 1

# 加噪声
X = X + torch.randn(400, 1, 28, 28) * 0.1
X = torch.clamp(X, 0, 1)

# 划分
idx = torch.randperm(400)
X_train, y_train = X[idx[:320]], y[idx[:320]]
X_test, y_test = X[idx[320:]], y[idx[320:]]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=80, shuffle=False)

# ========== 2. 两个网络：Plain vs ResNet ==========
class PlainNet(nn.Module):
    """普通网络：10 个 PlainBlock 堆叠"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.blocks = nn.Sequential(*[self._make_block() for _ in range(10)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)
    
    def _make_block(self):
        return nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNet(nn.Module):
    """残差网络：10 个 ResBlock 堆叠"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.blocks = nn.ModuleList([self._make_block() for _ in range(10)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)
    
    def _make_block(self):
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            def forward(self, x):
                out = torch.relu(self.conv1(x))
                out = self.conv2(out)
                return torch.relu(out + x)  # ← 高速公路在这里
        return ResBlock()
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========== 3. 训练并记录梯度 ==========
def train_and_track(model, name, epochs=10):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    train_losses, test_accs = [], []
    first_grads = []  # 记录第一层卷积的梯度大小
    
    for epoch in range(1, epochs+1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            out = model(data)
            loss = crit(out, target)
            loss.backward()
            
            # 记录第一层卷积的梯度
            for n, p in model.named_parameters():
                if 'conv.weight' in n and p.grad is not None:
                    first_grads.append(p.grad.abs().mean().item())
                    break
            
            opt.step()
        
        # 测试
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                _, pred = out.max(1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / total
        test_accs.append(acc)
        print(f"{name} Epoch {epoch}: Test Acc = {acc:.1f}%")
    
    return test_accs, first_grads

print("\n训练 PlainNet...")
plain_acc, plain_grad = train_and_track(PlainNet(), "Plain")

print("\n训练 ResNet...")
res_acc, res_grad = train_and_track(ResNet(), "ResNet")

# ========== 4. 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 准确率对比
axes[0].plot(plain_acc, 'r-o', label='PlainNet')
axes[0].plot(res_acc, 'b-s', label='ResNet')
axes[0].set_title('Test Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 梯度大小对比（取前 50 个 batch）
n = min(50, len(plain_grad))
axes[1].plot(plain_grad[:n], 'r-o', label='PlainNet', markersize=3)
axes[1].plot(res_grad[:n], 'b-s', label='ResNet', markersize=3)
axes[1].set_title('First Layer Gradient (first 50 batches)')
axes[1].set_xlabel('Batch')
axes[1].set_ylabel('Mean |gradient|')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resnet_demo.png', dpi=150)
plt.close()
print("\n图已保存: resnet_demo.png")
print("关键观察：ResNet 的第一层梯度应该比 PlainNet 大！")