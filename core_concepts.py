"""
core_concepts.py
Day 10 补完：卷积、ReLU、BN、ResBlock、ResNet 核心概念亲手验证
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("概念 1：卷积 Conv2d —— 滑动的小窗口")
print("=" * 60)

# 造一张 8×8 的图，中间一条竖线
img = torch.zeros(1, 1, 8, 8)
img[0, 0, :, 4] = 1.0  # 第4列是竖线

# 造一个"竖线检测器"核
kernel = torch.tensor([
    [-1,  1, -1],
    [-1,  1, -1],
    [-1,  1, -1],
], dtype=torch.float32).view(1, 1, 3, 3)

# 手动卷积
out = nn.functional.conv2d(img, kernel, padding=0)
print(f"输入:  {img.shape}   (1张, 1通道, 8×8)")
print(f"卷积核: {kernel.shape}  (1个, 1通道, 3×3)")
print(f"输出:  {out.shape}   (1张, 1通道, 6×6)  ← 8-3+1=6")

print("\n卷积核权重:")
print(kernel[0, 0])
print("→ 中间一列是 1，两边是 -1。遇到竖线时，中间亮两边暗，响应最强。")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(img[0, 0], cmap='gray'); axes[0].set_title("Input: Vertical Line"); axes[0].axis('off')
axes[1].imshow(kernel[0, 0], cmap='RdBu_r'); axes[1].set_title("Kernel"); axes[1].axis('off')
axes[2].imshow(out[0, 0].detach(), cmap='hot'); axes[2].set_title("Output: Strong Response"); axes[2].axis('off')
plt.savefig('concept_conv.png', dpi=150); plt.close()
print("图已保存: concept_conv.png")

# ========================
print("\n" + "=" * 60)
print("概念 2：ReLU —— 负数杀死，正数放行")
print("=" * 60)

x = torch.linspace(-5, 5, 100)
y = torch.relu(x)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, x, 'k--', label='y=x (no ReLU)', alpha=0.3)
ax.plot(x, y, 'r-', linewidth=2, label='ReLU(x)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title('ReLU: max(0, x)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('concept_relu.png', dpi=150); plt.close()
print("图已保存: concept_relu.png")

print("""
关键：
- x < 0: 输出 0，梯度 0 → 神经元"死亡"
- x > 0: 输出 x，梯度 1 → 信息无损通过
- 为什么需要它？没有 ReLU，10 层线性叠加 = 1 层线性
  y = W3(W2(W1x)) = (W3W2W1)x = W'x，还是直线，学不了曲线
""")

# ========================
print("\n" + "=" * 60)
print("概念 3：BatchNorm —— 把数据压回标准分布")
print("=" * 60)

# 模拟神经网络某层的输出：均值很大，方差也很大
bad_output = torch.randn(100, 16) * 10 + 50  # 均值≈50, 方差≈100

bn = nn.BatchNorm1d(16)
good_output = bn(bad_output)

print(f"BN 前: 均值={bad_output.mean():.2f},  方差={bad_output.std():.2f}")
print(f"BN 后: 均值={good_output.mean():.2f}, 方差={good_output.std():.2f}")
print("→ 强行拉回 均值≈0, 方差≈1")

# ========================
print("\n" + "=" * 60)
print("概念 4：ResBlock —— 输入和输出相加")
print("=" * 60)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
    
    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        return torch.relu(out + x)  # ← 关键：+ x

# 验证梯度保护
x = torch.randn(1, 8, 4, 4, requires_grad=True)
block = ResBlock(8)
out = block(x).sum()
out.backward()

print(f"输入梯度大小: {x.grad.abs().mean():.6f}")
print("→ 因为有 'out + x'，反向传播时梯度会 +1，不会消失")

# 对比普通 Block
x2 = torch.randn(1, 8, 4, 4, requires_grad=True)
plain = nn.Sequential(nn.Conv2d(8,8,3,padding=1), nn.ReLU(), nn.Conv2d(8,8,3,padding=1), nn.ReLU())
out2 = plain(x2).sum()
out2.backward()
print(f"Plain 输入梯度: {x2.grad.abs().mean():.6f}")
print("→ 通常更小，因为梯度要经过两层衰减")

# ========================
print("\n" + "=" * 60)
print("概念 5：ResNet —— 很多 ResBlock 堆起来")
print("=" * 60)

class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        # 堆 5 个 ResBlock
        self.blocks = nn.Sequential(*[ResBlock(16) for _ in range(5)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.blocks(x)   # ← 5 个 ResBlock
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TinyResNet()
print(f"TinyResNet 有 {sum(p.numel() for p in model.parameters()):,} 个参数")
print("→ 5 个 ResBlock = 10 层卷积 + 1 层输入卷积，共 11 层")
print("→ 没有残差连接，11 层已经很难训练；有了残差，可以堆 100+ 层")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
卷积  = 滑动窗口，可学习的模板匹配器
ReLU  = 开关，负数杀死，正数放行，引入非线性
BN    = 标准化，把每层输出压回均值0方差1，训练更稳
ResBlock = 卷积输出 + 原始输入，给梯度开高速公路
ResNet = 很多 ResBlock 堆起来，深层网络也能训练
""")

print("所有概念图已保存。去项目文件夹查看 concept_conv.png 和 concept_relu.png")