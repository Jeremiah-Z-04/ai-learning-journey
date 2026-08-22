import torch
import matplotlib.pyplot as plt

print("=" * 50)
print("线性回归：从零实现")
print("=" * 50)

# 生成模拟数据：y = 2x + 1 + 噪声
torch.manual_seed(42)  # 固定随机种子，结果可复现

n_samples = 100
X = torch.randn(n_samples, 1)  # 100个样本，1个特征
true_w = torch.tensor([[2.0]])  # 真实权重
true_b = torch.tensor([1.0])    # 真实偏置

# 生成标签：y = 2x + 1 + 噪声
noise = torch.randn(n_samples, 1) * 0.5  # 噪声
y = X @ true_w + true_b + noise  # @ 是矩阵乘法

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"真实参数: w={true_w.item():.2f}, b={true_b.item():.2f}")
print(f"前5个样本:\nX={X[:5].flatten()}\ny={y[:5].flatten()}")
# ========== 初始化参数 ==========
w = torch.randn(1, 1, requires_grad=True)  # 随机初始化
b = torch.zeros(1, requires_grad=True)      # 偏置从0开始

print(f"\n初始猜测: w={w.item():.4f}, b={b.item():.4f}")

# ========== 定义模型 ==========
def linear_model(X):
    return X @ w + b

# ========== 定义损失函数（均方误差 MSE） ==========
def squared_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# ========== 定义优化器（手动SGD） ==========
def sgd(params, lr):
    with torch.no_grad():  # 更新参数时不追踪梯度
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()  # 清零梯度
# ========== 训练 ==========
lr = 0.05          # 学习率
epochs = 100       # 训练轮数
loss_history = []  # 记录Loss，后面画图

print("\n开始训练...")
for epoch in range(epochs):
    # 1. 预测
    y_pred = linear_model(X)
    
    # 2. 算Loss
    loss = squared_loss(y_pred, y)
    loss_history.append(loss.item())
    
    # 3. 反向传播
    loss.backward()
    
    # 4. 更新参数
    sgd([w, b], lr)
    
    # 每20轮打印一次
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

# ========== 结果 ==========
print(f"\n训练完成！")
print(f"真实参数: w=2.00, b=1.00")
print(f"学到参数: w={w.item():.4f}, b={b.item():.4f}")

# ========== 可视化 ==========
plt.figure(figsize=(10, 4))

# 左图：数据 + 拟合线
plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='Data')
x_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = x_line @ w.detach() + b.detach()
plt.plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, label='Fitted line')
plt.plot(x_line.numpy(), 2*x_line.numpy() + 1, 'g--', linewidth=2, label='True line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')

# 右图：Loss下降曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.yscale('log')  # 对数坐标，看得更清楚

plt.tight_layout()
plt.savefig('linear_regression_result.png')
print("图片已保存: linear_regression_result.png")
plt.show()