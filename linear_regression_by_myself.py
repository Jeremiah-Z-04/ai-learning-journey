import torch
import matplotlib.pyplot as plt

# ========== 1. 生成数据 ==========
torch.manual_seed(42)

n_samples = 100
X = torch.randn(n_samples, 1)
true_w = torch.tensor([[2.0]])
true_b = torch.tensor([1.0])
noise = torch.randn(n_samples, 1) * 0.5

# 生成 y
y = X @ true_w + true_b + noise

print(f"数据形状: X={X.shape}, y={y.shape}")

# ========== 2. 初始化参数 ==========
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"初始: w={w.item():.4f}, b={b.item():.4f}")

# ========== 3. 定义模型、损失 ==========
def model(X):
    return X @ w + b

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

lr = 0.05
epochs = 100

# ========== 4. 训练循环 ==========
loss_history = []

for epoch in range(epochs):
    # 预测
    y_pred = model(X)
    
    # 算损失
    loss = mse_loss(y_pred, y)
    loss_history.append(loss.item())
    
    # 反向传播 + 更新参数 + 清零
    loss.backward()
    
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    w.grad.zero_()
    b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n真实: w=2.00, b=1.00")
print(f"学到: w={w.item():.4f}, b={b.item():.4f}")

# ========== 5. 可视化 ==========
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='Data')
x_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
with torch.no_grad():
    y_line = x_line @ w + b
plt.plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, label='Fitted')
plt.plot(x_line.numpy(), 2*x_line.numpy()+1, 'g--', linewidth=2, label='True')
plt.legend()
plt.title('Linear Regression')

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.yscale('log')

plt.savefig('my_result.png')
plt.show()