import torch

print("=" * 50)
print("自动求导：深度学习训练的核心")
print("=" * 50)

# 例子1：y = x², 求 dy/dx
print("\n【1】y = x², x=3, dy/dx=?")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"x = {x.item()}, y = {y.item()}, dy/dx = {x.grad.item()}")
print("→ 导数就是 2*x = 6.0，算对了！")

# 例子2：模拟一次参数更新
print("\n【2】模拟线性回归参数更新")
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(2.0)
y_true = torch.tensor(5.0)  # 假设真实值是5

y_pred = w * x + b
loss = (y_pred - y_true) ** 2

print(f"当前: w={w.item():.2f}, b={b.item():.2f}")
print(f"预测: {y_pred.item():.2f}, 真实: {y_true.item():.2f}")
print(f"Loss: {loss.item():.2f}")

loss.backward()
print(f"梯度: dL/dw={w.grad.item():.2f}, dL/db={b.grad.item():.2f}")
print("→ 梯度是负的，说明w和b要往大了调")

# 例子3：梯度会累积，必须清零
print("\n【3】梯度不清零会累加！")
w.grad.zero_()
b.grad.zero_()
print(f"清零后: w.grad={w.grad.item()}")

# 例子4：GPU上求导
print("\n【4】GPU上自动求导")
x = torch.tensor([1.0, 2.0], requires_grad=True).cuda()
y = (x ** 2).sum()
y.backward()
print(f"GPU上的梯度: {x.grad}")

print("\n✅ 完成！记住三件事：")
print("1. requires_grad=True 开启追踪")
print("2. backward() 算梯度")
print("3. grad.zero_() 每次迭代要清零")