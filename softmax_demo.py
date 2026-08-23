import torch
import torch.nn as nn

print("=" * 50)
print("Softmax 三分类：猫 / 狗 / 鸟")
print("=" * 50)

# 1. 手写 Softmax
def my_softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum()

x = torch.tensor([2.0, 1.0, 0.1])
print(f"手写 Softmax: {my_softmax(x)}")
print(f"PyTorch Softmax: {torch.softmax(x, dim=0)}")

# 2. 生成数据
torch.manual_seed(42)
n = 30

X_cat = torch.randn(n, 2) + torch.tensor([-2.0, -2.0])
X_dog = torch.randn(n, 2) + torch.tensor([0.0, 0.0])
X_bird = torch.randn(n, 2) + torch.tensor([2.0, 2.0])

X = torch.cat([X_cat, X_dog, X_bird], dim=0)
y = torch.cat([
    torch.zeros(n, dtype=torch.long),
    torch.ones(n, dtype=torch.long),
    torch.full((n,), 2, dtype=torch.long)
])

print(f"\n数据: X={X.shape}, y={y.shape}")

# 3. 模型
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
    
    def forward(self, x):
        return self.linear(x)

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. 训练
print("\n开始训练...")
for epoch in range(100):
    optimizer.zero_grad()
    scores = model(X)
    loss = criterion(scores, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            acc = (scores.argmax(dim=1) == y).float().mean().item()
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Acc: {acc*100:.1f}%")

# 5. 测试
with torch.no_grad():
    test = torch.tensor([[0.5, 0.5]])
    prob = torch.softmax(model(test), dim=1)
    print(f"\n测试 [0.5, 0.5] → 概率: {prob.squeeze()}")
    print(f"预测类别: {prob.argmax().item()} (0=猫, 1=狗, 2=鸟)")

print("\n✅ 完成")