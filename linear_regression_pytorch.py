import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# 1. 数据
n_samples = 100
X = torch.randn(n_samples, 1)
y = 2 * X + 1 + torch.randn(n_samples, 1) * 0.5

# 2. 模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. 损失 + 优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 4. 数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 5. 训练
epochs = 100
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w={w:.4f}, b={b:.4f}")

print(f"\n真实: w=2.00, b=1.00")
print(f"学到: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")