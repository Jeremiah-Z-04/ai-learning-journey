import torch
print("=" * 50)
print("PyTorch Tensor 基础操作")
print("=" * 50)

# ========== 1. 创建 Tensor ==========
print("\n[1]创建Tensor")

#从列表创建
x=torch.tensor([1.0,2.0,3.0,4.0])
print(f"向量 x: {x}")
print(f"形状：{x.shape},维度：{x.ndim}")

#创建二维张量（矩阵）
A = torch.tensor([[1,2,3],[4,5,6]])
print(f"\n矩阵A:\n{A}")
print(f"形状: {A.shape}")  #(2,3) = 2行3列

# 创建全0、全1、随机张量（深度学习最常用）
zeros = torch.zeros(3,3)
ones = torch.ones(2,2)
rand = torch.randn(3,3)
print(f"\n全0:\n{zeros}")
print(f"\n随机:\n{rand}")

# ========== 2. 基本运算 ==========
print("\n【2】基本计算")
a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")  # 逐元素相乘
print(f"a 的平方 = {a ** 2}")
# 矩阵乘法
A=torch.tensor([[1.0,2.0],[3.0,4.0]])
B=torch.tensor([[5.0,6.0],[7.0,8.0]])
c=torch.matmul(A,B)
print(f"\n矩阵乘法A @B:\n{c}")

# ========== 3. 索引和切片 ==========
print("\n【3】索引和切片")

x= torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(f"x:\n{x}")

print(f"x[0,1] = {x[0,1]}")
print(f"x[1,:] = {x[1,:]}")
print(f"x[:,0] = {x[:,0]}")
print(f"x[0:2,:] = \n{x[0:2,:]}")

# ========== 4. 变形操作 ==========
print("\n【4】变形操作 (reshape / view)")
x= torch.arange(12)
print(f"原始: {x}, 形状: {x.shape}")
x_3x4 =x.reshape(3,4)
print(f"reshape(3,4):\n{x_3x4}")

x_2x6 = x.view(2,6)
print(f"view(2,6):\n{x_2x6}")

# ========== 5. 广播机制 ==========
print("\n【5】广播机制 (Broadcasting)")
a=torch.tensor([[1],[2],[3]])
b=torch.tensor([10,20,30])
print(f"a:\n{a}")
print(f"b: {b}")
print(f"a+b = \n{a+b}")

# ========== 6. GPU / CPU 转换 ==========
print("\n【6】GPU / CPU 转换")
x_cpu = torch.randn(3,3)
print("在cpu:{x_cpu.device}")

if torch.cuda.is_available():
	x_gpu = x_cpu.cuda()
	print("在 GPU: {x_gpu.device}")

	# GPU 上计算
	y_gpu = x_gpu +1
	print("GPU 计算结果：\n{y_gpu}")

	# 转回 CPU
	y_cpu = y_gpu.cpu()
	print("回到 CPU：{y_cpu.device}")
else:
	print("没有GPU")
# ========== 7. 数据类型 ==========
print ("\n【7】数据类型")

x_float = torch.tensor([1.0,2.0])
x_int = torch.tensor([1,2])
print(f"默认浮点:{x_float.dtype}")
print(f"默认整型:{x_int.dtype}")

x_half = x_float.half()
print (f"半精度:{x_half.dtype}")

print("\n Tensor 基础操作完成！")