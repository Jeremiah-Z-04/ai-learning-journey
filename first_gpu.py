import torch
import time

print("=" * 50)
print("🚀 第一个 GPU 程序：矩阵乘法速度对比")
print("=" * 50)

# 创建两个大矩阵
a = torch.randn(5000, 5000).cuda()
b = torch.randn(5000, 5000).cuda()

# GPU 计算
torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"GPU 计算时间: {gpu_time*1000:.2f} ms")

# CPU 对比
a_cpu = a.cpu()
b_cpu = b.cpu()
start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start
print(f"CPU 计算时间: {cpu_time*1000:.2f} ms")
print(f"🎯 GPU 比 CPU 快了: {cpu_time/gpu_time:.1f} 倍")

# 显存信息
print(f"当前显存占用: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")