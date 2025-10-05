"""测试GPU训练简单神经网络"""
import torch
import torch.nn as nn
import time

print("="*60)
print("GPU训练测试")
print("="*60)

# 检查CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch版本: {torch.__version__}")

# 创建简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型和优化器
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练测试
batch_size = 256
num_iterations = 1000

print(f"\n开始训练测试...")
print(f"批次大小: {batch_size}")
print(f"迭代次数: {num_iterations}")

start = time.time()
for i in range(num_iterations):
    # 生成随机数据
    x = torch.randn(batch_size, 100, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 100 == 0:
        print(f"  迭代 {i+1}/{num_iterations}, Loss: {loss.item():.4f}")

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"\n训练完成!")
print(f"总时间: {elapsed:.2f}秒")
print(f"平均每次迭代: {elapsed/num_iterations*1000:.2f}ms")
print(f"吞吐量: {batch_size * num_iterations / elapsed:.0f} samples/sec")

# 显存使用
if torch.cuda.is_available():
    print(f"\n显存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

print("\n✅ GPU训练测试成功！")
print("="*60)


