"""测试GPU加速"""
import torch
import time

print("="*60)
print("GPU性能测试 - RTX 5070")
print("="*60)

# 检查CUDA
print(f"\n📦 PyTorch信息:")
print(f"  版本: {torch.__version__}")
print(f"  CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"\n🎮 GPU信息:")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    print(f"  当前GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  显存总量: {props.total_memory / 1024**3:.2f} GB")
    print(f"  计算能力: {props.major}.{props.minor}")
    print(f"  多处理器数量: {props.multi_processor_count}")
    
    # 显存使用情况
    print(f"\n💾 显存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # 性能测试
    print(f"\n⚡ 性能测试:")
    device = torch.device("cuda")
    
    # 小矩阵乘法（热身）
    print("  热身中...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 大矩阵乘法测试
    sizes = [5000, 10000, 15000]
    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        gflops = (2 * size**3) / (elapsed * 1e9)
        print(f"  {size}x{size} 矩阵乘法: {elapsed:.4f}秒 ({gflops:.2f} GFLOPS)")
    
    # 测试批量操作
    print(f"\n🔥 批量处理测试:")
    batch_size = 256
    input_size = 1000
    hidden_size = 512
    
    x = torch.randn(batch_size, input_size, device=device)
    w1 = torch.randn(input_size, hidden_size, device=device)
    w2 = torch.randn(hidden_size, hidden_size, device=device)
    
    start = time.time()
    for _ in range(100):
        h = torch.relu(torch.matmul(x, w1))
        out = torch.matmul(h, w2)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"  100次前向传播 (batch={batch_size}): {elapsed:.4f}秒")
    print(f"  平均每次: {elapsed/100*1000:.2f}ms")
    
    # 清理
    del a, b, c, x, w1, w2, h, out
    torch.cuda.empty_cache()
    
    print(f"\n✅ GPU加速正常工作！")
    print(f"   你的RTX 5070性能很强，非常适合训练深度RL算法！")
    
else:
    print("\n❌ CUDA不可用")
    print("   请检查:")
    print("   1. nvidia-smi命令是否正常")
    print("   2. PyTorch是否安装了GPU版本")
    print("   3. CUDA驱动是否正确安装")

print("\n" + "="*60)

