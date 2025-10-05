# 环境配置说明

## 系统配置

- **GPU**: NVIDIA GeForce RTX 5070 (12GB)
- **CUDA**: 12.8
- **驱动**: 570.133.07

## 已安装环境

### 核心包
- Python 3.10.18
- PyTorch 2.10.0 nightly (cu128)
- Gymnasium 1.2.0
- NumPy 1.26.0
- Stable-Baselines3 2.7.0
- TensorBoard 2.20.0

### 重要说明

**RTX 5070需要PyTorch nightly版本**：
- 稳定版PyTorch 2.5不支持RTX 5070（sm_120架构）
- 必须使用PyTorch nightly with CUDA 12.8

安装命令：
```bash
conda activate rl-env
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 性能测试结果

### GPU性能
- 5000x5000矩阵乘法: 0.014秒 (~18 TFLOPS)
- 10000x10000矩阵乘法: 0.099秒 (~20 TFLOPS)
- 批量训练吞吐量: ~376k samples/sec

### 测试脚本
```bash
python test_installation.py  # 环境测试
python test_gpu.py           # GPU性能测试
python test_gpu_training.py  # 训练测试
python test_environments.py  # 游戏环境测试
```

## 可用环境

### 核心环境（已安装）
- ✅ CartPole-v1
- ✅ MountainCar-v0
- ✅ Acrobot-v1
- ✅ Pendulum-v1

### 可选环境（需手动安装）
- ⚠️  Atari: `pip install ale-py AutoROM && AutoROM --accept-license`
- ⚠️  MuJoCo: `pip install mujoco`
- ⚠️  Box2D: `pip install box2d-py`（需要swig）

## 重新创建环境

如果需要重新创建环境：

```bash
# 1. 删除旧环境
conda env remove -n rl-env

# 2. 创建基础环境
conda env create -f environment.yml

# 3. 安装PyTorch nightly
conda activate rl-env
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. 验证
python test_gpu.py
```

## 故障排除

### Q: GPU不可用
检查：
```bash
nvidia-smi  # 确认驱动正常
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: CUDA版本不匹配
确保使用nightly版本：
```bash
python -c "import torch; print(torch.__version__)"
# 应该输出: 2.10.0.dev20xxxxxx+cu128
```

### Q: 显存不足
减少batch size或模型大小。

## 下一步

环境已完全配置，可以开始：
1. 实现第一个算法（推荐Q-Learning）
2. 在CartPole上训练DQN
3. 尝试GPU加速的深度RL算法

查看 [ALGORITHM_ROADMAP.md](ALGORITHM_ROADMAP.md) 了解学习路径。


