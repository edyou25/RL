"""测试环境安装"""
import sys

print("="*60)
print("强化学习环境测试")
print("="*60)
print(f"\nPython版本: {sys.version}\n")

# 测试基础包
print("📦 检查基础包...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch安装失败: {e}")

try:
    import gymnasium as gym
    print(f"✅ Gymnasium {gym.__version__}")
except ImportError as e:
    print(f"❌ Gymnasium安装失败: {e}")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy安装失败: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib安装失败: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas安装失败: {e}")

# 测试Gymnasium环境
print("\n🎮 测试Gymnasium环境...")
try:
    import gymnasium as gym
    
    # CartPole
    env = gym.make("CartPole-v1")
    print(f"✅ CartPole-v1 环境创建成功")
    env.close()
    
    # 检查Atari支持
    try:
        env = gym.make("ALE/Pong-v5")
        print(f"✅ Atari环境可用 (Pong)")
        env.close()
    except:
        print(f"⚠️  Atari环境未安装（可选）")
    
    # 检查Box2D支持
    try:
        env = gym.make("LunarLander-v2")
        print(f"✅ Box2D环境可用 (LunarLander)")
        env.close()
    except:
        print(f"⚠️  Box2D环境未安装（可选）")
    
    # 检查MuJoCo支持
    try:
        env = gym.make("HalfCheetah-v4")
        print(f"✅ MuJoCo环境可用 (HalfCheetah)")
        env.close()
    except:
        print(f"⚠️  MuJoCo环境未安装（可选）")
        
except Exception as e:
    print(f"❌ 环境创建失败: {e}")

# 测试自定义环境
print("\n🏗️  测试自定义环境...")
try:
    from environments.gridworld import GridWorld, create_example_gridworld
    env = create_example_gridworld()
    state = env.reset()
    print(f"✅ GridWorld自定义环境可用")
    print(f"   状态空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
except Exception as e:
    print(f"❌ GridWorld导入失败: {e}")

# 测试工具模块
print("\n🔧 测试工具模块...")
try:
    from utils.logger import Logger
    import tempfile
    import os
    
    # 使用临时目录测试
    temp_dir = tempfile.mkdtemp()
    logger = Logger(log_dir=temp_dir, experiment_name='test')
    logger.log_episode(100.0, 50, 1)
    print(f"✅ Logger工具可用")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir)
except Exception as e:
    print(f"❌ Logger测试失败: {e}")

try:
    from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    buffer = ReplayBuffer(capacity=1000)
    print(f"✅ ReplayBuffer工具可用")
except Exception as e:
    print(f"❌ ReplayBuffer导入失败: {e}")

try:
    from utils.networks import MLP, QNetwork, PolicyNetwork
    import torch
    net = QNetwork(state_dim=4, action_dim=2)
    print(f"✅ 神经网络模块可用")
except Exception as e:
    print(f"❌ 神经网络模块导入失败: {e}")

try:
    from utils.config import Config
    print(f"✅ Config工具可用")
except Exception as e:
    print(f"❌ Config导入失败: {e}")

# 测试日志记录
print("\n📝 测试日志系统...")
try:
    import tempfile
    import shutil
    from pathlib import Path
    
    temp_dir = tempfile.mkdtemp()
    logger = Logger(log_dir=temp_dir, experiment_name='log_test')
    
    # 测试各种日志级别
    logger.logger.info("这是一条INFO日志")
    logger.logger.debug("这是一条DEBUG日志")
    logger.logger.warning("这是一条WARNING日志")
    
    # 检查日志文件是否创建
    log_file = Path(temp_dir) / 'log_test' / 'training.log'
    if log_file.exists():
        print(f"✅ 日志文件创建成功: {log_file}")
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'INFO' in content and 'DEBUG' in content:
                print(f"✅ 日志内容正确记录")
    
    # 清理
    shutil.rmtree(temp_dir)
except Exception as e:
    print(f"❌ 日志系统测试失败: {e}")

# 总结
print("\n" + "="*60)
print("🎉 环境测试完成！")
print("="*60)
print("\n下一步:")
print("1. 查看 IMPLEMENTATION_GUIDE.md 了解开发建议")
print("2. 查看 ALGORITHM_ROADMAP.md 了解学习路径")
print("3. 开始实现第一个算法（推荐：Q-Learning）")
print("\n如需使用conda环境，请查看 CONDA_SETUP.md")
print("="*60)

