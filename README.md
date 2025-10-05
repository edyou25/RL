# 强化学习算法复现 🎮

从零实现经典到前沿的强化学习算法。

**系统配置**: RTX 5070 + CUDA 12.8 ⚡

## ✅ 环境已就绪

```bash
conda activate rl-env
python test_installation.py  # 测试基础环境
python test_gpu.py           # 测试GPU性能
python test_gpu_training.py  # 测试GPU训练
```

**已安装**：
- ✅ PyTorch 2.10 nightly (支持RTX 5070)
- ✅ CUDA 12.8
- ✅ Gymnasium 1.2.0
- ✅ Stable-Baselines3 2.7.0
- ✅ GPU加速完美工作（~20 TFLOPS）

## 📁 项目结构

```
RL/
├── algorithms/          # 算法实现
│   ├── tabular/        # Q-Learning, SARSA等
│   ├── value_based/    # DQN系列
│   ├── policy_based/   # REINFORCE, A2C
│   └── modern/         # PPO, SAC, TD3
├── environments/       # 自定义环境
│   └── gridworld.py   # 示例网格世界
├── utils/             # 通用工具
│   ├── logger.py      # 日志系统
│   ├── networks.py    # 神经网络模块
│   ├── replay_buffer.py
│   └── config.py
├── experiments/       # 训练脚本
├── notebooks/         # Jupyter教程
│   ├── tutorials/     # 学习笔记
│   ├── experiments/   # 快速实验
│   └── analysis/      # 结果分析
├── logs/              # 训练日志
└── results/           # 训练结果
```

## 🚀 快速开始

### 1. 测试环境

```bash
conda activate rl-env

# 测试所有环境
python test_environments.py

# 测试GridWorld
python environments/gridworld.py
```

### 2. 第一个算法

查看 **[ALGORITHM_ROADMAP.md](ALGORITHM_ROADMAP.md)** 了解从基础到前沿的完整算法清单。

**推荐起点**: Q-Learning → DQN → PPO

### 3. 开发方式

**混合模式（推荐）**：
- 📓 Jupyter Notebook：学习算法、快速实验
- 🐍 Python脚本：正式实现、长时间训练

配置Jupyter：
```bash
python -m ipykernel install --user --name=rl-env --display-name="Python (RL)"
jupyter lab
```

## 📦 已安装的包

**核心**：
- PyTorch 2.5+ (GPU版本)
- Gymnasium 1.2.0
- Stable-Baselines3 2.7.0
- TensorBoard 2.20.0

**环境**：
- ✅ 经典环境：CartPole, MountainCar, Pendulum等
- ⚠️  可选环境：Atari, MuJoCo, Box2D（需手动安装）

### 可选环境安装

**Atari游戏**（如需要DQN实验）：
```bash
conda activate rl-env
pip install ale-py AutoROM
AutoROM --accept-license
```

**MuJoCo连续控制**（如需要PPO/SAC实验）：
```bash
pip install mujoco
export MUJOCO_GL=egl
```

**Box2D物理引擎**：
```bash
sudo apt-get install swig build-essential
pip install box2d-py
```

## 🎯 算法路线图

### 基础（表格方法）
- Policy Iteration & Value Iteration
- Q-Learning & SARSA
- n-step TD & TD(λ)

### 进阶（深度RL）
- DQN系列：DQN, Double DQN, Dueling DQN, Rainbow
- 策略梯度：REINFORCE, A2C, A3C
- 现代方法：PPO, SAC, TD3

### 前沿
- Decision Transformer
- MuZero
- 多智能体RL
- 离线RL

详见 [ALGORITHM_ROADMAP.md](ALGORITHM_ROADMAP.md)

## 🛠️ 工具特性

### Logger（日志系统）
```python
from utils.logger import Logger

logger = Logger(log_dir='logs', experiment_name='dqn_cartpole')
logger.log_episode(reward=100.0, length=50, episode=1)
logger.plot_learning_curve()
logger.save_metrics()
```

日志自动保存到 `logs/` 文件夹，包括：
- `training.log` - 详细训练日志
- `metrics.json` - 训练指标
- `learning_curve.png` - 学习曲线图

### 神经网络模块
```python
from utils.networks import QNetwork, PolicyNetwork, ActorCritic

q_net = QNetwork(state_dim=4, action_dim=2)
policy = PolicyNetwork(state_dim=4, action_dim=2, discrete=True)
```

### 经验回放
```python
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

buffer = ReplayBuffer(capacity=100000)
per_buffer = PrioritizedReplayBuffer(capacity=100000)
```

## 📊 示例：测试GridWorld

```python
from environments.gridworld import create_example_gridworld
import numpy as np

env = create_example_gridworld()
state = env.reset()

for step in range(100):
    action = np.random.randint(0, 4)
    next_state, reward, done, info = env.step(action)
    if done:
        break

env.render()  # 可视化
```

## 🎓 学习建议

1. **入门路径**（2-3个月）
   - Week 1-2: GridWorld + Q-Learning
   - Week 3-4: DQN on CartPole
   - Week 5-8: A2C/PPO
   - Week 9-12: 复杂环境实验

2. **实践建议**
   - 从简单环境开始（CartPole）
   - 理解算法原理再实现
   - 记录所有实验结果
   - 对比参考实现（Stable-Baselines3）

## 📖 参考资源

- **教材**: [Sutton & Barto - RL: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- **教程**: [OpenAI Spinning Up](https://spinningup.openai.com/)
- **参考代码**: [CleanRL](https://github.com/vwxyzjn/cleanrl), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

## 🔥 开始你的第一个算法

```bash
# 1. 激活环境
conda activate rl-env

# 2. 创建算法文件
# algorithms/tabular/q_learning.py

# 3. 或者在notebook中学习
jupyter lab notebooks/tutorials/
```

---

**Happy Reinforcement Learning! 🚀**
