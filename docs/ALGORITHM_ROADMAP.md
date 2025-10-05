# 强化学习算法复现路线图

本文档列出了从基础到前沿的强化学习算法清单，按照难度和依赖关系组织。

## 📚 第一阶段：基础算法 (Tabular Methods)

### 1. 动态规划 (Dynamic Programming)
- [ ] **Policy Iteration** - 策略迭代
- [ ] **Value Iteration** - 价值迭代
- [ ] **Environment**: GridWorld (自定义简单环境)

### 2. 蒙特卡洛方法 (Monte Carlo Methods)
- [ ] **MC Prediction** - MC预测
- [ ] **MC Control with ε-greedy** - MC控制
- [ ] **Off-policy MC** - 离策略MC
- [ ] **Environment**: Blackjack, GridWorld

### 3. 时序差分学习 (Temporal Difference Learning)
- [ ] **TD(0)** - 单步TD
- [ ] **SARSA** - 在线策略TD控制
- [ ] **Q-Learning** - 离线策略TD控制
- [ ] **Expected SARSA** - 期望SARSA
- [ ] **n-step TD** - n步TD
- [ ] **TD(λ)** - TD(lambda)
- [ ] **Environment**: CliffWalking, FrozenLake, Taxi

## 🎯 第二阶段：价值函数逼近 (Value Function Approximation)

### 4. 深度Q网络系列 (DQN Family)
- [ ] **DQN** - Deep Q-Network (2015)
- [ ] **Double DQN** - 双重DQN (2016)
- [ ] **Dueling DQN** - 对决DQN (2016)
- [ ] **Prioritized Experience Replay (PER)** - 优先经验回放 (2016)
- [ ] **Noisy DQN** - 噪声网络 (2017)
- [ ] **Categorical DQN (C51)** - 分类DQN (2017)
- [ ] **Rainbow** - 集大成者 (2018)
- [ ] **Environment**: CartPole, LunarLander, Atari (Pong, Breakout)

## 🚀 第三阶段：策略梯度方法 (Policy Gradient Methods)

### 5. 基础策略梯度
- [ ] **REINFORCE** - 蒙特卡洛策略梯度 (1992)
- [ ] **REINFORCE with Baseline** - 带基线的REINFORCE
- [ ] **Actor-Critic** - 演员-评论家
- [ ] **Environment**: CartPole, LunarLander

### 6. 高级Actor-Critic
- [ ] **A2C** - Advantage Actor-Critic (2016)
- [ ] **A3C** - Asynchronous A2C (2016)
- [ ] **GAE** - Generalized Advantage Estimation (2016)
- [ ] **Environment**: Atari游戏

## 🎪 第四阶段：现代深度RL (Modern Deep RL)

### 7. 信赖域和自然梯度方法
- [ ] **TRPO** - Trust Region Policy Optimization (2015)
- [ ] **PPO** - Proximal Policy Optimization (2017) ⭐ 最流行
- [ ] **Environment**: MuJoCo (HalfCheetah, Walker2d, Humanoid)

### 8. 离线策略Actor-Critic
- [ ] **DDPG** - Deep Deterministic Policy Gradient (2016)
- [ ] **TD3** - Twin Delayed DDPG (2018)
- [ ] **SAC** - Soft Actor-Critic (2018) ⭐ 高效稳定
- [ ] **Environment**: MuJoCo连续控制任务

## 🌟 第五阶段：前沿算法 (Cutting-edge)

### 9. 模型基础RL (Model-Based RL)
- [ ] **Dyna-Q** - 整合规划和学习
- [ ] **MBPO** - Model-Based Policy Optimization (2019)
- [ ] **Dreamer** - 世界模型 (2020)
- [ ] **MuZero** - 无需了解规则的围棋AI (2020)

### 10. 多智能体RL (Multi-Agent RL)
- [ ] **IQL** - Independent Q-Learning
- [ ] **QMIX** - Q值混合网络 (2018)
- [ ] **MADDPG** - Multi-Agent DDPG (2017)
- [ ] **MAPPO** - Multi-Agent PPO (2021)

### 11. 离线RL (Offline RL)
- [ ] **BCQ** - Batch-Constrained Q-learning (2019)
- [ ] **CQL** - Conservative Q-Learning (2020)
- [ ] **IQL** - Implicit Q-Learning (2021)
- [ ] **Decision Transformer** - 序列建模方法 (2021)

### 12. 分层RL (Hierarchical RL)
- [ ] **Options Framework** - 选项框架
- [ ] **HAC** - Hierarchical Actor-Critic (2018)
- [ ] **HIRO** - Data-Efficient HRL (2018)

### 13. 元学习和迁移学习
- [ ] **MAML** - Model-Agnostic Meta-Learning (2017)
- [ ] **RL²** - Fast RL via Slow RL (2016)

### 14. 基于Transformer的方法
- [ ] **Decision Transformer** (2021) ⭐ 热门
- [ ] **Trajectory Transformer** (2021)
- [ ] **Gato** - 多任务通用Agent (2022)

## 🎯 推荐学习路径

### 初学者路径（2-3个月）
1. GridWorld + Policy/Value Iteration
2. Q-Learning + SARSA (FrozenLake, Taxi)
3. DQN (CartPole, LunarLander)
4. REINFORCE + A2C (CartPole)
5. PPO (简单MuJoCo任务)

### 进阶路径（3-6个月）
1. 完成初学者路径
2. Rainbow DQN (Atari游戏)
3. PPO + GAE (复杂环境)
4. SAC (连续控制)
5. 选择一个前沿方向深入

### 研究路径（6+个月）
1. 完成基础+进阶
2. 实现SOTA算法
3. 阅读最新论文
4. 尝试改进和创新

## 📊 环境难度评估

- **入门**: GridWorld, FrozenLake, CartPole
- **简单**: Taxi, CliffWalking, MountainCar
- **中等**: LunarLander, Atari (Pong)
- **困难**: Atari (Breakout), MuJoCo (HalfCheetah)
- **专家**: MuJoCo (Humanoid), 多智能体环境

## 🔗 参考资源

- **教材**: Sutton & Barto - Reinforcement Learning: An Introduction
- **课程**: DeepMind x UCL RL Course, Stanford CS234
- **论文**: [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- **代码**: CleanRL, Stable-Baselines3

