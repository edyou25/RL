"""神经网络模块定义"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLP(nn.Module):
    """多层感知机（全连接网络）"""
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int,
                 activation=nn.ReLU, output_activation=None):
        """
        初始化MLP
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度元组，如(64, 64)
            output_dim: 输出维度
            activation: 激活函数
            output_activation: 输出层激活函数
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class QNetwork(nn.Module):
    """Q网络（用于DQN）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64)):
        """
        初始化Q网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        self.network = MLP(state_dim, hidden_dims, action_dim)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            所有动作的Q值
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """对决Q网络（Dueling DQN）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64)):
        """
        初始化对决Q网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        # 特征提取层
        self.feature = MLP(state_dim, hidden_dims, hidden_dims[-1])
        
        # 价值流（Value Stream）
        self.value_stream = nn.Linear(hidden_dims[-1], 1)
        
        # 优势流（Advantage Stream）
        self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            Q值 = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 组合：Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class PolicyNetwork(nn.Module):
    """策略网络（用于策略梯度方法）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64),
                 discrete: bool = True):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度或动作数量
            hidden_dims: 隐藏层维度
            discrete: 是否为离散动作空间
        """
        super().__init__()
        self.discrete = discrete
        
        if discrete:
            # 离散动作：输出每个动作的概率
            self.network = MLP(state_dim, hidden_dims, action_dim)
        else:
            # 连续动作：输出均值和对数标准差
            self.mean_network = MLP(state_dim, hidden_dims, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """前向传播"""
        if self.discrete:
            logits = self.network(state)
            return F.softmax(logits, dim=-1)
        else:
            mean = self.mean_network(state)
            std = torch.exp(self.log_std)
            return mean, std


class ValueNetwork(nn.Module):
    """价值网络（用于Actor-Critic）"""
    
    def __init__(self, state_dim: int, hidden_dims=(64, 64)):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        self.network = MLP(state_dim, hidden_dims, 1)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            状态价值V(s)
        """
        return self.network(state).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64),
                 discrete: bool = True):
        """
        初始化Actor-Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            discrete: 是否为离散动作空间
        """
        super().__init__()
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dims, discrete)
        self.critic = ValueNetwork(state_dim, hidden_dims)
    
    def forward(self, state):
        """
        前向传播
        
        Returns:
            (action_dist, value)
        """
        action_dist = self.actor(state)
        value = self.critic(state)
        return action_dist, value

