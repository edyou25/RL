"""经验回放缓冲区实现"""
import numpy as np
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """标准经验回放缓冲区（用于DQN等算法）"""
    
    def __init__(self, capacity: int):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一条经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区（PER）"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数（0=均匀采样，1=完全优先）
            beta: 重要性采样指数（0=无修正，1=完全修正）
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """存储经验，初始优先级设为最大"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        按优先级采样
        
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            raise ValueError("缓冲区样本不足")
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 提取经验
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32),
                indices,
                np.array(weights, dtype=np.float32))
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级
        
        Args:
            indices: 经验索引
            priorities: 新的优先级（通常是TD-error的绝对值）
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 避免0优先级
    
    def __len__(self):
        return self.size

