"""日志和可视化工具"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from pathlib import Path
import json
import logging
from datetime import datetime


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        """
        初始化Logger
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置Python logging
        self.logger = self._setup_logging()
        
        # 存储指标
        self.metrics: Dict[str, List] = {}
        self.episode_rewards = []
        self.episode_lengths = []
        
        self.logger.info(f"初始化Logger，实验名称: {self.experiment_name}")
        self.logger.info(f"日志目录: {self.exp_dir}")
    
    def _setup_logging(self):
        """
        配置Python logging
        
        Returns:
            logger实例
        """
        # 创建logger
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.DEBUG)
        
        # 清除已有的handlers（避免重复）
        logger.handlers.clear()
        
        # 文件handler - 详细日志
        log_file = self.exp_dir / 'training.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # 控制台handler - 简洁输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 添加handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def log_scalar(self, name: str, value: float, step: int = None):
        """
        记录标量值
        
        Args:
            name: 指标名称
            value: 指标值
            step: 步数（可选）
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value) if step is not None else value)
        
        # 使用logging记录
        if step is not None:
            self.logger.debug(f"指标 {name} (步骤 {step}): {value:.4f}")
        else:
            self.logger.debug(f"指标 {name}: {value:.4f}")
    
    def log_episode(self, reward: float, length: int, episode: int):
        """
        记录episode信息
        
        Args:
            reward: episode总奖励
            length: episode长度
            episode: episode编号
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # 使用logging记录
        self.logger.debug(f"Episode {episode}: 奖励={reward:.2f}, 长度={length}")
        
        if episode % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.logger.info(f"Episode {episode} | "
                           f"近100次平均奖励: {avg_reward:.2f} | "
                           f"当前奖励: {reward:.2f} | "
                           f"长度: {length}")
    
    def plot_learning_curve(self, window: int = 100, save: bool = True):
        """
        绘制学习曲线
        
        Args:
            window: 移动平均窗口大小
            save: 是否保存图像
        """
        if not self.episode_rewards:
            self.logger.warning("没有可绘制的数据")
            return
        
        self.logger.info("开始绘制学习曲线...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制奖励曲线
        episodes = np.arange(len(self.episode_rewards))
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, label='Raw Reward')
        
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                     np.ones(window)/window, 
                                     mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 
                    linewidth=2, label=f'{window}-Episode Moving Avg')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Reward over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制episode长度
        ax2.plot(episodes, self.episode_lengths, alpha=0.6, color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.exp_dir / 'learning_curve.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"学习曲线已保存至: {save_path}")
        
        plt.show()
    
    def save_metrics(self):
        """保存所有指标到JSON文件"""
        metrics_dict = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'other_metrics': self.metrics
        }
        
        save_path = self.exp_dir / 'metrics.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"指标已保存至: {save_path}")
    
    def print_summary(self):
        """打印训练摘要"""
        if not self.episode_rewards:
            self.logger.warning("没有训练数据")
            return
        
        summary = "\n" + "="*50 + "\n"
        summary += "训练摘要\n"
        summary += "="*50 + "\n"
        summary += f"总Episode数: {len(self.episode_rewards)}\n"
        summary += f"平均奖励: {np.mean(self.episode_rewards):.2f}\n"
        summary += f"最大奖励: {np.max(self.episode_rewards):.2f}\n"
        summary += f"最小奖励: {np.min(self.episode_rewards):.2f}\n"
        summary += f"最后100个episode平均奖励: {np.mean(self.episode_rewards[-100:]):.2f}\n"
        summary += f"平均episode长度: {np.mean(self.episode_lengths):.2f}\n"
        summary += "="*50
        
        self.logger.info(summary)

