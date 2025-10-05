"""简单的GridWorld环境实现"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class GridWorld:
    """
    简单的网格世界环境
    
    - Agent从起点移动到终点
    - 可以设置障碍物和奖励
    - 支持4个动作：上、下、左、右
    """
    
    # 动作定义
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(self, 
                 size: int = 5,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (4, 4),
                 obstacles: list = None,
                 step_reward: float = -0.1,
                 goal_reward: float = 1.0,
                 obstacle_reward: float = -1.0):
        """
        初始化GridWorld
        
        Args:
            size: 网格大小 (size x size)
            start: 起始位置 (row, col)
            goal: 目标位置 (row, col)
            obstacles: 障碍物位置列表 [(row, col), ...]
            step_reward: 每步的奖励
            goal_reward: 到达目标的奖励
            obstacle_reward: 碰到障碍物的奖励
        """
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or []
        
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_reward = obstacle_reward
        
        # 状态空间和动作空间
        self.observation_space = size * size
        self.action_space = 4
        
        # 当前位置
        self.current_pos = None
        self.reset()
    
    def reset(self) -> int:
        """
        重置环境
        
        Returns:
            初始状态（整数编码）
        """
        self.current_pos = self.start
        return self._pos_to_state(self.current_pos)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        执行动作
        
        Args:
            action: 动作 (0: 上, 1: 下, 2: 左, 3: 右)
            
        Returns:
            (next_state, reward, done, info)
        """
        row, col = self.current_pos
        
        # 执行动作
        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.DOWN:
            row = min(self.size - 1, row + 1)
        elif action == self.LEFT:
            col = max(0, col - 1)
        elif action == self.RIGHT:
            col = min(self.size - 1, col + 1)
        
        new_pos = (row, col)
        
        # 计算奖励
        if new_pos in self.obstacles:
            reward = self.obstacle_reward
            new_pos = self.current_pos  # 撞墙后停留原地
        elif new_pos == self.goal:
            reward = self.goal_reward
        else:
            reward = self.step_reward
        
        # 更新位置
        self.current_pos = new_pos
        
        # 判断是否结束
        done = (new_pos == self.goal)
        
        next_state = self._pos_to_state(new_pos)
        info = {'position': new_pos}
        
        return next_state, reward, done, info
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """将位置转换为状态编号"""
        row, col = pos
        return row * self.size + col
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """将状态编号转换为位置"""
        row = state // self.size
        col = state % self.size
        return (row, col)
    
    def render(self, mode='human'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
        """
        grid = np.zeros((self.size, self.size))
        
        # 标记障碍物
        for obs in self.obstacles:
            grid[obs] = -1
        
        # 标记目标
        grid[self.goal] = 2
        
        # 标记当前位置
        grid[self.current_pos] = 1
        
        # 可视化
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=2)
        
        # 添加网格线
        for i in range(self.size + 1):
            plt.axhline(i - 0.5, color='black', linewidth=1)
            plt.axvline(i - 0.5, color='black', linewidth=1)
        
        # 添加标签
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.current_pos:
                    plt.text(j, i, 'A', ha='center', va='center', 
                            fontsize=20, color='blue', weight='bold')
                elif (i, j) == self.goal:
                    plt.text(j, i, 'G', ha='center', va='center', 
                            fontsize=20, color='green', weight='bold')
                elif (i, j) in self.obstacles:
                    plt.text(j, i, 'X', ha='center', va='center', 
                            fontsize=20, color='red', weight='bold')
        
        plt.title('GridWorld Environment')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    
    def get_action_name(self, action: int) -> str:
        """获取动作名称"""
        action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        return action_names.get(action, '?')


# 示例：创建一个带障碍物的网格世界
def create_example_gridworld():
    """创建示例GridWorld"""
    return GridWorld(
        size=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=[(1, 1), (1, 2), (2, 1), (3, 3)],
        step_reward=-0.1,
        goal_reward=10.0,
        obstacle_reward=-5.0
    )


if __name__ == "__main__":
    # 测试环境
    env = create_example_gridworld()
    state = env.reset()
    print(f"初始状态: {state}, 位置: {env.current_pos}")
    
    # 随机执行几步
    for i in range(10):
        action = np.random.randint(0, 4)
        next_state, reward, done, info = env.step(action)
        print(f"步骤 {i+1}: 动作={env.get_action_name(action)}, "
              f"奖励={reward:.1f}, 新状态={next_state}, "
              f"位置={info['position']}, 结束={done}")
        
        if done:
            print("到达目标!")
            break
    
    # 渲染环境
    env.render()

