"""测试各个游戏环境的可用性"""
import gymnasium as gym

print("="*60)
print("游戏环境测试")
print("="*60)

# 核心环境（应该都能工作）
print("\n✅ 核心环境（已包含）:")
core_envs = [
    'CartPole-v1',
    'MountainCar-v0',
    'Acrobot-v1',
    'Pendulum-v1'
]

for env_name in core_envs:
    try:
        env = gym.make(env_name)
        obs_space = env.observation_space
        act_space = env.action_space
        print(f"  ✅ {env_name:20s} obs:{obs_space.shape if hasattr(obs_space, 'shape') else obs_space.n} act:{act_space.n if hasattr(act_space, 'n') else 'continuous'}")
        env.close()
    except Exception as e:
        print(f"  ❌ {env_name:20s} {str(e)[:40]}")

# 可选环境
print("\n⚠️  可选环境（需要额外安装）:")
optional_envs = [
    ('ALE/Pong-v5', 'Atari', 'pip install ale-py AutoROM && AutoROM --accept-license'),
    ('LunarLander-v2', 'Box2D', 'sudo apt install swig && pip install box2d-py'),
    ('HalfCheetah-v4', 'MuJoCo', 'pip install mujoco'),
]

for env_name, env_type, install_cmd in optional_envs:
    try:
        env = gym.make(env_name)
        print(f"  ✅ {env_name:20s} ({env_type})")
        env.close()
    except Exception as e:
        print(f"  ⚠️  {env_name:20s} ({env_type}) - 未安装")
        print(f"      安装: {install_cmd}")

print("\n" + "="*60)
print("提示:")
print("  - 核心环境已可用，可以开始学习基础算法")
print("  - 可选环境按需安装，详见 POST_INSTALL.md")
print("="*60)


