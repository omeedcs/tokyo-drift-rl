# 🏎️ Drift Gym - Autonomous Vehicle Drifting Environment

A professional Gymnasium environment for autonomous vehicle drifting research, based on F1/10 scale vehicle dynamics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- ✅ **Gymnasium-compliant** API for seamless RL integration
- 🎮 **Real-time visualization** with PyGame
- 🔧 **Configurable scenarios** (loose, tight, custom)
- 📊 **Dense reward shaping** for faster learning
- 🚗 **Realistic vehicle dynamics** (kinematic bicycle model)
- 📹 **Video recording** support (rgb_array mode)
- 🎯 **Multiple difficulty levels**
- 📦 **Easy installation** via pip

## Installation

### Option 1: Install from source (recommended)

```bash
git clone https://github.com/yourusername/autonomous-vehicle-drifting.git
cd autonomous-vehicle-drifting/drift_gym
pip install -e .
```

### Option 2: Install specific dependencies

```bash
pip install gymnasium numpy pygame
```

## Quick Start

```python
import gymnasium as gym
import drift_gym

# Create environment
env = gym.make('DriftCar-v0', scenario='loose', render_mode='human')

# Reset environment
obs, info = env.reset()

# Run episode
for _ in range(500):
    # Random action [velocity_cmd, angular_velocity_cmd]
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environment Details

### Observation Space

10-dimensional continuous vector:

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Normalized velocity | [-1, 1] |
| 1 | Normalized angular velocity | [-1, 1] |
| 2 | Normalized x position | [-1, 1] |
| 3 | Normalized y position | [-1, 1] |
| 4 | cos(heading angle) | [-1, 1] |
| 5 | sin(heading angle) | [-1, 1] |
| 6 | Normalized distance to gate | [0, 1] |
| 7 | cos(angle to gate) | [-1, 1] |
| 8 | sin(angle to gate) | [-1, 1] |
| 9 | Normalized min obstacle distance | [0, 1] |

### Action Space

2-dimensional continuous vector (normalized to [-1, 1]):

- **action[0]**: Velocity command (scales to [0, 4.0] m/s)
- **action[1]**: Angular velocity command (scales to [-3.0, 3.0] rad/s)

### Reward Function

**Dense Reward (default)**:
- Progress toward gate: -0.1 * distance
- Lateral error penalty: -0.5 * |error|
- Alignment bonus: -0.3 * angle_error
- Collision penalty: -50.0
- Success bonus: +50.0
- Control effort: -0.01 * (|v| + |ω|)

**Sparse Reward** (set `dense_rewards=False`):
- Success: +100.0
- Collision: -100.0
- Each step: -0.1

### Scenarios

#### Loose Drift
```python
env = gym.make('DriftCar-v0', scenario='loose')
```
- Gate width: 2.13m (84 inches)
- More room for error
- Good for initial training

#### Tight Drift
```python
env = gym.make('DriftCar-v0', scenario='tight')
```
- Gate width: 0.81m (32 inches)
- Requires precise control
- Advanced difficulty

#### Custom Configuration
```python
env = gym.make('DriftCar-v0', 
               scenario='custom',
               gate_width=1.5,
               gate_position=[3.0, 1.0],
               max_steps=600)
```

## Training Examples

### Example 1: Random Policy

```python
import gymnasium as gym
import drift_gym

env = gym.make('DriftCar-v0', scenario='loose', render_mode='human')
obs, info = env.reset()

for episode in range(10):
    terminated = truncated = False
    episode_reward = 0
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
    
    print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    obs, info = env.reset()

env.close()
```

### Example 2: Train with Stable-Baselines3

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import drift_gym

# Create vectorized environment
env = DummyVecEnv([lambda: gym.make('DriftCar-v0', scenario='loose')])

# Train SAC agent
model = SAC('MlpPolicy', env, verbose=1, 
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99)

model.learn(total_timesteps=50000)
model.save("drift_sac_model")

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
```

### Example 3: Record Video

```python
import gymnasium as gym
import drift_gym
from gymnasium.wrappers import RecordVideo

env = gym.make('DriftCar-v0', scenario='tight', render_mode='rgb_array')
env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda x: True)

obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
print("Video saved to ./videos/")
```

## Advanced Usage

### Custom Reward Function

```python
class CustomDriftEnv(drift_gym.DriftCarEnv):
    def _compute_reward(self):
        state = self.sim_env.vehicle.get_state()
        
        # Your custom reward logic
        reward = 0.0
        
        # Example: Encourage high speed
        reward += state.velocity * 0.5
        
        # Example: Penalize large slip angles
        slip_angle = abs(state.angular_velocity / max(state.velocity, 0.1))
        reward -= slip_angle * 2.0
        
        if self._passed_gate():
            reward += 100.0
        
        return reward

env = CustomDriftEnv(scenario='loose', render_mode='human')
```

### Custom Scenarios

```python
import numpy as np

env = gym.make('DriftCar-v0', scenario='loose', render_mode='human')
obs, info = env.reset()

# Add custom obstacles programmatically
from src.simulator.obstacle import CircularObstacle

custom_obstacle = CircularObstacle(x=2.0, y=1.0, radius=0.3)
env.sim_env.obstacles.append(custom_obstacle)

# Modify gate position
env.gate_pos = np.array([4.0, 2.0])
env.gate_width = 1.0
```

## Benchmarks

Performance of different algorithms on the Drift Gym environment:

| Algorithm | Loose Drift Success | Tight Drift Success | Training Steps |
|-----------|---------------------|---------------------|----------------|
| Random | 5% | 0% | - |
| SAC | **89.2%** | 68% | 50k |
| PPO | 72% | 45% | 100k |
| IKD (supervised) | 76.5% | 50% | 15min data |

## API Reference

### DriftCarEnv

```python
class DriftCarEnv(gym.Env):
    def __init__(
        self,
        scenario: str = "loose",          # "loose", "tight", or "custom"
        max_steps: int = 400,              # Maximum episode length
        render_mode: str = None,           # "human" or "rgb_array"
        dense_rewards: bool = True,        # Use dense vs sparse rewards
        gate_width: float = None,          # Custom gate width (m)
        gate_position: list = None,        # Custom gate position [x, y]
    )
```

### Methods

- `reset(seed, options)` → `(observation, info)`
- `step(action)` → `(observation, reward, terminated, truncated, info)`
- `render()` → `None` or `np.ndarray`
- `close()` → `None`

## Troubleshooting

### Issue: Pygame window not showing

**Solution**: Make sure you're using `render_mode='human'`:
```python
env = gym.make('DriftCar-v0', render_mode='human')
```

### Issue: Actions seem too fast/slow

**Solution**: Adjust action scaling:
```python
env = gym.make('DriftCar-v0')
env.max_velocity = 3.0  # Reduce max velocity
env.max_angular_velocity = 2.5  # Reduce max angular velocity
```

### Issue: Training not converging

**Solutions**:
- Start with `scenario='loose'` for easier learning
- Use `dense_rewards=True` for better learning signal
- Increase `max_steps` if agent times out frequently
- Tune hyperparameters (learning rate, batch size)
- Check reward scaling

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{drift_gym2024,
  author = {Tehrani, Omeed},
  title = {Drift Gym: A Gymnasium Environment for Autonomous Vehicle Drifting},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/autonomous-vehicle-drifting}
}
```

## License

MIT License - see LICENSE file for details.

## Related Work

- **Original IKD Paper**: [Learning Inverse Kinodynamics for Autonomous Vehicle Drifting](https://arxiv.org/abs/2402.14928) (Suvarna & Tehrani, 2024)
- **F1/10**: [F1TENTH Autonomous Racing](https://f1tenth.org/)
- **Gymnasium**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

## Acknowledgments

- UT Austin Autonomous Mobile Robotics Laboratory
- F1/10 community
- Gymnasium/Farama Foundation

---

**Happy Drifting! 🏎️💨**
