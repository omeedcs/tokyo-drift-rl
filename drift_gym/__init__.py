"""
Drift Gym - Gymnasium Environment for Autonomous Vehicle Drifting

A professional environment for autonomous vehicle drifting research based on F1/10 dynamics.
"""

from gymnasium.envs.registration import register

__version__ = "0.1.0"

# Register environment with Gymnasium
register(
    id="DriftCar-v0",
    entry_point="drift_gym.envs:DriftCarEnv",
    max_episode_steps=400,
    reward_threshold=45.0,
)

# Import main environment class
from drift_gym.envs.drift_car_env import DriftCarEnv

__all__ = ["DriftCarEnv", "__version__"]
