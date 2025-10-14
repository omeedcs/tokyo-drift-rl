"""Drift Gym environments."""

from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv

# Alias for backward compatibility
DriftCarEnv = AdvancedDriftCarEnv

__all__ = ["DriftCarEnv", "AdvancedDriftCarEnv"]
