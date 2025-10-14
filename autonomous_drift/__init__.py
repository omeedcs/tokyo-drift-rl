"""
Autonomous Drift: Research-grade autonomous vehicle drifting with deep RL.

A comprehensive framework for training and evaluating deep reinforcement learning
algorithms on autonomous drifting tasks.
"""

__version__ = "1.0.0"
__author__ = "Mihir Suvarna, Omeed Tehrani"
__license__ = "CC BY 4.0"

from autonomous_drift.envs import DriftEnv, DriftEnv3D

__all__ = [
    "DriftEnv",
    "DriftEnv3D",
]
