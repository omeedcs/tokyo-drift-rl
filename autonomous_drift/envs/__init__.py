"""
Gymnasium environments for autonomous drifting.
"""

from gymnasium.envs.registration import register

# Register 2D environment
register(
    id='DriftCar-v0',
    entry_point='autonomous_drift.envs.drift_env:DriftEnv',
    max_episode_steps=500,
    reward_threshold=100.0,
)

# Register 3D environment
register(
    id='DriftCar3D-v0',
    entry_point='autonomous_drift.envs.drift_env_3d:DriftEnv3D',
    max_episode_steps=500,
    reward_threshold=100.0,
)

# Register tight scenario
register(
    id='DriftCarTight-v0',
    entry_point='autonomous_drift.envs.drift_env:DriftEnv',
    max_episode_steps=500,
    kwargs={'scenario': 'tight', 'dense_rewards': True},
)

# Lazy imports
def _get_DriftEnv():
    from autonomous_drift.envs.drift_env import GymDriftEnv
    return GymDriftEnv

def _get_DriftEnv3D():
    try:
        from autonomous_drift.envs.drift_env_3d import DriftEnv3D
        return DriftEnv3D
    except ImportError:
        print("Warning: Panda3D not installed. 3D environment unavailable.")
        print("Install with: pip install autonomous-drift[3d]")
        return None

# Public API
__all__ = [
    'DriftEnv',
    'DriftEnv3D',
]

# Make available
DriftEnv = _get_DriftEnv()
DriftEnv3D = _get_DriftEnv3D()
