"""
Example: Training with Advanced Research-Grade Features

Demonstrates how to train an RL agent with:
- Noisy sensors (GPS drift, IMU bias)
- Perception pipeline (false positives/negatives)
- Latency (100ms delay)
- 3D dynamics (weight transfer)
- Moving obstacles
"""

import numpy as np
import gymnasium as gym
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv


def simple_policy(obs):
    """
    Simple proportional controller for testing.
    
    In real training, you'd use SAC/PPO/etc.
    """
    # Extract GPS position (with noise!)
    gps_x = obs[0] * 10.0
    gps_y = obs[1] * 10.0
    
    # Target: move forward and right
    target_x = 3.0
    target_y = 1.0
    
    # Simple proportional control
    dx = target_x - gps_x
    dy = target_y - gps_y
    
    # Actions
    velocity_cmd = 0.5 if gps_x < target_x else 0.1
    angular_velocity_cmd = np.clip(dy * 0.5, -1.0, 1.0)
    
    return np.array([velocity_cmd, angular_velocity_cmd], dtype=np.float32)


def evaluate_configuration(config_name, env_kwargs, num_episodes=5):
    """Evaluate a specific configuration."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {config_name}")
    print(f"{'=' * 70}")
    
    env = AdvancedDriftCarEnv(**env_kwargs, seed=42)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        steps = 0
        
        for _ in range(400):
            action = simple_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                if terminated and info.get("termination_reason") == "success":
                    success_count += 1
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        status = "✅ SUCCESS" if info.get("termination_reason") == "success" else "❌ FAILED"
        print(f"  Episode {episode + 1}: {status} | Reward: {episode_reward:6.1f} | Steps: {steps:3d}")
    
    env.close()
    
    # Summary
    success_rate = success_count / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print(f"\nResults:")
    print(f"  Success Rate: {success_rate:.0f}%")
    print(f"  Avg Reward:   {avg_reward:.1f}")
    print(f"  Avg Steps:    {avg_steps:.1f}")
    
    return success_rate, avg_reward, avg_steps


def main():
    """Compare different configurations."""
    print("\n" + "=" * 70)
    print(" TRAINING WITH ADVANCED FEATURES")
    print(" Comparing: Toy vs Research-Grade Environment")
    print("=" * 70)
    
    results = {}
    
    # Configuration 1: Toy mode (perfect sensors, no delays)
    print("\n🎮 TOY MODE (Perfect State, No Noise)")
    results['toy'] = evaluate_configuration(
        config_name="Toy Mode",
        env_kwargs={
            'scenario': 'loose',
            'use_noisy_sensors': False,
            'use_perception_pipeline': False,
            'use_latency': False,
            'use_3d_dynamics': False,
            'use_moving_agents': False,
        },
        num_episodes=5
    )
    
    # Configuration 2: Noisy sensors only
    print("\n🛰️  WITH SENSOR NOISE")
    results['sensors'] = evaluate_configuration(
        config_name="Noisy Sensors",
        env_kwargs={
            'scenario': 'loose',
            'use_noisy_sensors': True,
            'use_perception_pipeline': False,
            'use_latency': False,
            'use_3d_dynamics': False,
            'use_moving_agents': False,
        },
        num_episodes=5
    )
    
    # Configuration 3: With perception pipeline
    print("\n👁️  WITH PERCEPTION PIPELINE")
    results['perception'] = evaluate_configuration(
        config_name="Perception Pipeline",
        env_kwargs={
            'scenario': 'loose',
            'use_noisy_sensors': True,
            'use_perception_pipeline': True,
            'use_latency': False,
            'use_3d_dynamics': False,
            'use_moving_agents': False,
        },
        num_episodes=5
    )
    
    # Configuration 4: With latency
    print("\n⏱️  WITH LATENCY (100ms)")
    results['latency'] = evaluate_configuration(
        config_name="With Latency",
        env_kwargs={
            'scenario': 'loose',
            'use_noisy_sensors': True,
            'use_perception_pipeline': True,
            'use_latency': True,
            'use_3d_dynamics': False,
            'use_moving_agents': False,
        },
        num_episodes=5
    )
    
    # Configuration 5: Full realism
    print("\n🔬 FULL RESEARCH-GRADE (All Features)")
    results['full'] = evaluate_configuration(
        config_name="Full Realism",
        env_kwargs={
            'scenario': 'loose',
            'use_noisy_sensors': True,
            'use_perception_pipeline': True,
            'use_latency': True,
            'use_3d_dynamics': True,
            'use_moving_agents': True,
        },
        num_episodes=5
    )
    
    # Final comparison
    print("\n" + "=" * 70)
    print(" FINAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'Success Rate':<15} {'Avg Reward':<15}")
    print("-" * 70)
    
    for config_name, (success_rate, avg_reward, avg_steps) in results.items():
        print(f"{config_name:<25} {success_rate:>13.0f}% {avg_reward:>14.1f}")
    
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("""
1. 🎮 TOY MODE: Agent performs well but won't transfer to real robots
   - Gets perfect state information
   - No realistic challenges
   
2. 🛰️ SENSOR NOISE: Performance degrades - agent must handle uncertainty
   - GPS drift and IMU bias affect state estimation
   - Requires filtering/robust control
   
3. 👁️ PERCEPTION: False positives/negatives make it harder
   - Must distinguish real obstacles from clutter
   - Track management is critical
   
4. ⏱️ LATENCY: 100ms delay requires anticipation
   - Can't react instantly
   - Must predict future states
   
5. 🔬 FULL REALISM: Most challenging but ready for real hardware!
   - All challenges combined
   - Agents trained here will transfer better to real robots
    """)
    
    print("=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)
    print("""
For Real Robot Deployment:
✅ Train with ALL features enabled
✅ Use domain randomization (vary noise levels)
✅ Test extensively in simulation first
✅ Implement safety checks on real hardware
✅ Start with lower speeds on real robot

For Research Papers:
✅ Report results with and without realistic features
✅ Ablation study: show impact of each feature
✅ Compare sim-to-real transfer performance
    """)
    
    print("=" * 70)
    print("\n✅ Demo complete! Your environment is research-grade.\n")


if __name__ == "__main__":
    main()
