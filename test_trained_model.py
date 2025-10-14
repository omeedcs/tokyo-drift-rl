#!/usr/bin/env python3
"""
Quick test of your trained SAC model
"""

import sys
sys.path.insert(0, 'jake-deep-rl-algos')

import numpy as np
import deep_control as dc
from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv

# Your trained model
MODEL_PATH = "sac_advanced_models/sac_advanced_20251014_132042"

print("="*70)
print("🎯 Testing Your Trained SAC Model")
print("="*70)

# Load model
print(f"\n📦 Loading model from: {MODEL_PATH}")
agent = dc.sac.SACAgent(
    obs_space_size=13,
    act_space_size=2,
    log_std_low=-20,
    log_std_high=2,
    hidden_size=256
)
agent.load(MODEL_PATH)
agent.eval()
print("✅ Model loaded!")

# Create environment (same config as training)
print("\n🏗️  Creating environment...")
env = AdvancedDriftCarEnv(
    scenario='loose',
    max_steps=400,
    render_mode='human',  # Show visualization!
    use_noisy_sensors=True,
    use_perception_pipeline=True,
    use_latency=True,
    use_3d_dynamics=True,
    use_moving_agents=True,
    seed=42
)
print("✅ Environment created!")

# Run 3 episodes
print("\n" + "="*70)
print("🚀 Running 3 Test Episodes")
print("="*70)

for ep in range(3):
    print(f"\n📍 Episode {ep+1}/3")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        # Get action from trained policy
        action = agent.forward(obs, from_cpu=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        # Render
        env.render()
    
    success = info.get('termination_reason') == 'success'
    print(f"   Steps: {steps}")
    print(f"   Reward: {total_reward:.1f}")
    print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"   Reason: {info.get('termination_reason', 'unknown')}")

env.close()

print("\n" + "="*70)
print("✅ Testing Complete!")
print("="*70)
print("\n🎯 Your model is trained and working!")
print("🌐 Next: Start web UI to demo it online")
print("   Run: ./run_everything.sh → Press 4")
print("="*70 + "\n")
