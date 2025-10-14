#!/usr/bin/env python3
"""
Train SAC Policy on Advanced Drift Gym Environment

Optimized for Apple M1 Max with MPS acceleration.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import json

# Add jake's RL library
sys.path.insert(0, 'jake-deep-rl-algos')

import deep_control as dc
from drift_gym.envs.drift_car_env_advanced import AdvancedDriftCarEnv


def get_device():
    """Get best available device for M1 Max."""
    if torch.backends.mps.is_available():
        print("✅ Using MPS (Metal Performance Shaders) - Apple Silicon GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("⚠️  Using CPU (MPS not available)")
        return torch.device("cpu")


def print_m1_max_info():
    """Print M1 Max specific information."""
    print("\n" + "="*70)
    print("🖥️  SYSTEM INFORMATION (Apple M1 Max)")
    print("="*70)
    print(f"  CPU Cores:        10 (8 performance + 2 efficiency)")
    print(f"  GPU Cores:        24-32 (integrated)")
    print(f"  PyTorch Version:  {torch.__version__}")
    print(f"  MPS Available:    {torch.backends.mps.is_available()}")
    print(f"  Device:           {get_device()}")
    print("="*70 + "\n")


def create_sac_config(use_advanced_features=True):
    """
    Create SAC configuration optimized for M1 Max.
    
    Args:
        use_advanced_features: If True, use research-grade features
                              If False, use toy mode for faster training
    """
    config = {
        # Environment
        'scenario': 'loose',
        'max_episode_steps': 400,
        
        # Advanced features (toggle for ablation study)
        'use_noisy_sensors': use_advanced_features,
        'use_perception_pipeline': use_advanced_features,
        'use_latency': use_advanced_features,
        'use_3d_dynamics': use_advanced_features,
        'use_moving_agents': use_advanced_features,
        
        # SAC Hyperparameters (tuned for M1 Max)
        'hidden_layers': [256, 256],  # 2 hidden layers
        'learning_rate': 3e-4,
        'buffer_size': 100000,  # Reduced for M1 Max memory
        'batch_size': 256,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,  # Entropy temperature
        'auto_tune_alpha': True,
        
        # Training
        'total_timesteps': 50000,  # Adjust based on time available
        'warmup_steps': 1000,
        'update_frequency': 1,
        'save_frequency': 5000,
        
        # Evaluation
        'eval_frequency': 2500,
        'eval_episodes': 10,
        
        # Device
        'device': str(get_device()),
    }
    
    return config


def train_sac(config, save_dir='sac_advanced_models'):
    """
    Train SAC agent on drift gym.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save models
    """
    print("\n" + "="*70)
    print("🚀 TRAINING SAC ON ADVANCED DRIFT GYM")
    print("="*70)
    
    # Create save directory
    # Note: deep_control saves to dc_saves/ by default, so we just use a simple name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"sac_advanced_{timestamp}"
    
    # This will actually save to dc_saves/sac_advanced_TIMESTAMP/
    # due to deep_control's make_process_dirs() function
    config_dir = os.path.join(save_dir, model_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Save config to our directory
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Config saved to: {config_path}")
    
    # Create environment
    print("\n📦 Creating environment...")
    print(f"   Scenario: {config['scenario']}")
    print(f"   Advanced Features: {config['use_noisy_sensors']}")
    
    env = AdvancedDriftCarEnv(
        scenario=config['scenario'],
        max_steps=config['max_episode_steps'],
        render_mode=None,  # No rendering during training
        use_noisy_sensors=config['use_noisy_sensors'],
        use_perception_pipeline=config['use_perception_pipeline'],
        use_latency=config['use_latency'],
        use_3d_dynamics=config['use_3d_dynamics'],
        use_moving_agents=config['use_moving_agents'],
        seed=42
    )
    
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Create SAC agent
    print("\n🤖 Creating SAC agent...")
    device = get_device()
    
    # Note: Jake's deep_control uses old gym API, need wrapper
    class GymWrapper:
        """Convert new Gymnasium API to old gym API."""
        def __init__(self, env):
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        
        def reset(self):
            obs, _ = self.env.reset()
            return obs
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return obs, reward, done, info
        
        def close(self):
            return self.env.close()
    
    wrapped_env = GymWrapper(env)
    
    agent = dc.sac.SACAgent(
        obs_space_size=env.observation_space.shape[0],
        act_space_size=env.action_space.shape[0],
        log_std_low=-20,
        log_std_high=2,
        hidden_size=config['hidden_layers'][0]  # Use first layer size
    )
    
    # Move to device
    agent.to(device)
    
    # Create replay buffer
    buffer = dc.replay.ReplayBuffer(
        config['buffer_size'],
        state_shape=env.observation_space.shape,
        state_dtype=float,
        action_shape=env.action_space.shape
    )
    
    print(f"   Hidden size: {config['hidden_layers'][0]}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Device: {device}")
    
    # Training loop using Jake's SAC implementation
    print("\n" + "="*70)
    print("🎓 STARTING TRAINING (using deep_control.sac)")
    print("="*70)
    print(f"\nTotal timesteps: {config['total_timesteps']:,}")
    print(f"Model name: {model_name}")
    print(f"Eval frequency: {config['eval_frequency']:,}")
    print("")
    
    # Train using Jake's deep_control SAC
    print("📊 Training will show progress bar below...")
    print("")
    
    # Note: deep_control will save to dc_saves/{model_name}_N/
    dc.sac.sac(
        agent=agent,
        train_env=wrapped_env,
        test_env=wrapped_env,
        buffer=buffer,
        num_steps=config['total_timesteps'],
        batch_size=config['batch_size'],
        warmup_steps=config['warmup_steps'],
        gamma=config['gamma'],
        tau=config['tau'],
        actor_lr=config['learning_rate'],
        critic_lr=config['learning_rate'],
        alpha_lr=config['learning_rate'],
        eval_interval=config['eval_frequency'],
        eval_episodes=10,
        save_interval=config['save_frequency'],
        verbosity=1,
        save_to_disk=True,
        log_to_disk=True,
        name=model_name,  # Just the name, dc_saves/ is added automatically
        max_episode_steps=config['max_episode_steps'],
        target_entropy='auto',
        init_alpha=config['alpha']
    )
    
    env.close()
    
    # Find actual save location (deep_control appends _0, _1, etc.)
    actual_save_dir = None
    for i in range(100):
        candidate = os.path.join("dc_saves", f"{model_name}_{i}")
        if os.path.exists(candidate):
            actual_save_dir = candidate
    
    # Training summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    if actual_save_dir:
        print(f"\n📁 Models saved to: {actual_save_dir}")
        print(f"   - actor.pt")
        print(f"   - critic1.pt")
        print(f"   - critic2.pt")
    print(f"\n🎯 Model name for web UI: {model_name}")
    print("="*70 + "\n")
    
    return actual_save_dir if actual_save_dir else f"dc_saves/{model_name}_0", agent


def main():
    """Main training function."""
    print_m1_max_info()
    
    print("\n🎯 Training Configuration")
    print("="*70)
    print("\nChoose training mode:")
    print("  1. Research-Grade (all advanced features) - Slower, better sim-to-real")
    print("  2. Toy Mode (no advanced features) - Faster, good for debugging")
    print("  3. Quick Test (5k steps) - Fast validation")
    
    choice = input("\nEnter choice (1/2/3) [default=1]: ").strip() or "1"
    
    if choice == "1":
        print("\n✅ Training with RESEARCH-GRADE features")
        config = create_sac_config(use_advanced_features=True)
    elif choice == "2":
        print("\n✅ Training with TOY MODE (no advanced features)")
        config = create_sac_config(use_advanced_features=False)
    elif choice == "3":
        print("\n✅ QUICK TEST MODE (5k steps)")
        config = create_sac_config(use_advanced_features=False)
        config['total_timesteps'] = 5000
        config['eval_frequency'] = 2000
        config['save_frequency'] = 2000
    else:
        print("Invalid choice, using research-grade")
        config = create_sac_config(use_advanced_features=True)
    
    # Train
    model_path, agent = train_sac(config)
    
    print("\n✅ All done! Model ready for web UI demo.")
    print(f"\n📝 To use in web UI, update simulation_server.py to load from:")
    print(f"   {model_path}")


if __name__ == "__main__":
    main()
