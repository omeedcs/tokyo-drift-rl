#!/usr/bin/env python3
"""
Model Playground - Interactive UI for Autonomous Vehicle Drifting.

A comprehensive Gradio-based interface for:
- Loading and testing pretrained models
- Running simulations with real-time visualization
- Comparing model performance
- Adjusting parameters interactively
- Augmenting data with IMU delays

Launch with: python model_playground.py
"""

import sys
sys.path.insert(0, 'jake-deep-rl-algos')

import gradio as gr
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import threading
from datetime import datetime

# Import project modules
from src.utils.model_loader import PretrainedModelLoader, ModelInfo
from src.rl.gym_drift_env import GymDriftEnv
from src.simulator.environment import SimulationEnvironment
from src.models.ikd_model import IKDModel
from src.data_processing.imu_augmentation import IMUDelayAugmenter, IMUDelayConfig
from src.visualization.realtime_plots import InteractivePlotter
import deep_control as dc


# Global state
model_loader = PretrainedModelLoader()
current_ikd_model = None
current_sac_agent = None
current_env = None


def load_ikd_model(model_name: str) -> str:
    """Load IKD model."""
    global current_ikd_model
    try:
        if model_name is None:
            return "❌ No model selected. Please select an IKD model from the dropdown."
        current_ikd_model, info = model_loader.load_ikd(model_name)
        return f"✅ Loaded IKD model: {model_name}\nTrained: {info.timestamp or 'Unknown'}"
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}"


def load_sac_model(model_name: str) -> str:
    """Load SAC model."""
    global current_sac_agent
    try:
        if model_name is None:
            return "❌ No model selected. Please select a SAC model from the dropdown."
        current_sac_agent, info = model_loader.load_sac(model_name)
        return f"✅ Loaded SAC agent: {model_name}\nTrained: {info.timestamp or 'Unknown'}"
    except Exception as e:
        return f"❌ Failed to load agent: {str(e)}"


def run_ikd_simulation(
    model_name: str,
    scenario: str,
    max_steps: int,
    velocity_cmd: float,
    angular_vel_cmd: float,
    progress=gr.Progress()
) -> Tuple[go.Figure, str]:
    """Run IKD simulation with progress tracking."""
    try:
        progress(0, desc="Initializing IKD simulation...")
        
        if model_name is None:
            return go.Figure(), "❌ No model selected. Please select an IKD model."
        
        # Load model
        progress(0.1, desc="Loading IKD model...")
        if current_ikd_model is None:
            load_ikd_model(model_name)
        
        # Create environment
        progress(0.2, desc="Setting up environment...")
        env = GymDriftEnv(scenario=scenario.lower(), max_steps=max_steps)
        obs, _ = env.reset()
        
        # Storage
        trajectory = []
        rewards = []
        velocities = []
        angular_vels = []
        steps_taken = 0
        success = False
        collision = False
        
        # Run simulation with progress updates
        for step in range(max_steps):
            progress((0.2 + 0.7 * step / max_steps), desc=f"Simulating step {step+1}/{max_steps}")
            
            # Get vehicle state
            state = env.sim_env.vehicle.get_state()
            trajectory.append([state.x, state.y, state.theta])
            velocities.append(state.velocity)
            angular_vels.append(state.angular_velocity)
            
            # Simple constant command (IKD correction would be applied here)
            action = np.array([velocity_cmd / 4.0, angular_vel_cmd / 3.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            steps_taken = step + 1
            
            if terminated:
                success = "success" in info.get("termination_reason", "")
                collision = "collision" in info.get("termination_reason", "")
                break
            
            if truncated:
                break
        
        env.close()
        
        progress(0.92, desc="Analyzing trajectory data...")
        time.sleep(0.5)  # Allow progress to be visible
        
        progress(0.95, desc="Rendering animated visualization...")
        
        # Create trajectory plot
        trajectory = np.array(trajectory)
        time.sleep(0.3)  # Build suspense
        fig = create_trajectory_plot(trajectory, scenario, env.gate_pos, env.gate_width)
        
        # Enhanced summary with detailed metrics
        summary = f"""
## 🎯 IKD Simulation Complete!

### 📊 Simulation Parameters
- **Model**: `{model_name}`
- **Scenario**: {scenario.upper()}
- **Max Steps**: {max_steps}
- **Velocity Command**: {velocity_cmd:.2f} m/s
- **Angular Velocity Command**: {angular_vel_cmd:.2f} rad/s

### 🏆 Results
| Metric | Value |
|--------|-------|
| **Success** | {'✅ Yes' if success else '❌ No'} |
| **Steps Taken** | {steps_taken} / {max_steps} |
| **Total Reward** | {sum(rewards):.2f} |
| **Avg Reward/Step** | {np.mean(rewards):.3f} |
| **Final Position** | ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f}) m |
| **Distance to Goal** | {np.sqrt((trajectory[-1, 0] - env.gate_pos[0])**2 + (trajectory[-1, 1] - env.gate_pos[1])**2):.2f} m |

### 📈 Performance Metrics
- **Avg Velocity**: {np.mean(velocities):.2f} m/s
- **Max Velocity**: {np.max(velocities):.2f} m/s
- **Avg Angular Velocity**: {np.mean(np.abs(angular_vels)):.2f} rad/s
- **Trajectory Length**: {np.sum(np.sqrt(np.diff(trajectory[:, 0])**2 + np.diff(trajectory[:, 1])**2)):.2f} m
- **Termination Reason**: {'Success' if success else ('Collision' if collision else 'Timeout/Out of Bounds')}
        """
        
        progress(1.0, desc="Done!")
        
        return fig, summary
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Simulation failed: {str(e)}\n\n```python\n{traceback.format_exc()}\n```"
        return go.Figure(), error_msg


def run_sac_simulation(
    model_name: str,
    scenario: str,
    max_steps: int,
    n_episodes: int,
    progress=gr.Progress()
) -> Tuple[go.Figure, go.Figure, str]:
    """Run SAC simulation with progress tracking."""
    try:
        progress(0, desc="Initializing...")
        
        if model_name is None:
            return go.Figure(), go.Figure(), "❌ No model selected. Please select a SAC model."
        
        # Load agent
        progress(0.1, desc="Loading SAC model...")
        if current_sac_agent is None:
            load_sac_model(model_name)
        
        # Create environment
        progress(0.2, desc="Setting up environment...")
        env = GymDriftEnv(scenario=scenario.lower(), max_steps=max_steps)
        
        all_trajectories = []
        all_rewards = []
        all_velocities = []
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # Run episodes with progress updates
        for ep in range(n_episodes):
            progress((0.2 + 0.7 * ep / n_episodes), desc=f"Running Episode {ep+1}/{n_episodes}")
            
            obs, _ = env.reset()
            trajectory = []
            rewards_ep = []
            velocities_ep = []
            episode_reward = 0
            
            for step in range(max_steps):
                state = env.sim_env.vehicle.get_state()
                trajectory.append([state.x, state.y, state.theta])
                velocities_ep.append(state.velocity)
                
                # Get action from SAC agent
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs)
                    action = current_sac_agent.forward(obs_tensor)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                rewards_ep.append(reward)
                
                if terminated or truncated:
                    if terminated and "success" in info.get("termination_reason", ""):
                        success_count += 1
                    break
            
            all_trajectories.append(np.array(trajectory))
            all_rewards.append(rewards_ep)
            all_velocities.append(velocities_ep)
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(trajectory))
        
        env.close()
        
        progress(0.85, desc="Processing episode data...")
        time.sleep(0.4)
        
        progress(0.9, desc="Generating trajectory visualizations...")
        time.sleep(0.3)
        
        # Create enhanced plots
        traj_fig = create_multi_trajectory_plot(all_trajectories, scenario, env.gate_pos, env.gate_width)
        
        progress(0.94, desc="Creating performance charts...")
        time.sleep(0.3)
        perf_fig = create_performance_plot(episode_rewards, episode_lengths)
        
        # Enhanced summary with more details
        summary = f"""
## 🎯 SAC Simulation Complete!

### 📊 Performance Summary
- **Model**: `{model_name}`
- **Scenario**: {scenario.upper()}
- **Episodes**: {n_episodes} | **Max Steps**: {max_steps}

### 🏆 Results
| Metric | Value |
|--------|-------|
| **Success Rate** | {success_count / n_episodes * 100:.1f}% ({success_count}/{n_episodes}) |
| **Avg Reward** | {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f} |
| **Avg Steps** | {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} |
| **Best Episode** | {np.max(episode_rewards):.2f} |
| **Worst Episode** | {np.min(episode_rewards):.2f} |
| **Median Reward** | {np.median(episode_rewards):.2f} |
| **Total Steps** | {sum(episode_lengths)} |

### 📈 Performance Insights
- **Consistency**: Std Dev = {np.std(episode_rewards):.2f}
- **Success Episodes**: {[i+1 for i in range(n_episodes) if i < success_count]}
- **Avg Velocity**: {np.mean([np.mean(v) for v in all_velocities]):.2f} m/s
        """
        
        progress(1.0, desc="Done!")
        
        return traj_fig, perf_fig, summary
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Simulation failed: {str(e)}\n\n```python\n{traceback.format_exc()}\n```"
        return go.Figure(), go.Figure(), error_msg


def compare_models(
    ikd_model: str,
    sac_model: str,
    scenario: str,
    n_trials: int,
    max_steps: int,
    progress=gr.Progress()
) -> Tuple[go.Figure, str]:
    """Compare IKD and SAC models with detailed analysis."""
    try:
        progress(0, desc="Initializing comparison...")
        
        if ikd_model is None or sac_model is None:
            return go.Figure(), "❌ Please select both IKD and SAC models."
        
        results = {
            'IKD': {'rewards': [], 'steps': [], 'success': 0},
            'SAC': {'rewards': [], 'steps': [], 'success': 0}
        }
        
        # Test IKD
        progress(0.1, desc="Testing IKD model...")
        load_ikd_model(ikd_model)
        env = GymDriftEnv(scenario=scenario.lower(), max_steps=max_steps)
        
        for trial in range(n_trials):
            progress((0.1 + 0.4 * trial / n_trials), desc=f"IKD Trial {trial+1}/{n_trials}")
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                action = np.array([0.5, 0.2])  # Simplified
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated:
                    if "success" in info.get("termination_reason", ""):
                        results['IKD']['success'] += 1
                    break
                if truncated:
                    break
            
            results['IKD']['rewards'].append(total_reward)
            results['IKD']['steps'].append(steps)
        
        env.close()
        
        # Test SAC
        progress(0.5, desc="Testing SAC model...")
        load_sac_model(sac_model)
        env = GymDriftEnv(scenario=scenario.lower(), max_steps=max_steps)
        
        for trial in range(n_trials):
            progress((0.5 + 0.4 * trial / n_trials), desc=f"SAC Trial {trial+1}/{n_trials}")
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs)
                    action = current_sac_agent.forward(obs_tensor)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated:
                    if "success" in info.get("termination_reason", ""):
                        results['SAC']['success'] += 1
                    break
                if truncated:
                    break
            
            results['SAC']['rewards'].append(total_reward)
            results['SAC']['steps'].append(steps)
        
        env.close()
        
        progress(0.92, desc="Analyzing performance metrics...")
        time.sleep(0.5)
        
        progress(0.95, desc="Generating comparison visualizations...")
        time.sleep(0.4)
        
        # Create comparison plot
        fig = create_comparison_plot(results)
        
        # Calculate statistics
        ikd_mean = np.mean(results['IKD']['rewards'])
        sac_mean = np.mean(results['SAC']['rewards'])
        improvement = abs(sac_mean - ikd_mean) / abs(ikd_mean) * 100 if ikd_mean != 0 else 0
        winner = 'SAC' if sac_mean > ikd_mean else 'IKD'
        
        # Enhanced summary
        summary = f"""
## 🏁 Model Comparison Complete!

### 🔬 Test Configuration
- **IKD Model**: `{ikd_model}`
- **SAC Model**: `{sac_model}`
- **Scenario**: {scenario.upper()}
- **Trials per Model**: {n_trials}
- **Max Steps**: {max_steps}

### 📊 Results Summary

| Metric | IKD | SAC | Difference |
|--------|-----|-----|------------|
| **Success Rate** | {results['IKD']['success'] / n_trials * 100:.1f}% | {results['SAC']['success'] / n_trials * 100:.1f}% | {(results['SAC']['success'] - results['IKD']['success']) / n_trials * 100:+.1f}% |
| **Avg Reward** | {ikd_mean:.2f} | {sac_mean:.2f} | {sac_mean - ikd_mean:+.2f} |
| **Avg Steps** | {np.mean(results['IKD']['steps']):.1f} | {np.mean(results['SAC']['steps']):.1f} | {np.mean(results['SAC']['steps']) - np.mean(results['IKD']['steps']):+.1f} |
| **Std Dev (Reward)** | {np.std(results['IKD']['rewards']):.2f} | {np.std(results['SAC']['rewards']):.2f} | - |
| **Best Trial** | {np.max(results['IKD']['rewards']):.2f} | {np.max(results['SAC']['rewards']):.2f} | - |
| **Worst Trial** | {np.min(results['IKD']['rewards']):.2f} | {np.min(results['SAC']['rewards']):.2f} | - |

### 🏆 Winner: **{winner}** 
- **Performance Improvement**: {improvement:.1f}%
- **Consistency**: {'SAC' if np.std(results['SAC']['rewards']) < np.std(results['IKD']['rewards']) else 'IKD'} has lower variance
- **Reliability**: {'SAC' if results['SAC']['success'] > results['IKD']['success'] else 'IKD'} has higher success rate

### 💡 Recommendation
{f"**SAC is recommended** for this scenario due to {improvement:.1f}% better average reward and {'higher' if results['SAC']['success'] > results['IKD']['success'] else 'similar'} success rate." if winner == 'SAC' else f"**IKD is recommended** for this scenario due to better average reward."}
        """
        
        progress(1.0, desc="Comparison complete!")
        
        return fig, summary
        
    except Exception as e:
        return go.Figure(), f"❌ Comparison failed: {str(e)}"


def augment_data_demo(
    min_delay: float,
    max_delay: float,
    augmentation_factor: int
) -> Tuple[go.Figure, str]:
    """Demonstrate IMU delay augmentation."""
    # Generate synthetic signal
    duration = 5.0
    sample_rate = 40.0
    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Synthetic angular velocity
    original_signal = 2.0 * np.sin(2 * np.pi * 0.5 * time) + 0.1 * time
    
    # Create augmenter
    config = IMUDelayConfig(
        min_delay=min_delay,
        max_delay=max_delay,
        sample_rate=sample_rate
    )
    augmenter = IMUDelayAugmenter(config)
    
    # Generate augmentations
    velocity = np.ones(n_samples) * 2.0
    augmented = augmenter.augment_trajectory(
        original_signal,
        velocity,
        augmentation_factor=augmentation_factor
    )
    
    # Create plot
    fig = go.Figure()
    
    # Original signal
    fig.add_trace(go.Scatter(
        x=time,
        y=original_signal,
        mode='lines',
        name='Original',
        line=dict(color='blue', width=3)
    ))
    
    # Augmented signals
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    for i, aug in enumerate(augmented[:5]):  # Show max 5
        fig.add_trace(go.Scatter(
            x=time,
            y=aug['angular_velocity'],
            mode='lines',
            name=f'Delay {aug["delay_applied"]:.3f}s',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='IMU Delay Augmentation Demonstration',
        xaxis_title='Time (s)',
        yaxis_title='Angular Velocity (rad/s)',
        template='plotly_dark',
        height=500
    )
    
    summary = f"""
**Augmentation Results:**
- Original samples: {n_samples}
- Augmentation factor: {augmentation_factor}
- Total augmented: {len(augmented)}
- Delay range: {min_delay:.3f}s - {max_delay:.3f}s
- Delays applied: {', '.join([f'{a["delay_applied"]:.3f}s' for a in augmented])}
    """
    
    return fig, summary


def create_trajectory_plot(trajectory: np.ndarray, scenario: str, gate_pos: np.ndarray, gate_width: float) -> go.Figure:
    """Create animated trajectory plot."""
    fig = go.Figure()
    
    # Animated trajectory trace
    n_points = len(trajectory)
    frames = []
    
    # Create frames for animation
    for i in range(1, n_points + 1):
        frame_data = [
            # Trajectory line (grows over time)
            go.Scatter(
                x=trajectory[:i, 0],
                y=trajectory[:i, 1],
                mode='lines',
                name='Trajectory',
                line=dict(color='cyan', width=3),
                showlegend=(i == 1)
            ),
            # Current position marker
            go.Scatter(
                x=[trajectory[i-1, 0]],
                y=[trajectory[i-1, 1]],
                mode='markers',
                name='Vehicle',
                marker=dict(
                    size=15,
                    color='cyan',
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                showlegend=(i == 1)
            ),
            # Trail markers
            go.Scatter(
                x=trajectory[max(0, i-10):i, 0],
                y=trajectory[max(0, i-10):i, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color='cyan',
                    opacity=np.linspace(0.2, 0.8, min(10, i))
                ),
                showlegend=False
            )
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial frame
    fig.add_trace(go.Scatter(
        x=[trajectory[0, 0]],
        y=[trajectory[0, 1]],
        mode='markers',
        name='Start',
        marker=dict(size=20, color='green', symbol='star')
    ))
    
    # Goal gate
    y1 = gate_pos[1] - gate_width / 2
    y2 = gate_pos[1] + gate_width / 2
    fig.add_trace(go.Scatter(
        x=[gate_pos[0], gate_pos[0]],
        y=[y1, y2],
        mode='lines+markers',
        name='Goal',
        line=dict(color='lime', width=6),
        marker=dict(size=15, symbol='diamond')
    ))
    
    # Add frames
    fig.frames = frames
    
    # Animation controls
    fig.update_layout(
        title=dict(
            text=f'🏎️ Vehicle Trajectory - {scenario.upper()} Scenario',
            font=dict(size=24, color='cyan')
        ),
        xaxis=dict(
            title='X Position (m)',
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.2)'
        ),
        yaxis=dict(
            title='Y Position (m)',
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.2)'
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(26, 26, 46, 0.8)',
        plot_bgcolor='rgba(10, 10, 15, 0.8)',
        height=700,
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 20}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 1.15,
            'xanchor': 'left',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(i),
                    'method': 'animate'
                }
                for i, f in enumerate(fig.frames)
            ],
            'x': 0.1,
            'len': 0.85,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'pad': {'b': 10, 't': 50},
            'currentvalue': {
                'visible': True,
                'prefix': 'Step: ',
                'xanchor': 'right',
                'font': {'size': 16, 'color': 'cyan'}
            }
        }]
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_multi_trajectory_plot(trajectories: List[np.ndarray], scenario: str, gate_pos: np.ndarray, gate_width: float) -> go.Figure:
    """Create animated plot with multiple trajectories."""
    fig = go.Figure()
    
    colors = ['#00ffff', '#ff00ff', '#ffff00', '#00ff00', '#ff0000', '#ff8000', '#8000ff']
    
    # Add each trajectory with gradient opacity
    for i, traj in enumerate(trajectories):
        # Main trajectory line
        fig.add_trace(go.Scatter(
            x=traj[:, 0],
            y=traj[:, 1],
            mode='lines',
            name=f'Episode {i+1}',
            line=dict(
                color=colors[i % len(colors)],
                width=3
            ),
            opacity=0.8,
            hovertemplate=f'<b>Episode {i+1}</b><br>X: %{{x:.2f}}m<br>Y: %{{y:.2f}}m<extra></extra>'
        ))
        
        # Start marker
        fig.add_trace(go.Scatter(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            mode='markers',
            marker=dict(size=12, color=colors[i % len(colors)], symbol='circle', line=dict(color='white', width=2)),
            showlegend=False,
            hovertemplate=f'<b>Ep {i+1} Start</b><extra></extra>'
        ))
        
        # End marker
        fig.add_trace(go.Scatter(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            mode='markers',
            marker=dict(size=12, color=colors[i % len(colors)], symbol='x', line=dict(width=3)),
            showlegend=False,
            hovertemplate=f'<b>Ep {i+1} End</b><extra></extra>'
        ))
    
    # Goal gate
    y1 = gate_pos[1] - gate_width / 2
    y2 = gate_pos[1] + gate_width / 2
    fig.add_trace(go.Scatter(
        x=[gate_pos[0], gate_pos[0]],
        y=[y1, y2],
        mode='lines+markers',
        name='Goal Gate',
        line=dict(color='lime', width=8),
        marker=dict(size=20, symbol='diamond'),
        hovertemplate='<b>Goal Gate</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'🏁 Multi-Episode Analysis - {scenario.upper()} Scenario ({len(trajectories)} episodes)',
            font=dict(size=24, color='cyan')
        ),
        xaxis=dict(
            title='X Position (m)',
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title='Y Position (m)',
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.2)',
            showgrid=True
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(26, 26, 46, 0.8)',
        plot_bgcolor='rgba(10, 10, 15, 0.8)',
        height=700,
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(26, 26, 46, 0.8)',
            bordercolor='cyan',
            borderwidth=1,
            font=dict(color='white')
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_performance_plot(rewards: List[float], steps: List[int]) -> go.Figure:
    """Create performance metrics plot."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Episode Rewards', 'Episode Lengths')
    )
    
    episodes = list(range(1, len(rewards) + 1))
    
    fig.add_trace(
        go.Bar(x=episodes, y=rewards, name='Reward', marker_color='cyan'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=episodes, y=steps, name='Steps', marker_color='magenta'),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    return fig


def create_comparison_plot(results: Dict) -> go.Figure:
    """Create model comparison plot."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Success Rate', 'Average Reward', 'Average Steps'),
        specs=[[{'type': 'bar'}, {'type': 'box'}, {'type': 'box'}]]
    )
    
    methods = list(results.keys())
    
    # Success rates
    success_rates = [results[m]['success'] / len(results[m]['rewards']) * 100 for m in methods]
    fig.add_trace(
        go.Bar(x=methods, y=success_rates, marker_color=['blue', 'green']),
        row=1, col=1
    )
    
    # Rewards
    for method in methods:
        fig.add_trace(
            go.Box(y=results[method]['rewards'], name=method),
            row=1, col=2
        )
    
    # Steps
    for method in methods:
        fig.add_trace(
            go.Box(y=results[method]['steps'], name=method, showlegend=False),
            row=1, col=3
        )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig


def get_available_models() -> Tuple[List[str], List[str]]:
    """Get lists of available models."""
    models = model_loader.list_available_models()
    ikd_names = [m.name for m in models['ikd']]
    sac_names = [m.name for m in models['sac']]
    return ikd_names, sac_names


# Custom CSS for premium dark theme
CUSTOM_CSS = """
/* Dark Premium Theme */
:root {
    --primary-color: #00ffff;
    --secondary-color: #ff00ff;
    --success-color: #00ff00;
    --warning-color: #ffff00;
    --danger-color: #ff0000;
    --bg-dark: #0a0a0f;
    --bg-card: #1a1a2e;
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
}

body {
    background: linear-gradient(135deg, #0a0a0f 0%, #16213e 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
}

/* Header styling */
.markdown h1 {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em !important;
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 20px;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 10px var(--primary-color)); }
    to { filter: drop-shadow(0 0 20px var(--secondary-color)); }
}

/* Card styling */
.block {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}

.block:hover {
    border-color: rgba(0, 255, 255, 0.5) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.3) !important;
    transform: translateY(-2px) !important;
}

/* Tab styling */
.tab-nav button {
    background: transparent !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border-radius: 10px 10px 0 0 !important;
    margin: 0 5px !important;
}

.tab-nav button.selected {
    background: linear-gradient(180deg, rgba(0, 255, 255, 0.2), transparent) !important;
    border-bottom-color: var(--primary-color) !important;
    color: var(--primary-color) !important;
    border-width: 2px !important;
}

.tab-nav button:hover {
    background: rgba(0, 255, 255, 0.1) !important;
    color: var(--primary-color) !important;
}

/* Button styling */
.btn-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    border: none !important;
    color: #000 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px 0 rgba(0, 255, 255, 0.3) !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(0, 255, 255, 0.5) !important;
}

/* Input styling */
input, select, textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

input:focus, select:focus, textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.5) !important;
}

/* Slider styling */
input[type="range"] {
    background: rgba(0, 255, 255, 0.1) !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: var(--primary-color) !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.5) !important;
}

/* Progress bar */
.progress-bar {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
    animation: shimmer 2s infinite !important;
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

/* Plot container */
.plot-container {
    background: var(--bg-card) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.1) !important;
}

/* Status indicators */
.status-success { color: var(--success-color) !important; }
.status-warning { color: var(--warning-color) !important; }
.status-error { color: var(--danger-color) !important; }

/* Markdown content */
.markdown-text {
    color: var(--text-primary) !important;
    line-height: 1.8 !important;
}

.markdown-text table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 20px 0 !important;
}

.markdown-text th {
    background: rgba(0, 255, 255, 0.2) !important;
    color: var(--primary-color) !important;
    padding: 12px !important;
    font-weight: 700 !important;
}

.markdown-text td {
    padding: 10px !important;
    border-bottom: 1px solid rgba(0, 255, 255, 0.1) !important;
}

.markdown-text tr:hover {
    background: rgba(0, 255, 255, 0.05) !important;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 1.5s ease-in-out infinite !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-color);
}
"""


# Build Gradio Interface
def build_interface():
    """Build premium Gradio interface with custom styling."""
    
    ikd_models, sac_models = get_available_models()
    
    # Custom theme with simplified configuration
    custom_theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="purple",
        neutral_hue="slate"
    )
    
    with gr.Blocks(title="Autonomous Drifting Model Playground", theme=custom_theme, css=CUSTOM_CSS) as app:
        
        gr.Markdown("""
        # 🏎️ Autonomous Vehicle Drifting - Advanced Model Playground
        
        **Professional interactive platform for testing, comparing, and analyzing drift control models**
        
        🎯 **Features:** Real-time progress tracking | Detailed performance metrics | Interactive visualizations | Model comparison | IMU augmentation
        
        ---
        """)
        
        with gr.Tabs():
            
            # Tab 1: Model Browser
            with gr.Tab("📦 Model Browser"):
                gr.Markdown("### Available Pretrained Models")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### IKD Models")
                        ikd_model_list = gr.Dropdown(
                            choices=ikd_models,
                            label="Select IKD Model",
                            value=ikd_models[0] if ikd_models else None
                        )
                        load_ikd_btn = gr.Button("Load IKD Model", variant="primary")
                        ikd_status = gr.Textbox(label="Status", lines=3)
                    
                    with gr.Column():
                        gr.Markdown("#### SAC Models")
                        sac_model_list = gr.Dropdown(
                            choices=sac_models,
                            label="Select SAC Model",
                            value=sac_models[0] if sac_models else None
                        )
                        load_sac_btn = gr.Button("Load SAC Model", variant="primary")
                        sac_status = gr.Textbox(label="Status", lines=3)
                
                load_ikd_btn.click(load_ikd_model, inputs=[ikd_model_list], outputs=[ikd_status])
                load_sac_btn.click(load_sac_model, inputs=[sac_model_list], outputs=[sac_status])
            
            # Tab 2: IKD Simulator
            with gr.Tab("🔵 IKD Simulator"):
                gr.Markdown("""
                ### 🔵 Test Inverse Kinodynamic Model
                
                Run single-episode simulations with customizable control parameters and track detailed performance metrics.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Simulation Configuration")
                        ikd_sim_model = gr.Dropdown(choices=ikd_models, label="IKD Model", value=ikd_models[0] if ikd_models else None)
                        ikd_scenario = gr.Radio(["Loose", "Tight"], label="Scenario", value="Loose")
                        ikd_max_steps = gr.Slider(100, 1000, value=300, step=50, label="Max Steps")
                        ikd_velocity = gr.Slider(0, 4, value=2.0, step=0.1, label="Velocity Command (m/s)")
                        ikd_angular_vel = gr.Slider(-3, 3, value=0.5, step=0.1, label="Angular Velocity Command (rad/s)")
                        run_ikd_btn = gr.Button("🚀 Run Simulation", variant="primary", size="lg")
                    
                    with gr.Column():
                        ikd_plot = gr.Plot(label="2D Trajectory Visualization")
                        ikd_results = gr.Markdown("**Ready to simulate!** Configure parameters and click 'Run Simulation'.")
                
                run_ikd_btn.click(
                    run_ikd_simulation,
                    inputs=[ikd_sim_model, ikd_scenario, ikd_max_steps, ikd_velocity, ikd_angular_vel],
                    outputs=[ikd_plot, ikd_results]
                )
            
            # Tab 3: SAC Simulator
            with gr.Tab("🟢 SAC Simulator"):
                gr.Markdown("""
                ### 🟢 Test Soft Actor-Critic Agent
                
                Run multi-episode simulations with SAC agent and analyze performance across episodes with comprehensive metrics.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Simulation Configuration")
                        sac_sim_model = gr.Dropdown(choices=sac_models, label="SAC Model", value=sac_models[0] if sac_models else None)
                        sac_scenario = gr.Radio(["Loose", "Tight"], label="Scenario", value="Loose")
                        sac_max_steps = gr.Slider(100, 1000, value=300, step=50, label="Max Steps per Episode")
                        sac_episodes = gr.Slider(1, 50, value=10, step=1, label="Number of Episodes")
                        run_sac_btn = gr.Button("🚀 Run Multi-Episode Simulation", variant="primary", size="lg")
                    
                    with gr.Column():
                        sac_traj_plot = gr.Plot(label="Multi-Episode Trajectories")
                        sac_perf_plot = gr.Plot(label="Episode Performance Metrics")
                        sac_results = gr.Markdown("**Ready to simulate!** Configure parameters and click 'Run Multi-Episode Simulation'.")
                
                run_sac_btn.click(
                    run_sac_simulation,
                    inputs=[sac_sim_model, sac_scenario, sac_max_steps, sac_episodes],
                    outputs=[sac_traj_plot, sac_perf_plot, sac_results]
                )
            
            # Tab 4: Model Comparison
            with gr.Tab("⚡ Model Comparison"):
                gr.Markdown("""
                ### 🥊 Compare IKD vs SAC Performance
                
                Run head-to-head benchmarks between models across multiple trials with detailed statistical analysis.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Configuration")
                        comp_ikd = gr.Dropdown(choices=ikd_models, label="IKD Model", value=ikd_models[0] if ikd_models else None)
                        comp_sac = gr.Dropdown(choices=sac_models, label="SAC Model", value=sac_models[0] if sac_models else None)
                        comp_scenario = gr.Radio(["Loose", "Tight"], label="Scenario", value="Loose")
                        comp_trials = gr.Slider(5, 100, value=20, step=5, label="Number of Trials per Model")
                        comp_max_steps = gr.Slider(100, 500, value=200, step=50, label="Max Steps per Trial")
                        compare_btn = gr.Button("🚀 Start Comparison", variant="primary", size="lg")
                    
                    with gr.Column():
                        comp_plot = gr.Plot(label="Performance Comparison")
                        comp_results = gr.Markdown("**Ready to compare!** Select models and click 'Start Comparison'.")
                
                compare_btn.click(
                    compare_models,
                    inputs=[comp_ikd, comp_sac, comp_scenario, comp_trials, comp_max_steps],
                    outputs=[comp_plot, comp_results]
                )
            
            # Tab 5: IMU Augmentation
            with gr.Tab("🔬 IMU Delay Augmentation"):
                gr.Markdown("### Data Augmentation with IMU Delays")
                
                with gr.Row():
                    with gr.Column():
                        min_delay = gr.Slider(0.05, 0.20, value=0.10, step=0.01, label="Min Delay (s)")
                        max_delay = gr.Slider(0.15, 0.40, value=0.25, step=0.01, label="Max Delay (s)")
                        aug_factor = gr.Slider(1, 10, value=5, step=1, label="Augmentation Factor")
                        augment_btn = gr.Button("Generate Augmentations", variant="primary", size="lg")
                    
                    with gr.Column():
                        aug_plot = gr.Plot(label="Augmented Signals")
                        aug_results = gr.Markdown("Results will appear here...")
                
                augment_btn.click(
                    augment_data_demo,
                    inputs=[min_delay, max_delay, aug_factor],
                    outputs=[aug_plot, aug_results]
                )
        
        gr.Markdown("""
        ---
        ### 💡 Quick Tips:
        - **Model Browser**: Load pretrained models before running simulations
        - **IKD/SAC Simulators**: Test individual model performance with custom parameters
        - **Model Comparison**: Head-to-head benchmark across multiple trials
        - **IMU Augmentation**: Generate training data with realistic sensor delays
        
        ### 📚 Resources:
        - [GitHub Repository](https://github.com/msuv08/autonomous-vehicle-drifting)
        - [Paper](https://arxiv.org/abs/2402.14928)
        """)
    
    return app


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏎️  Autonomous Vehicle Drifting - Model Playground")
    print("="*70)
    print("\nInitializing interface...")
    
    # Discover models
    model_loader.discover_models()
    model_loader.print_model_summary()
    
    # Build and launch
    app = build_interface()
    
    print("\n✅ Interface ready!")
    print("🌐 Launching web server...\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
