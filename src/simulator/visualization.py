"""
Visualization tools for simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Dict
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend


def plot_trajectory(
    trajectory: np.ndarray,
    obstacles: List = None,
    title: str = "Vehicle Trajectory",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot vehicle trajectory with obstacles.
    
    Args:
        trajectory: Nx2 array of (x, y) positions
        obstacles: List of Obstacle objects
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectory
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    
    # Plot obstacles
    if obstacles:
        for obs in obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
            ax.add_patch(circle)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_simulation_data(
    data: Dict,
    title: str = "Simulation Results",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot comprehensive simulation results.
    
    Args:
        data: Dictionary of simulation data
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    time = np.array(data['time'])
    
    # Velocities
    axes[0, 0].plot(time, data['commanded_velocity'], label='Commanded', linewidth=2)
    axes[0, 0].plot(time, data['measured_velocity'], label='Measured', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Linear Velocity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Angular velocities
    axes[0, 1].plot(time, data['commanded_angular_velocity'], label='Commanded', linewidth=2)
    axes[0, 1].plot(time, data['measured_angular_velocity'], label='Measured (IMU)', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trajectory (XY plot)
    axes[1, 0].plot(data['x'], data['y'], 'b-', linewidth=2)
    axes[1, 0].plot(data['x'][0], data['y'][0], 'go', markersize=10, label='Start')
    axes[1, 0].plot(data['x'][-1], data['y'][-1], 'ro', markersize=10, label='End')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].set_title('Trajectory')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Heading
    axes[1, 1].plot(time, np.rad2deg(data['theta']), 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Heading (degrees)')
    axes[1, 1].set_title('Heading Angle')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Curvature
    axes[2, 0].plot(time, data['curvature'], 'b-', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Curvature (1/m)')
    axes[2, 0].set_title('Path Curvature')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Angular velocity error
    av_error = np.array(data['commanded_angular_velocity']) - np.array(data['measured_angular_velocity'])
    axes[2, 1].plot(time, av_error, 'r-', linewidth=2)
    axes[2, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('AV Error (rad/s)')
    axes[2, 1].set_title('Angular Velocity Error (Cmd - Measured)')
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved simulation plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compare_baseline_vs_ikd(
    baseline_data: Dict,
    ikd_data: Dict,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Compare baseline and IKD-corrected trajectories.
    
    Args:
        baseline_data: Simulation data without IKD
        ikd_data: Simulation data with IKD correction
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trajectories
    axes[0, 0].plot(baseline_data['x'], baseline_data['y'], 'r-', linewidth=2, label='Baseline', alpha=0.7)
    axes[0, 0].plot(ikd_data['x'], ikd_data['y'], 'b-', linewidth=2, label='IKD Corrected')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Angular velocities
    time_baseline = np.array(baseline_data['time'])
    time_ikd = np.array(ikd_data['time'])
    
    axes[0, 1].plot(time_baseline, baseline_data['measured_angular_velocity'], 
                    'r-', linewidth=2, label='Baseline', alpha=0.7)
    axes[0, 1].plot(time_ikd, ikd_data['measured_angular_velocity'], 
                    'b-', linewidth=2, label='IKD Corrected')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocity Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Curvatures
    axes[1, 0].plot(time_baseline, baseline_data['curvature'], 
                    'r-', linewidth=2, label='Baseline', alpha=0.7)
    axes[1, 0].plot(time_ikd, ikd_data['curvature'], 
                    'b-', linewidth=2, label='IKD Corrected')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Curvature (1/m)')
    axes[1, 0].set_title('Curvature Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error comparison
    baseline_error = np.array(baseline_data['commanded_angular_velocity']) - \
                    np.array(baseline_data['measured_angular_velocity'])
    ikd_error = np.array(ikd_data['commanded_angular_velocity']) - \
               np.array(ikd_data['measured_angular_velocity'])
    
    axes[1, 1].plot(time_baseline, np.abs(baseline_error), 
                    'r-', linewidth=2, label=f'Baseline (MAE={np.mean(np.abs(baseline_error)):.4f})', alpha=0.7)
    axes[1, 1].plot(time_ikd, np.abs(ikd_error), 
                    'b-', linewidth=2, label=f'IKD (MAE={np.mean(np.abs(ikd_error)):.4f})')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('|AV Error| (rad/s)')
    axes[1, 1].set_title('Angular Velocity Error Magnitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Baseline vs IKD-Corrected Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_animation(
    data: Dict,
    obstacles: List = None,
    save_path: Optional[str] = None,
    fps: int = 20
):
    """
    Create animation of vehicle trajectory.
    
    Args:
        data: Simulation data dictionary
        obstacles: List of obstacles
        save_path: Path to save animation (MP4 or GIF)
        fps: Frames per second
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot obstacles
    if obstacles:
        for obs in obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
            ax.add_patch(circle)
    
    # Initialize trajectory line and vehicle marker
    line, = ax.plot([], [], 'b-', linewidth=2, label='Trajectory')
    vehicle, = ax.plot([], [], 'go', markersize=15, label='Vehicle')
    
    # Set plot limits
    x_data = np.array(data['x'])
    y_data = np.array(data['y'])
    margin = 1.0
    ax.set_xlim(x_data.min() - margin, x_data.max() + margin)
    ax.set_ylim(y_data.min() - margin, y_data.max() + margin)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Vehicle Simulation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    def init():
        line.set_data([], [])
        vehicle.set_data([], [])
        return line, vehicle
    
    def animate(frame):
        line.set_data(x_data[:frame], y_data[:frame])
        vehicle.set_data([x_data[frame]], [y_data[frame]])
        return line, vehicle
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(x_data), interval=1000/fps,
        blit=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Saved animation to {save_path}")
    
    plt.close()
    return anim
