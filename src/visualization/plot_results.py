"""
Visualization utilities for IKD model results and data analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, List, Tuple
from pathlib import Path

# Set style
matplotlib.use('Agg')  # Non-interactive backend
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


def plot_angular_velocity_comparison(
    commanded_av: np.ndarray,
    true_av: np.ndarray,
    predicted_av: np.ndarray,
    title: str = "Angular Velocity Comparison",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot comparison of commanded, true, and predicted angular velocities.
    
    Args:
        commanded_av: Commanded angular velocities
        true_av: Ground truth angular velocities from IMU
        predicted_av: Model predictions
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    time_steps = np.arange(len(commanded_av))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_steps, commanded_av, label='Commanded AV', alpha=0.7, linewidth=1.5)
    ax.plot(time_steps, true_av, label='True AV (IMU)', alpha=0.7, linewidth=1.5)
    ax.plot(time_steps, predicted_av, label='Predicted AV (IKD)', alpha=0.8, linewidth=1.5, linestyle='--')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training History",
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    epochs = np.arange(1, len(train_losses) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=3, linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=3, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add minimum markers
    min_train_idx = np.argmin(train_losses)
    min_val_idx = np.argmin(val_losses)
    ax.plot(epochs[min_train_idx], train_losses[min_train_idx], 'ro', markersize=8, label=f'Min Train: {train_losses[min_train_idx]:.6f}')
    ax.plot(epochs[min_val_idx], val_losses[min_val_idx], 'gs', markersize=8, label=f'Min Val: {val_losses[min_val_idx]:.6f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_curvature_distribution(
    velocities: np.ndarray,
    angular_velocities: np.ndarray,
    title: str = "Curvature Distribution",
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot histogram of curvature distribution.
    
    Args:
        velocities: Linear velocities
        angular_velocities: Angular velocities
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save figure
        show: Whether to display plot
    """
    # Calculate curvatures
    valid_mask = np.abs(velocities) > 1e-3
    curvatures = angular_velocities[valid_mask] / velocities[valid_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(curvatures, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Curvature (1/m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'Mean: {np.mean(curvatures):.3f}\nStd: {np.std(curvatures):.3f}\nMedian: {np.median(curvatures):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Prediction Error Distribution",
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plot distribution of prediction errors.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        title: Plot title
        bins: Number of histogram bins
        save_path: Path to save figure
        show: Whether to display plot
    """
    errors = predictions - targets
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error histogram
    ax1.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Prediction Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Histogram', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Error vs target scatter
    ax2.scatter(targets, errors, alpha=0.3, s=10)
    ax2.axhline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Target Value', fontsize=12)
    ax2.set_ylabel('Prediction Error', fontsize=12)
    ax2.set_title('Error vs Target', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_circle_comparison(
    commanded_curvature: float,
    baseline_radius: float,
    ikd_radius: float,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Visualize circle trajectory comparison for baseline vs IKD-corrected.
    
    Args:
        commanded_curvature: Commanded curvature value
        baseline_radius: Measured radius without IKD
        ikd_radius: Measured radius with IKD
        save_path: Path to save figure
        show: Whether to display plot
    """
    from src.evaluation.metrics import CircleMetrics
    
    # Calculate curvatures
    baseline_curv = CircleMetrics.compute_curvature_from_radius(baseline_radius)
    ikd_curv = CircleMetrics.compute_curvature_from_radius(ikd_radius)
    commanded_radius = CircleMetrics.compute_radius_from_curvature(commanded_curvature)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw circles
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Commanded circle
    x_cmd = commanded_radius * np.cos(theta)
    y_cmd = commanded_radius * np.sin(theta)
    ax.plot(x_cmd, y_cmd, 'g-', linewidth=2, label=f'Commanded (r={commanded_radius:.2f}m, c={commanded_curvature:.3f})')
    
    # Baseline circle
    x_base = baseline_radius * np.cos(theta)
    y_base = baseline_radius * np.sin(theta)
    ax.plot(x_base, y_base, 'r--', linewidth=2, label=f'Baseline (r={baseline_radius:.2f}m, c={baseline_curv:.3f})')
    
    # IKD circle
    x_ikd = ikd_radius * np.cos(theta)
    y_ikd = ikd_radius * np.sin(theta)
    ax.plot(x_ikd, y_ikd, 'b-.', linewidth=2, label=f'IKD Corrected (r={ikd_radius:.2f}m, c={ikd_curv:.3f})')
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Circle Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-max(commanded_radius, baseline_radius, ikd_radius) * 1.2, 
                max(commanded_radius, baseline_radius, ikd_radius) * 1.2)
    ax.set_ylim(-max(commanded_radius, baseline_radius, ikd_radius) * 1.2,
                max(commanded_radius, baseline_radius, ikd_radius) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
