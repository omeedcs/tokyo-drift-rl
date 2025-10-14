"""
Real-time Plotting System for Training and Evaluation.

Provides live visualization of metrics during training and evaluation using
matplotlib animation and plotly for interactive plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import threading
import time


class RealtimePlotter:
    """
    Real-time plotter for training metrics.
    
    Features:
    - Live updating plots
    - Multiple metrics support
    - Rolling window display
    - Thread-safe updates
    - Save snapshots
    
    Example:
        plotter = RealtimePlotter(metrics=['reward', 'loss'])
        plotter.start()
        
        for step in range(1000):
            plotter.update('reward', step, reward_value)
            plotter.update('loss', step, loss_value)
        
        plotter.stop()
    """
    
    def __init__(
        self,
        metrics: List[str],
        window_size: int = 1000,
        update_interval: int = 50,
        figsize: Tuple[int, int] = (12, 8),
        style: str = 'dark_background'
    ):
        """
        Initialize real-time plotter.
        
        Args:
            metrics: List of metric names to plot
            window_size: Number of points to display in rolling window
            update_interval: Update interval in milliseconds
            figsize: Figure size
            style: Matplotlib style
        """
        self.metrics = metrics
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Data storage (thread-safe with locks)
        self.data_lock = threading.Lock()
        self.data = {metric: {'x': deque(maxlen=window_size), 'y': deque(maxlen=window_size)} 
                     for metric in metrics}
        
        # Setup matplotlib
        plt.style.use(style)
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.setup_subplots()
        
        # Animation
        self.anim = None
        self.running = False
    
    def setup_subplots(self):
        """Setup subplot grid."""
        n_metrics = len(self.metrics)
        
        if n_metrics == 1:
            self.axes = [self.fig.add_subplot(111)]
        elif n_metrics == 2:
            self.axes = [
                self.fig.add_subplot(211),
                self.fig.add_subplot(212)
            ]
        elif n_metrics <= 4:
            self.axes = [
                self.fig.add_subplot(221),
                self.fig.add_subplot(222),
                self.fig.add_subplot(223),
                self.fig.add_subplot(224)
            ][:n_metrics]
        else:
            # Create grid for more metrics
            rows = int(np.ceil(n_metrics / 3))
            cols = min(3, n_metrics)
            self.axes = [self.fig.add_subplot(rows, cols, i+1) for i in range(n_metrics)]
        
        # Configure axes
        for ax, metric in zip(self.axes, self.metrics):
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, pad=10)
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        self.lines = [ax.plot([], [], lw=2, alpha=0.8)[0] for ax in self.axes]
        
    def update(self, metric: str, x: float, y: float):
        """
        Update metric value (thread-safe).
        
        Args:
            metric: Metric name
            x: X value (typically step/episode number)
            y: Y value (metric value)
        """
        if metric not in self.data:
            return
        
        with self.data_lock:
            self.data[metric]['x'].append(x)
            self.data[metric]['y'].append(y)
    
    def _animate(self, frame):
        """Animation update function."""
        with self.data_lock:
            for i, metric in enumerate(self.metrics):
                if len(self.data[metric]['x']) > 0:
                    x_data = list(self.data[metric]['x'])
                    y_data = list(self.data[metric]['y'])
                    
                    self.lines[i].set_data(x_data, y_data)
                    
                    # Auto-scale axes
                    if len(x_data) > 1:
                        self.axes[i].set_xlim(min(x_data), max(x_data))
                        y_min, y_max = min(y_data), max(y_data)
                        y_range = y_max - y_min if y_max != y_min else 1
                        self.axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        return self.lines
    
    def start(self):
        """Start real-time plotting."""
        self.running = True
        self.anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=self.update_interval,
            blit=True,
            cache_frame_data=False
        )
        plt.show(block=False)
        print(f"✅ Real-time plotter started for: {', '.join(self.metrics)}")
    
    def stop(self):
        """Stop real-time plotting."""
        self.running = False
        if self.anim:
            self.anim.event_source.stop()
        print("🛑 Real-time plotter stopped")
    
    def save_snapshot(self, filepath: str):
        """Save current plot state."""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot to: {filepath}")
    
    def clear(self):
        """Clear all data."""
        with self.data_lock:
            for metric in self.metrics:
                self.data[metric]['x'].clear()
                self.data[metric]['y'].clear()


class InteractivePlotter:
    """
    Interactive plotter using Plotly for web-based visualization.
    
    Features:
    - Interactive zooming/panning
    - Hover tooltips
    - Export to HTML
    - Multiple traces per subplot
    
    Example:
        plotter = InteractivePlotter()
        plotter.add_trace('reward', x_data, y_data, subplot=1)
        plotter.add_trace('loss', x_data, y_data, subplot=2)
        plotter.show()
    """
    
    def __init__(
        self,
        n_rows: int = 2,
        n_cols: int = 1,
        subplot_titles: Optional[List[str]] = None,
        height: int = 800
    ):
        """
        Initialize interactive plotter.
        
        Args:
            n_rows: Number of subplot rows
            n_cols: Number of subplot columns
            subplot_titles: Titles for subplots
            height: Figure height in pixels
        """
        self.fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        self.fig.update_layout(
            height=height,
            template='plotly_dark',
            showlegend=True,
            hovermode='x unified'
        )
        
        self.n_rows = n_rows
        self.n_cols = n_cols
    
    def add_trace(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        row: int = 1,
        col: int = 1,
        mode: str = 'lines',
        color: Optional[str] = None,
        **kwargs
    ):
        """
        Add a trace to subplot.
        
        Args:
            name: Trace name
            x: X data
            y: Y data
            row: Subplot row
            col: Subplot column
            mode: Plot mode ('lines', 'markers', 'lines+markers')
            color: Line color
            **kwargs: Additional plotly trace arguments
        """
        trace_kwargs = dict(
            x=x,
            y=y,
            name=name,
            mode=mode,
            **kwargs
        )
        
        if color:
            trace_kwargs['line'] = dict(color=color)
        
        self.fig.add_trace(
            go.Scatter(**trace_kwargs),
            row=row,
            col=col
        )
    
    def add_scatter(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        row: int = 1,
        col: int = 1,
        **kwargs
    ):
        """Add scatter plot."""
        self.add_trace(name, x, y, row, col, mode='markers', **kwargs)
    
    def update_layout(self, **kwargs):
        """Update figure layout."""
        self.fig.update_layout(**kwargs)
    
    def update_xaxis(self, title: str, row: int = 1, col: int = 1):
        """Update x-axis title."""
        self.fig.update_xaxes(title_text=title, row=row, col=col)
    
    def update_yaxis(self, title: str, row: int = 1, col: int = 1):
        """Update y-axis title."""
        self.fig.update_yaxes(title_text=title, row=row, col=col)
    
    def show(self):
        """Show interactive plot."""
        self.fig.show()
    
    def save_html(self, filepath: str):
        """Save to HTML file."""
        self.fig.write_html(filepath)
        print(f"💾 Saved interactive plot to: {filepath}")
    
    def save_image(self, filepath: str, width: int = 1200, height: int = 800):
        """Save as static image (requires kaleido)."""
        try:
            self.fig.write_image(filepath, width=width, height=height)
            print(f"💾 Saved plot image to: {filepath}")
        except Exception as e:
            print(f"⚠️  Failed to save image: {e}")
            print("Install kaleido for image export: pip install kaleido")


class TrajectoryPlotter:
    """
    Real-time trajectory visualization for vehicle path tracking.
    
    Features:
    - Live vehicle trajectory
    - Goal and obstacle visualization
    - Path comparison
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 10),
        xlim: Tuple[float, float] = (-2, 5),
        ylim: Tuple[float, float] = (-2, 3)
    ):
        """
        Initialize trajectory plotter.
        
        Args:
            figsize: Figure size
            xlim: X-axis limits
            ylim: Y-axis limits
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title('Vehicle Trajectory', fontsize=14)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Trajectory line
        self.trajectory_line, = self.ax.plot([], [], 'b-', lw=2, alpha=0.7, label='Trajectory')
        
        # Vehicle marker
        self.vehicle_marker, = self.ax.plot([], [], 'ro', markersize=10, label='Vehicle')
        
        # Goal gate
        self.gate_line = None
        
        # Obstacles
        self.obstacle_patches = []
        
        self.ax.legend(loc='upper right')
        
        # Data
        self.trajectory = []
    
    def set_goal(self, x: float, y: float, width: float):
        """Set goal gate position."""
        y1 = y - width / 2
        y2 = y + width / 2
        
        if self.gate_line:
            self.gate_line.remove()
        
        self.gate_line, = self.ax.plot([x, x], [y1, y2], 'g-', lw=4, alpha=0.8, label='Goal')
        self.ax.plot([x], [y1], 'go', markersize=8)
        self.ax.plot([x], [y2], 'go', markersize=8)
    
    def add_obstacle(self, x: float, y: float, radius: float):
        """Add obstacle to plot."""
        circle = plt.Circle((x, y), radius, color='red', alpha=0.3)
        self.ax.add_patch(circle)
        self.obstacle_patches.append(circle)
    
    def update_trajectory(self, x: float, y: float):
        """Update trajectory with new point."""
        self.trajectory.append((x, y))
        
        if len(self.trajectory) > 1:
            xs, ys = zip(*self.trajectory)
            self.trajectory_line.set_data(xs, ys)
        
        # Update vehicle position
        self.vehicle_marker.set_data([x], [y])
        
        plt.pause(0.01)
    
    def clear(self):
        """Clear trajectory."""
        self.trajectory = []
        self.trajectory_line.set_data([], [])
        self.vehicle_marker.set_data([], [])
    
    def show(self):
        """Show plot."""
        plt.show()
    
    def save(self, filepath: str):
        """Save current plot."""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Saved trajectory plot to: {filepath}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot training history with interactive plotly.
    
    Args:
        history: Dictionary of metric_name -> values
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
        
    Example:
        history = {
            'reward': [1, 2, 3, 4],
            'loss': [10, 8, 6, 4],
            'success_rate': [0.2, 0.4, 0.6, 0.8]
        }
        fig = plot_training_history(history)
    """
    n_metrics = len(history)
    
    plotter = InteractivePlotter(
        n_rows=n_metrics,
        n_cols=1,
        subplot_titles=list(history.keys()),
        height=300 * n_metrics
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (metric, values) in enumerate(history.items()):
        x = np.arange(len(values))
        plotter.add_trace(
            name=metric,
            x=x,
            y=np.array(values),
            row=i+1,
            col=1,
            color=colors[i % len(colors)]
        )
        plotter.update_xaxis('Step', row=i+1, col=1)
        plotter.update_yaxis(metric.replace('_', ' ').title(), row=i+1, col=1)
    
    plotter.update_layout(title_text='Training History')
    
    if save_path:
        plotter.save_html(save_path)
    
    return plotter.fig


if __name__ == "__main__":
    # Demo: Real-time plotter
    print("Testing real-time plotter...")
    
    plotter = RealtimePlotter(
        metrics=['reward', 'loss', 'success_rate'],
        window_size=500
    )
    
    plotter.start()
    
    # Simulate training
    for step in range(1000):
        reward = np.sin(step / 50) * 10 + np.random.randn() * 2
        loss = 100 / (step + 1) + np.random.randn() * 0.5
        success = min(1.0, step / 800 + np.random.randn() * 0.1)
        
        plotter.update('reward', step, reward)
        plotter.update('loss', step, loss)
        plotter.update('success_rate', step, success)
        
        time.sleep(0.01)
    
    plotter.stop()
    plotter.save_snapshot('realtime_plot_demo.png')
    
    plt.show()
