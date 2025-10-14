"""
Visualization script for drifting data with cone obstacle.
Plots commanded vs. true angular velocities over time.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_drift_data(csv_path='./dataset/drift_with_cone.csv'):
    """
    Plot commanded and true angular velocities from drift data.
    
    Args:
        csv_path: Path to the CSV file containing drift data
    """
    # Read the CSV file
    data = pd.read_csv(csv_path)
    
    # Extract joystick and executed data
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    data_array = np.concatenate((joystick, executed), axis=1)
    
    # Parse data columns
    linear_velocity = data_array[:, 0]
    commanded_angular_velocity = data_array[:, 1]
    true_angular_velocity = data_array[:, 2]
    
    # Generate time array
    time_steps = range(len(joystick))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, commanded_angular_velocity, 
            label='Commanded Angular Velocity', linewidth=2)
    ax.plot(time_steps, true_angular_velocity, 
            label='True Angular Velocity', linewidth=2, alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax.set_title('Drift with Cone: Commanded vs. True Angular Velocity', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"[INFO] Plotted {len(joystick)} data points")
    print(f"[INFO] First true angular velocity value: {true_angular_velocity[0]:.4f} rad/s")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        plot_drift_data(csv_path)
    else:
        plot_drift_data()