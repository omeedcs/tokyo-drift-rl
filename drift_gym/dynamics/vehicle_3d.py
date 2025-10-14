"""
3D Vehicle Dynamics with Pitch, Roll, and Weight Transfer

Extends the 2D bicycle model to include vertical dynamics critical for drifting.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Vehicle3DState:
    """Complete 3D vehicle state."""
    # Position (meters)
    x: float
    y: float
    z: float  # Height above ground
    
    # Orientation (radians)
    roll: float  # φ (phi)
    pitch: float  # θ (theta)
    yaw: float  # ψ (psi)
    
    # Linear velocity (m/s)
    vx: float  # Forward
    vy: float  # Lateral
    vz: float  # Vertical
    
    # Angular velocity (rad/s)
    wx: float  # Roll rate
    wy: float  # Pitch rate
    wz: float  # Yaw rate


class Vehicle3DDynamics:
    """
    3D vehicle dynamics with weight transfer.
    
    Implements:
    - Load transfer during acceleration/braking
    - Roll dynamics during cornering
    - Pitch dynamics
    - Individual wheel loads
    - Tire force coupling with normal load
    """
    
    def __init__(
        self,
        mass: float = 1.5,  # kg
        wheelbase: float = 0.25,  # meters
        track_width: float = 0.19,  # meters
        cog_height: float = 0.05,  # meters (center of gravity height)
        l_f: float = 0.125,  # Front axle to CoG
        l_r: float = 0.125,  # Rear axle to CoG
        roll_stiffness: float = 100.0,  # N*m/rad
        roll_damping: float = 5.0,  # N*m*s/rad
        pitch_stiffness: float = 150.0,  # N*m/rad
        pitch_damping: float = 8.0,  # N*m*s/rad
        gravity: float = 9.81  # m/s^2
    ):
        self.mass = mass
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.cog_height = cog_height
        self.l_f = l_f
        self.l_r = l_r
        self.roll_stiffness = roll_stiffness
        self.roll_damping = roll_damping
        self.pitch_stiffness = pitch_stiffness
        self.pitch_damping = pitch_damping
        self.gravity = gravity
        
        # Moments of inertia (approximations for F1/10 vehicle)
        self.I_xx = mass * (track_width**2 + cog_height**2) / 12  # Roll
        self.I_yy = mass * (wheelbase**2 + cog_height**2) / 12  # Pitch
        self.I_zz = mass * (wheelbase**2 + track_width**2) / 12  # Yaw
        
    def compute_wheel_loads(
        self,
        state: Vehicle3DState,
        ax: float,  # Longitudinal acceleration
        ay: float   # Lateral acceleration
    ) -> Tuple[float, float, float, float]:
        """
        Compute individual wheel normal loads with weight transfer.
        
        Args:
            state: Current vehicle state
            ax: Longitudinal acceleration (m/s^2)
            ay: Lateral acceleration (m/s^2)
            
        Returns:
            Tuple of (FL, FR, RL, RR) wheel loads in Newtons
        """
        # Static load distribution
        static_front = self.mass * self.gravity * (self.l_r / self.wheelbase)
        static_rear = self.mass * self.gravity * (self.l_f / self.wheelbase)
        
        # Longitudinal load transfer (pitch)
        # During braking: weight shifts forward
        # During acceleration: weight shifts backward
        dFz_long = self.mass * ax * (self.cog_height / self.wheelbase)
        
        # Lateral load transfer (roll)
        # During left turn: weight shifts to right wheels
        # During right turn: weight shifts to left wheels
        dFz_lat = self.mass * ay * (self.cog_height / self.track_width)
        
        # Front wheels
        F_front_total = static_front - dFz_long
        FL = F_front_total / 2 - dFz_lat
        FR = F_front_total / 2 + dFz_lat
        
        # Rear wheels
        F_rear_total = static_rear + dFz_long
        RL = F_rear_total / 2 - dFz_lat
        RR = F_rear_total / 2 + dFz_lat
        
        # Ensure non-negative (wheels can't pull on ground)
        FL = max(0.0, FL)
        FR = max(0.0, FR)
        RL = max(0.0, RL)
        RR = max(0.0, RR)
        
        return FL, FR, RL, RR
    
    def compute_roll_pitch(
        self,
        state: Vehicle3DState,
        ay: float,  # Lateral acceleration
        ax: float,  # Longitudinal acceleration
        dt: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute roll and pitch angles and rates.
        
        Args:
            state: Current state
            ay: Lateral acceleration
            ax: Longitudinal acceleration
            dt: Time step
            
        Returns:
            Tuple of (roll, roll_rate, pitch, pitch_rate)
        """
        # Roll dynamics (due to lateral acceleration)
        # Moment = m * ay * h
        roll_moment = self.mass * ay * self.cog_height
        
        # Spring and damper forces
        roll_restoring = -self.roll_stiffness * state.roll
        roll_damping_force = -self.roll_damping * state.wx
        
        # Roll acceleration
        roll_accel = (roll_moment + roll_restoring + roll_damping_force) / self.I_xx
        
        # Integrate
        roll_rate = state.wx + roll_accel * dt
        roll = state.roll + roll_rate * dt
        
        # Limit roll angle (physical limits)
        roll = np.clip(roll, -0.3, 0.3)  # ±17 degrees
        
        # Pitch dynamics (due to longitudinal acceleration)
        pitch_moment = self.mass * ax * self.cog_height
        
        # Spring and damper
        pitch_restoring = -self.pitch_stiffness * state.pitch
        pitch_damping_force = -self.pitch_damping * state.wy
        
        # Pitch acceleration
        pitch_accel = (pitch_moment + pitch_restoring + pitch_damping_force) / self.I_yy
        
        # Integrate
        pitch_rate = state.wy + pitch_accel * dt
        pitch = state.pitch + pitch_rate * dt
        
        # Limit pitch angle
        pitch = np.clip(pitch, -0.2, 0.2)  # ±11 degrees
        
        return roll, roll_rate, pitch, pitch_rate
    
    def step(
        self,
        state: Vehicle3DState,
        steering_angle: float,
        throttle: float,
        dt: float
    ) -> Vehicle3DState:
        """
        Integrate vehicle dynamics for one timestep.
        
        Args:
            state: Current state
            steering_angle: Steering input (radians)
            throttle: Throttle [-1, 1]
            dt: Time step (seconds)
            
        Returns:
            New Vehicle3DState
        """
        # Simplified force model (you'd use Pacejka here in full impl)
        # For now, use kinematic model with 3D extensions
        
        # Compute slip angle
        if abs(state.vx) > 0.1:
            beta = np.arctan2(state.vy, state.vx)
        else:
            beta = 0.0
        
        # Accelerations in body frame
        ax = throttle * 3.0  # Simplified
        ay = state.vx * state.wz  # Centripetal acceleration
        
        # Compute roll and pitch
        roll, roll_rate, pitch, pitch_rate = self.compute_roll_pitch(
            state, ay, ax, dt
        )
        
        # Compute wheel loads
        FL, FR, RL, RR = self.compute_wheel_loads(state, ax, ay)
        
        # Total normal force affects tire performance
        # (In full implementation, this would affect Pacejka coefficients)
        total_load = FL + FR + RL + RR
        load_factor = total_load / (self.mass * self.gravity)
        
        # Update velocities (simplified)
        vx_new = state.vx + ax * dt
        yaw_rate = (state.vx / (self.l_f + self.l_r)) * np.tan(steering_angle)
        
        # Limit yaw rate based on roll angle (stability)
        max_yaw_rate = 3.0 * (1.0 - abs(roll) / 0.3)
        yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)
        
        yaw_new = state.yaw + yaw_rate * dt
        
        # Update position
        x_new = state.x + state.vx * np.cos(state.yaw) * dt
        y_new = state.y + state.vx * np.sin(state.yaw) * dt
        
        # Z position (vertical) - simple suspension model
        # CoG height changes with roll and pitch
        z_new = self.cog_height * (1.0 + abs(roll) * 0.1 + abs(pitch) * 0.05)
        
        new_state = Vehicle3DState(
            x=x_new,
            y=y_new,
            z=z_new,
            roll=roll,
            pitch=pitch,
            yaw=yaw_new,
            vx=vx_new,
            vy=state.vy,  # Simplified
            vz=0.0,
            wx=roll_rate,
            wy=pitch_rate,
            wz=yaw_rate
        )
        
        return new_state


def test_3d_dynamics():
    """Test 3D vehicle dynamics."""
    print("Testing 3D Vehicle Dynamics")
    print("=" * 60)
    
    dynamics = Vehicle3DDynamics()
    
    # Initial state
    state = Vehicle3DState(
        x=0.0, y=0.0, z=0.05,
        roll=0.0, pitch=0.0, yaw=0.0,
        vx=5.0, vy=0.0, vz=0.0,
        wx=0.0, wy=0.0, wz=0.0
    )
    
    print("\nSimulating hard left turn at 5 m/s:")
    print(f"{'Time':>6} {'Roll':>8} {'Pitch':>8} {'FL':>8} {'FR':>8} {'RL':>8} {'RR':>8}")
    print("-" * 60)
    
    for step in range(20):
        t = step * 0.05
        
        # Hard left turn
        steering = 0.4  # radians
        throttle = 0.2
        
        # Compute wheel loads
        ay = state.vx * state.wz
        ax = throttle * 3.0
        FL, FR, RL, RR = dynamics.compute_wheel_loads(state, ax, ay)
        
        print(f"{t:6.2f} {np.degrees(state.roll):8.2f}° {np.degrees(state.pitch):8.2f}° "
              f"{FL:8.2f} {FR:8.2f} {RL:8.2f} {RR:8.2f}")
        
        # Step dynamics
        state = dynamics.step(state, steering, throttle, dt=0.05)
    
    print(f"\nFinal state:")
    print(f"  Position: ({state.x:.2f}, {state.y:.2f})")
    print(f"  Roll: {np.degrees(state.roll):.2f}°")
    print(f"  Pitch: {np.degrees(state.pitch):.2f}°")
    print(f"  Velocity: {state.vx:.2f} m/s")


if __name__ == "__main__":
    test_3d_dynamics()
