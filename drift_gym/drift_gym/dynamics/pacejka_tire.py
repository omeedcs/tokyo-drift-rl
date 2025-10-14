"""
Pacejka Magic Formula Tire Model

Implements the simplified Pacejka tire model for realistic tire forces during drifting.
"""

import numpy as np
from typing import Tuple


class PacejkaTireModel:
    """
    Pacejka Magic Formula for tire force calculation.
    
    The Magic Formula provides realistic tire forces as a function of slip ratio
    and slip angle, capturing the nonlinear behavior critical for drifting.
    
    Magic Formula: F = D * sin(C * arctan(B * slip - E * (B * slip - arctan(B * slip))))
    
    Where:
    - B: Stiffness factor
    - C: Shape factor  
    - D: Peak factor (peak force)
    - E: Curvature factor
    """
    
    def __init__(
        self,
        B_x: float = 10.0,
        C_x: float = 1.9,
        D_x: float = 1.0,
        E_x: float = 0.97,
        B_y: float = 8.0,
        C_y: float = 1.3,
        D_y: float = 1.0,
        E_y: float = -1.6,
        mu_peak: float = 1.0,
        mu_slide: float = 0.7,
    ):
        """
        Initialize Pacejka tire model.
        
        Args:
            B_x, C_x, D_x, E_x: Longitudinal force coefficients
            B_y, C_y, D_y, E_y: Lateral force coefficients
            mu_peak: Peak friction coefficient
            mu_slide: Sliding friction coefficient
        """
        # Longitudinal (forward/backward) coefficients
        self.B_x = B_x
        self.C_x = C_x
        self.D_x = D_x
        self.E_x = E_x
        
        # Lateral (sideways) coefficients
        self.B_y = B_y
        self.C_y = C_y
        self.D_y = D_y
        self.E_y = E_y
        
        # Friction coefficients
        self.mu_peak = mu_peak
        self.mu_slide = mu_slide
    
    def magic_formula(
        self,
        slip: float,
        B: float,
        C: float,
        D: float,
        E: float
    ) -> float:
        """
        Compute Magic Formula.
        
        Args:
            slip: Slip ratio or slip angle
            B, C, D, E: Pacejka coefficients
            
        Returns:
            Force from tire
        """
        # Prevent numerical issues
        slip = np.clip(slip, -0.95, 0.95)
        
        # Magic Formula
        term = B * slip
        arctan_term = np.arctan(term)
        F = D * np.sin(C * np.arctan(term - E * (term - arctan_term)))
        
        return F
    
    def longitudinal_force(
        self,
        slip_ratio: float,
        normal_force: float
    ) -> float:
        """
        Calculate longitudinal tire force (forward/backward).
        
        Args:
            slip_ratio: (v_wheel - v_vehicle) / max(v_vehicle, 0.1)
            normal_force: Normal force on tire (N)
            
        Returns:
            Longitudinal force (N), positive = forward
        """
        # Scale peak force by normal load and friction
        D_x_scaled = self.D_x * self.mu_peak * normal_force
        
        # Compute force using Magic Formula
        F_x = self.magic_formula(
            slip_ratio,
            self.B_x,
            self.C_x,
            D_x_scaled,
            self.E_x
        )
        
        return F_x
    
    def lateral_force(
        self,
        slip_angle: float,
        normal_force: float
    ) -> float:
        """
        Calculate lateral tire force (sideways).
        
        Args:
            slip_angle: Angle between wheel direction and velocity (rad)
            normal_force: Normal force on tire (N)
            
        Returns:
            Lateral force (N), positive = left
        """
        # Scale peak force by normal load and friction
        D_y_scaled = self.D_y * self.mu_peak * normal_force
        
        # Compute force using Magic Formula
        F_y = self.magic_formula(
            slip_angle,
            self.B_y,
            self.C_y,
            D_y_scaled,
            self.E_y
        )
        
        return F_y
    
    def combined_forces(
        self,
        slip_ratio: float,
        slip_angle: float,
        normal_force: float,
        velocity: float
    ) -> Tuple[float, float]:
        """
        Calculate combined longitudinal and lateral forces.
        
        During combined slip (both longitudinal and lateral), the tire cannot
        produce full force in both directions simultaneously (friction circle).
        
        Args:
            slip_ratio: Longitudinal slip
            slip_angle: Lateral slip (rad)
            normal_force: Normal force (N)
            velocity: Vehicle velocity (m/s)
            
        Returns:
            Tuple of (F_x, F_y) forces in tire frame (N)
        """
        # Calculate individual forces
        F_x_pure = self.longitudinal_force(slip_ratio, normal_force)
        F_y_pure = self.lateral_force(slip_angle, normal_force)
        
        # Combined slip reduction (simplified friction ellipse)
        # When both slips are active, reduce forces
        combined_slip = np.sqrt(slip_ratio**2 + slip_angle**2)
        
        if combined_slip > 0.01:
            # Friction circle approximation
            max_force = self.mu_peak * normal_force
            combined_force = np.sqrt(F_x_pure**2 + F_y_pure**2)
            
            if combined_force > max_force:
                # Scale down to friction limit
                scale = max_force / combined_force
                F_x = F_x_pure * scale
                F_y = F_y_pure * scale
            else:
                F_x = F_x_pure
                F_y = F_y_pure
        else:
            F_x = F_x_pure
            F_y = F_y_pure
        
        # Velocity-dependent damping (at low speeds, less force)
        speed_factor = np.tanh(velocity / 0.5)  # Smooth transition
        F_x *= speed_factor
        F_y *= speed_factor
        
        return F_x, F_y
    
    def get_slip_angle(
        self,
        v_x: float,
        v_y: float,
        wheel_angle: float = 0.0
    ) -> float:
        """
        Calculate tire slip angle from velocity components.
        
        Args:
            v_x: Longitudinal velocity in wheel frame (m/s)
            v_y: Lateral velocity in wheel frame (m/s)
            wheel_angle: Steering angle (rad)
            
        Returns:
            Slip angle (rad)
        """
        # Slip angle is angle between wheel direction and velocity
        if abs(v_x) < 0.1:
            return 0.0
        
        velocity_angle = np.arctan2(v_y, v_x)
        slip_angle = velocity_angle - wheel_angle
        
        # Normalize to [-pi, pi]
        slip_angle = np.arctan2(np.sin(slip_angle), np.cos(slip_angle))
        
        return slip_angle
    
    def get_slip_ratio(
        self,
        wheel_speed: float,
        vehicle_speed: float
    ) -> float:
        """
        Calculate longitudinal slip ratio.
        
        Args:
            wheel_speed: Wheel tangential velocity (m/s)
            vehicle_speed: Vehicle longitudinal velocity (m/s)
            
        Returns:
            Slip ratio (dimensionless)
        """
        if abs(vehicle_speed) < 0.1:
            return 0.0
        
        slip_ratio = (wheel_speed - vehicle_speed) / max(abs(vehicle_speed), 0.1)
        
        return np.clip(slip_ratio, -1.0, 1.0)


def test_pacejka_model():
    """Test Pacejka tire model with typical values."""
    tire = PacejkaTireModel()
    
    print("Testing Pacejka Tire Model")
    print("=" * 50)
    
    # Test lateral force vs slip angle
    print("\nLateral Force vs Slip Angle (Normal Force = 10N):")
    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        F_y = tire.lateral_force(alpha, normal_force=10.0)
        print(f"  Slip Angle: {np.degrees(alpha):5.1f}° → F_y: {F_y:6.2f} N")
    
    # Test combined slip
    print("\nCombined Slip (slip_ratio=0.1, slip_angle=0.1, v=3.0 m/s):")
    F_x, F_y = tire.combined_forces(0.1, 0.1, normal_force=10.0, velocity=3.0)
    print(f"  F_x: {F_x:6.2f} N")
    print(f"  F_y: {F_y:6.2f} N")
    print(f"  |F|: {np.sqrt(F_x**2 + F_y**2):6.2f} N")
    

if __name__ == "__main__":
    test_pacejka_model()
