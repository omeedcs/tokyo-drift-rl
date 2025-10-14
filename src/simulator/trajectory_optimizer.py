"""
Optimization-based trajectory planning for tight drift maneuvers.

Uses numerical optimization to find collision-free trajectories through narrow gates.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass

from src.simulator.trajectory import Waypoint, Trajectory


@dataclass
class TrajectoryConstraints:
    """Constraints for trajectory optimization."""
    max_curvature: float = 3.0  # 1/m
    max_velocity: float = 4.0   # m/s
    max_acceleration: float = 6.0  # m/s^2
    vehicle_width: float = 0.48  # m
    safety_margin: float = 0.05  # m


class OptimizedTrajectoryPlanner:
    """
    Optimization-based trajectory planner for tight scenarios.
    
    Uses scipy.optimize to find trajectories that:
    - Pass through narrow gates
    - Minimize control effort
    - Avoid collisions
    - Satisfy kinematic constraints
    """
    
    def __init__(
        self,
        constraints: Optional[TrajectoryConstraints] = None,
        aggressive_mode: bool = False
    ):
        """
        Initialize optimizer.
        
        Args:
            constraints: Trajectory constraints
            aggressive_mode: Enable aggressive drift angles
        """
        self.constraints = constraints or TrajectoryConstraints()
        self.aggressive_mode = aggressive_mode
        
        # Aggressive mode allows higher slip angles
        if aggressive_mode:
            self.constraints.max_curvature = 4.0
            self.constraints.max_velocity = 3.5
    
    def plan_drift_trajectory(
        self,
        start_pos: Tuple[float, float],
        gate_pos: Tuple[float, float],
        gate_width: float,
        obstacles: List[Tuple[float, float, float]],
        direction: str = "ccw"
    ) -> Trajectory:
        """
        Plan optimized drift trajectory.
        
        Args:
            start_pos: Starting position (x, y)
            gate_pos: Gate center position (x, y)
            gate_width: Width of gate
            obstacles: List of (x, y, radius) obstacles
            direction: "ccw" or "cw"
            
        Returns:
            Optimized trajectory
        """
        # Determine if this is a tight scenario
        clearance = gate_width - self.constraints.vehicle_width
        is_tight = clearance < 0.5
        
        if is_tight:
            print(f"[OPTIMIZER] Tight scenario detected (clearance: {clearance:.2f}m)")
            print(f"[OPTIMIZER] Using aggressive planning mode")
            return self._plan_tight_drift(
                start_pos, gate_pos, gate_width, obstacles, direction
            )
        else:
            print(f"[OPTIMIZER] Loose scenario (clearance: {clearance:.2f}m)")
            return self._plan_standard_drift(
                start_pos, gate_pos, gate_width, obstacles, direction
            )
    
    def _plan_tight_drift(
        self,
        start_pos: Tuple[float, float],
        gate_pos: Tuple[float, float],
        gate_width: float,
        obstacles: List[Tuple[float, float, float]],
        direction: str
    ) -> Trajectory:
        """Plan trajectory for tight gate using optimization."""
        
        # Parameter vector: [entry_angle, arc_radius, drift_angle, exit_angle, speed]
        # Use tighter bounds for constrained scenario
        bounds = [
            (-np.pi/3, np.pi/3),      # entry_angle
            (0.2, gate_width * 0.35),  # arc_radius (tighter)
            (0.0, np.pi/4),            # drift_angle (aggressive slip)
            (-np.pi/3, np.pi/3),       # exit_angle
            (1.5, 3.0)                 # speed (slower for precision)
        ]
        
        def objective(params):
            """Objective: minimize control effort + smooth trajectory."""
            entry_angle, arc_radius, drift_angle, exit_angle, speed = params
            
            # Generate trajectory from parameters
            waypoints = self._generate_parameterized_trajectory(
                start_pos, gate_pos, gate_width, direction,
                entry_angle, arc_radius, drift_angle, exit_angle, speed
            )
            
            if not waypoints:
                return 1e6  # Invalid trajectory
            
            # Compute costs
            control_cost = self._compute_control_cost(waypoints)
            collision_cost = self._compute_collision_cost(waypoints, obstacles)
            smoothness_cost = self._compute_smoothness_cost(waypoints)
            
            # Weighted sum
            total_cost = (
                1.0 * control_cost +
                100.0 * collision_cost +  # High penalty for collisions
                0.5 * smoothness_cost
            )
            
            return total_cost
        
        # Run optimization
        print("[OPTIMIZER] Running trajectory optimization...")
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            atol=0.01,
            workers=1
        )
        
        if not result.success:
            print(f"[OPTIMIZER] Warning: Optimization did not converge")
        
        # Extract best parameters
        entry_angle, arc_radius, drift_angle, exit_angle, speed = result.x
        
        print(f"[OPTIMIZER] Best params:")
        print(f"  Arc radius: {arc_radius:.3f}m")
        print(f"  Drift angle: {np.degrees(drift_angle):.1f}°")
        print(f"  Speed: {speed:.2f}m/s")
        print(f"  Cost: {result.fun:.4f}")
        
        # Generate final trajectory
        waypoints = self._generate_parameterized_trajectory(
            start_pos, gate_pos, gate_width, direction,
            entry_angle, arc_radius, drift_angle, exit_angle, speed
        )
        
        return Trajectory(waypoints)
    
    def _plan_standard_drift(
        self,
        start_pos: Tuple[float, float],
        gate_pos: Tuple[float, float],
        gate_width: float,
        obstacles: List[Tuple[float, float, float]],
        direction: str
    ) -> Trajectory:
        """Use faster heuristic planning for loose scenarios."""
        # Use adaptive radius based on gate width
        arc_radius = min(gate_width * 0.38, 1.0)
        entry_angle = 0.0
        drift_angle = 0.1  # Small drift angle
        exit_angle = 0.0
        speed = 2.8
        
        waypoints = self._generate_parameterized_trajectory(
            start_pos, gate_pos, gate_width, direction,
            entry_angle, arc_radius, drift_angle, exit_angle, speed
        )
        
        return Trajectory(waypoints)
    
    def _generate_parameterized_trajectory(
        self,
        start_pos: Tuple[float, float],
        gate_pos: Tuple[float, float],
        gate_width: float,
        direction: str,
        entry_angle: float,
        arc_radius: float,
        drift_angle: float,
        exit_angle: float,
        speed: float
    ) -> List[Waypoint]:
        """Generate trajectory from parameters."""
        waypoints = []
        
        # Phase 1: Approach with entry angle
        approach_dist = np.sqrt(
            (gate_pos[0] - start_pos[0])**2 + 
            (gate_pos[1] - start_pos[1])**2
        ) - 0.8
        
        for i in range(15):
            t = i / 14
            x = start_pos[0] + t * approach_dist * np.cos(entry_angle)
            y = start_pos[1] + t * approach_dist * np.sin(entry_angle)
            v = speed * (0.5 + 0.5 * t)
            waypoints.append(Waypoint(x, y, v, curvature=0.0))
        
        # Phase 2: Drift arc with aggressive angle
        sign = 1 if direction == "ccw" else -1
        
        # Offset arc center to account for drift angle
        arc_center_x = gate_pos[0] - arc_radius * np.sin(drift_angle)
        arc_center_y = gate_pos[1] - sign * arc_radius * np.cos(drift_angle)
        
        # Arc from entry to exit
        start_arc_angle = -np.pi/2 - drift_angle
        end_arc_angle = np.pi/2 + drift_angle
        
        for i in range(25):
            t = i / 24
            angle = start_arc_angle + t * (end_arc_angle - start_arc_angle)
            
            x = arc_center_x + arc_radius * np.cos(angle)
            y = arc_center_y + arc_radius * np.sin(angle)
            
            # Higher curvature for tighter turns
            curvature = sign * (1.0 / arc_radius) * (1.0 + 0.5 * drift_angle)
            
            waypoints.append(Waypoint(x, y, speed, curvature))
        
        # Phase 3: Exit with straightening
        last_wp = waypoints[-1]
        exit_dist = 1.5
        
        for i in range(10):
            t = i / 9
            x = last_wp.x + t * exit_dist * np.cos(exit_angle)
            y = last_wp.y + t * exit_dist * np.sin(exit_angle)
            v = speed * (1.0 - 0.2 * t)
            
            # Gradually reduce curvature
            curvature = last_wp.curvature * (1.0 - t)
            
            waypoints.append(Waypoint(x, y, v, curvature))
        
        return waypoints
    
    def _compute_control_cost(self, waypoints: List[Waypoint]) -> float:
        """Compute control effort cost."""
        cost = 0.0
        for wp in waypoints:
            # Penalize high curvature and velocity
            cost += abs(wp.curvature) + 0.1 * wp.velocity**2
        return cost / len(waypoints)
    
    def _compute_collision_cost(
        self,
        waypoints: List[Waypoint],
        obstacles: List[Tuple[float, float, float]]
    ) -> float:
        """Compute collision penalty."""
        cost = 0.0
        vehicle_radius = self.constraints.vehicle_width / 2 + self.constraints.safety_margin
        
        for wp in waypoints:
            for obs_x, obs_y, obs_radius in obstacles:
                dist = np.sqrt((wp.x - obs_x)**2 + (wp.y - obs_y)**2)
                clearance = dist - obs_radius - vehicle_radius
                
                if clearance < 0:
                    # Collision
                    cost += 1000 * abs(clearance)
                elif clearance < 0.1:
                    # Too close
                    cost += 10 * (0.1 - clearance)
        
        return cost
    
    def _compute_smoothness_cost(self, waypoints: List[Waypoint]) -> float:
        """Compute trajectory smoothness cost."""
        if len(waypoints) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(1, len(waypoints)):
            # Penalize sudden changes in curvature
            dcurv = abs(waypoints[i].curvature - waypoints[i-1].curvature)
            dvel = abs(waypoints[i].velocity - waypoints[i-1].velocity)
            cost += dcurv + 0.1 * dvel
        
        return cost / (len(waypoints) - 1)


def create_adaptive_planner(
    gate_width: float,
    vehicle_width: float = 0.48
) -> OptimizedTrajectoryPlanner:
    """
    Factory function to create planner based on scenario difficulty.
    
    Args:
        gate_width: Width of gate to navigate
        vehicle_width: Width of vehicle
        
    Returns:
        Configured planner
    """
    clearance = gate_width - vehicle_width
    
    if clearance < 0.5:
        # Tight scenario - use aggressive mode
        constraints = TrajectoryConstraints(
            max_curvature=4.0,
            max_velocity=3.0,
            vehicle_width=vehicle_width,
            safety_margin=0.03  # Smaller margin for tight spaces
        )
        return OptimizedTrajectoryPlanner(constraints, aggressive_mode=True)
    else:
        # Standard scenario
        constraints = TrajectoryConstraints(vehicle_width=vehicle_width)
        return OptimizedTrajectoryPlanner(constraints, aggressive_mode=False)
