"""
Trajectory planning for drift maneuvers.

Provides path planning algorithms for navigating through obstacles
during drift scenarios.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Waypoint:
    """Represents a waypoint along a trajectory."""
    x: float
    y: float
    velocity: float
    curvature: float = 0.0
    
    def distance_to(self, x: float, y: float) -> float:
        """Compute Euclidean distance to a point."""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)


class Trajectory:
    """Container for a planned trajectory."""
    
    def __init__(self, waypoints: List[Waypoint]):
        """
        Initialize trajectory.
        
        Args:
            waypoints: List of waypoints defining the path
        """
        self.waypoints = waypoints
        self.current_idx = 0
    
    def get_closest_point(self, x: float, y: float) -> Tuple[int, Waypoint]:
        """
        Find closest waypoint to current position.
        
        Args:
            x, y: Current position
            
        Returns:
            Tuple of (index, waypoint)
        """
        min_dist = float('inf')
        closest_idx = 0
        
        # Search ahead from current index
        search_range = range(self.current_idx, min(self.current_idx + 20, len(self.waypoints)))
        
        for i in search_range:
            dist = self.waypoints[i].distance_to(x, y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx, self.waypoints[closest_idx]
    
    def get_lookahead_point(self, x: float, y: float, lookahead_dist: float) -> Optional[Waypoint]:
        """
        Get lookahead point for pure pursuit.
        
        Args:
            x, y: Current position
            lookahead_dist: Lookahead distance
            
        Returns:
            Waypoint at lookahead distance, or None if trajectory complete
        """
        # Find closest point
        closest_idx, _ = self.get_closest_point(x, y)
        self.current_idx = closest_idx
        
        # Search forward for lookahead point
        for i in range(closest_idx, len(self.waypoints)):
            wp = self.waypoints[i]
            dist = wp.distance_to(x, y)
            
            if dist >= lookahead_dist:
                return wp
        
        # Return last waypoint if we're near the end
        if len(self.waypoints) > 0:
            return self.waypoints[-1]
        
        return None
    
    def is_complete(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """
        Check if trajectory is complete.
        
        Args:
            x, y: Current position
            threshold: Distance threshold to final waypoint
            
        Returns:
            True if near final waypoint
        """
        if len(self.waypoints) == 0:
            return True
        
        final_wp = self.waypoints[-1]
        dist = final_wp.distance_to(x, y)
        return dist < threshold
    
    def reset(self):
        """Reset trajectory tracking."""
        self.current_idx = 0


class DriftTrajectoryPlanner:
    """
    Plans trajectories for drift maneuvers through obstacles.
    
    Uses a combination of:
    - Entry straight line (approach)
    - Circular arc (drift through gate)
    - Exit straight line (recovery)
    """
    
    def __init__(self, vehicle_width: float = 0.48):
        """
        Initialize trajectory planner.
        
        Args:
            vehicle_width: Width of vehicle for clearance calculations
        """
        self.vehicle_width = vehicle_width
        self.safety_margin = 0.1  # Extra clearance
    
    def plan_drift_through_gate(
        self,
        start_pos: Tuple[float, float],
        gate_center: Tuple[float, float],
        gate_width: float,
        drift_direction: str = "ccw",
        approach_speed: float = 3.0,
        drift_speed: float = 2.5
    ) -> Trajectory:
        """
        Plan a drift trajectory through a gate.
        
        Args:
            start_pos: Starting (x, y) position
            gate_center: Center of gate (x, y)
            gate_width: Width of gate opening
            drift_direction: "ccw" (counter-clockwise) or "cw" (clockwise)
            approach_speed: Speed during approach phase
            drift_speed: Speed during drift phase
            
        Returns:
            Trajectory object with waypoints
        """
        waypoints = []
        
        # Check if gate is wide enough
        required_width = self.vehicle_width + 2 * self.safety_margin
        if gate_width < required_width:
            print(f"⚠️  Warning: Gate width {gate_width:.2f}m may be too narrow "
                  f"(need {required_width:.2f}m)")
        
        # Phase 1: Approach trajectory (straight line)
        approach_waypoints = self._plan_approach(
            start_pos, gate_center, approach_speed, num_points=20
        )
        waypoints.extend(approach_waypoints)
        
        # Phase 2: Drift arc through gate
        arc_radius = gate_width * 0.4  # Radius of drift arc
        drift_waypoints = self._plan_drift_arc(
            gate_center, arc_radius, drift_direction, drift_speed, num_points=30
        )
        waypoints.extend(drift_waypoints)
        
        # Phase 3: Exit trajectory (straighten out)
        last_wp = drift_waypoints[-1] if drift_waypoints else approach_waypoints[-1]
        exit_waypoints = self._plan_exit(
            (last_wp.x, last_wp.y), drift_direction, drift_speed, num_points=15
        )
        waypoints.extend(exit_waypoints)
        
        return Trajectory(waypoints)
    
    def _plan_approach(
        self,
        start: Tuple[float, float],
        gate_center: Tuple[float, float],
        speed: float,
        num_points: int
    ) -> List[Waypoint]:
        """Plan straight approach to gate."""
        waypoints = []
        
        # Compute approach line (offset to align for drift entry)
        dx = gate_center[0] - start[0]
        dy = gate_center[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Reduce approach distance to leave room for drift entry
        approach_distance = max(distance - 1.0, 0.5)
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start[0] + t * dx * (approach_distance / distance)
            y = start[1] + t * dy * (approach_distance / distance)
            
            # Gradually increase speed
            v = speed * min(1.0, 0.3 + 0.7 * t)
            
            waypoints.append(Waypoint(x, y, v, curvature=0.0))
        
        return waypoints
    
    def _plan_drift_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        direction: str,
        speed: float,
        num_points: int
    ) -> List[Waypoint]:
        """Plan circular arc for drift maneuver."""
        waypoints = []
        
        # Arc angles
        if direction == "ccw":
            start_angle = -np.pi / 2  # Start from bottom
            end_angle = np.pi / 2     # End at top
            sign = 1
        else:  # cw
            start_angle = np.pi / 2   # Start from top
            end_angle = -np.pi / 2    # End at bottom
            sign = -1
        
        for i in range(num_points):
            t = i / (num_points - 1)
            angle = start_angle + t * (end_angle - start_angle)
            
            # Position on circle
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            # Curvature for circular path
            curvature = sign * (1.0 / radius)
            
            waypoints.append(Waypoint(x, y, speed, curvature))
        
        return waypoints
    
    def _plan_exit(
        self,
        start: Tuple[float, float],
        direction: str,
        speed: float,
        num_points: int
    ) -> List[Waypoint]:
        """Plan exit trajectory after drift."""
        waypoints = []
        
        # Exit straight ahead with decreasing curvature
        exit_distance = 2.0
        
        if direction == "ccw":
            exit_angle = 0.0  # Exit to the right
        else:
            exit_angle = np.pi  # Exit to the left
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Position
            x = start[0] + t * exit_distance * np.cos(exit_angle)
            y = start[1] + t * exit_distance * np.sin(exit_angle)
            
            # Gradually reduce curvature to straighten out
            curvature = (1.0 - t) * 0.5 * (1.0 if direction == "ccw" else -1.0)
            
            # Gradually reduce speed
            v = speed * (1.0 - 0.3 * t)
            
            waypoints.append(Waypoint(x, y, v, curvature))
        
        return waypoints


class CircleTrajectoryPlanner:
    """Plans circular trajectories for circle navigation tests."""
    
    @staticmethod
    def plan_circle(
        center: Tuple[float, float],
        radius: float,
        velocity: float,
        direction: str = "ccw",
        num_points: int = 100
    ) -> Trajectory:
        """
        Plan a circular trajectory.
        
        Args:
            center: Center of circle (x, y)
            radius: Radius of circle
            velocity: Constant velocity
            direction: "ccw" or "cw"
            num_points: Number of waypoints
            
        Returns:
            Trajectory object
        """
        waypoints = []
        
        sign = 1 if direction == "ccw" else -1
        curvature = sign / radius
        
        for i in range(num_points):
            angle = sign * 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            waypoints.append(Waypoint(x, y, velocity, curvature))
        
        return Trajectory(waypoints)
