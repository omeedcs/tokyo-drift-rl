"""
Moving Obstacle Agents with Realistic Behaviors

Implements other vehicles and pedestrians with behavior models.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AgentBehavior(Enum):
    """Agent behavior types."""
    STATIONARY = 0
    STRAIGHT = 1
    CIRCULAR = 2
    LANE_FOLLOW = 3
    CUT_IN = 4
    JAYWALKING = 5
    RANDOM_WALK = 6


@dataclass
class MovingAgent:
    """Moving agent (vehicle or pedestrian)."""
    id: int
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    heading: float  # radians
    size: Tuple[float, float]  # (length, width)
    behavior: AgentBehavior
    max_speed: float
    
    # Internal state
    target_position: Optional[np.ndarray] = None
    path_progress: float = 0.0
    behavior_timer: float = 0.0


class CarFollowingModel:
    """
    Intelligent Driver Model (IDM) for car-following behavior.
    
    Classic model used in traffic simulation.
    """
    
    def __init__(
        self,
        desired_speed: float = 3.0,  # m/s
        safe_time_headway: float = 1.5,  # seconds
        min_spacing: float = 2.0,  # meters
        max_acceleration: float = 2.0,  # m/s^2
        comfortable_braking: float = 3.0,  # m/s^2
        acceleration_exponent: float = 4.0
    ):
        self.desired_speed = desired_speed
        self.safe_time_headway = safe_time_headway
        self.min_spacing = min_spacing
        self.max_acceleration = max_acceleration
        self.comfortable_braking = comfortable_braking
        self.acceleration_exponent = acceleration_exponent
    
    def compute_acceleration(
        self,
        speed: float,
        lead_vehicle_distance: Optional[float],
        lead_vehicle_speed: Optional[float]
    ) -> float:
        """
        Compute acceleration using IDM.
        
        Args:
            speed: Current speed (m/s)
            lead_vehicle_distance: Distance to lead vehicle (m), None if no leader
            lead_vehicle_speed: Speed of lead vehicle (m/s), None if no leader
            
        Returns:
            Acceleration (m/s^2)
        """
        # Free-flow acceleration
        free_flow_accel = self.max_acceleration * (
            1.0 - (speed / self.desired_speed) ** self.acceleration_exponent
        )
        
        if lead_vehicle_distance is None or lead_vehicle_speed is None:
            # No leader, just accelerate to desired speed
            return free_flow_accel
        
        # Desired spacing
        speed_diff = speed - lead_vehicle_speed
        desired_spacing = (
            self.min_spacing +
            speed * self.safe_time_headway +
            (speed * speed_diff) / (2 * np.sqrt(self.max_acceleration * self.comfortable_braking))
        )
        
        # Interaction acceleration
        if lead_vehicle_distance > 0:
            interaction_accel = -self.max_acceleration * (desired_spacing / lead_vehicle_distance) ** 2
        else:
            interaction_accel = -self.comfortable_braking
        
        return free_flow_accel + interaction_accel


class MovingAgentSimulator:
    """
    Simulates multiple moving agents with behaviors.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.agents: List[MovingAgent] = []
        self.next_agent_id = 0
        self.car_following_model = CarFollowingModel()
        
    def add_agent(
        self,
        position: np.ndarray,
        behavior: AgentBehavior,
        size: Tuple[float, float] = (1.0, 0.5),
        max_speed: float = 3.0
    ) -> MovingAgent:
        """Add a new agent to the simulation."""
        agent = MovingAgent(
            id=self.next_agent_id,
            position=position.copy(),
            velocity=np.zeros(2),
            heading=0.0,
            size=size,
            behavior=behavior,
            max_speed=max_speed
        )
        
        self.agents.append(agent)
        self.next_agent_id += 1
        
        return agent
    
    def _update_straight(self, agent: MovingAgent, dt: float):
        """Update agent moving straight."""
        # Accelerate to max speed
        current_speed = np.linalg.norm(agent.velocity)
        if current_speed < agent.max_speed:
            acceleration = 1.0  # m/s^2
            new_speed = min(agent.max_speed, current_speed + acceleration * dt)
            
            if current_speed > 0:
                agent.velocity = agent.velocity / current_speed * new_speed
            else:
                # Start moving in heading direction
                agent.velocity = np.array([
                    agent.max_speed * np.cos(agent.heading),
                    agent.max_speed * np.sin(agent.heading)
                ])
        
        agent.position += agent.velocity * dt
    
    def _update_circular(self, agent: MovingAgent, dt: float):
        """Update agent moving in circle."""
        if agent.target_position is None:
            # Initialize circular path
            agent.target_position = agent.position.copy()
        
        # Circular motion parameters
        radius = 5.0  # meters
        angular_velocity = agent.max_speed / radius
        
        agent.path_progress += angular_velocity * dt
        
        # Compute position on circle
        offset = np.array([
            radius * np.cos(agent.path_progress),
            radius * np.sin(agent.path_progress)
        ])
        
        agent.position = agent.target_position + offset
        
        # Update velocity (tangent to circle)
        agent.velocity = np.array([
            -agent.max_speed * np.sin(agent.path_progress),
            agent.max_speed * np.cos(agent.path_progress)
        ])
        
        agent.heading = agent.path_progress + np.pi / 2
    
    def _update_lane_follow(self, agent: MovingAgent, dt: float):
        """Update agent following a lane."""
        # Simple lane following along x-axis
        if agent.target_position is None:
            agent.target_position = agent.position + np.array([100.0, 0.0])
        
        # Direction to target
        direction = agent.target_position - agent.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:
            # Reached target, set new one
            agent.target_position = agent.position + np.array([100.0, 0.0])
            direction = agent.target_position - agent.position
            distance = np.linalg.norm(direction)
        
        direction = direction / distance
        
        # Use car-following model if there's a lead vehicle
        lead_distance = None
        lead_speed = None
        
        # Find closest agent ahead
        for other in self.agents:
            if other.id == agent.id:
                continue
            
            # Check if ahead
            relative_pos = other.position - agent.position
            forward_distance = np.dot(relative_pos, direction)
            
            if forward_distance > 0 and forward_distance < 20.0:
                lateral_distance = abs(np.cross(relative_pos, direction))
                if lateral_distance < 2.0:  # In same lane
                    if lead_distance is None or forward_distance < lead_distance:
                        lead_distance = forward_distance
                        lead_speed = np.linalg.norm(other.velocity)
        
        # Compute acceleration
        current_speed = np.linalg.norm(agent.velocity)
        acceleration = self.car_following_model.compute_acceleration(
            current_speed, lead_distance, lead_speed
        )
        
        # Update speed
        new_speed = current_speed + acceleration * dt
        new_speed = np.clip(new_speed, 0.0, agent.max_speed)
        
        # Update velocity
        agent.velocity = direction * new_speed
        agent.heading = np.arctan2(direction[1], direction[0])
        
        # Update position
        agent.position += agent.velocity * dt
    
    def _update_cut_in(self, agent: MovingAgent, dt: float):
        """Update agent performing cut-in maneuver."""
        # Initialize if needed
        if agent.target_position is None:
            agent.target_position = agent.position + np.array([10.0, 3.0])
            agent.behavior_timer = 0.0
        
        agent.behavior_timer += dt
        
        # Phase 1: Accelerate forward (0-2s)
        if agent.behavior_timer < 2.0:
            agent.velocity = np.array([agent.max_speed, 0.0])
        # Phase 2: Cut in (2-4s)
        elif agent.behavior_timer < 4.0:
            t = (agent.behavior_timer - 2.0) / 2.0
            lateral_speed = 1.5 * np.sin(t * np.pi)
            agent.velocity = np.array([agent.max_speed, lateral_speed])
        # Phase 3: Stabilize (4s+)
        else:
            agent.velocity = np.array([agent.max_speed, 0.0])
        
        agent.position += agent.velocity * dt
        agent.heading = np.arctan2(agent.velocity[1], agent.velocity[0])
    
    def _update_jaywalking(self, agent: MovingAgent, dt: float):
        """Update pedestrian jaywalking."""
        # Walk across road
        if agent.target_position is None:
            agent.target_position = agent.position + np.array([0.0, 5.0])
        
        direction = agent.target_position - agent.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            # Reached other side, walk back
            agent.target_position = agent.position - np.array([0.0, 5.0])
            direction = agent.target_position - agent.position
            distance = np.linalg.norm(direction)
        
        direction = direction / distance
        agent.velocity = direction * agent.max_speed
        agent.position += agent.velocity * dt
    
    def _update_random_walk(self, agent: MovingAgent, dt: float):
        """Update agent with random walk behavior."""
        # Change direction randomly
        if agent.behavior_timer <= 0:
            new_heading = self.rng.uniform(-np.pi, np.pi)
            agent.velocity = np.array([
                agent.max_speed * np.cos(new_heading),
                agent.max_speed * np.sin(new_heading)
            ])
            agent.heading = new_heading
            agent.behavior_timer = self.rng.uniform(1.0, 3.0)
        
        agent.behavior_timer -= dt
        agent.position += agent.velocity * dt
    
    def step(self, dt: float):
        """Update all agents."""
        for agent in self.agents:
            if agent.behavior == AgentBehavior.STATIONARY:
                pass  # Don't move
            elif agent.behavior == AgentBehavior.STRAIGHT:
                self._update_straight(agent, dt)
            elif agent.behavior == AgentBehavior.CIRCULAR:
                self._update_circular(agent, dt)
            elif agent.behavior == AgentBehavior.LANE_FOLLOW:
                self._update_lane_follow(agent, dt)
            elif agent.behavior == AgentBehavior.CUT_IN:
                self._update_cut_in(agent, dt)
            elif agent.behavior == AgentBehavior.JAYWALKING:
                self._update_jaywalking(agent, dt)
            elif agent.behavior == AgentBehavior.RANDOM_WALK:
                self._update_random_walk(agent, dt)
    
    def get_agents_in_range(
        self,
        position: np.ndarray,
        max_range: float
    ) -> List[MovingAgent]:
        """Get agents within range of a position."""
        nearby = []
        for agent in self.agents:
            distance = np.linalg.norm(agent.position - position)
            if distance <= max_range:
                nearby.append(agent)
        return nearby
    
    def reset(self):
        """Remove all agents."""
        self.agents = []
        self.next_agent_id = 0


def test_moving_agents():
    """Test moving agent behaviors."""
    print("Testing Moving Agent Behaviors")
    print("=" * 60)
    
    simulator = MovingAgentSimulator(seed=42)
    
    # Add various agents
    simulator.add_agent(
        position=np.array([0.0, 0.0]),
        behavior=AgentBehavior.LANE_FOLLOW,
        size=(2.0, 1.0),
        max_speed=5.0
    )
    
    simulator.add_agent(
        position=np.array([10.0, 0.0]),
        behavior=AgentBehavior.LANE_FOLLOW,
        size=(2.0, 1.0),
        max_speed=3.0
    )
    
    simulator.add_agent(
        position=np.array([5.0, 5.0]),
        behavior=AgentBehavior.CIRCULAR,
        size=(0.5, 0.5),
        max_speed=2.0
    )
    
    print("\nSimulation:")
    print(f"{'Time':>6} {'Agent 0':>20} {'Agent 1':>20} {'Agent 2':>20}")
    print("-" * 70)
    
    for step in range(20):
        t = step * 0.1
        simulator.step(dt=0.1)
        
        print(f"{t:6.2f}", end="")
        for agent in simulator.agents[:3]:
            speed = np.linalg.norm(agent.velocity)
            print(f" ({agent.position[0]:5.1f},{agent.position[1]:5.1f}) {speed:4.1f}", end="")
        print()
    
    print(f"\nFinal positions:")
    for agent in simulator.agents:
        print(f"  Agent {agent.id} ({agent.behavior.name}): {agent.position}")


if __name__ == "__main__":
    test_moving_agents()
