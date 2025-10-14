"""
Scenario Generator for Drift Gym

Generates diverse, randomized scenarios for training robust policies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Obstacle definition."""
    x: float
    y: float
    radius: float
    

@dataclass
class Gate:
    """Gate definition."""
    x: float
    y: float
    width: float
    orientation: float = 0.0  # radians


@dataclass
class Scenario:
    """Complete scenario definition."""
    name: str
    gates: List[Gate]
    obstacles: List[Obstacle]
    start_position: Tuple[float, float]
    start_heading: float
    bounds: Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
    difficulty: float  # 0.0 to 1.0
    

class ScenarioGenerator:
    """
    Generates diverse scenarios with controlled difficulty.
    
    Supports:
    - Predefined scenarios (loose, tight, slalom)
    - Randomized scenarios
    - Curriculum learning (progressive difficulty)
    - Procedural generation
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize scenario generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self._scenarios = self._initialize_scenarios()
    
    def _initialize_scenarios(self) -> Dict[str, Scenario]:
        """Initialize predefined scenarios."""
        scenarios = {}
        
        # Loose Drift (Easy)
        scenarios['loose'] = Scenario(
            name='loose',
            gates=[Gate(x=3.0, y=1.065, width=2.13)],
            obstacles=[
                Obstacle(x=1.5, y=0.5, radius=0.3),
                Obstacle(x=2.0, y=1.8, radius=0.35),
            ],
            start_position=(0.0, 1.0),
            start_heading=0.0,
            bounds=(-1.0, 5.0, -1.0, 4.0),
            difficulty=0.3
        )
        
        # Tight Drift (Hard)
        scenarios['tight'] = Scenario(
            name='tight',
            gates=[Gate(x=3.0, y=0.405, width=0.81)],
            obstacles=[
                Obstacle(x=1.2, y=0.2, radius=0.25),
                Obstacle(x=1.8, y=0.7, radius=0.3),
                Obstacle(x=2.5, y=0.1, radius=0.25),
            ],
            start_position=(0.0, 0.4),
            start_heading=0.0,
            bounds=(-1.0, 5.0, -0.5, 2.0),
            difficulty=0.8
        )
        
        # Slalom (Medium-Hard)
        gates = []
        for i in range(5):
            x = 2.0 + i * 2.0
            y = 1.5 + (0.5 if i % 2 == 0 else -0.5)
            gates.append(Gate(x=x, y=y, width=1.5))
        
        scenarios['slalom'] = Scenario(
            name='slalom',
            gates=gates,
            obstacles=[
                Obstacle(x=3.0, y=1.0, radius=0.3),
                Obstacle(x=5.0, y=2.0, radius=0.3),
                Obstacle(x=7.0, y=1.0, radius=0.3),
            ],
            start_position=(0.0, 1.5),
            start_heading=0.0,
            bounds=(-1.0, 12.0, -1.0, 4.0),
            difficulty=0.6
        )
        
        # Figure-8 (Very Hard)
        scenarios['figure8'] = Scenario(
            name='figure8',
            gates=[
                Gate(x=3.0, y=1.5, width=1.2),
                Gate(x=6.0, y=3.0, width=1.2),
                Gate(x=9.0, y=1.5, width=1.2),
                Gate(x=6.0, y=0.0, width=1.2),
            ],
            obstacles=[
                Obstacle(x=4.5, y=2.2, radius=0.4),
                Obstacle(x=7.5, y=2.2, radius=0.4),
                Obstacle(x=4.5, y=0.8, radius=0.4),
                Obstacle(x=7.5, y=0.8, radius=0.4),
            ],
            start_position=(0.0, 1.5),
            start_heading=0.0,
            bounds=(-1.0, 11.0, -1.0, 5.0),
            difficulty=0.9
        )
        
        return scenarios
    
    def get_scenario(
        self,
        name: str,
        randomize: bool = False,
        difficulty_scale: float = 1.0
    ) -> Scenario:
        """
        Get a scenario by name.
        
        Args:
            name: Scenario name ('loose', 'tight', 'slalom', 'figure8')
            randomize: Apply randomization to scenario
            difficulty_scale: Scale difficulty (0.5 = easier, 2.0 = harder)
            
        Returns:
            Scenario object
        """
        if name not in self._scenarios:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(self._scenarios.keys())}")
        
        scenario = self._scenarios[name]
        
        if randomize:
            scenario = self.randomize_scenario(scenario, difficulty_scale)
        
        return scenario
    
    def randomize_scenario(
        self,
        scenario: Scenario,
        difficulty_scale: float = 1.0
    ) -> Scenario:
        """
        Randomize a scenario while maintaining its structure.
        
        Args:
            scenario: Base scenario
            difficulty_scale: Difficulty multiplier
            
        Returns:
            Randomized scenario
        """
        # Randomize gates
        new_gates = []
        for gate in scenario.gates:
            # Randomize position
            x_noise = self.rng.uniform(-0.3, 0.3)
            y_noise = self.rng.uniform(-0.3, 0.3)
            
            # Randomize width (narrower = harder)
            width_factor = 1.0 / difficulty_scale if difficulty_scale > 0 else 1.0
            width_noise = self.rng.uniform(0.9, 1.1) * width_factor
            new_width = max(0.6, gate.width * width_noise)
            
            new_gates.append(Gate(
                x=gate.x + x_noise,
                y=gate.y + y_noise,
                width=new_width,
                orientation=gate.orientation
            ))
        
        # Randomize obstacles
        new_obstacles = []
        for obs in scenario.obstacles:
            # Randomize position
            x_noise = self.rng.uniform(-0.4, 0.4)
            y_noise = self.rng.uniform(-0.4, 0.4)
            
            # Randomize size
            radius_noise = self.rng.uniform(0.8, 1.2) * difficulty_scale
            new_radius = np.clip(obs.radius * radius_noise, 0.15, 0.6)
            
            new_obstacles.append(Obstacle(
                x=obs.x + x_noise,
                y=obs.y + y_noise,
                radius=new_radius
            ))
        
        # Optionally add/remove obstacles based on difficulty
        if difficulty_scale > 1.2 and self.rng.random() < 0.5:
            # Add an obstacle
            new_obstacles.append(self._generate_random_obstacle(scenario.bounds))
        elif difficulty_scale < 0.8 and len(new_obstacles) > 1 and self.rng.random() < 0.5:
            # Remove an obstacle
            new_obstacles = new_obstacles[:-1]
        
        # Randomize start
        start_x_noise = self.rng.uniform(-0.3, 0.3)
        start_y_noise = self.rng.uniform(-0.2, 0.2)
        new_start = (
            scenario.start_position[0] + start_x_noise,
            scenario.start_position[1] + start_y_noise
        )
        
        start_heading_noise = self.rng.uniform(-0.2, 0.2)
        new_heading = scenario.start_heading + start_heading_noise
        
        return Scenario(
            name=f"{scenario.name}_randomized",
            gates=new_gates,
            obstacles=new_obstacles,
            start_position=new_start,
            start_heading=new_heading,
            bounds=scenario.bounds,
            difficulty=scenario.difficulty * difficulty_scale
        )
    
    def generate_random_scenario(
        self,
        num_gates: int = 3,
        num_obstacles: int = 5,
        difficulty: float = 0.5
    ) -> Scenario:
        """
        Generate a completely random scenario.
        
        Args:
            num_gates: Number of gates
            num_obstacles: Number of obstacles
            difficulty: Difficulty level (0.0 to 1.0)
            
        Returns:
            Random scenario
        """
        bounds = (-1.0, 10.0, -1.0, 5.0)
        
        # Generate gates along a path
        gates = []
        for i in range(num_gates):
            x = 2.0 + i * (8.0 / num_gates)
            y = 2.0 + self.rng.uniform(-1.5, 1.5)
            
            # Gate width inversely proportional to difficulty
            width = self.rng.uniform(2.5 - difficulty * 1.5, 3.0 - difficulty * 1.5)
            width = max(0.6, width)
            
            gates.append(Gate(x=x, y=y, width=width))
        
        # Generate obstacles
        obstacles = []
        for _ in range(num_obstacles):
            obstacles.append(self._generate_random_obstacle(bounds, difficulty))
        
        return Scenario(
            name=f'random_{self.rng.randint(10000)}',
            gates=gates,
            obstacles=obstacles,
            start_position=(0.0, 2.0),
            start_heading=0.0,
            bounds=bounds,
            difficulty=difficulty
        )
    
    def _generate_random_obstacle(
        self,
        bounds: Tuple[float, float, float, float],
        difficulty: float = 0.5
    ) -> Obstacle:
        """Generate a random obstacle within bounds."""
        min_x, max_x, min_y, max_y = bounds
        
        x = self.rng.uniform(min_x + 1.0, max_x - 1.0)
        y = self.rng.uniform(min_y + 0.5, max_y - 0.5)
        
        # Larger obstacles = harder
        radius = self.rng.uniform(0.2, 0.3 + difficulty * 0.3)
        
        return Obstacle(x=x, y=y, radius=radius)
    
    def get_curriculum_scenario(
        self,
        level: int,
        max_level: int = 10
    ) -> Scenario:
        """
        Get scenario for curriculum learning.
        
        Args:
            level: Current difficulty level (0 to max_level)
            max_level: Maximum difficulty level
            
        Returns:
            Scenario appropriate for current level
        """
        # Normalize level to [0, 1]
        difficulty = level / max_level
        
        # Map level to scenario progression
        if level < 2:
            # Start with loose scenarios
            return self.get_scenario('loose', randomize=True, difficulty_scale=0.5 + difficulty * 0.5)
        elif level < 5:
            # Progress to medium scenarios
            return self.get_scenario('loose', randomize=True, difficulty_scale=1.0 + difficulty * 0.5)
        elif level < 8:
            # Introduce tight scenarios
            return self.get_scenario('tight', randomize=True, difficulty_scale=0.8 + difficulty * 0.4)
        else:
            # Final level: slalom and figure-8
            if self.rng.random() < 0.5:
                return self.get_scenario('slalom', randomize=True, difficulty_scale=1.0 + difficulty * 0.3)
            else:
                return self.get_scenario('tight', randomize=True, difficulty_scale=1.2 + difficulty * 0.5)


# Test
if __name__ == "__main__":
    gen = ScenarioGenerator(seed=42)
    
    print("Testing Scenario Generator")
    print("=" * 60)
    
    # Test predefined scenarios
    for name in ['loose', 'tight', 'slalom']:
        scenario = gen.get_scenario(name)
        print(f"\n{scenario.name.upper()}:")
        print(f"  Gates: {len(scenario.gates)}")
        print(f"  Obstacles: {len(scenario.obstacles)}")
        print(f"  Difficulty: {scenario.difficulty:.2f}")
    
    # Test randomization
    print("\n\nRANDOMIZED TIGHT (5 samples):")
    for i in range(5):
        scenario = gen.get_scenario('tight', randomize=True, difficulty_scale=1.0)
        print(f"  Sample {i+1}: {len(scenario.obstacles)} obstacles, gate width: {scenario.gates[0].width:.2f}m")
    
    # Test curriculum
    print("\n\nCURRICULUM PROGRESSION:")
    for level in [0, 2, 5, 7, 10]:
        scenario = gen.get_curriculum_scenario(level, max_level=10)
        print(f"  Level {level:2d}: {scenario.name:20s} difficulty={scenario.difficulty:.2f}")
