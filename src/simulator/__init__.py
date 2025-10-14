"""
Simulator module for testing IKD models without physical hardware.
Based on UT AUTOmata F1/10 vehicle specifications from the paper.
"""

from src.simulator.vehicle import F110Vehicle
from src.simulator.environment import SimulationEnvironment
from src.simulator.sensors import IMUSensor, VelocitySensor
from src.simulator.controller import VirtualJoystick

__all__ = [
    'F110Vehicle',
    'SimulationEnvironment',
    'IMUSensor',
    'VelocitySensor',
    'VirtualJoystick'
]
