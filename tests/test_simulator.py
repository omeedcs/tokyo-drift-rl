"""
Unit tests for simulator components.
"""
import unittest
import numpy as np
import torch

from src.simulator.vehicle import F110Vehicle, VehicleState
from src.simulator.sensors import IMUSensor, VelocitySensor
from src.simulator.environment import SimulationEnvironment
from src.simulator.controller import VirtualJoystick


class TestF110Vehicle(unittest.TestCase):
    """Test cases for F1/10 vehicle simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vehicle = F110Vehicle(dt=0.05)
    
    def test_initialization(self):
        """Test vehicle initializes correctly."""
        self.assertEqual(self.vehicle.state.x, 0.0)
        self.assertEqual(self.vehicle.state.y, 0.0)
        self.assertEqual(self.vehicle.state.theta, 0.0)
        self.assertEqual(self.vehicle.state.velocity, 0.0)
    
    def test_reset(self):
        """Test vehicle reset."""
        self.vehicle.state.x = 5.0
        self.vehicle.state.y = 3.0
        self.vehicle.reset()
        self.assertEqual(self.vehicle.state.x, 0.0)
        self.assertEqual(self.vehicle.state.y, 0.0)
    
    def test_motor_constraints(self):
        """Test motor ERPM limits are enforced."""
        # Request speed beyond limit
        excessive_speed = 10.0
        constrained = self.vehicle._apply_motor_constraints(excessive_speed)
        self.assertLessEqual(abs(constrained), self.vehicle.MAX_SPEED)
    
    def test_servo_constraints(self):
        """Test servo constraints are enforced."""
        # Request angle beyond limit
        excessive_angle = 1.0
        constrained = self.vehicle._apply_servo_constraints(excessive_angle)
        # Servo limits result in max angles (from servo range 0.05-0.95)
        max_angle_from_min = (self.vehicle.SERVO_MIN - self.vehicle.STEERING_TO_SERVO_OFFSET) / self.vehicle.STEERING_TO_SERVO_GAIN
        max_angle_from_max = (self.vehicle.SERVO_MAX - self.vehicle.STEERING_TO_SERVO_OFFSET) / self.vehicle.STEERING_TO_SERVO_GAIN
        max_achievable = max(abs(max_angle_from_min), abs(max_angle_from_max))
        self.assertLessEqual(abs(constrained), max_achievable + 0.01)
    
    def test_forward_motion(self):
        """Test vehicle moves forward."""
        self.vehicle.set_control(velocity=1.0, steering_angle=0.0)
        
        for _ in range(20):
            self.vehicle.step()
        
        # Should have moved forward
        self.assertGreater(self.vehicle.state.x, 0.0)
        self.assertAlmostEqual(self.vehicle.state.y, 0.0, places=2)
    
    def test_turning(self):
        """Test vehicle turns."""
        self.vehicle.set_control(velocity=2.0, steering_angle=0.2)
        
        for _ in range(100):
            self.vehicle.step()
        
        # Should have turned
        self.assertNotEqual(self.vehicle.state.theta, 0.0)
        self.assertNotEqual(self.vehicle.state.y, 0.0)
    
    def test_circular_motion(self):
        """Test vehicle can execute circular motion."""
        velocity = 2.0
        curvature = 0.5
        angular_velocity = velocity * curvature
        
        # Convert to steering angle
        steering_angle = np.arctan(curvature * self.vehicle.WHEELBASE)
        
        self.vehicle.set_control(velocity, steering_angle)
        
        initial_x = self.vehicle.state.x
        
        # Complete roughly one circle
        steps = int(2 * np.pi / angular_velocity / self.vehicle.dt)
        for _ in range(steps):
            self.vehicle.step()
        
        # Should return near starting position
        self.assertAlmostEqual(self.vehicle.state.x, initial_x, delta=0.5)


class TestIMUSensor(unittest.TestCase):
    """Test cases for IMU sensor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.imu = IMUSensor(noise_std=0.01, delay=0.1, sample_rate=20.0)
    
    def test_measurement(self):
        """Test IMU produces measurements."""
        true_av = 1.0
        
        # Fill buffer first (due to delay)
        for _ in range(self.imu.buffer_size):
            self.imu.measure(true_av)
        
        # Now measurement should be close to true value (with noise)
        measured = self.imu.measure(true_av)
        self.assertAlmostEqual(measured, true_av, delta=0.5)
    
    def test_delay(self):
        """Test IMU delay."""
        # First measurement should be zero (buffer initialized)
        measured = self.imu.measure(1.0)
        self.assertEqual(measured, 0.0)
        
        # After filling buffer, should get delayed measurements
        for _ in range(10):
            self.imu.measure(1.0)
        
        measured = self.imu.measure(1.0)
        self.assertGreater(measured, 0.0)
    
    def test_reset(self):
        """Test sensor reset."""
        for _ in range(10):
            self.imu.measure(1.0)
        
        self.imu.reset()
        measured = self.imu.measure(1.0)
        self.assertEqual(measured, 0.0)


class TestSimulationEnvironment(unittest.TestCase):
    """Test cases for simulation environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = SimulationEnvironment(dt=0.05)
    
    def test_initialization(self):
        """Test environment initializes."""
        self.assertIsNotNone(self.env.vehicle)
        self.assertIsNotNone(self.env.imu)
        self.assertEqual(self.env.time, 0.0)
    
    def test_reset(self):
        """Test environment reset."""
        self.env.time = 5.0
        self.env.vehicle.state.x = 10.0
        
        self.env.reset()
        
        self.assertEqual(self.env.time, 0.0)
        self.assertEqual(self.env.vehicle.state.x, 0.0)
    
    def test_step(self):
        """Test environment step."""
        self.env.set_control(velocity=1.0, angular_velocity=0.0)
        measurements = self.env.step()
        
        self.assertIn('velocity', measurements)
        self.assertIn('angular_velocity', measurements)
        self.assertIn('time', measurements)
    
    def test_recording(self):
        """Test data recording."""
        self.env.start_recording()
        self.env.set_control(1.0, 0.0)
        
        for _ in range(10):
            self.env.step()
        
        self.env.stop_recording()
        data = self.env.get_recorded_data()
        
        self.assertEqual(len(data['time']), 10)
        self.assertEqual(len(data['x']), 10)
    
    def test_circle_test_setup(self):
        """Test circle test setup."""
        velocity, angular_velocity = self.env.setup_circle_test(2.0, 0.7)
        self.assertEqual(velocity, 2.0)
        self.assertAlmostEqual(angular_velocity, 1.4)
    
    def test_obstacle_collision(self):
        """Test collision detection."""
        self.env.reset()
        self.env.add_obstacle(x=1.0, y=0.0, radius=0.5)
        
        # Move vehicle into obstacle
        self.env.vehicle.state.x = 1.0
        self.env.vehicle.state.y = 0.0
        
        collision = self.env.check_collision()
        self.assertTrue(collision)


class TestVirtualJoystick(unittest.TestCase):
    """Test cases for virtual joystick."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.joystick = VirtualJoystick()
    
    def test_manual_control(self):
        """Test manual control mode."""
        self.joystick.set_manual_control(2.0, 0.5)
        
        velocity, av = self.joystick.get_control(0.0, 0.0)
        
        self.assertEqual(velocity, 2.0)
        self.assertEqual(av, 0.5)
    
    def test_circle_mode(self):
        """Test circle mode."""
        self.joystick.set_circle_mode(2.0, 0.7)
        
        velocity, av = self.joystick.get_control(0.0, 0.0)
        
        self.assertEqual(velocity, 2.0)
        self.assertAlmostEqual(av, 1.4)


if __name__ == '__main__':
    unittest.main()
