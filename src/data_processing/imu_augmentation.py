"""
IMU Delay Augmentation for Data Generation.

Augments training data by simulating various IMU delays to improve
model robustness to sensor latency variations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import deque
import copy


@dataclass
class IMUDelayConfig:
    """Configuration for IMU delay augmentation."""
    min_delay: float = 0.10  # Minimum delay in seconds
    max_delay: float = 0.30  # Maximum delay in seconds
    delay_std: float = 0.02  # Delay variation (Gaussian noise)
    sample_rate: float = 40.0  # IMU sampling rate in Hz
    noise_std: float = 0.01  # Additional measurement noise
    bias_range: Tuple[float, float] = (-0.05, 0.05)  # Bias range


class IMUDelayAugmenter:
    """
    Augment IMU data with variable delays for robust training.
    
    Features:
    - Variable delay simulation
    - Realistic noise injection
    - Bias drift simulation
    - Multiple augmentation strategies
    
    Use cases:
    1. Training data augmentation for IKD models
    2. Testing model robustness to sensor delays
    3. Generating diverse training scenarios
    
    Example:
        augmenter = IMUDelayAugmenter(config)
        augmented_data = augmenter.augment_dataset(original_data)
    """
    
    def __init__(self, config: Optional[IMUDelayConfig] = None):
        """
        Initialize IMU delay augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or IMUDelayConfig()
        self.dt = 1.0 / self.config.sample_rate
    
    def apply_delay(
        self,
        signal: np.ndarray,
        delay_seconds: float
    ) -> np.ndarray:
        """
        Apply time delay to signal.
        
        Args:
            signal: Input signal array (1D time series)
            delay_seconds: Delay in seconds
            
        Returns:
            Delayed signal (same shape as input)
        """
        delay_samples = int(delay_seconds * self.config.sample_rate)
        
        if delay_samples <= 0:
            return signal
        
        # Shift signal and pad with initial value
        delayed = np.zeros_like(signal)
        delayed[:delay_samples] = signal[0]  # Hold first value
        delayed[delay_samples:] = signal[:-delay_samples]
        
        return delayed
    
    def apply_variable_delay(
        self,
        signal: np.ndarray,
        base_delay: float,
        time_varying: bool = True
    ) -> np.ndarray:
        """
        Apply time-varying delay to signal.
        
        Args:
            signal: Input signal array
            base_delay: Base delay in seconds
            time_varying: Whether delay varies over time
            
        Returns:
            Signal with variable delay
        """
        if not time_varying:
            return self.apply_delay(signal, base_delay)
        
        delayed_signal = np.zeros_like(signal)
        buffer_size = int(self.config.max_delay * self.config.sample_rate) + 10
        buffer = deque([signal[0]] * buffer_size, maxlen=buffer_size)
        
        for i in range(len(signal)):
            # Add current sample to buffer
            buffer.append(signal[i])
            
            # Sample delay with variation
            current_delay = max(0, base_delay + np.random.normal(0, self.config.delay_std))
            delay_samples = int(current_delay * self.config.sample_rate)
            delay_samples = min(delay_samples, len(buffer) - 1)
            
            # Get delayed sample
            delayed_signal[i] = buffer[-delay_samples - 1] if delay_samples < len(buffer) else buffer[0]
        
        return delayed_signal
    
    def add_measurement_noise(
        self,
        signal: np.ndarray,
        noise_std: Optional[float] = None
    ) -> np.ndarray:
        """
        Add measurement noise to signal.
        
        Args:
            signal: Input signal
            noise_std: Noise standard deviation (uses config default if None)
            
        Returns:
            Noisy signal
        """
        if noise_std is None:
            noise_std = self.config.noise_std
        
        noise = np.random.normal(0, noise_std, size=signal.shape)
        return signal + noise
    
    def add_bias_drift(
        self,
        signal: np.ndarray,
        bias: Optional[float] = None
    ) -> np.ndarray:
        """
        Add constant or drifting bias to signal.
        
        Args:
            signal: Input signal
            bias: Constant bias (random if None)
            
        Returns:
            Signal with bias
        """
        if bias is None:
            bias = np.random.uniform(*self.config.bias_range)
        
        return signal + bias
    
    def augment_trajectory(
        self,
        angular_velocity: np.ndarray,
        velocity: np.ndarray,
        augmentation_factor: int = 3
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate multiple augmented versions of a trajectory.
        
        Args:
            angular_velocity: True angular velocity measurements
            velocity: Velocity measurements
            augmentation_factor: Number of augmented versions to generate
            
        Returns:
            List of augmented data dictionaries
        """
        augmented_data = []
        
        for i in range(augmentation_factor):
            # Sample random delay
            delay = np.random.uniform(self.config.min_delay, self.config.max_delay)
            
            # Apply augmentations to angular velocity (IMU measurement)
            aug_angular_vel = angular_velocity.copy()
            aug_angular_vel = self.apply_variable_delay(aug_angular_vel, delay, time_varying=(i > 0))
            aug_angular_vel = self.add_measurement_noise(aug_angular_vel)
            
            # Optionally add bias
            if np.random.rand() > 0.5:
                aug_angular_vel = self.add_bias_drift(aug_angular_vel)
            
            # Velocity typically has less noise and no significant delay
            aug_velocity = self.add_measurement_noise(velocity, noise_std=0.001)
            
            augmented_data.append({
                'angular_velocity': aug_angular_vel,
                'velocity': aug_velocity,
                'delay_applied': delay,
                'augmentation_id': i
            })
        
        return augmented_data
    
    def augment_dataset(
        self,
        dataset: Dict[str, np.ndarray],
        augmentation_factor: int = 3,
        include_original: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Augment entire dataset with IMU delays.
        
        Args:
            dataset: Original dataset with keys:
                - 'commanded_velocity' or 'velocity': velocity commands
                - 'commanded_angular_velocity': angular velocity commands  
                - 'true_angular_velocity' or 'angular_velocity': actual measurements
            augmentation_factor: Number of augmented copies per sample
            include_original: Whether to include original data
            
        Returns:
            Augmented dataset (all arrays expanded)
            
        Example:
            original_data = {
                'velocity': np.array([...]),
                'commanded_angular_velocity': np.array([...]),
                'true_angular_velocity': np.array([...])
            }
            augmented = augmenter.augment_dataset(original_data, augmentation_factor=5)
        """
        # Identify field names (support different naming conventions)
        vel_key = 'velocity' if 'velocity' in dataset else 'commanded_velocity'
        ang_vel_key = 'true_angular_velocity' if 'true_angular_velocity' in dataset else 'angular_velocity'
        cmd_ang_vel_key = 'commanded_angular_velocity'
        
        if vel_key not in dataset or ang_vel_key not in dataset:
            raise ValueError(f"Dataset must contain velocity and angular velocity fields. Found: {dataset.keys()}")
        
        original_velocity = dataset[vel_key]
        original_angular_velocity = dataset[ang_vel_key]
        original_commanded_ang_vel = dataset.get(cmd_ang_vel_key, original_angular_velocity)
        
        # Detect if data is 2D (batch) or 1D (single trajectory)
        is_batched = original_velocity.ndim > 1
        
        if is_batched:
            # Process each trajectory separately
            augmented_lists = {key: [] for key in dataset.keys()}
            augmented_lists['delay_applied'] = []
            
            for idx in range(len(original_velocity)):
                # Extract single trajectory
                traj_vel = original_velocity[idx]
                traj_ang_vel = original_angular_velocity[idx]
                
                # Augment
                augmented_trajs = self.augment_trajectory(
                    traj_ang_vel,
                    traj_vel,
                    augmentation_factor
                )
                
                # Include original?
                if include_original:
                    augmented_lists[vel_key].append(traj_vel)
                    augmented_lists[ang_vel_key].append(traj_ang_vel)
                    augmented_lists['delay_applied'].append(0.0)
                    if cmd_ang_vel_key in dataset:
                        augmented_lists[cmd_ang_vel_key].append(original_commanded_ang_vel[idx])
                
                # Add augmented versions
                for aug_data in augmented_trajs:
                    augmented_lists[vel_key].append(aug_data['velocity'])
                    augmented_lists[ang_vel_key].append(aug_data['angular_velocity'])
                    augmented_lists['delay_applied'].append(aug_data['delay_applied'])
                    if cmd_ang_vel_key in dataset:
                        augmented_lists[cmd_ang_vel_key].append(original_commanded_ang_vel[idx])
            
            # Convert lists to arrays
            return {key: np.array(val) for key, val in augmented_lists.items()}
        
        else:
            # Single trajectory
            augmented_trajs = self.augment_trajectory(
                original_angular_velocity,
                original_velocity,
                augmentation_factor
            )
            
            # Combine results
            augmented_dataset = {}
            
            if include_original:
                augmented_dataset = {key: [val] for key, val in dataset.items()}
                augmented_dataset['delay_applied'] = [0.0]
            else:
                augmented_dataset = {key: [] for key in dataset.keys()}
                augmented_dataset['delay_applied'] = []
            
            for aug_data in augmented_trajs:
                augmented_dataset[vel_key].append(aug_data['velocity'])
                augmented_dataset[ang_vel_key].append(aug_data['angular_velocity'])
                augmented_dataset['delay_applied'].append(aug_data['delay_applied'])
                if cmd_ang_vel_key in dataset:
                    augmented_dataset[cmd_ang_vel_key].append(original_commanded_ang_vel)
            
            return {key: np.array(val) for key, val in augmented_dataset.items()}
    
    def generate_delay_profiles(
        self,
        n_profiles: int = 10,
        duration: float = 10.0
    ) -> List[np.ndarray]:
        """
        Generate realistic delay profiles for testing.
        
        Args:
            n_profiles: Number of delay profiles to generate
            duration: Duration in seconds
            
        Returns:
            List of delay profile arrays
        """
        n_samples = int(duration * self.config.sample_rate)
        profiles = []
        
        for _ in range(n_profiles):
            # Base delay
            base_delay = np.random.uniform(self.config.min_delay, self.config.max_delay)
            
            # Add slow drift
            drift = np.linspace(0, np.random.uniform(-0.05, 0.05), n_samples)
            
            # Add fast variations
            variations = np.random.normal(0, self.config.delay_std, n_samples)
            
            # Combine
            profile = base_delay + drift + variations
            profile = np.clip(profile, self.config.min_delay, self.config.max_delay)
            
            profiles.append(profile)
        
        return profiles


def visualize_delay_effects(
    original_signal: np.ndarray,
    augmenter: IMUDelayAugmenter,
    n_augmentations: int = 3
):
    """
    Visualize effects of delay augmentation on a signal.
    
    Args:
        original_signal: Original signal to augment
        augmenter: Augmenter instance
        n_augmentations: Number of augmented versions to show
    """
    import matplotlib.pyplot as plt
    
    time = np.arange(len(original_signal)) / augmenter.config.sample_rate
    
    fig, axes = plt.subplots(n_augmentations + 1, 1, figsize=(12, 8), sharex=True)
    
    # Original signal
    axes[0].plot(time, original_signal, 'b-', lw=2, alpha=0.8)
    axes[0].set_ylabel('Original', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('IMU Delay Augmentation Comparison', fontsize=12)
    
    # Augmented versions
    for i in range(n_augmentations):
        delay = np.random.uniform(augmenter.config.min_delay, augmenter.config.max_delay)
        
        aug_signal = augmenter.apply_variable_delay(original_signal, delay)
        aug_signal = augmenter.add_measurement_noise(aug_signal)
        
        axes[i+1].plot(time, aug_signal, 'r-', lw=1.5, alpha=0.7)
        axes[i+1].plot(time, original_signal, 'b--', lw=1, alpha=0.3, label='Original')
        axes[i+1].set_ylabel(f'Delay {delay:.3f}s', fontsize=10)
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('imu_delay_augmentation_demo.png', dpi=300, bbox_inches='tight')
    print("💾 Saved visualization to: imu_delay_augmentation_demo.png")
    plt.show()


if __name__ == "__main__":
    # Demo usage
    print("Testing IMU Delay Augmentation\n")
    print("="*60)
    
    # Create synthetic signal
    duration = 5.0  # seconds
    sample_rate = 40.0  # Hz
    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Synthetic angular velocity (sine wave + drift)
    true_signal = 2.0 * np.sin(2 * np.pi * 0.5 * time) + 0.1 * time
    velocity = np.ones(n_samples) * 2.0  # Constant velocity
    
    # Create augmenter
    config = IMUDelayConfig(
        min_delay=0.10,
        max_delay=0.25,
        delay_std=0.02,
        sample_rate=sample_rate
    )
    augmenter = IMUDelayAugmenter(config)
    
    # Test single trajectory augmentation
    print("\n1. Augmenting single trajectory...")
    augmented = augmenter.augment_trajectory(true_signal, velocity, augmentation_factor=5)
    print(f"   Generated {len(augmented)} augmented versions")
    
    for i, aug in enumerate(augmented):
        print(f"   Version {i}: delay={aug['delay_applied']:.3f}s")
    
    # Test dataset augmentation
    print("\n2. Augmenting dataset...")
    dataset = {
        'velocity': velocity,
        'true_angular_velocity': true_signal,
        'commanded_angular_velocity': true_signal * 1.1
    }
    
    augmented_dataset = augmenter.augment_dataset(
        dataset,
        augmentation_factor=3,
        include_original=True
    )
    
    print(f"   Original size: {len(dataset['velocity'])}")
    print(f"   Augmented size: {len(augmented_dataset['velocity'])}")
    print(f"   Augmentation ratio: {len(augmented_dataset['velocity']) / len(dataset['velocity']):.1f}x")
    
    # Visualize
    print("\n3. Generating visualization...")
    visualize_delay_effects(true_signal, augmenter, n_augmentations=4)
    
    print("\n" + "="*60)
    print("✅ IMU Delay Augmentation test complete!")
