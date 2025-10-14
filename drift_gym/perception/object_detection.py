"""
Perception Pipeline with Object Detection

Simulates realistic object detection with false positives, false negatives,
and uncertainty estimation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ObjectClass(Enum):
    """Object classification types."""
    VEHICLE = 0
    PEDESTRIAN = 1
    OBSTACLE = 2
    UNKNOWN = 3


@dataclass
class Detection:
    """Single object detection."""
    position: np.ndarray  # [x, y] in vehicle frame
    velocity: np.ndarray  # [vx, vy] in vehicle frame
    size: Tuple[float, float]  # (length, width)
    object_class: ObjectClass
    confidence: float  # 0.0 to 1.0
    covariance: np.ndarray  # 2x2 position uncertainty
    is_false_positive: bool = False


class ObjectDetector:
    """
    Simulates object detection pipeline with realistic errors.
    
    Implements:
    - Detection range limits
    - False positives (clutter, ghosts)
    - False negatives (missed detections)
    - Confidence scores
    - Position uncertainty
    - Occlusion handling
    """
    
    def __init__(
        self,
        max_range: float = 50.0,  # meters
        fov_angle: float = np.pi,  # radians (180 degrees)
        false_positive_rate: float = 0.05,  # 5% FP rate
        false_negative_rate: float = 0.10,  # 10% FN rate
        position_noise_std: float = 0.3,  # meters
        confidence_threshold: float = 0.5,
        seed: Optional[int] = None
    ):
        self.max_range = max_range
        self.fov_angle = fov_angle
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.position_noise_std = position_noise_std
        self.confidence_threshold = confidence_threshold
        
        self.rng = np.random.RandomState(seed)
        
    def detect_objects(
        self,
        true_objects: List[dict],
        ego_position: np.ndarray,
        ego_heading: float
    ) -> List[Detection]:
        """
        Detect objects with realistic errors.
        
        Args:
            true_objects: List of dicts with 'position', 'velocity', 'size', 'class'
            ego_position: Ego vehicle position [x, y]
            ego_heading: Ego vehicle heading (radians)
            
        Returns:
            List of Detections (may include false positives, miss some objects)
        """
        detections = []
        
        # Transform objects to ego frame
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        for obj in true_objects:
            # Transform to ego frame
            rel_pos = obj['position'] - ego_position
            ego_frame_pos = rotation_matrix @ rel_pos
            
            # Check if in range
            distance = np.linalg.norm(ego_frame_pos)
            if distance > self.max_range:
                continue
            
            # Check if in FOV
            angle = np.arctan2(ego_frame_pos[1], ego_frame_pos[0])
            if abs(angle) > self.fov_angle / 2:
                continue
            
            # False negative (missed detection)
            # Higher probability at longer range
            fn_prob = self.false_negative_rate * (1.0 + distance / self.max_range)
            if self.rng.random() < fn_prob:
                continue  # Miss this object
            
            # Add position noise
            noise = self.rng.randn(2) * self.position_noise_std
            noisy_position = ego_frame_pos + noise
            
            # Transform velocity to ego frame
            if 'velocity' in obj and obj['velocity'] is not None:
                ego_frame_vel = rotation_matrix @ obj['velocity']
                # Add velocity noise
                vel_noise = self.rng.randn(2) * (self.position_noise_std * 0.5)
                noisy_velocity = ego_frame_vel + vel_noise
            else:
                noisy_velocity = np.zeros(2)
            
            # Compute confidence (decreases with range)
            base_confidence = 0.95 - (distance / self.max_range) * 0.3
            # Add random variation
            confidence = np.clip(base_confidence + self.rng.randn() * 0.1, 0.0, 1.0)
            
            # Uncertainty increases with range
            uncertainty = self.position_noise_std**2 * (1.0 + distance / self.max_range)
            covariance = np.eye(2) * uncertainty
            
            # Determine class
            obj_class = obj.get('class', ObjectClass.UNKNOWN)
            
            # Size with some uncertainty
            true_size = obj.get('size', (1.0, 1.0))
            noisy_size = (
                max(0.5, true_size[0] + self.rng.randn() * 0.2),
                max(0.5, true_size[1] + self.rng.randn() * 0.2)
            )
            
            detection = Detection(
                position=noisy_position,
                velocity=noisy_velocity,
                size=noisy_size,
                object_class=obj_class,
                confidence=confidence,
                covariance=covariance,
                is_false_positive=False
            )
            
            detections.append(detection)
        
        # Add false positives (clutter)
        num_false_positives = self.rng.poisson(self.false_positive_rate * 5)
        
        for _ in range(num_false_positives):
            # Random position in FOV
            range_fp = self.rng.uniform(5.0, self.max_range * 0.8)
            angle_fp = self.rng.uniform(-self.fov_angle/2, self.fov_angle/2)
            
            fp_position = np.array([
                range_fp * np.cos(angle_fp),
                range_fp * np.sin(angle_fp)
            ])
            
            # False positives have lower confidence
            fp_confidence = self.rng.uniform(self.confidence_threshold * 0.5, self.confidence_threshold * 1.2)
            
            # Random velocity
            fp_velocity = self.rng.randn(2) * 2.0
            
            false_detection = Detection(
                position=fp_position,
                velocity=fp_velocity,
                size=(self.rng.uniform(0.5, 2.0), self.rng.uniform(0.5, 2.0)),
                object_class=ObjectClass.UNKNOWN,
                confidence=fp_confidence,
                covariance=np.eye(2) * (self.position_noise_std * 3)**2,
                is_false_positive=True
            )
            
            detections.append(false_detection)
        
        # Filter by confidence threshold
        detections = [d for d in detections if d.confidence >= self.confidence_threshold]
        
        return detections
    
    def reset(self):
        """Reset detector state."""
        pass  # Stateless for now


class TrackingFilter:
    """
    Simple tracking filter for associating detections over time.
    
    Uses nearest-neighbor association and exponential smoothing.
    """
    
    def __init__(
        self,
        association_threshold: float = 2.0,  # meters
        alpha: float = 0.3  # Smoothing factor
    ):
        self.association_threshold = association_threshold
        self.alpha = alpha
        
        self.tracks: List[dict] = []
        self.next_track_id = 0
        
    def update(self, detections: List[Detection], dt: float) -> List[dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: New detections
            dt: Time since last update
            
        Returns:
            List of tracks with 'id', 'position', 'velocity', 'age', 'confidence'
        """
        # Predict tracks forward
        for track in self.tracks:
            track['position'] += track['velocity'] * dt
            track['age'] += dt
            track['confidence'] *= 0.95  # Decay confidence if not updated
        
        # Associate detections to tracks
        unmatched_detections = list(detections)
        matched_tracks = set()
        
        for detection in detections:
            best_track_idx = None
            best_distance = self.association_threshold
            
            for idx, track in enumerate(self.tracks):
                if idx in matched_tracks:
                    continue
                
                distance = np.linalg.norm(detection.position - track['position'])
                
                if distance < best_distance:
                    best_distance = distance
                    best_track_idx = idx
            
            if best_track_idx is not None:
                # Update track with detection
                track = self.tracks[best_track_idx]
                
                # Exponential smoothing
                track['position'] = (
                    self.alpha * detection.position +
                    (1 - self.alpha) * track['position']
                )
                track['velocity'] = (
                    self.alpha * detection.velocity +
                    (1 - self.alpha) * track['velocity']
                )
                track['confidence'] = max(track['confidence'], detection.confidence)
                track['age'] = 0.0  # Reset age on update
                
                matched_tracks.add(best_track_idx)
                unmatched_detections.remove(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            if detection.confidence > 0.7:  # Higher threshold for new tracks
                new_track = {
                    'id': self.next_track_id,
                    'position': detection.position.copy(),
                    'velocity': detection.velocity.copy(),
                    'size': detection.size,
                    'class': detection.object_class,
                    'confidence': detection.confidence,
                    'age': 0.0
                }
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] < 2.0 and t['confidence'] > 0.3]
        
        return self.tracks
    
    def reset(self):
        """Reset all tracks."""
        self.tracks = []
        self.next_track_id = 0


def test_perception():
    """Test perception pipeline."""
    print("Testing Perception Pipeline")
    print("=" * 60)
    
    detector = ObjectDetector(max_range=30.0, seed=42)
    tracker = TrackingFilter()
    
    # Simulate objects
    true_objects = [
        {
            'position': np.array([10.0, 5.0]),
            'velocity': np.array([2.0, 0.0]),
            'size': (4.0, 2.0),
            'class': ObjectClass.VEHICLE
        },
        {
            'position': np.array([15.0, -3.0]),
            'velocity': np.array([0.0, 1.0]),
            'size': (0.5, 0.5),
            'class': ObjectClass.PEDESTRIAN
        }
    ]
    
    ego_position = np.array([0.0, 0.0])
    ego_heading = 0.0
    
    print("\nDetections over 5 timesteps:")
    for step in range(5):
        detections = detector.detect_objects(true_objects, ego_position, ego_heading)
        tracks = tracker.update(detections, dt=0.1)
        
        print(f"\nStep {step}:")
        print(f"  Detections: {len(detections)} (FP: {sum(d.is_false_positive for d in detections)})")
        print(f"  Tracks: {len(tracks)}")
        
        for track in tracks:
            print(f"    Track {track['id']}: pos={track['position']}, conf={track['confidence']:.2f}")
        
        # Move objects slightly
        for obj in true_objects:
            obj['position'] += obj['velocity'] * 0.1


if __name__ == "__main__":
    test_perception()
