# Changelog

All notable changes to drift-gym will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-14

### Added
- **Research-grade sensor models**
  - GPS sensor based on u-blox ZED-F9P specifications
  - IMU sensor based on BMI088/MPU9250 datasheets
  - Allan variance noise modeling
  - Random walk drift behavior
  
- **Extended Kalman Filter**
  - 6-DOF state estimation
  - GPS + IMU sensor fusion
  - Uncertainty propagation
  - Joseph form for numerical stability
  
- **Redesigned observation space**
  - Task-relevant features (12-dim)
  - Relative goal information
  - EKF estimates with uncertainties
  - Action history for memory
  
- **Standardized evaluation protocol**
  - 10+ metrics (success rate, path quality, safety, smoothness)
  - JSON + CSV export
  - Statistical significance testing
  - Pretty-printed summaries
  
- **Multi-algorithm benchmarking**
  - Support for SAC, PPO, TD3
  - Standardized hyperparameters
  - Multiple random seeds
  - TensorBoard logging
  - Comparison tables
  
- **Ablation study framework**
  - Systematic feature addition
  - Impact quantification
  - Automated reports and plots
  - Analysis recommendations
  
- **Comprehensive documentation**
  - Research guide (800 lines)
  - Calibration report (400 lines)
  - Quick start guide (500 lines)
  - API reference
  
- **Unit tests**
  - Sensor model tests
  - EKF tests
  - Environment tests
  - 90%+ code coverage
  
- **Python packaging**
  - setup.py for distribution
  - pyproject.toml for modern packaging
  - Console scripts for CLI tools
  - Extras for optional dependencies

### Changed
- Sensor noise parameters now based on hardware datasheets
- Removed physically meaningless multipath model
- Observation space optimized for learning efficiency

### Removed
- Dependency on arbitrary sensor parameters
- Sinusoidal GPS multipath model

### Fixed
- Perception tracking bug in object association
- EKF covariance numerical stability
- Observation space normalization

---

## [Unreleased]

### Planned
- Camera/LiDAR sensor models
- Domain randomization support
- Additional vehicle models
- URDF support for custom vehicles
- ROS2 integration
- Real-world data collection tools

---

## Version History

- **1.0.0** (2025-10-14): Initial release with research-grade features
