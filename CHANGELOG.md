# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2024-10-14

### Added
- Comprehensive `.gitignore` for Python, macOS, and data files
- `requirements.txt` with all Python dependencies
- Organized code structure under `src/` directory
- Proper `__init__.py` files for package imports
- Extensive documentation in README.md
- CONTRIBUTING.md guidelines
- This CHANGELOG.md

### Changed
- Reorganized code into logical subdirectories:
  - `src/models/` - Model definitions and training
  - `src/data_processing/` - Data extraction and alignment
  - `src/visualization/` - Plotting scripts
  - `src/utils/` - Utility functions
- Updated all import statements to use proper package structure
- Refactored `make_predictions.py` with function-based approach
- Simplified `merge_csv.py` to remove redundant file operations
- Fixed duplicate imports in `plot_turning_data.py`
- Enhanced all scripts with proper docstrings and comments

### Fixed
- **Critical Bug**: Fixed test loop in `ikd_training.py` that was using training data
- Added `torch.no_grad()` context manager for evaluation
- Fixed incorrect import in `make_predictions.py` (was importing from wrong module)
- Corrected test data indexing bounds

### Removed
- Empty file `graph_loose_drift.py`
- Duplicate training script `train.py` (kept `ikd_training.py`)
- Redundant file I/O operations in `merge_csv.py`
- 100+ lines of commented code in `make_predictions.py`

### Improved
- Code readability with consistent formatting
- Documentation with detailed docstrings
- Error handling in prediction scripts
- Plot aesthetics with better labels and styling

## [Original] - 2023-04-25

Initial research implementation with:
- Basic IKD model architecture
- Training pipeline
- Data collection and processing scripts
- Visualization tools
- Test datasets (loose/tight, CW/CCW)
