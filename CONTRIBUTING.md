# Contributing Guidelines

Thank you for your interest in contributing to this autonomous vehicle drifting research project!

## Code Style

### Python
- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### File Organization
```
src/
├── models/           # Neural network models and training
├── data_processing/  # Data extraction and preprocessing
├── visualization/    # Plotting and analysis scripts
└── utils/           # Helper utilities
```

## Adding New Features

### 1. Data Processing
If adding new data processing scripts:
- Place in `src/data_processing/`
- Document input/output format
- Handle edge cases gracefully

### 2. Model Architectures
If proposing new models:
- Extend or create new classes in `src/models/`
- Document architecture clearly
- Include training script
- Report baseline metrics

### 3. Visualization
If adding plots or analysis:
- Place in `src/visualization/`
- Use consistent styling
- Label axes clearly
- Save figures with descriptive names

## Testing

Before submitting changes:
1. Test on sample data
2. Verify all imports work correctly
3. Check for Python syntax errors
4. Ensure documentation is updated

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update documentation
5. Test your changes
6. Submit a pull request

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions
