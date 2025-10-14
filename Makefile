# Makefile for IKD Autonomous Vehicle Drifting
# Provides convenient commands for common tasks

.PHONY: help install install-dev test train evaluate clean lint format docs benchmark

# Default target
help:
	@echo "IKD Autonomous Vehicle Drifting - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package and dependencies"
	@echo "  make install-dev    Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train         Train default model"
	@echo "  make train-loose   Train on loose drifting data"
	@echo "  make train-tight   Train on tight drifting data"
	@echo "  make evaluate      Evaluate best model"
	@echo "  make benchmark     Run comprehensive benchmarks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove generated files"
	@echo "  make validate      Validate training data"
	@echo "  make docs          Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x --ff

# Training
train:
	python train.py

train-loose:
	python train.py --config configs/loose_drifting.yaml

train-tight:
	python train.py --config configs/tight_drifting.yaml

train-circle:
	python train.py --config configs/circle_navigation.yaml

train-validate:
	python train.py --validate-data

# Evaluation
evaluate:
	python evaluate.py \
		--checkpoint experiments/ikd_baseline/checkpoints/best_model.pt \
		--plot-results \
		--save-predictions

evaluate-all:
	@echo "Evaluating all test datasets..."
	@for dataset in loose_ccw loose_cw tight_ccw tight_cw; do \
		echo "Evaluating $$dataset..."; \
		python evaluate.py \
			--checkpoint experiments/ikd_baseline/checkpoints/best_model.pt \
			--dataset dataset/$$dataset.csv \
			--output-dir evaluation_results/$$dataset; \
	done

# Simulation
simulate-circle:
	python simulate.py --mode circle --velocity 2.0 --curvature 0.7 --save-data

simulate-drift:
	python simulate.py --mode drift-loose --duration 15.0 --save-data

simulate-ikd:
	python simulate.py \
		--mode circle \
		--velocity 2.0 \
		--curvature 0.7 \
		--use-ikd \
		--model experiments/ikd_baseline/checkpoints/best_model.pt \
		--save-data

simulate-compare:
	python simulate.py \
		--mode circle \
		--velocity 2.0 \
		--curvature 0.7 \
		--compare \
		--model experiments/ikd_baseline/checkpoints/best_model.pt

test-simulator:
	pytest tests/test_simulator.py -v

# Benchmarking
benchmark:
	python scripts/run_benchmarks.py

benchmark-fast:
	python scripts/run_benchmarks.py --skip-training

# Code Quality
lint:
	@echo "Running flake8..."
	-flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503
	@echo "Running mypy..."
	-mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code with black..."
	black src/ tests/ train.py evaluate.py --line-length=120

format-check:
	black src/ tests/ train.py evaluate.py --check --line-length=120

# Data Validation
validate:
	python train.py --validate-data --config configs/default.yaml

validate-all:
	@echo "Validating all datasets..."
	@for dataset in dataset/*.csv; do \
		echo "Validating $$dataset..."; \
		python -c "from src.data_processing.validators import DataValidator; \
			validator = DataValidator(); \
			result = validator.validate_csv('$$dataset'); \
			print('✓ Valid' if result.is_valid else '✗ Invalid')"; \
	done

# Cleaning
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf dist build
	@echo "Clean complete!"

clean-experiments:
	@echo "WARNING: This will delete all experiment results!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf experiments/; \
		echo "Experiments deleted."; \
	fi

# Documentation
docs:
	@echo "Documentation is in docs/ folder"
	@echo "  - GETTING_STARTED.md: Tutorial and quick start"
	@echo "  - REPRODUCING_PAPER.md: Paper reproduction guide"
	@echo ""
	@echo "To view README in browser:"
	@echo "  open README.md  # macOS"
	@echo "  xdg-open README.md  # Linux"

# Quick start
quickstart: install train evaluate
	@echo ""
	@echo "✓ Quickstart complete!"
	@echo "  - Package installed"
	@echo "  - Model trained"
	@echo "  - Results evaluated"
	@echo ""
	@echo "Check results in: experiments/ikd_baseline/"

# CI/CD simulation
ci: clean install-dev test lint
	@echo ""
	@echo "✓ CI checks passed!"

# Full pipeline
full-pipeline: clean install-dev test train evaluate benchmark
	@echo ""
	@echo "✓ Full pipeline complete!"
	@echo ""
	@echo "Results:"
	@echo "  - Tests: pytest results"
	@echo "  - Training: experiments/ikd_baseline/"
	@echo "  - Evaluation: evaluation_results/"
	@echo "  - Benchmarks: benchmark_results.json"
