"""
drift_gym: Research-Grade Drift Control Environment

A Gymnasium environment for autonomous vehicle drift control with realistic
sensors, state estimation, and comprehensive evaluation tools.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="drift-gym",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Research-grade drift control environment with realistic sensors and state estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/drift-gym",
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
        ],
        "experiments": [
            "stable-baselines3>=2.0.0",
            "tensorboard>=2.0.0",
            "seaborn>=0.13.0",
        ],
    },
    include_package_data=True,
    package_data={
        "drift_gym": ["*.yaml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "drift-gym-benchmark=experiments.benchmark_algorithms:main",
            "drift-gym-ablation=experiments.ablation_study:main",
        ],
    },
)
