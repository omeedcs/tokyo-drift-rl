"""
Setup script for IKD Autonomous Vehicle Drifting package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ikd-drifting",
    version="1.0.0",
    author="Mihir Suvarna, Omeed Tehrani",
    author_email="msuvarna@cs.utexas.edu, omeed@cs.utexas.edu",
    description="Learning Inverse Kinodynamics for Autonomous Vehicle Drifting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msuv08/autonomous-vehicle-drifting",
    project_urls={
        "Bug Tracker": "https://github.com/msuv08/autonomous-vehicle-drifting/issues",
        "Paper": "https://arxiv.org/abs/2402.14928",
        "Documentation": "https://github.com/msuv08/autonomous-vehicle-drifting#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International License (CC BY 4.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0,<2.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0.0",
        "scikit-learn>=1.3.0",
        "gymnasium>=0.29.0",
        "gym==0.26.2",
        "pygame>=2.5.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "experiment": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ikd-train=train:main",
            "ikd-evaluate=evaluate:main",
        ],
    },
)
