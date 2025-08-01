"""
RL Training Package - Reinforcement Learning training for boid simulation

This package provides PPO training capabilities that build on the existing
simulation, reward, and configuration systems. It loads supervised learning
models and fine-tunes them with reinforcement learning.

Components:
- environment: Gym environment wrapper for the simulation
- models: Model loading and integration with stable-baselines3
- training: PPO training scripts and configuration
- utils: Utility functions and helpers
- tests: Comprehensive testing suite
"""

from .environment import BoidEnvironment
from .training import PPOTrainer
from .models import TransformerModelLoader

__version__ = "1.0.0"
__all__ = ['BoidEnvironment', 'PPOTrainer', 'TransformerModelLoader']