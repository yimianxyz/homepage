"""
RL Training Package - Environment and training utilities

This package contains the RL environment wrapper and training loop
for the boids predator-prey system.
"""

from .environment import BoidsEnvironment
from .trainer import PPOTrainer
from .utils import TrainingUtils

__all__ = ['BoidsEnvironment', 'PPOTrainer', 'TrainingUtils'] 