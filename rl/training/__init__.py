"""
Training Package - PPO training for transformer models

This package provides PPO training functionality that loads pre-trained
supervised learning models and continues training with reinforcement learning.
"""

from .ppo_trainer import PPOTrainer
from .config import TrainingConfig

__all__ = ['PPOTrainer', 'TrainingConfig']