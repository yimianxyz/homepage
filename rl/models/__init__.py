"""
Models Package - Model loading and integration for RL training

This package provides functionality to load supervised learning models
and integrate them with stable-baselines3 for reinforcement learning.
"""

from .transformer_model import TransformerModel, TransformerModelLoader
from .policy_wrapper import TransformerPolicy

__all__ = ['TransformerModel', 'TransformerModelLoader', 'TransformerPolicy']