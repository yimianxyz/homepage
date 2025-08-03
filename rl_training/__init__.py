"""
RL Training Package - PPO finetuning for transformer models using TorchRL

This package provides a clean interface between the existing simulation
and TorchRL's PPO implementation for continued training from SL checkpoints.
"""

from .rl_environment import BoidsEnvironment
from .rl_policy import TransformerRLPolicy
from .ppo_config import PPOConfig

__all__ = ['BoidsEnvironment', 'TransformerRLPolicy', 'PPOConfig']