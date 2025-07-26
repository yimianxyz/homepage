"""
RL Models Package - PPO-enhanced transformer models

This package contains the PPO actor-critic model that builds upon 
the supervised learning transformer checkpoint.
"""

from .ppo_model import PPOModel

__all__ = ['PPOModel'] 