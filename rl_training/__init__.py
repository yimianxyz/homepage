"""
RL Training Package - PPO fine-tuning for pretrained transformer policies

This package provides a complete PPO training pipeline that fine-tunes
pretrained transformer models using reinforcement learning.

Main Components:
- PPOEnvironment: Gym wrapper for existing simulation
- PPOTransformerPolicy: Actor-critic policy with pretrained initialization
- ObservationWrapper: Handles structured observations
- PPOPolicyWrapper: Adapter for existing evaluation system
- Training Pipeline: Complete training and evaluation workflow

Usage:
    from rl_training.train_ppo import train_ppo_model
    
    model, results = train_ppo_model(
        pretrained_path="checkpoints/best_model.pt",
        total_timesteps=100000
    )
"""

from .ppo_environment import create_boids_environment, BoidsEnvironment
from .observation_wrapper import (
    create_wrapped_environment, 
    StructuredObservationWrapper,
    create_custom_policy_class
)
from .ppo_policy_wrapper import (
    create_policy_wrapper, 
    evaluate_ppo_model,
    PPOPolicyWrapper,
    PPOTransformerPolicyWrapper
)
from .train_ppo import train_ppo_model

__all__ = [
    'create_boids_environment',
    'BoidsEnvironment', 
    'create_wrapped_environment',
    'StructuredObservationWrapper',
    'create_custom_policy_class',
    'create_policy_wrapper',
    'evaluate_ppo_model', 
    'PPOPolicyWrapper',
    'PPOTransformerPolicyWrapper',
    'train_ppo_model'
]

__version__ = "1.0.0"