"""
RL Training Package - Reinforcement Learning for Boids Predator-Prey System

This package implements PPO (Proximal Policy Optimization) for fine-tuning 
the supervised learning checkpoint on the boids predator-prey simulation.

Features:
- PPO algorithm for continuous control
- Loads supervised learning checkpoint as starting point
- Uses existing simulation, rewards, and config systems
- Modular design for easy testing and debugging
- Extensive logging and monitoring

Usage:
    python rl/main.py --checkpoint checkpoints/best_model.pt --episodes 1000
"""

from .models import PPOModel
from .algorithms import PPOAlgorithm
from .training import BoidsEnvironment, PPOTrainer
from .evaluation import ModelEvaluator

__all__ = [
    'PPOModel',
    'PPOAlgorithm', 
    'BoidsEnvironment',
    'PPOTrainer',
    'ModelEvaluator'
]

__version__ = '1.0.0' 