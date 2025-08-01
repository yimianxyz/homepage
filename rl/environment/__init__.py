"""
Environment Package - Gym environment wrapper for boid simulation

This package provides a Gym-compatible environment that wraps the existing
simulation system, making it compatible with stable-baselines3 and other
RL frameworks.
"""

from .boid_env import BoidEnvironment

__all__ = ['BoidEnvironment']