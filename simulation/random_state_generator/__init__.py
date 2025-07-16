"""
Random State Generator Package - Generate random boid and predator states

This package provides random state generation for the simulation system.
Both Python and JavaScript versions are maintained for 100% identical behavior.
"""

from .random_state_generator import RandomStateGenerator, generate_random_state

__all__ = ['RandomStateGenerator', 'generate_random_state'] 