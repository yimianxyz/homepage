"""
Simulation Package - Data conversion layer for neural network training

This package provides input and action processors that are 100% identical
between Python and JavaScript implementations.
"""

from .processors import InputProcessor, ActionProcessor
from .state_manager import StateManager
from .random_state_generator import RandomStateGenerator

__all__ = ['InputProcessor', 'ActionProcessor', 'StateManager', 'RandomStateGenerator'] 