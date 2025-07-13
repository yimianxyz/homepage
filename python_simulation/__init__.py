"""
Python Simulation Package - Exact port of JavaScript simulation environment

This package provides a 100% exact replica of the JavaScript simulation
environment for training neural networks in Python while maintaining
compatibility with the browser-based inference.
"""

from .constants import CONSTANTS
from .vector import Vector
from .boid import Boid
from .predator import Predator
from .simulation import Simulation
from .input_processor import InputProcessor
from .action_processor import ActionProcessor

__all__ = [
    'CONSTANTS',
    'Vector',
    'Boid',
    'Predator',
    'Simulation',
    'InputProcessor',
    'ActionProcessor'
] 