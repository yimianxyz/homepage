"""
Processors Package - Data conversion layer for neural network training

This package provides input and action processors that are 100% identical
between Python and JavaScript implementations.
"""

from .input_processor import InputProcessor
from .action_processor import ActionProcessor

__all__ = ['InputProcessor', 'ActionProcessor'] 