"""
Action Processor - Exact port of JavaScript src/ai/action_processor.js

Neural output to game actions. Converts neural network outputs to game forces.
This MUST match exactly with the JavaScript implementation.
"""

from typing import List
from .constants import CONSTANTS

class ActionProcessor:
    def __init__(self):
        # Scale neural outputs to match actual predator force limits
        self.force_scale = CONSTANTS.PREDATOR_MAX_FORCE
    
    def process_action(self, neural_outputs: List[float]) -> List[float]:
        """
        Convert neural network outputs to game actions
        
        Args:
            neural_outputs: Neural network outputs [x, y] in [-1, 1] range
            
        Returns:
            Action forces [force_x, force_y] in game units
        """
        return [
            neural_outputs[0] * self.force_scale,
            neural_outputs[1] * self.force_scale
        ] 