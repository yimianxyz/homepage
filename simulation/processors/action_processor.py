"""
Action Processor - Universal policy output interface

This is a data conversion layer that converts policy outputs to game actions.
Works with any policy type (neural networks, rule-based, human, etc.).
This MUST match exactly with the JavaScript implementation.
"""

from typing import List
from config.constants import CONSTANTS

class ActionProcessor:
    def __init__(self):
        # Scale neural outputs to match actual predator force limits
        self.force_scale = CONSTANTS.PREDATOR_MAX_FORCE
    
    def process_action(self, policy_outputs: List[float]) -> List[float]:
        """
        Convert policy outputs to game actions
        
        This universal interface works with any policy type:
        - Neural networks: normalized outputs [-1, 1]
        - Rule-based policies: computed steering forces
        - Human policies: input-based actions
        - Hybrid policies: combined outputs
        
        Args:
            policy_outputs: Policy outputs [x, y] in [-1, 1] range
            
        Returns:
            Action forces [force_x, force_y] in game units
        """
        return [
            policy_outputs[0] * self.force_scale,
            policy_outputs[1] * self.force_scale
        ] 