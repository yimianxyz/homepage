"""
Teacher Policy - Simple pursuit behavior for supervised learning

This exactly matches the JavaScript teacher policy implementation
to provide consistent supervision signals.
"""

import torch
import math
from typing import Dict, List, Any
from python_simulation import CONSTANTS

class TeacherPolicy:
    """Simple pursuit teacher policy for supervised learning"""
    
    def __init__(self):
        # Use same force scaling as ActionProcessor for consistency  
        self.max_force = CONSTANTS.PREDATOR_MAX_FORCE
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get action based on structured inputs (matches JavaScript implementation)
        
        Args:
            structured_inputs: Same format as transformer input
            
        Returns:
            Action forces [force_x, force_y]
        """
        # If no boids, return zero force
        if not structured_inputs['boids'] or len(structured_inputs['boids']) == 0:
            return [0.0, 0.0]
        
        # Find the closest boid in the structured format
        target_boid = None
        min_distance = float('inf')
        
        for boid in structured_inputs['boids']:
            # Calculate distance to this boid
            distance = math.sqrt(boid['relX'] ** 2 + boid['relY'] ** 2)
            
            if distance < min_distance:
                min_distance = distance
                target_boid = boid
        
        # If no valid target found, return zero force
        if target_boid is None or min_distance < 0.001:
            return [0.0, 0.0]
        
        # Simple seeking: move toward target boid
        # Normalized direction to target
        dir_x = target_boid['relX'] / min_distance
        dir_y = target_boid['relY'] / min_distance
        
        # Apply force scaling (matches ActionProcessor)
        return [dir_x * self.max_force, dir_y * self.max_force]
    
    def get_action_batch(self, structured_inputs_batch: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Get actions for a batch of structured inputs
        
        Args:
            structured_inputs_batch: List of structured inputs
            
        Returns:
            Tensor of shape [batch_size, 2] with action forces
        """
        batch_actions = []
        
        for structured_inputs in structured_inputs_batch:
            action = self.get_action(structured_inputs)
            batch_actions.append(action)
        
        return torch.tensor(batch_actions, dtype=torch.float32)
    
    def get_normalized_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized action in [-1, 1] range (for direct comparison with model output)
        
        Args:
            structured_inputs: Same format as transformer input
            
        Returns:
            Normalized action forces [force_x, force_y] in [-1, 1] range
        """
        action = self.get_action(structured_inputs)
        
        # Normalize by max force to get [-1, 1] range
        normalized_action = [
            action[0] / self.max_force,
            action[1] / self.max_force
        ]
        
        # Clamp to ensure [-1, 1] range
        normalized_action[0] = max(-1.0, min(1.0, normalized_action[0]))
        normalized_action[1] = max(-1.0, min(1.0, normalized_action[1]))
        
        return normalized_action
    
    def get_normalized_action_batch(self, structured_inputs_batch: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Get normalized actions for a batch of structured inputs
        
        Args:
            structured_inputs_batch: List of structured inputs
            
        Returns:
            Tensor of shape [batch_size, 2] with normalized action forces in [-1, 1]
        """
        batch_actions = []
        
        for structured_inputs in structured_inputs_batch:
            action = self.get_normalized_action(structured_inputs)
            batch_actions.append(action)
        
        return torch.tensor(batch_actions, dtype=torch.float32)

def create_teacher_policy():
    """Create teacher policy instance"""
    policy = TeacherPolicy()
    print(f"Created TeacherPolicy:")
    print(f"  Max force: {policy.max_force}")
    print(f"  Strategy: Simple pursuit (closest boid)")
    return policy

if __name__ == "__main__":
    # Test teacher policy
    teacher = create_teacher_policy()
    
    # Test with dummy data
    test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    }
    
    action = teacher.get_action(test_input)
    normalized_action = teacher.get_normalized_action(test_input)
    
    print(f"Test action: {action}")
    print(f"Test normalized action: {normalized_action}")
    
    # Test batch processing
    batch_actions = teacher.get_normalized_action_batch([test_input, test_input])
    print(f"Batch actions shape: {batch_actions.shape}")
    print(f"Batch actions: {batch_actions}") 