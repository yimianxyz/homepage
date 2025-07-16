"""
Closest Pursuit Policy - Simple pursuit behavior targeting the closest boid

This policy implements a greedy pursuit strategy where the predator always
moves toward the closest boid. It's designed to be 100% identical between
Python and JavaScript versions.

Interface:
- Input: structured_inputs (same format as transformer input)
- Output: action forces [force_x, force_y]
"""

import math
from typing import Dict, List, Any

class ClosestPursuitPolicy:
    """Greedy pursuit policy that always targets the closest boid"""
    
    def __init__(self):
        # Constants that must match JavaScript exactly
        self.PREDATOR_MAX_FORCE = 0.001
        self.PREDATOR_FORCE_SCALE = 200
        
        # Calculate max force (matches ActionProcessor)
        self.max_force = self.PREDATOR_MAX_FORCE * self.PREDATOR_FORCE_SCALE
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get action based on structured inputs (matches JavaScript implementation)
        
        Args:
            structured_inputs: Same format as transformer input
            - context: {canvasWidth: float, canvasHeight: float}
            - predator: {velX: float, velY: float}
            - boids: [{relX: float, relY: float, velX: float, velY: float}, ...]
            
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

def create_closest_pursuit_policy():
    """Create closest pursuit policy instance"""
    policy = ClosestPursuitPolicy()
    print(f"Created ClosestPursuitPolicy:")
    print(f"  Max force: {policy.max_force}")
    print(f"  Strategy: Greedy pursuit (always targets closest boid)")
    return policy

if __name__ == "__main__":
    # Test closest pursuit policy
    policy = create_closest_pursuit_policy()
    
    # Test with dummy data
    test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    }
    
    action = policy.get_action(test_input)
    normalized_action = policy.get_normalized_action(test_input)
    
    print(f"Test action: {action}")
    print(f"Test normalized action: {normalized_action}") 