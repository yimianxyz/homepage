"""
Closest Pursuit Policy - Simple pursuit behavior targeting the closest boid

This policy implements a greedy pursuit strategy where the predator always
moves toward the closest boid. It's designed to be 100% identical between
Python and JavaScript versions.

Interface:
- Input: structured_inputs (same format as universal policy input)
- Output: normalized policy outputs [x, y] in [-1, 1] range
- The ActionProcessor handles scaling to game forces
"""

import math
from typing import Dict, List, Any

class ClosestPursuitPolicy:
    """Greedy pursuit policy that always targets the closest boid"""
    
    def __init__(self):
        # No max_force needed - ActionProcessor handles scaling
        pass
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized policy outputs (compatible with ActionProcessor)
        
        Args:
            structured_inputs: Same format as universal policy input
            - context: {canvasWidth: float, canvasHeight: float}
            - predator: {velX: float, velY: float}
            - boids: [{relX: float, relY: float, velX: float, velY: float}, ...]
            
        Returns:
            Normalized policy outputs [x, y] in [-1, 1] range
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
        # Normalized direction to target (already in [-1, 1] range)
        dir_x = target_boid['relX'] / min_distance
        dir_y = target_boid['relY'] / min_distance
        
        # Return normalized policy outputs - ActionProcessor handles scaling
        return [dir_x, dir_y]
    
    def get_normalized_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized action in [-1, 1] range (deprecated - use get_action instead)
        
        Args:
            structured_inputs: Same format as universal policy input
            
        Returns:
            Normalized policy outputs [x, y] in [-1, 1] range
        """
        # Now get_action already returns normalized values
        return self.get_action(structured_inputs)

def create_closest_pursuit_policy():
    """Create closest pursuit policy instance"""
    policy = ClosestPursuitPolicy()
    print(f"Created ClosestPursuitPolicy:")
    print(f"  Output: Normalized policy outputs in [-1, 1] range")
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
    
    policy_output = policy.get_action(test_input)
    
    print(f"Policy output (normalized): {policy_output}")
    print(f"Note: Use ActionProcessor to convert to game forces") 