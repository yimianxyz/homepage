"""
Random Policy - Pure random action generation

This policy generates completely random actions uniformly distributed in [-1, 1].
It serves as a baseline for comparison and testing. Designed to be 100% identical
between Python and JavaScript versions.

Interface:
- Input: structured_inputs (ignored for random policy)
- Output: normalized policy outputs [x, y] in [-1, 1] range
- The ActionProcessor handles scaling to game forces
"""

import random
from typing import Dict, List, Any

class RandomPolicy:
    """Pure random policy that generates uniform random actions"""
    
    def __init__(self, seed: int = None):
        """
        Initialize random policy
        
        Args:
            seed: Optional seed for reproducible random actions
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized random policy outputs (compatible with ActionProcessor)
        
        Args:
            structured_inputs: Same format as universal policy input (ignored)
            - context: {canvasWidth: float, canvasHeight: float}
            - predator: {velX: float, velY: float}
            - boids: [{relX: float, relY: float, velX: float, velY: float}, ...]
            
        Returns:
            Random normalized policy outputs [x, y] in [-1, 1] range
        """
        # Generate uniform random values in [-1, 1] range
        return [
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0)
        ]
    
    def get_normalized_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized action in [-1, 1] range (deprecated - use get_action instead)
        
        Args:
            structured_inputs: Same format as universal policy input (ignored)
            
        Returns:
            Random normalized policy outputs [x, y] in [-1, 1] range
        """
        # Now get_action already returns normalized values
        return self.get_action(structured_inputs)

def create_random_policy(seed: int = None):
    """Create random policy instance"""
    policy = RandomPolicy(seed)
    print(f"Created RandomPolicy:")
    print(f"  Output: Random normalized outputs in [-1, 1] range")
    print(f"  Strategy: Pure random uniform distribution")
    print(f"  Seed: {seed if seed else 'None (non-deterministic)'}")
    return policy

if __name__ == "__main__":
    # Test random policy
    policy = create_random_policy(seed=42)
    
    # Test with dummy data (will be ignored)
    test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    }
    
    # Generate a few random actions
    print(f"Random actions:")
    for i in range(5):
        action = policy.get_action(test_input)
        print(f"  Action {i+1}: [{action[0]:.3f}, {action[1]:.3f}]")
    
    print(f"Note: Use ActionProcessor to convert to game forces")