"""
Random State Generator - Generate random boid and predator states for simulation

This generates random initial states in the format expected by StateManager.
The Python and JavaScript versions MUST be 100% identical.
"""

import random
import math
from typing import Dict, List, Any

# Import centralized constants
from config.constants import CONSTANTS

class RandomStateGenerator:
    """Generate random boid and predator states for simulation"""
    
    def __init__(self, seed: int = None):
        """
        Initialize random state generator
        
        Args:
            seed: Optional seed for reproducible random generation
        """
        self.seed = seed
        if seed is not None:
            # Use seeded random number generator if provided
            self.random = self._seeded_random(seed)
        else:
            # Use Python's random for unseeded generation
            self.random = random.random
    
    def generate_random_state(self, 
                            num_boids: int, 
                            canvas_width: float, 
                            canvas_height: float) -> Dict[str, Any]:
        """
        Generate completely random state for simulation
        
        Args:
            num_boids: Number of boids to generate
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            State dict in StateManager format:
            {
                'boids_states': [
                    {
                        'position': {'x': float, 'y': float},
                        'velocity': {'x': float, 'y': float}
                    },
                    ...
                ],
                'predator_state': {
                    'position': {'x': float, 'y': float},
                    'velocity': {'x': float, 'y': float}
                },
                'canvas_width': float,
                'canvas_height': float
            }
        """
        # Generate random boids
        boids_states = []
        for i in range(num_boids):
            boid_state = self._generate_random_boid(canvas_width, canvas_height)
            boids_states.append(boid_state)
        
        # Generate random predator
        predator_state = self._generate_random_predator(canvas_width, canvas_height)
        
        return {
            'boids_states': boids_states,
            'predator_state': predator_state,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height
        }
    
    def _generate_random_boid(self, canvas_width: float, canvas_height: float) -> Dict[str, Any]:
        """
        Generate single random boid state
        
        Args:
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            Boid state dict
        """
        # Random position within canvas bounds
        position = {
            'x': self.random() * canvas_width,
            'y': self.random() * canvas_height
        }
        
        # Random velocity with random direction and speed
        angle = self.random() * 2 * math.pi
        speed = 0.5 + self.random() * (CONSTANTS.BOID_MAX_SPEED - 0.5)
        velocity = {
            'x': math.cos(angle) * speed,
            'y': math.sin(angle) * speed
        }
        
        return {
            'position': position,
            'velocity': velocity
        }
    
    def _generate_random_predator(self, canvas_width: float, canvas_height: float) -> Dict[str, Any]:
        """
        Generate random predator state
        
        Args:
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            Predator state dict
        """
        # Random position within canvas bounds
        position = {
            'x': self.random() * canvas_width,
            'y': self.random() * canvas_height
        }
        
        # Random velocity with random direction and speed
        angle = self.random() * 2 * math.pi
        speed = 0.5 + self.random() * (CONSTANTS.PREDATOR_MAX_SPEED - 0.5)
        velocity = {
            'x': math.cos(angle) * speed,
            'y': math.sin(angle) * speed
        }
        
        return {
            'position': position,
            'velocity': velocity
        }
    
    def _seeded_random(self, seed: int):
        """
        Simple seeded random number generator (LCG - Linear Congruential Generator)
        This ensures reproducible random sequences when seed is provided
        Matches the JavaScript implementation exactly
        
        Args:
            seed: Seed value
            
        Returns:
            Random number generator function
        """
        m = 0x80000000  # 2**31
        a = 1103515245
        c = 12345
        
        state = seed if seed else int(random.random() * (m - 1))
        
        def next_random():
            nonlocal state
            state = (a * state + c) % m
            return state / (m - 1)
        
        return next_random

def generate_random_state(num_boids: int, 
                         canvas_width: float, 
                         canvas_height: float, 
                         seed: int = None) -> Dict[str, Any]:
    """
    Convenience function to generate random state
    
    Args:
        num_boids: Number of boids to generate
        canvas_width: Canvas width
        canvas_height: Canvas height
        seed: Optional seed for reproducible generation
        
    Returns:
        Random state dict in StateManager format
    """
    generator = RandomStateGenerator(seed)
    return generator.generate_random_state(num_boids, canvas_width, canvas_height)

# Example usage
if __name__ == "__main__":
    # Test random state generation
    state = generate_random_state(num_boids=10, canvas_width=800, canvas_height=600, seed=42)
    
    print("Generated state:")
    print(f"  Boids: {len(state['boids_states'])}")
    print(f"  Canvas: {state['canvas_width']}x{state['canvas_height']}")
    print(f"  First boid position: {state['boids_states'][0]['position']}")
    print(f"  First boid velocity: {state['boids_states'][0]['velocity']}")
    print(f"  Predator position: {state['predator_state']['position']}")
    print(f"  Predator velocity: {state['predator_state']['velocity']}") 