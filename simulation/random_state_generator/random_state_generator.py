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
        Generate completely random (scattered) state for simulation (default method)
        
        Args:
            num_boids: Number of boids to generate
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            State dict in StateManager format
        """
        return self.generate_scattered_state(num_boids, canvas_width, canvas_height)
    
    def generate_clustered_state(self, 
                               num_boids: int, 
                               canvas_width: float, 
                               canvas_height: float) -> Dict[str, Any]:
        """
        Generate clustered state - boids clustered around a random starting point
        
        Args:
            num_boids: Number of boids to generate
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            State dict in StateManager format
        """
        boids_states = []
        start_x = math.floor(self.random() * canvas_width)
        start_y = math.floor(self.random() * canvas_height)
        
        # Create boids clustered around a random starting point
        for i in range(num_boids):
            random_angle = self.random() * 2 * math.pi
            boids_states.append({
                'id': i,  # Unique ID for each boid
                'position': {
                    'x': start_x + (self.random() - 0.5) * 100,
                    'y': start_y + (self.random() - 0.5) * 100
                },
                'velocity': {
                    'x': math.cos(random_angle),
                    'y': math.sin(random_angle)
                }
            })
        
        # Create predator in center
        predator_state = {
            'position': {
                'x': canvas_width / 2,
                'y': canvas_height / 2
            },
            'velocity': {
                'x': self.random() * 2 - 1,
                'y': self.random() * 2 - 1
            }
        }
        
        return {
            'boids_states': boids_states,
            'predator_state': predator_state,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height
        }
    
    def generate_scattered_state(self, 
                               num_boids: int, 
                               canvas_width: float, 
                               canvas_height: float) -> Dict[str, Any]:
        """
        Generate scattered state - boids scattered randomly across the canvas
        
        Args:
            num_boids: Number of boids to generate
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            State dict in StateManager format
        """
        # Generate random boids
        boids_states = []
        for i in range(num_boids):
            boid_state = self._generate_random_boid(canvas_width, canvas_height, boid_id=i)
            boids_states.append(boid_state)
        
        # Generate random predator
        predator_state = self._generate_random_predator(canvas_width, canvas_height)
        
        return {
            'boids_states': boids_states,
            'predator_state': predator_state,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height
        }
    
    def clone_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clone an existing state
        
        Args:
            state: State to clone
            
        Returns:
            Cloned state
        """
        cloned_boids = []
        for boid in state['boids_states']:
            cloned_boids.append({
                'id': boid['id'],  # Preserve boid ID
                'position': {'x': boid['position']['x'], 'y': boid['position']['y']},
                'velocity': {'x': boid['velocity']['x'], 'y': boid['velocity']['y']}
            })
        
        return {
            'boids_states': cloned_boids,
            'predator_state': {
                'position': {'x': state['predator_state']['position']['x'], 'y': state['predator_state']['position']['y']},
                'velocity': {'x': state['predator_state']['velocity']['x'], 'y': state['predator_state']['velocity']['y']}
            },
            'canvas_width': state['canvas_width'],
            'canvas_height': state['canvas_height']
        }
    
    def _generate_random_boid(self, canvas_width: float, canvas_height: float, boid_id: int = None) -> Dict[str, Any]:
        """
        Generate single random boid state
        
        Args:
            canvas_width: Canvas width
            canvas_height: Canvas height
            boid_id: Unique ID for the boid
            
        Returns:
            Boid state dict
        """
        # Random position within canvas bounds
        position = {
            'x': self.random() * canvas_width,
            'y': self.random() * canvas_height
        }
        
        # Random velocity with uniform distribution across all possible speeds
        angle = self.random() * 2 * math.pi
        speed = self.random() * CONSTANTS.BOID_MAX_SPEED
        velocity = {
            'x': math.cos(angle) * speed,
            'y': math.sin(angle) * speed
        }
        
        return {
            'id': boid_id if boid_id is not None else 0,  # Include unique ID
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
        
        # Random velocity with uniform distribution across all possible speeds
        angle = self.random() * 2 * math.pi
        speed = self.random() * CONSTANTS.PREDATOR_MAX_SPEED
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

# Convenience functions for easy use
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

def generate_clustered_state(num_boids: int, 
                           canvas_width: float, 
                           canvas_height: float, 
                           seed: int = None) -> Dict[str, Any]:
    """
    Convenience function to generate clustered state
    
    Args:
        num_boids: Number of boids to generate
        canvas_width: Canvas width
        canvas_height: Canvas height
        seed: Optional seed for reproducible generation
        
    Returns:
        Clustered state dict in StateManager format
    """
    generator = RandomStateGenerator(seed)
    return generator.generate_clustered_state(num_boids, canvas_width, canvas_height)

def generate_scattered_state(num_boids: int, 
                           canvas_width: float, 
                           canvas_height: float, 
                           seed: int = None) -> Dict[str, Any]:
    """
    Convenience function to generate scattered state
    
    Args:
        num_boids: Number of boids to generate
        canvas_width: Canvas width
        canvas_height: Canvas height
        seed: Optional seed for reproducible generation
        
    Returns:
        Scattered state dict in StateManager format
    """
    generator = RandomStateGenerator(seed)
    return generator.generate_scattered_state(num_boids, canvas_width, canvas_height)

def clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to clone state
    
    Args:
        state: State to clone
        
    Returns:
        Cloned state
    """
    generator = RandomStateGenerator()
    return generator.clone_state(state)

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