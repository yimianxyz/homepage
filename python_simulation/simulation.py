"""
Simulation class - Exact port of JavaScript src/simulation/simulation.js

Main simulation controller that orchestrates the entire ecosystem.
This MUST match exactly with the JavaScript implementation.
"""

import random
from typing import List, Optional

from .boid import Boid
from .predator import Predator
from .constants import CONSTANTS

class Simulation:
    def __init__(self, canvas_width: int = 800, canvas_height: int = 600):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Flocking behavior multipliers (matches JavaScript defaults)
        self.separation_multiplier = 2
        self.cohesion_multiplier = 1
        self.alignment_multiplier = 1
        
        # Entities
        self.boids: List[Boid] = []
        self.predator: Optional[Predator] = None
    
    def initialize(self, skip_predator: bool = False) -> None:
        """Initialize simulation with boids and optionally predator"""
        self.boids = []
        
        # Initialize boids at random starting position (matches JavaScript)
        start_x = random.randint(0, self.canvas_width - 1)
        start_y = random.randint(0, self.canvas_height - 1)
        
        for i in range(CONSTANTS.NUM_BOIDS):
            boid = Boid(start_x, start_y, self)
            self.boids.append(boid)
        
        # Initialize predator at center (matches JavaScript)
        if not skip_predator:
            predator_x = self.canvas_width / 2
            predator_y = self.canvas_height / 2
            self.predator = Predator(predator_x, predator_y, self)
    
    def step(self) -> None:
        """Single simulation step"""
        # Update all boids
        for boid in self.boids:
            boid.run(self.boids)
        
        # Update predator if it exists
        if self.predator:
            self.predator.update(self.boids)
            
            # Check for caught boids
            caught_boids = self.predator.checkForPrey(self.boids)
            
            # Remove caught boids in reverse order to maintain indices
            for i in reversed(caught_boids):
                self.boids.pop(i)
    
    def get_state(self) -> dict:
        """Get current simulation state for neural network input"""
        return {
            'boids': [
                {
                    'position': {'x': boid.position.x, 'y': boid.position.y},
                    'velocity': {'x': boid.velocity.x, 'y': boid.velocity.y}
                }
                for boid in self.boids
            ],
            'predator': {
                'position': {'x': self.predator.position.x, 'y': self.predator.position.y},
                'velocity': {'x': self.predator.velocity.x, 'y': self.predator.velocity.y}
            } if self.predator else None,
            'canvas_width': self.canvas_width,
            'canvas_height': self.canvas_height
        }
    
    def set_predator_acceleration(self, force_x: float, force_y: float) -> None:
        """Set predator acceleration from neural network output"""
        if self.predator:
            self.predator.acceleration.x = force_x
            self.predator.acceleration.y = force_y
    
    def is_episode_complete(self) -> bool:
        """Check if episode is complete (≤20 boids remaining)"""
        return len(self.boids) <= 20
    
    def get_boid_count(self) -> int:
        """Get current number of boids"""
        return len(self.boids)
    
    def reset(self) -> None:
        """Reset simulation to initial state"""
        self.initialize() 