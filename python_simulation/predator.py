"""
Predator class - Exact port of JavaScript src/simulation/predator.js

Base Predator Class - provides core functionality for predator entities.
This MUST match exactly with the JavaScript implementation.
"""

import random
from typing import List, TYPE_CHECKING

from .vector import Vector
from .constants import CONSTANTS

if TYPE_CHECKING:
    from .simulation import Simulation
    from .boid import Boid

class Predator:
    def __init__(self, x: float, y: float, simulation: 'Simulation'):
        self.position = Vector(x, y)
        self.velocity = Vector(random.random() * 2 - 1, random.random() * 2 - 1)
        self.acceleration = Vector(0, 0)
        self.simulation = simulation
        self.size = CONSTANTS.PREDATOR_SIZE
    
    def seek(self, target_position: Vector) -> Vector:
        """Seek towards target position"""
        desired_vector = target_position.subtract(self.position)
        desired_vector.iFastSetMagnitude(CONSTANTS.PREDATOR_MAX_SPEED)
        steering_vector = desired_vector.subtract(self.velocity)
        steering_vector.iFastLimit(CONSTANTS.PREDATOR_MAX_FORCE)
        return steering_vector
    
    def bound(self) -> None:
        """Handle boundary wrapping"""
        if self.position.x > self.simulation.canvas_width + CONSTANTS.PREDATOR_BORDER_OFFSET:
            self.position.x = -CONSTANTS.PREDATOR_BORDER_OFFSET
        if self.position.x < -CONSTANTS.PREDATOR_BORDER_OFFSET:
            self.position.x = self.simulation.canvas_width + CONSTANTS.PREDATOR_BORDER_OFFSET
        if self.position.y > self.simulation.canvas_height + CONSTANTS.PREDATOR_BORDER_OFFSET:
            self.position.y = -CONSTANTS.PREDATOR_BORDER_OFFSET
        if self.position.y < -CONSTANTS.PREDATOR_BORDER_OFFSET:
            self.position.y = self.simulation.canvas_height + CONSTANTS.PREDATOR_BORDER_OFFSET
    
    def checkForPrey(self, boids: List['Boid']) -> List[int]:
        """Check for caught boids and return their indices"""
        caught_boids = []
        catch_radius = self.size * 0.7
        
        for i, boid in enumerate(boids):
            distance = self.position.getDistance(boid.position)
            if distance < catch_radius:
                caught_boids.append(i)
        
        return caught_boids
    
    def update(self, boids: List['Boid']) -> None:
        """Update predator physics"""
        self.velocity.iAdd(self.acceleration)
        self.velocity.iFastLimit(CONSTANTS.PREDATOR_MAX_SPEED)
        self.position.iAdd(self.velocity)
        
        self.bound()
        self.acceleration.iMultiplyBy(0)  # Reset acceleration 