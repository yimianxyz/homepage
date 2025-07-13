"""
Boid class - Exact port of JavaScript src/simulation/boid.js

This MUST match exactly with the JavaScript implementation to ensure
identical flocking behavior.
"""

import math
import random
from typing import List, Optional, TYPE_CHECKING

from .vector import Vector
from .constants import CONSTANTS

if TYPE_CHECKING:
    from .simulation import Simulation
    from .predator import Predator

class Boid:
    def __init__(self, x: float, y: float, simulation: 'Simulation'):
        # Initialize with random direction (matches JavaScript)
        random_angle = random.random() * 2 * math.pi
        self.velocity = Vector(math.cos(random_angle), math.sin(random_angle))
        self.position = Vector(x, y)
        self.acceleration = Vector(0, 0)
        self.simulation = simulation
        self.render_size = CONSTANTS.BOID_RENDER_SIZE
    
    def getCohesionVector(self, boids: List['Boid']) -> Vector:
        """Calculate cohesion force - move towards average position of neighbors"""
        total_position = Vector(0, 0)
        neighbor_count = 0
        
        for boid in boids:
            if self == boid:
                continue
            
            distance = self.position.getDistance(boid.position) + CONSTANTS.EPSILON
            if distance <= CONSTANTS.BOID_NEIGHBOR_DISTANCE:
                total_position = total_position.add(boid.position)
                neighbor_count += 1
        
        if neighbor_count > 0:
            average_position = total_position.divideBy(neighbor_count)
            return self.seek(average_position)
        else:
            return Vector(0, 0)
    
    def seek(self, target_position: Vector) -> Vector:
        """Seek towards target position"""
        desired_vector = target_position.subtract(self.position)
        desired_vector.iFastSetMagnitude(CONSTANTS.BOID_MAX_SPEED)
        steering_vector = desired_vector.subtract(self.velocity)
        steering_vector.iFastLimit(CONSTANTS.BOID_MAX_FORCE)
        return steering_vector
    
    def getSeparationVector(self, boids: List['Boid']) -> Vector:
        """Calculate separation force - avoid crowding neighbors"""
        steering_vector = Vector(0, 0)
        neighbor_count = 0
        
        for boid in boids:
            if self == boid:
                continue
            
            distance = self.position.getDistance(boid.position) + CONSTANTS.EPSILON
            if distance > 0 and distance < CONSTANTS.BOID_DESIRED_SEPARATION:
                delta_vector = self.position.subtract(boid.position)
                delta_vector.iNormalize()
                delta_vector.iDivideBy(distance)
                steering_vector.iAdd(delta_vector)
                neighbor_count += 1
        
        if neighbor_count > 0:
            average_steering_vector = steering_vector.divideBy(neighbor_count)
            average_steering_vector.iFastSetMagnitude(CONSTANTS.BOID_MAX_SPEED)
            average_steering_vector.iSubtract(self.velocity)
            average_steering_vector.iFastLimit(CONSTANTS.BOID_MAX_FORCE)
            return average_steering_vector
        else:
            return Vector(0, 0)
    
    def getAlignmentVector(self, boids: List['Boid']) -> Vector:
        """Calculate alignment force - steer towards average heading of neighbors"""
        perceived_flock_velocity = Vector(0, 0)
        neighbor_count = 0
        
        for boid in boids:
            if self == boid:
                continue
            
            distance = self.position.getDistance(boid.position) + CONSTANTS.EPSILON
            if distance > 0 and distance < CONSTANTS.BOID_NEIGHBOR_DISTANCE:
                perceived_flock_velocity.iAdd(boid.velocity)
                neighbor_count += 1
        
        if neighbor_count > 0:
            average_velocity = perceived_flock_velocity.divideBy(neighbor_count)
            average_velocity.iFastSetMagnitude(CONSTANTS.BOID_MAX_SPEED)
            steering_vector = average_velocity.subtract(self.velocity)
            steering_vector.iFastLimit(CONSTANTS.BOID_MAX_FORCE)
            return steering_vector
        else:
            return Vector(0, 0)
    
    def getPredatorAvoidanceVector(self, predator: Optional['Predator']) -> Vector:
        """Calculate predator avoidance force"""
        if not predator:
            return Vector(0, 0)
        
        distance = self.position.getDistance(predator.position) + CONSTANTS.EPSILON
        if distance > 0 and distance < CONSTANTS.PREDATOR_RANGE:
            avoidance_vector = self.position.subtract(predator.position)
            avoidance_vector.iFastNormalize()
            
            avoidance_strength = (CONSTANTS.PREDATOR_RANGE - distance) / CONSTANTS.PREDATOR_RANGE
            avoidance_vector.iMultiplyBy(avoidance_strength * CONSTANTS.PREDATOR_TURN_FACTOR)
            avoidance_vector.iFastLimit(CONSTANTS.BOID_MAX_FORCE * 1.5)
            
            return avoidance_vector
        
        return Vector(0, 0)
    
    def flock(self, boids: List['Boid']) -> None:
        """Apply flocking behavior"""
        cohesion_vector = self.getCohesionVector(boids)
        separation_vector = self.getSeparationVector(boids)
        alignment_vector = self.getAlignmentVector(boids)
        
        # Apply multipliers (from simulation)
        separation_vector.iMultiplyBy(self.simulation.separation_multiplier)
        cohesion_vector.iMultiplyBy(self.simulation.cohesion_multiplier)
        alignment_vector.iMultiplyBy(self.simulation.alignment_multiplier)
        
        self.acceleration.iAdd(cohesion_vector)
        self.acceleration.iAdd(separation_vector)
        self.acceleration.iAdd(alignment_vector)
        
        # Add predator avoidance if predator exists
        if self.simulation.predator:
            predator_avoidance_vector = self.getPredatorAvoidanceVector(self.simulation.predator)
            self.acceleration.iAdd(predator_avoidance_vector)
    
    def bound(self) -> None:
        """Handle boundary wrapping"""
        if self.position.x > self.simulation.canvas_width + CONSTANTS.BOID_BORDER_OFFSET:
            self.position.x = -CONSTANTS.BOID_BORDER_OFFSET
        if self.position.x < -CONSTANTS.BOID_BORDER_OFFSET:
            self.position.x = self.simulation.canvas_width + CONSTANTS.BOID_BORDER_OFFSET
        if self.position.y > self.simulation.canvas_height + CONSTANTS.BOID_BORDER_OFFSET:
            self.position.y = -CONSTANTS.BOID_BORDER_OFFSET
        if self.position.y < -CONSTANTS.BOID_BORDER_OFFSET:
            self.position.y = self.simulation.canvas_height + CONSTANTS.BOID_BORDER_OFFSET
    
    def update(self) -> None:
        """Update boid physics"""
        self.velocity.iAdd(self.acceleration)
        self.velocity.iFastLimit(CONSTANTS.BOID_MAX_SPEED)
        self.position.iAdd(self.velocity)
        self.bound()
        self.acceleration.iMultiplyBy(0)  # Reset acceleration
    
    def run(self, boids: List['Boid']) -> None:
        """Main update loop - flocking + update"""
        self.flock(boids)
        self.update()
        # Note: render() is not needed for training, only for visualization 