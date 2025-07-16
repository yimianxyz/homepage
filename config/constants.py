"""
Centralized Constants - Single source of truth for all simulation parameters

This file contains all constants used by the Python simulation and training code.
There is a corresponding constants.js file that MUST contain identical values.
Run 'python config/validate_constants.py' to verify they match.
"""

class SimulationConstants:
    # Boid behavior constants
    BOID_MAX_SPEED = 3.5
    BOID_MAX_FORCE = 0.1
    BOID_DESIRED_SEPARATION = 40
    BOID_NEIGHBOR_DISTANCE = 60
    BOID_BORDER_OFFSET = 10
    BOID_RENDER_SIZE = 9
    
    # Predator behavior constants
    PREDATOR_MAX_SPEED = 2
    PREDATOR_MAX_FORCE = 0.2
    PREDATOR_SIZE = 18
    PREDATOR_RANGE = 80
    PREDATOR_TURN_FACTOR = 0.3
    PREDATOR_BORDER_OFFSET = 20
    
    # Device-independent normalization constants
    MAX_DISTANCE = 1000         # Maximum expected distance for normalization
    
    # Simulation constants
    DEFAULT_NUM_BOIDS = 50
    EPSILON = 0.0000001
    
    # Flocking behavior multipliers
    SEPARATION_MULTIPLIER = 2.0
    COHESION_MULTIPLIER = 1.0
    ALIGNMENT_MULTIPLIER = 1.0

# Global constants instance
CONSTANTS = SimulationConstants()

# Export all constants for easy access
__all__ = ['CONSTANTS', 'SimulationConstants'] 