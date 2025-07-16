"""
Centralized Constants - Single source of truth for all simulation parameters

This eliminates duplicated constants and ensures neural network components
use the same values as the actual simulation.

These values MUST match exactly with src/config/constants.js
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
    
    # Legacy MLP constants (kept for backward compatibility)
    MAX_BOIDS = 50              # Maximum number of boids in simulation
    BOID_VECTOR_SIZE = 4        # Features per boid: [rel_x, rel_y, vel_x, vel_y]
    PREDATOR_VECTOR_SIZE = 4    # Features for predator: [canvas_width_norm, canvas_height_norm, vel_x, vel_y]
    TOTAL_ENTITIES = 51         # 50 boids + 1 predator
    NEURAL_INPUT_SIZE = 204     # 51 entities × 4 features each
    
    # Transformer architecture constants
    TRANSFORMER_EMBED_DIM = 128     # Embedding dimension for transformer
    TRANSFORMER_NUM_HEADS = 8       # Number of attention heads
    TRANSFORMER_NUM_LAYERS = 6      # Number of transformer layers
    TRANSFORMER_FF_DIM = 512        # Feed-forward network dimension
    TRANSFORMER_DROPOUT = 0.1       # Dropout rate
    
    # Entity type IDs for transformer processing
    ENTITY_TYPE_CONTEXT = 0         # Context information
    ENTITY_TYPE_PREDATOR = 1        # Predator entity
    ENTITY_TYPE_BOID = 2            # Boid entity
    
    # Device-independent normalization constants
    MAX_DISTANCE = 1000         # Maximum expected distance for normalization
    
    # Simulation constants
    NUM_BOIDS = 50
    EPSILON = 0.0000001
    
    # Training constants
    TARGET_CHANGE_INTERVAL = 3000

# Global constants instance
CONSTANTS = SimulationConstants() 