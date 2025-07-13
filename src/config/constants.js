/**
 * Centralized Constants - Single source of truth for all simulation parameters
 * 
 * This eliminates duplicated constants and ensures neural network components
 * use the same values as the actual simulation.
 */

window.SIMULATION_CONSTANTS = {
    // Boid behavior constants
    BOID_MAX_SPEED: 3.5,
    BOID_MAX_FORCE: 0.1,
    BOID_DESIRED_SEPARATION: 40,
    BOID_NEIGHBOR_DISTANCE: 60,
    BOID_BORDER_OFFSET: 10,
    BOID_RENDER_SIZE: 9,
    
    // Predator behavior constants
    PREDATOR_MAX_SPEED: 2,
    PREDATOR_MAX_FORCE: 0.001,
    PREDATOR_SIZE: 18,
    PREDATOR_RANGE: 80,
    PREDATOR_TURN_FACTOR: 0.3,
    PREDATOR_BORDER_OFFSET: 20,
    PREDATOR_FORCE_SCALE: 200,  // Neural network force scaling for smooth turning
    
    // Neural network constants - New encoding system
    MAX_BOIDS: 50,              // Maximum number of boids in simulation
    BOID_VECTOR_SIZE: 4,        // Features per boid: [rel_x, rel_y, vel_x, vel_y]
    PREDATOR_VECTOR_SIZE: 4,    // Features for predator: [canvas_width_norm, canvas_height_norm, vel_x, vel_y]
    TOTAL_ENTITIES: 51,         // 50 boids + 1 predator
    NEURAL_INPUT_SIZE: 204,     // 51 entities × 4 features each
    
    // Device-independent normalization constants
    MAX_DISTANCE: 1000,         // Maximum expected distance for normalization
    
    // Simulation constants
    NUM_BOIDS: 50,
    EPSILON: 0.0000001,
    
    // Training constants
    TARGET_CHANGE_INTERVAL: 3000
}; 