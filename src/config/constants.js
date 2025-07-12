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
    
    // Neural network constants
    VISION_WIDTH: 400,          // Rectangular vision width
    VISION_HEIGHT: 568,         // Rectangular vision height (smallest phone height)
    MAX_VISIBLE_BOIDS: 5,       // Maximum boids in input vector
    
    // Simulation constants
    NUM_BOIDS: 50,
    EPSILON: 0.0000001,
    
    // Training constants
    TARGET_CHANGE_INTERVAL: 3000
}; 