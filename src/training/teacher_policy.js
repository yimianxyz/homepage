/**
 * Teacher Policy - Simple pursuit behavior for supervised learning
 */

function TeacherPolicy() {
    // Use same force scaling as ActionProcessor for consistency
    this.maxForce = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * window.SIMULATION_CONSTANTS.PREDATOR_FORCE_SCALE;
    this.visionHalfWidth = window.SIMULATION_CONSTANTS.VISION_WIDTH / 2;
    this.visionHalfHeight = window.SIMULATION_CONSTANTS.VISION_HEIGHT / 2;
}

TeacherPolicy.prototype.getAction = function(inputs) {
    // Extract nearest boid (first 4 values: relX, relY, velX, velY)
    var relX = inputs[0] * this.visionHalfWidth;
    var relY = inputs[1] * this.visionHalfHeight;
    
    // If no boids visible, return zero force
    if (relX === 0 && relY === 0) {
        return [0, 0];
    }
    
    // Simple seeking: move toward nearest boid
    var distance = Math.sqrt(relX * relX + relY * relY);
    if (distance < 0.001) return [0, 0];
    
    // Normalized direction to target
    var dirX = relX / distance;
    var dirY = relY / distance;
    
    // Apply force scaling (now matches ActionProcessor)
    return [dirX * this.maxForce, dirY * this.maxForce];
}; 