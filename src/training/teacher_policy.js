/**
 * Teacher Policy - Simple pursuit behavior for supervised learning
 * Updated for new encoding system with 51 entities (50 boids + 1 predator)
 */

function TeacherPolicy() {
    // Use same force scaling as ActionProcessor for consistency
    this.maxForce = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * window.SIMULATION_CONSTANTS.PREDATOR_FORCE_SCALE;
    this.boidVectorSize = window.SIMULATION_CONSTANTS.BOID_VECTOR_SIZE;
    this.maxBoids = window.SIMULATION_CONSTANTS.MAX_BOIDS;
}

TeacherPolicy.prototype.getAction = function(inputs) {
    // Find the closest non-zero boid in the input vector
    var targetBoidIndex = -1;
    var minDistance = Infinity;
    
    for (var i = 0; i < this.maxBoids; i++) {
        var baseIndex = i * this.boidVectorSize;
        var relX = inputs[baseIndex];
        var relY = inputs[baseIndex + 1];
        
        // Skip if this boid slot is empty (all zeros)
        if (relX === 0 && relY === 0) {
            continue;
        }
        
        // Calculate distance to this boid
        var distance = Math.sqrt(relX * relX + relY * relY);
        if (distance < minDistance) {
            minDistance = distance;
            targetBoidIndex = i;
        }
    }
    
    // If no boids found, return zero force
    if (targetBoidIndex === -1) {
        return [0, 0];
    }
    
    // Get the closest boid's relative position (already normalized to [-1, 1])
    var baseIndex = targetBoidIndex * this.boidVectorSize;
    var relX = inputs[baseIndex];
    var relY = inputs[baseIndex + 1];
    
    // Simple seeking: move toward target boid
    var distance = Math.sqrt(relX * relX + relY * relY);
    if (distance < 0.001) return [0, 0];
    
    // Normalized direction to target
    var dirX = relX / distance;
    var dirY = relY / distance;
    
    // Apply force scaling (matches ActionProcessor)
    return [dirX * this.maxForce, dirY * this.maxForce];
}; 