/**
 * Teacher Policy - Simple pursuit behavior for supervised learning
 * Updated for transformer architecture with structured inputs
 */

function TeacherPolicy() {
    // Use same force scaling as ActionProcessor for consistency
            this.maxForce = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE;
}

/**
 * Get action based on structured inputs
 * @param {Object} structuredInputs - {context, predator, boids}
 * @returns {Array} Action forces [forceX, forceY]
 */
TeacherPolicy.prototype.getAction = function(structuredInputs) {
    // If no boids, return zero force
    if (!structuredInputs.boids || structuredInputs.boids.length === 0) {
        return [0, 0];
    }
    
    // Find the closest boid in the structured format
    var targetBoid = null;
    var minDistance = Infinity;
    
    for (var i = 0; i < structuredInputs.boids.length; i++) {
        var boid = structuredInputs.boids[i];
        
        // Calculate distance to this boid
        var distance = Math.sqrt(boid.relX * boid.relX + boid.relY * boid.relY);
        
        if (distance < minDistance) {
            minDistance = distance;
            targetBoid = boid;
        }
    }
    
    // If no valid target found, return zero force
    if (!targetBoid || minDistance < 0.001) {
        return [0, 0];
    }
    
    // Simple seeking: move toward target boid
    // Normalized direction to target
    var dirX = targetBoid.relX / minDistance;
    var dirY = targetBoid.relY / minDistance;
    
    // Apply force scaling (matches ActionProcessor)
    return [dirX * this.maxForce, dirY * this.maxForce];
}; 