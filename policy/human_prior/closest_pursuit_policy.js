/**
 * Closest Pursuit Policy - Simple pursuit behavior targeting the closest boid
 * 
 * This policy implements a greedy pursuit strategy where the predator always
 * moves toward the closest boid. It's designed to be 100% identical between
 * Python and JavaScript versions.
 * 
 * Interface:
 * - Input: structured_inputs (same format as universal policy input)
 * - Output: normalized policy outputs [x, y] in [-1, 1] range
 * - The ActionProcessor handles scaling to game forces
 */

/**
 * Greedy pursuit policy that always targets the closest boid
 * @constructor
 */
function ClosestPursuitPolicy() {
    // No max_force needed - ActionProcessor handles scaling
}

/**
 * Get normalized policy outputs (compatible with ActionProcessor)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input
 * - context: {canvasWidth: number, canvasHeight: number}
 * - predator: {velX: number, velY: number}
 * - boids: [{relX: number, relY: number, velX: number, velY: number}, ...]
 * 
 * @returns {Array} Normalized policy outputs [x, y] in [-1, 1] range
 */
ClosestPursuitPolicy.prototype.getAction = function(structured_inputs) {
    // If no boids, return zero force
    if (!structured_inputs.boids || structured_inputs.boids.length === 0) {
        return [0.0, 0.0];
    }
    
    // Find the closest boid in the structured format
    var target_boid = null;
    var min_distance = Infinity;
    
    for (var i = 0; i < structured_inputs.boids.length; i++) {
        var boid = structured_inputs.boids[i];
        
        // Calculate distance to this boid
        var distance = Math.sqrt(boid.relX * boid.relX + boid.relY * boid.relY);
        
        if (distance < min_distance) {
            min_distance = distance;
            target_boid = boid;
        }
    }
    
    // If no valid target found, return zero force
    if (target_boid === null || min_distance < 0.001) {
        return [0.0, 0.0];
    }
    
    // Simple seeking: move toward target boid
    // Normalized direction to target (already in [-1, 1] range)
    var dir_x = target_boid.relX / min_distance;
    var dir_y = target_boid.relY / min_distance;
    
    // Return normalized policy outputs - ActionProcessor handles scaling
    return [dir_x, dir_y];
};

/**
 * Get normalized action in [-1, 1] range (deprecated - use getAction instead)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input
 * 
 * @returns {Array} Normalized policy outputs [x, y] in [-1, 1] range
 */
ClosestPursuitPolicy.prototype.getNormalizedAction = function(structured_inputs) {
    // Now getAction already returns normalized values
    return this.getAction(structured_inputs);
};

/**
 * Create closest pursuit policy instance
 * @returns {ClosestPursuitPolicy} Policy instance
 */
function createClosestPursuitPolicy() {
    var policy = new ClosestPursuitPolicy();
    console.log("Created ClosestPursuitPolicy:");
    console.log("  Output: Normalized policy outputs in [-1, 1] range");
    console.log("  Strategy: Greedy pursuit (always targets closest boid)");
    return policy;
}

// Test code (uncomment to run)
/*
if (typeof module === 'undefined') {
    // Test closest pursuit policy
    var policy = createClosestPursuitPolicy();
    
    // Test with dummy data
    var test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    };
    
    var policy_output = policy.getAction(test_input);
    
    console.log("Policy output (normalized): " + policy_output);
    console.log("Note: Use ActionProcessor to convert to game forces");
}
*/

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ClosestPursuitPolicy: ClosestPursuitPolicy,
        createClosestPursuitPolicy: createClosestPursuitPolicy
    };
} 