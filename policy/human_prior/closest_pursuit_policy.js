/**
 * Closest Pursuit Policy - Simple pursuit behavior targeting the closest boid
 * 
 * This policy implements a greedy pursuit strategy where the predator always
 * moves toward the closest boid. It's designed to be 100% identical between
 * Python and JavaScript versions.
 * 
 * Interface:
 * - Input: structured_inputs (same format as transformer input)
 * - Output: action forces [force_x, force_y]
 */

/**
 * Greedy pursuit policy that always targets the closest boid
 * @constructor
 */
function ClosestPursuitPolicy() {
    // Constants that must match Python exactly
    this.PREDATOR_MAX_FORCE = 0.001;
    this.PREDATOR_FORCE_SCALE = 200;
    
    // Calculate max force (matches ActionProcessor)
    this.max_force = this.PREDATOR_MAX_FORCE * this.PREDATOR_FORCE_SCALE;
}

/**
 * Get action based on structured inputs (matches Python implementation)
 * 
 * @param {Object} structured_inputs - Same format as transformer input
 * - context: {canvasWidth: number, canvasHeight: number}
 * - predator: {velX: number, velY: number}
 * - boids: [{relX: number, relY: number, velX: number, velY: number}, ...]
 * 
 * @returns {Array} Action forces [force_x, force_y]
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
    // Normalized direction to target
    var dir_x = target_boid.relX / min_distance;
    var dir_y = target_boid.relY / min_distance;
    
    // Apply force scaling (matches ActionProcessor)
    return [dir_x * this.max_force, dir_y * this.max_force];
};

/**
 * Get normalized action in [-1, 1] range (for direct comparison with model output)
 * 
 * @param {Object} structured_inputs - Same format as transformer input
 * 
 * @returns {Array} Normalized action forces [force_x, force_y] in [-1, 1] range
 */
ClosestPursuitPolicy.prototype.getNormalizedAction = function(structured_inputs) {
    var action = this.getAction(structured_inputs);
    
    // Normalize by max force to get [-1, 1] range
    var normalized_action = [
        action[0] / this.max_force,
        action[1] / this.max_force
    ];
    
    // Clamp to ensure [-1, 1] range
    normalized_action[0] = Math.max(-1.0, Math.min(1.0, normalized_action[0]));
    normalized_action[1] = Math.max(-1.0, Math.min(1.0, normalized_action[1]));
    
    return normalized_action;
};

/**
 * Create closest pursuit policy instance
 * @returns {ClosestPursuitPolicy} Policy instance
 */
function createClosestPursuitPolicy() {
    var policy = new ClosestPursuitPolicy();
    console.log("Created ClosestPursuitPolicy:");
    console.log("  Max force: " + policy.max_force);
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
    
    var action = policy.getAction(test_input);
    var normalized_action = policy.getNormalizedAction(test_input);
    
    console.log("Test action: " + action);
    console.log("Test normalized action: " + normalized_action);
}
*/

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ClosestPursuitPolicy: ClosestPursuitPolicy,
        createClosestPursuitPolicy: createClosestPursuitPolicy
    };
} 