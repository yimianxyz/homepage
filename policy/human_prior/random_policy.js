/**
 * Random Policy - Pure random action generation
 * 
 * This policy generates completely random actions uniformly distributed in [-1, 1].
 * It serves as a baseline for comparison and testing. Designed to be 100% identical
 * between Python and JavaScript versions.
 * 
 * Interface:
 * - Input: structured_inputs (ignored for random policy)
 * - Output: normalized policy outputs [x, y] in [-1, 1] range
 * - The ActionProcessor handles scaling to game forces
 */

/**
 * Pure random policy that generates uniform random actions
 * @param {number} seed - Optional seed for reproducible random actions
 * @constructor
 */
function RandomPolicy(seed) {
    this.seed = seed;
    
    // If seed provided, use seeded random number generator
    if (typeof seed === 'number') {
        this.random = this._seededRandom(seed);
    } else {
        this.random = Math.random;
    }
}

/**
 * Get normalized random policy outputs (compatible with ActionProcessor)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input (ignored)
 * - context: {canvasWidth: number, canvasHeight: number}
 * - predator: {velX: number, velY: number}
 * - boids: [{relX: number, relY: number, velX: number, velY: number}, ...]
 * 
 * @returns {Array} Random normalized policy outputs [x, y] in [-1, 1] range
 */
RandomPolicy.prototype.getAction = function(structured_inputs) {
    // Generate uniform random values in [-1, 1] range
    return [
        this.random() * 2.0 - 1.0,  // Convert [0,1] to [-1,1]
        this.random() * 2.0 - 1.0   // Convert [0,1] to [-1,1]
    ];
};

/**
 * Get normalized action in [-1, 1] range (deprecated - use getAction instead)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input (ignored)
 * 
 * @returns {Array} Random normalized policy outputs [x, y] in [-1, 1] range
 */
RandomPolicy.prototype.getNormalizedAction = function(structured_inputs) {
    // Now getAction already returns normalized values
    return this.getAction(structured_inputs);
};

/**
 * Simple seeded random number generator (LCG - Linear Congruential Generator)
 * This ensures reproducible random sequences when seed is provided
 * Matches the Python implementation exactly
 * 
 * @param {number} seed - Seed value
 * @returns {Function} Random number generator function
 */
RandomPolicy.prototype._seededRandom = function(seed) {
    var m = 0x80000000; // 2**31
    var a = 1103515245;
    var c = 12345;
    
    var state = seed || Math.floor(Math.random() * (m - 1));
    
    return function() {
        state = (a * state + c) % m;
        return state / (m - 1);
    };
};

/**
 * Create random policy instance
 * @param {number} seed - Optional seed for reproducible random actions
 * @returns {RandomPolicy} Policy instance
 */
function createRandomPolicy(seed) {
    var policy = new RandomPolicy(seed);
    console.log("Created RandomPolicy:");
    console.log("  Output: Random normalized outputs in [-1, 1] range");
    console.log("  Strategy: Pure random uniform distribution");
    console.log("  Seed: " + (seed !== undefined ? seed : "None (non-deterministic)"));
    return policy;
}

// Test code (uncomment to run)
/*
if (typeof module === 'undefined') {
    // Test random policy
    var policy = createRandomPolicy(42);
    
    // Test with dummy data (will be ignored)
    var test_input = {
        'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
        'predator': {'velX': 0.1, 'velY': -0.2},
        'boids': [
            {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
            {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
        ]
    };
    
    // Generate a few random actions
    console.log("Random actions:");
    for (var i = 0; i < 5; i++) {
        var action = policy.getAction(test_input);
        console.log("  Action " + (i+1) + ": [" + action[0].toFixed(3) + ", " + action[1].toFixed(3) + "]");
    }
    
    console.log("Note: Use ActionProcessor to convert to game forces");
}
*/

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        RandomPolicy: RandomPolicy,
        createRandomPolicy: createRandomPolicy
    };
}