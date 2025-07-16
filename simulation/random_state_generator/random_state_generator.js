/**
 * Random State Generator - Generate random boid and predator states for simulation
 * 
 * This generates random initial states in the format expected by StateManager.
 * The Python and JavaScript versions MUST be 100% identical.
 */

/**
 * Generate random boid and predator states for simulation
 * @constructor
 * @param {number} seed - Optional seed for reproducible random generation
 */
function RandomStateGenerator(seed) {
    this.seed = seed;
    if (seed !== undefined && seed !== null) {
        // Use seeded random number generator if provided
        this.random = this.seededRandom(seed);
    } else {
        // Use Math.random for unseeded generation
        this.random = Math.random;
    }
}

/**
 * Generate completely random state for simulation
 * 
 * @param {number} numBoids - Number of boids to generate
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} State dict in StateManager format:
 *   {
 *       boids_states: [
 *           {
 *               position: {x: float, y: float},
 *               velocity: {x: float, y: float}
 *           },
 *           ...
 *       ],
 *       predator_state: {
 *           position: {x: float, y: float},
 *           velocity: {x: float, y: float}
 *       },
 *       canvas_width: float,
 *       canvas_height: float
 *   }
 */
RandomStateGenerator.prototype.generateRandomState = function(numBoids, canvasWidth, canvasHeight) {
    // Generate random boids
    var boidsStates = [];
    for (var i = 0; i < numBoids; i++) {
        var boidState = this._generateRandomBoid(canvasWidth, canvasHeight);
        boidsStates.push(boidState);
    }
    
    // Generate random predator
    var predatorState = this._generateRandomPredator(canvasWidth, canvasHeight);
    
    return {
        boids_states: boidsStates,
        predator_state: predatorState,
        canvas_width: canvasWidth,
        canvas_height: canvasHeight
    };
};

/**
 * Generate single random boid state
 * 
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} Boid state dict
 * @private
 */
RandomStateGenerator.prototype._generateRandomBoid = function(canvasWidth, canvasHeight) {
    // Random position within canvas bounds
    var position = {
        x: this.random() * canvasWidth,
        y: this.random() * canvasHeight
    };
    
    // Random velocity with random direction and speed
    var angle = this.random() * 2 * Math.PI;
    var speed = 0.5 + this.random() * (window.SIMULATION_CONSTANTS.BOID_MAX_SPEED - 0.5);
    var velocity = {
        x: Math.cos(angle) * speed,
        y: Math.sin(angle) * speed
    };
    
    return {
        position: position,
        velocity: velocity
    };
};

/**
 * Generate random predator state
 * 
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} Predator state dict
 * @private
 */
RandomStateGenerator.prototype._generateRandomPredator = function(canvasWidth, canvasHeight) {
    // Random position within canvas bounds
    var position = {
        x: this.random() * canvasWidth,
        y: this.random() * canvasHeight
    };
    
    // Random velocity with random direction and speed
    var angle = this.random() * 2 * Math.PI;
    var speed = 0.5 + this.random() * (window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED - 0.5);
    var velocity = {
        x: Math.cos(angle) * speed,
        y: Math.sin(angle) * speed
    };
    
    return {
        position: position,
        velocity: velocity
    };
};

/**
 * Simple seeded random number generator (LCG - Linear Congruential Generator)
 * This ensures reproducible random sequences when seed is provided
 * 
 * @param {number} seed - Seed value
 * @returns {function} Random number generator function
 * @private
 */
RandomStateGenerator.prototype.seededRandom = function(seed) {
    var m = 0x80000000; // 2**31
    var a = 1103515245;
    var c = 12345;
    
    var state = seed ? seed : Math.floor(Math.random() * (m - 1));
    
    return function() {
        state = (a * state + c) % m;
        return state / (m - 1);
    };
};

/**
 * Convenience function to generate random state
 * 
 * @param {number} numBoids - Number of boids to generate
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @param {number} seed - Optional seed for reproducible generation
 * @returns {Object} Random state dict in StateManager format
 */
function generateRandomState(numBoids, canvasWidth, canvasHeight, seed) {
    var generator = new RandomStateGenerator(seed);
    return generator.generateRandomState(numBoids, canvasWidth, canvasHeight);
}

// Example usage for testing
if (typeof window === 'undefined') {
    // Node.js environment - for testing
    console.log("RandomStateGenerator module loaded (Node.js)");
} else {
    // Browser environment
    console.log("RandomStateGenerator module loaded (Browser)");
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RandomStateGenerator: RandomStateGenerator, generateRandomState: generateRandomState };
} 