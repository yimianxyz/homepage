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
 * Generate completely random (scattered) state for simulation (default method)
 * 
 * @param {number} numBoids - Number of boids to generate
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} State dict in StateManager format
 */
RandomStateGenerator.prototype.generateRandomState = function(numBoids, canvasWidth, canvasHeight) {
    return this.generateScatteredState(numBoids, canvasWidth, canvasHeight);
};

/**
 * Generate clustered state - boids clustered around a random starting point
 * 
 * @param {number} numBoids - Number of boids to generate
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} State dict in StateManager format
 */
RandomStateGenerator.prototype.generateClusteredState = function(numBoids, canvasWidth, canvasHeight) {
    var boidsStates = [];
    var startX = Math.floor(this.random() * canvasWidth);
    var startY = Math.floor(this.random() * canvasHeight);
    
    // Create boids clustered around a random starting point
    for (var i = 0; i < numBoids; i++) {
        var randomAngle = this.random() * 2 * Math.PI;
        boidsStates.push({
            id: i,  // Unique ID for each boid
            position: {
                x: startX + (this.random() - 0.5) * 100,
                y: startY + (this.random() - 0.5) * 100
            },
            velocity: {
                x: Math.cos(randomAngle),
                y: Math.sin(randomAngle)
            }
        });
    }
    
    // Create predator in center
    var predatorState = {
        position: {
            x: canvasWidth / 2,
            y: canvasHeight / 2
        },
        velocity: {
            x: this.random() * 2 - 1,
            y: this.random() * 2 - 1
        }
    };
    
    return {
        boids_states: boidsStates,
        predator_state: predatorState,
        canvas_width: canvasWidth,
        canvas_height: canvasHeight
    };
};

/**
 * Generate scattered state - boids scattered randomly across the canvas
 * 
 * @param {number} numBoids - Number of boids to generate
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} State dict in StateManager format
 */
RandomStateGenerator.prototype.generateScatteredState = function(numBoids, canvasWidth, canvasHeight) {
    // Generate random boids
    var boidsStates = [];
    for (var i = 0; i < numBoids; i++) {
        var boidState = this._generateRandomBoid(canvasWidth, canvasHeight, i);
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
 * Clone an existing state
 * 
 * @param {Object} state - State to clone
 * @returns {Object} Cloned state
 */
RandomStateGenerator.prototype.cloneState = function(state) {
    var clonedBoids = [];
    for (var i = 0; i < state.boids_states.length; i++) {
        var boid = state.boids_states[i];
        clonedBoids.push({
            id: boid.id,  // Preserve boid ID
            position: { x: boid.position.x, y: boid.position.y },
            velocity: { x: boid.velocity.x, y: boid.velocity.y }
        });
    }
    
    return {
        boids_states: clonedBoids,
        predator_state: {
            position: { x: state.predator_state.position.x, y: state.predator_state.position.y },
            velocity: { x: state.predator_state.velocity.x, y: state.predator_state.velocity.y }
        },
        canvas_width: state.canvas_width,
        canvas_height: state.canvas_height
    };
};

/**
 * Generate single random boid state
 * 
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @param {number} boidId - Unique ID for the boid
 * @returns {Object} Boid state dict
 * @private
 */
RandomStateGenerator.prototype._generateRandomBoid = function(canvasWidth, canvasHeight, boidId) {
    // Random position within canvas bounds
    var position = {
        x: this.random() * canvasWidth,
        y: this.random() * canvasHeight
    };
    
    // Random velocity with uniform distribution across all possible speeds
    var angle = this.random() * 2 * Math.PI;
    var speed = this.random() * window.SIMULATION_CONSTANTS.BOID_MAX_SPEED;
    var velocity = {
        x: Math.cos(angle) * speed,
        y: Math.sin(angle) * speed
    };
    
    return {
        id: boidId !== undefined ? boidId : 0,  // Include unique ID
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
    
    // Random velocity with uniform distribution across all possible speeds
    var angle = this.random() * 2 * Math.PI;
    var speed = this.random() * window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED;
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

// Convenience functions for easy use
function generateRandomState(numBoids, canvasWidth, canvasHeight, seed) {
    var generator = new RandomStateGenerator(seed);
    return generator.generateRandomState(numBoids, canvasWidth, canvasHeight);
}

function generateClusteredState(numBoids, canvasWidth, canvasHeight, seed) {
    var generator = new RandomStateGenerator(seed);
    return generator.generateClusteredState(numBoids, canvasWidth, canvasHeight);
}

function generateScatteredState(numBoids, canvasWidth, canvasHeight, seed) {
    var generator = new RandomStateGenerator(seed);
    return generator.generateScatteredState(numBoids, canvasWidth, canvasHeight);
}

function cloneState(state) {
    var generator = new RandomStateGenerator();
    return generator.cloneState(state);
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
    module.exports = { 
        RandomStateGenerator: RandomStateGenerator, 
        generateRandomState: generateRandomState,
        generateClusteredState: generateClusteredState,
        generateScatteredState: generateScatteredState,
        cloneState: cloneState
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.RandomStateGenerator = RandomStateGenerator;
    window.generateRandomState = generateRandomState;
    window.generateClusteredState = generateClusteredState;
    window.generateScatteredState = generateScatteredState;
    window.cloneState = cloneState;
} 