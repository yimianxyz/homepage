/**
 * Input Processor - Transformer-ready format
 * 
 * Architecture:
 * - Context: canvas dimensions for world boundary awareness
 * - Predator: velocity information  
 * - Boids: dynamic array of relative positions and velocities
 * - No hardcoded sequence length - supports variable number of boids
 */
function InputProcessor() {
    // Use centralized constants
    this.maxDistance = window.SIMULATION_CONSTANTS.MAX_DISTANCE;
    
    // Unified velocity normalization - use the maximum of boid and predator speeds
    this.unifiedMaxVelocity = Math.max(
        window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
}

/**
 * Convert game state to transformer-friendly format
 * @param {Array} boids - Array of all boid objects
 * @param {Object} predatorPos - Predator position {x, y}
 * @param {Object} predatorVel - Predator velocity {x, y}
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} Structured input with context, predator, and boids arrays
 */
InputProcessor.prototype.processInputs = function(boids, predatorPos, predatorVel, canvasWidth, canvasHeight) {
    // Context information for world boundary awareness
    var context = {
        canvasWidth: canvasWidth / this.maxDistance,  // Normalized canvas width
        canvasHeight: canvasHeight / this.maxDistance  // Normalized canvas height
    };
    
    // Predator velocity information
    var predator = {
        velX: this.clamp(predatorVel.x / this.unifiedMaxVelocity),
        velY: this.clamp(predatorVel.y / this.unifiedMaxVelocity)
    };
    
    // Dynamic array of boids with relative positions and velocities
    var boidArray = [];
    for (var i = 0; i < boids.length; i++) {
        var boidData = this.encodeBoid(boids[i], predatorPos, canvasWidth, canvasHeight);
        boidArray.push({
            relX: boidData[0],
            relY: boidData[1], 
            velX: boidData[2],
            velY: boidData[3]
        });
    }
    
    return {
        context: context,
        predator: predator,
        boids: boidArray
    };
};

/**
 * Encode a single boid as relative position and velocity
 * @param {Object} boid - Boid object with position and velocity
 * @param {Object} predatorPos - Predator position for relative calculation
 * @param {number} canvasWidth - Canvas width for edge wrapping calculation
 * @param {number} canvasHeight - Canvas height for edge wrapping calculation
 * @returns {Array} [rel_x, rel_y, vel_x, vel_y] - normalized values
 */
InputProcessor.prototype.encodeBoid = function(boid, predatorPos, canvasWidth, canvasHeight) {
    // Calculate relative position with edge wrapping support
    var relativePos = this.calculateRelativePosition(
        boid.position, predatorPos, canvasWidth, canvasHeight
    );
    
    // Normalize relative position by fixed maximum distance
    var relX = relativePos.x / this.maxDistance;
    var relY = relativePos.y / this.maxDistance;
    
    // Normalize and clamp velocity by unified max velocity
    var velX = this.clamp(boid.velocity.x / this.unifiedMaxVelocity);
    var velY = this.clamp(boid.velocity.y / this.unifiedMaxVelocity);
    
    return [relX, relY, velX, velY];
};

/**
 * Calculate relative position with edge wrapping support
 * @param {Object} boidPos - Boid position {x, y}
 * @param {Object} predatorPos - Predator position {x, y}
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Object} Relative position {x, y}
 */
InputProcessor.prototype.calculateRelativePosition = function(boidPos, predatorPos, canvasWidth, canvasHeight) {
    // Calculate basic relative position
    var dx = boidPos.x - predatorPos.x;
    var dy = boidPos.y - predatorPos.y;
    
    // Handle edge wrapping (toroidal topology)
    // Choose the shorter distance considering wrapping
    if (Math.abs(dx) > canvasWidth / 2) {
        dx = dx > 0 ? dx - canvasWidth : dx + canvasWidth;
    }
    if (Math.abs(dy) > canvasHeight / 2) {
        dy = dy > 0 ? dy - canvasHeight : dy + canvasHeight;
    }
    
    return { x: dx, y: dy };
};

/**
 * Clamp value to [-1, 1] range
 * @param {number} value - Value to clamp
 * @returns {number} Clamped value
 */
InputProcessor.prototype.clamp = function(value) {
    return Math.max(-1, Math.min(1, value));
};

 