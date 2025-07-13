/**
 * Input Processor - Complete information encoding system
 * 
 * New Architecture:
 * - 50 boid vectors + 1 predator vector = 51 total entities
 * - Each entity: 4 features = 204 total inputs
 * - No vision limitations - all boids included
 * - Device-independent normalization for consistency
 * 
 * Boid Vector Format: [rel_x, rel_y, vel_x, vel_y]
 * Predator Vector Format: [canvas_width_norm, canvas_height_norm, vel_x, vel_y]
 */
function InputProcessor() {
    // Use centralized constants for new encoding system
    this.maxBoids = window.SIMULATION_CONSTANTS.MAX_BOIDS;
    this.boidVectorSize = window.SIMULATION_CONSTANTS.BOID_VECTOR_SIZE;
    this.predatorVectorSize = window.SIMULATION_CONSTANTS.PREDATOR_VECTOR_SIZE;
    this.totalInputSize = window.SIMULATION_CONSTANTS.NEURAL_INPUT_SIZE;
    
    // Unified velocity normalization - use the maximum of boid and predator speeds
    // This ensures all velocities are normalized consistently
    this.unifiedMaxVelocity = Math.max(
        window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
    
    // Device-independent normalization bounds from centralized config
    this.maxDistance = window.SIMULATION_CONSTANTS.MAX_DISTANCE;
}

/**
 * Convert complete game state to neural network inputs
 * @param {Array} boids - Array of all boid objects
 * @param {Object} predatorPos - Predator position {x, y}
 * @param {Object} predatorVel - Predator velocity {x, y}
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Array} 204-element input vector (51 entities × 4 features)
 */
InputProcessor.prototype.processInputs = function(boids, predatorPos, predatorVel, canvasWidth, canvasHeight) {
    var inputs = new Array(this.totalInputSize);
    var inputIndex = 0;
    
    // Encode all boid vectors (50 slots)
    for (var i = 0; i < this.maxBoids; i++) {
        if (i < boids.length) {
            // Encode existing boid
            var boidVector = this.encodeBoid(boids[i], predatorPos, canvasWidth, canvasHeight);
            for (var j = 0; j < this.boidVectorSize; j++) {
                inputs[inputIndex++] = boidVector[j];
            }
        } else {
            // Pad with zeros for missing boids
            for (var j = 0; j < this.boidVectorSize; j++) {
                inputs[inputIndex++] = 0.0;
            }
        }
    }
    
    // Encode predator vector (1 slot)
    var predatorVector = this.encodePredator(predatorVel, canvasWidth, canvasHeight);
    for (var j = 0; j < this.predatorVectorSize; j++) {
        inputs[inputIndex++] = predatorVector[j];
    }
    
    return inputs;
};

/**
 * Encode a single boid as a 4-feature vector
 * @param {Object} boid - Boid object with position and velocity
 * @param {Object} predatorPos - Predator position for relative calculation
 * @param {number} canvasWidth - Canvas width for edge wrapping calculation
 * @param {number} canvasHeight - Canvas height for edge wrapping calculation
 * @returns {Array} [rel_x, rel_y, vel_x, vel_y] - positions normalized, velocities clamped
 */
InputProcessor.prototype.encodeBoid = function(boid, predatorPos, canvasWidth, canvasHeight) {
    // Calculate relative position with edge wrapping support
    var relativePos = this.calculateRelativePosition(
        boid.position, predatorPos, canvasWidth, canvasHeight
    );
    
    // Normalize relative position by fixed maximum distance (no clamping - keep real spatial info)
    var relX = relativePos.x / this.maxDistance;
    var relY = relativePos.y / this.maxDistance;
    
    // Normalize and clamp velocity by unified max velocity (velocities have natural bounds)
    var velX = this.clamp(boid.velocity.x / this.unifiedMaxVelocity);
    var velY = this.clamp(boid.velocity.y / this.unifiedMaxVelocity);
    
    return [relX, relY, velX, velY];
};

/**
 * Encode predator as a 4-feature vector with world context and velocity
 * @param {Object} predatorVel - Predator velocity {x, y}
 * @param {number} canvasWidth - Canvas width
 * @param {number} canvasHeight - Canvas height
 * @returns {Array} [canvas_width_norm, canvas_height_norm, vel_x, vel_y] - canvas size normalized, velocities clamped
 */
InputProcessor.prototype.encodePredator = function(predatorVel, canvasWidth, canvasHeight) {
    // Normalize canvas dimensions (no clamping - keep real world boundary info)
    // This gives the network context about world boundaries
    var canvasWidthNorm = canvasWidth / this.maxDistance;
    var canvasHeightNorm = canvasHeight / this.maxDistance;
    
    // Normalize and clamp velocity by unified max velocity (velocities have natural bounds)
    // Keep velocity in positions 2,3 like boids for consistency
    var velX = this.clamp(predatorVel.x / this.unifiedMaxVelocity);
    var velY = this.clamp(predatorVel.y / this.unifiedMaxVelocity);
    
    return [canvasWidthNorm, canvasHeightNorm, velX, velY];
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

 