/**
 * Input Processor - Vision-based neural input system
 * 
 * Uses fixed rectangular vision area to eliminate screen size dependencies.
 * Only considers boids within predator's limited vision range.
 * 
 * Vision Design:
 * - Fixed rectangular vision: 400px width × 568px height
 * - Only top 5 boids within vision area
 * - Relative position normalization using vision dimensions
 * - Fixed velocity normalization: 3.0 units/frame max
 * - Screen size independent by design
 */
function InputProcessor() {
    // Use centralized constants - no duplicated values
    this.visionWidth = window.SIMULATION_CONSTANTS.VISION_WIDTH;
    this.visionHeight = window.SIMULATION_CONSTANTS.VISION_HEIGHT;
    this.maxVelocity = window.SIMULATION_CONSTANTS.BOID_MAX_SPEED;      // Use actual boid max speed
    this.maxBoids = window.SIMULATION_CONSTANTS.MAX_VISIBLE_BOIDS;
}

/**
 * Convert game state to neural network inputs (vision-based)
 * @param {Array} boids - Array of boid objects
 * @param {Object} predatorPos - Predator position {x, y}
 * @param {Object} predatorVel - Predator velocity {x, y}
 * @param {number} canvasWidth - Canvas width (for edge wrapping)
 * @param {number} canvasHeight - Canvas height (for edge wrapping)
 * @returns {Array} 22-element input vector
 */
InputProcessor.prototype.processInputs = function(boids, predatorPos, predatorVel, canvasWidth, canvasHeight) {
    var inputs = new Array(22);
    
    // Find boids within vision range (with edge wrapping support)
    var visibleBoids = this.findBoidsInVision(boids, predatorPos, canvasWidth, canvasHeight);
    
    // Fill first 20 elements with boid data (5 boids × 4 values each)
    for (var i = 0; i < this.maxBoids; i++) {
        var baseIndex = i * 4;
        
        if (i < visibleBoids.length) {
            var boidData = visibleBoids[i];
            
            // Relative position normalized by vision dimensions
            var relX = boidData.relativePos.x / (this.visionWidth / 2);
            var relY = boidData.relativePos.y / (this.visionHeight / 2);
            
            // Boid velocity normalized by max velocity
            var velX = boidData.boid.velocity.x / this.maxVelocity;
            var velY = boidData.boid.velocity.y / this.maxVelocity;
            
            inputs[baseIndex] = this.clamp(relX);
            inputs[baseIndex + 1] = this.clamp(relY);
            inputs[baseIndex + 2] = this.clamp(velX);
            inputs[baseIndex + 3] = this.clamp(velY);
        } else {
            // No boid available - use zero values
            inputs[baseIndex] = 0;
            inputs[baseIndex + 1] = 0;
            inputs[baseIndex + 2] = 0;
            inputs[baseIndex + 3] = 0;
        }
    }
    
    // Last 2 elements: predator velocity (normalized by max velocity)
    inputs[20] = this.clamp(predatorVel.x / this.maxVelocity);
    inputs[21] = this.clamp(predatorVel.y / this.maxVelocity);
    
    return inputs;
};

/**
 * Find boids within rectangular vision range (supports edge wrapping)
 * @param {Array} boids - Array of boid objects
 * @param {Object} predatorPos - Predator position {x, y}
 * @param {number} canvasWidth - Canvas width for edge wrapping
 * @param {number} canvasHeight - Canvas height for edge wrapping
 * @returns {Array} Up to 5 nearest boids within vision, with relative positions
 */
InputProcessor.prototype.findBoidsInVision = function(boids, predatorPos, canvasWidth, canvasHeight) {
    var visibleBoids = [];
    var halfWidth = this.visionWidth / 2;
    var halfHeight = this.visionHeight / 2;
    
    for (var i = 0; i < boids.length; i++) {
        var boid = boids[i];
        
        // Calculate relative position with edge wrapping
        var relativePos = this.calculateRelativePosition(
            boid.position, predatorPos, canvasWidth, canvasHeight
        );
        
        // Check if boid is within rectangular vision area
        if (Math.abs(relativePos.x) <= halfWidth && Math.abs(relativePos.y) <= halfHeight) {
            // Calculate distance for sorting
            var distance = Math.sqrt(relativePos.x * relativePos.x + relativePos.y * relativePos.y);
            
            visibleBoids.push({
                boid: boid,
                relativePos: relativePos,
                distance: distance
            });
        }
    }
    
    // Sort by distance and take closest ones
    visibleBoids.sort(function(a, b) { return a.distance - b.distance; });
    return visibleBoids.slice(0, this.maxBoids);
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

 