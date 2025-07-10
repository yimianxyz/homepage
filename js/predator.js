/**
 * Base Predator Class
 * 
 * This file contains the base predator functionality that is extended by NeuralPredator.
 * It provides core mechanics like movement, feeding, and boundary handling with device-independent speed and size for training consistency.
 * 
 * The actual predator behavior is implemented in neural_predator.js which uses
 * a neural network for intelligent hunting and learning.
 */

// Base Predator Configuration (Extended by NeuralPredator)
var PREDATOR_BASE_MAX_SPEED = 2.5; // Base speed for reference screen size
var PREDATOR_MAX_FORCE = 0.05;
var PREDATOR_SIZE = 17; // Fixed middle size for consistent training behavior

// Device-independent speed scaling
function getPredatorMaxSpeed(canvasWidth, canvasHeight) {
    // Scale speed based on screen size to maintain consistent relative movement
    var referenceScreenSize = 1000; // Reference screen dimension
    var currentScreenSize = (canvasWidth + canvasHeight) / 2; // Average screen dimension
    var speedScale = currentScreenSize / referenceScreenSize;
    return PREDATOR_BASE_MAX_SPEED * speedScale;
}

/**
 * Base Predator Class
 * 
 * Provides core functionality for predator entities.
 * Extended by NeuralPredator for AI behavior.
 */
function Predator(x, y, simulation) {
    this.position = new Vector(x, y);
    this.velocity = new Vector(Math.random() * 2 - 1, Math.random() * 2 - 1);
    this.acceleration = new Vector(0, 0);
    this.simulation = simulation;
    
    // Fixed size for consistent training behavior
    this.baseSize = PREDATOR_SIZE;
    this.currentSize = PREDATOR_SIZE; // Fixed size, no growth/decay for training consistency
    this.lastFeedTime = 0;
    this.feedCooldown = 100; // Minimum time between feeding (ms)
}

Predator.prototype = {
    
    // Basic steering behavior - seek a target position
    seek: function(targetPosition) {
        var desiredVector = targetPosition.subtract(this.position);
        var maxSpeed = getPredatorMaxSpeed(this.simulation.canvasWidth, this.simulation.canvasHeight);
        desiredVector.iFastSetMagnitude(maxSpeed);
        var steeringVector = desiredVector.subtract(this.velocity);
        steeringVector.iFastLimit(PREDATOR_MAX_FORCE);
        return steeringVector;
    },
    
    // Wrap-around boundary handling (similar to boids)
    bound: function() {
        var BORDER_OFFSET = 20;
        
        if (this.position.x > this.simulation.canvasWidth + BORDER_OFFSET) {
            this.position.x = -BORDER_OFFSET;
        }
        if (this.position.x < -BORDER_OFFSET) {
            this.position.x = this.simulation.canvasWidth + BORDER_OFFSET;
        }
        if (this.position.y > this.simulation.canvasHeight + BORDER_OFFSET) {
            this.position.y = -BORDER_OFFSET;
        }
        if (this.position.y < -BORDER_OFFSET) {
            this.position.y = this.simulation.canvasHeight + BORDER_OFFSET;
        }
    },
    
    // Check for boid collisions and handle feeding
    checkForPrey: function(boids) {
        var currentTime = Date.now();
        if (currentTime - this.lastFeedTime < this.feedCooldown) {
            return []; // Still digesting
        }
        
        var caughtBoids = [];
        var catchRadius = this.currentSize * 0.7; // Catch radius scales with size
        
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < catchRadius) {
                caughtBoids.push(i);
                this.feed();
                break; // Only catch one boid at a time for smooth animation
            }
        }
        
        return caughtBoids;
    },
    
    // Handle feeding - fixed size for training consistency
    feed: function() {
        // Size remains constant for consistent training behavior
        this.lastFeedTime = Date.now();
    },
    
    // Size decay disabled for training consistency
    decaySize: function() {
        // Size remains constant for consistent training behavior
    },

    // Base update method - extended by subclasses for behavior
    update: function(boids) {
        // Note: Steering force application is handled by subclasses
        // This base method only handles physics and maintenance
        
        // Update velocity and position
        this.velocity.iAdd(this.acceleration);
        var maxSpeed = getPredatorMaxSpeed(this.simulation.canvasWidth, this.simulation.canvasHeight);
        this.velocity.iFastLimit(maxSpeed);
        this.position.iAdd(this.velocity);
        
        // Handle boundaries
        this.bound();
        
        // Handle size decay
        this.decaySize();
        
        // Reset acceleration for next frame
        this.acceleration.iMultiplyBy(0);
    },
    
    // Base render method - should be overridden by subclasses
    render: function() {
        // This is a placeholder - subclasses should implement their own rendering
        console.warn('Base Predator render() called - should be overridden by subclass');
    }
}; 