/**
 * Simple Reward Processor - Instant reward calculation for RL training
 * 
 * This module calculates instant rewards for each simulation step based on:
 * 1. Catch reward - dominant reward for catching boids (instant)
 * 2. Approaching reward - small reward for getting closer to boids (instant)
 * 
 * Design principle: Occam's razor - simple, clean, single-step processing
 */

/**
 * Simple reward processor for RL training with instant rewards
 * @constructor
 */
function RewardProcessor() {
    // Use centralized constants from config
    this.baseCatchReward = window.SIMULATION_CONSTANTS.BASE_CATCH_REWARD;
    this.proximityMultiplier = window.SIMULATION_CONSTANTS.APPROACHING_PROXIMITY_MULTIPLIER;
    this.velocityMultiplier = window.SIMULATION_CONSTANTS.APPROACHING_VELOCITY_MULTIPLIER;  
    this.alignmentMultiplier = window.SIMULATION_CONSTANTS.APPROACHING_ALIGNMENT_MULTIPLIER;
    this.minDistanceThreshold = window.SIMULATION_CONSTANTS.MIN_DISTANCE_THRESHOLD;
    
    // Normalization constants
    this.maxDistance = window.SIMULATION_CONSTANTS.MAX_DISTANCE;
    this.unifiedMaxVelocity = Math.max(
        window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
}

/**
 * Calculate instant reward for a single simulation step
 * 
 * @param {Object} stepInput - Single step input object:
 *   {
 *     state: {context, predator, boids},      // Policy standard input format
 *     action: [x, y],                         // Policy standard output format  
 *     caughtBoids: [boidId1, boidId2, ...]    // IDs of boids caught this step
 *   }
 * @returns {Object} Reward object:
 *   {
 *     total: number,        // Total reward for this step
 *     approaching: number,  // Approaching reward component
 *     catch: number         // Catch reward component
 *   }
 */
RewardProcessor.prototype.calculateStepReward = function(stepInput) {
    if (!stepInput || !stepInput.state || !stepInput.action) {
        return {
            total: 0.0,
            approaching: 0.0,
            catch: 0.0
        };
    }
    
    // Calculate approaching reward (same logic as before, simplified)
    var approachingReward = this._calculateApproachingReward(stepInput.state, stepInput.action);
    
    // Calculate catch reward (simple: count * base reward)
    var catchReward = this._calculateCatchReward(stepInput.caughtBoids || []);
    
    // Combine rewards
    var totalReward = approachingReward + catchReward;
    
    return {
        total: totalReward,
        approaching: approachingReward,
        catch: catchReward
    };
};

/**
 * Calculate approaching reward - rewards getting closer to the nearest boid
 * 
 * @param {Object} state - Policy standard input format
 * @param {Array} action - Policy output [x, y]
 * @returns {number} Approaching reward value
 * @private
 */
RewardProcessor.prototype._calculateApproachingReward = function(state, action) {
    if (!state.boids || state.boids.length === 0) {
        return 0.0;
    }
    
    // Find closest boid
    var closestBoid = null;
    var minDistance = Infinity;
    
    for (var i = 0; i < state.boids.length; i++) {
        var boid = state.boids[i];
        var distance = Math.sqrt(boid.relX * boid.relX + boid.relY * boid.relY);
        
        if (distance < minDistance) {
            minDistance = distance;
            closestBoid = boid;
        }
    }
    
    if (!closestBoid || minDistance < this.minDistanceThreshold) {
        return 0.0;
    }
    
    // Calculate approach direction (normalized)
    var approachDirX = closestBoid.relX / minDistance;
    var approachDirY = closestBoid.relY / minDistance;
    
    // Action alignment reward - how well action aligns with approach direction
    var actionAlignment = action[0] * approachDirX + action[1] * approachDirY;
    var alignmentReward = Math.max(0, actionAlignment) * this.alignmentMultiplier;
    
    // Proximity reward - closer is better (max reward at distance 0)
    var proximityReward = Math.max(0, 1 - minDistance) * this.proximityMultiplier;
    
    // Velocity convergence reward - predator and boid moving toward each other
    var predatorVel = state.predator;
    var relativeVelX = closestBoid.velX - predatorVel.velX;
    var relativeVelY = closestBoid.velY - predatorVel.velY;
    var convergenceRate = -(relativeVelX * approachDirX + relativeVelY * approachDirY);
    var velocityReward = Math.max(0, convergenceRate) * this.velocityMultiplier;
    
    return proximityReward + velocityReward + alignmentReward;
};

/**
 * Calculate catch reward - simple count-based reward
 * 
 * @param {Array} caughtBoids - Array of caught boid IDs
 * @returns {number} Catch reward value
 * @private
 */
RewardProcessor.prototype._calculateCatchReward = function(caughtBoids) {
    if (!caughtBoids || caughtBoids.length === 0) {
        return 0.0;
    }
    
    // Simple: base reward per boid caught
    return caughtBoids.length * this.baseCatchReward;
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        RewardProcessor: RewardProcessor
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.RewardProcessor = RewardProcessor;
} 