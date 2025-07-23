/**
 * Reward Processor - RL reward calculation with retroactive catch rewards
 * 
 * This module calculates rewards for reinforcement learning training based on:
 * 1. Approaching reward - immediate reward for getting closer to boids
 * 2. Catch retro reward - retroactive reward for actions that lead to catches
 * 
 * The processor looks ahead up to MAX_RETRO_REWARD_STEPS to assign retroactive
 * rewards when catches occur in the future.
 */

/**
 * Reward Processor Constructor
 * @constructor
 */
function RewardProcessor() {
    // Use centralized constants
    this.maxRetroSteps = window.SIMULATION_CONSTANTS.MAX_RETRO_REWARD_STEPS;
    this.maxDistance = window.SIMULATION_CONSTANTS.MAX_DISTANCE;
    this.unifiedMaxVelocity = Math.max(
        window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
}

/**
 * Process rewards for a sequence of steps
 * 
 * @param {Array} rewardInputs - Array of reward input objects:
 *   [{
 *     caughtBoids: [boidId1, boidId2, ...],  // IDs of boids caught this step
 *     state: {context, predator, boids},      // Policy standard input format
 *     action: [x, y]                          // Policy standard output format
 *   }, ...]
 * @param {boolean} isEpisodeEnd - True if the last step is episode end
 * @returns {Array} Array of reward objects:
 *   [{
 *     total: number,        // Total reward for this step
 *     approaching: number,  // Approaching reward component
 *     catchRetro: number    // Catch retro reward component
 *   }, ...]
 */
RewardProcessor.prototype.processRewards = function(rewardInputs, isEpisodeEnd) {
    if (!rewardInputs || rewardInputs.length === 0) {
        return [];
    }
    
    // Determine output length based on episode end
    var outputLength;
    if (isEpisodeEnd) {
        outputLength = rewardInputs.length;
    } else {
        outputLength = Math.max(0, rewardInputs.length - this.maxRetroSteps);
    }
    
    if (outputLength === 0) {
        return [];
    }
    
    var rewards = [];
    
    // Calculate rewards for each output step
    for (var i = 0; i < outputLength; i++) {
        var approachingReward = this._calculateApproachingReward(rewardInputs[i]);
        var catchRetroReward = this._calculateCatchRetroReward(rewardInputs, i, isEpisodeEnd);
        
        var totalReward = approachingReward + catchRetroReward;
        
        rewards.push({
            total: totalReward,
            approaching: approachingReward,
            catchRetro: catchRetroReward
        });
    }
    
    return rewards;
};

/**
 * Calculate approaching reward for a single step
 * Rewards getting closer to the nearest boid and moving toward it
 * 
 * @param {Object} rewardInput - Single reward input object
 * @returns {number} Approaching reward value
 * @private
 */
RewardProcessor.prototype._calculateApproachingReward = function(rewardInput) {
    var state = rewardInput.state;
    var action = rewardInput.action;
    
    if (!state.boids || state.boids.length === 0) {
        return 0;
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
    
    if (!closestBoid || minDistance < 0.001) {
        return 0;
    }
    
    // Calculate approach direction (normalized)
    var approachDirX = closestBoid.relX / minDistance;
    var approachDirY = closestBoid.relY / minDistance;
    
    // Calculate how well the action aligns with approach direction
    var actionAlignment = action[0] * approachDirX + action[1] * approachDirY;
    
    // Calculate proximity reward (closer = better, max reward at distance 0)
    var proximityReward = Math.max(0, 1 - minDistance);
    
    // Calculate relative velocity reward (boid and predator moving toward each other)
    var predatorVel = state.predator;
    var relativeVelX = closestBoid.velX - predatorVel.velX;
    var relativeVelY = closestBoid.velY - predatorVel.velY;
    var convergenceRate = -(relativeVelX * approachDirX + relativeVelY * approachDirY);
    var velocityReward = Math.max(0, convergenceRate) * 0.5;
    
    // Combine rewards (keep approaching reward much smaller than catch reward)
    var baseReward = proximityReward * 0.05 + velocityReward * 0.03;
    var alignmentBonus = Math.max(0, actionAlignment) * baseReward * 0.1;
    
    return baseReward + alignmentBonus;
};

/**
 * Calculate catch retro reward for a single step
 * Looks ahead for catches and assigns retroactive rewards
 * 
 * @param {Array} rewardInputs - All reward input objects
 * @param {number} currentStep - Index of current step
 * @param {boolean} isEpisodeEnd - True if episode ends
 * @returns {number} Catch retro reward value
 * @private
 */
RewardProcessor.prototype._calculateCatchRetroReward = function(rewardInputs, currentStep, isEpisodeEnd) {
    var totalRetroReward = 0;
    
    // Determine how far ahead we can look
    var lookAheadLimit;
    if (isEpisodeEnd) {
        lookAheadLimit = Math.min(rewardInputs.length, currentStep + this.maxRetroSteps + 1);
    } else {
        lookAheadLimit = currentStep + this.maxRetroSteps + 1;
    }
    

    
            // Look ahead for catches
        for (var futureStep = currentStep + 1; futureStep < lookAheadLimit; futureStep++) {
            var caughtBoids = rewardInputs[futureStep].caughtBoids;
            
            if (caughtBoids && caughtBoids.length > 0) {
                // For each caught boid, find when they started approaching each other
                for (var c = 0; c < caughtBoids.length; c++) {
                    var caughtBoidId = caughtBoids[c];
                    var firstApproachStep = this._findFirstApproachStep(
                        rewardInputs, 
                        currentStep, 
                        futureStep, 
                        caughtBoidId
                    );
                    
                    if (firstApproachStep !== -1) {
                        // Calculate retro reward for this catch
                        var stepsFromFirst = currentStep - firstApproachStep;
                        var totalSteps = futureStep - firstApproachStep;
                        
                        if (currentStep >= firstApproachStep && totalSteps > 0) {
                            // Linear increase: closer to catch = higher reward
                            var progressRatio = (stepsFromFirst + 1) / (totalSteps + 1);
                            var catchReward = progressRatio * 10.0; // Base catch reward
                            totalRetroReward += catchReward;
                        }
                    }
                }
            }
        }
    
    return totalRetroReward;
};

/**
 * Find the first step where predator and boid started approaching each other
 * 
 * @param {Array} rewardInputs - All reward input objects  
 * @param {number} startStep - Start looking from this step
 * @param {number} catchStep - Step where catch occurred
 * @param {number} boidId - ID of the caught boid
 * @returns {number} Step index where approach started, or -1 if not found
 * @private
 */
RewardProcessor.prototype._findFirstApproachStep = function(rewardInputs, startStep, catchStep, boidId) {
    // Look backwards from catch step to find first approach
    var maxLookback = Math.max(0, catchStep - this.maxRetroSteps);
    var searchStart = Math.max(startStep, maxLookback);
    

    
    for (var step = searchStart; step <= catchStep; step++) {
        var state = rewardInputs[step].state;
        
        // Find the target boid in this step's state
        var targetBoid = null;
        for (var i = 0; i < state.boids.length; i++) {
            if (state.boids[i].id === boidId) {
                targetBoid = state.boids[i];
                break;
            }
        }
        
        if (targetBoid) {
            // Check if they're moving toward each other
            var isApproaching = this._areMovingTowardEachOther(state.predator, targetBoid);
            if (isApproaching) {
                return step;
            }
        }
    }
    
    return -1;
};

/**
 * Check if predator and boid are moving toward each other
 * 
 * @param {Object} predator - Predator state {velX, velY}
 * @param {Object} boid - Boid state {relX, relY, velX, velY}
 * @returns {boolean} True if moving toward each other
 * @private
 */
RewardProcessor.prototype._areMovingTowardEachOther = function(predator, boid) {
    // Calculate relative position vector (from predator to boid)
    var distance = Math.sqrt(boid.relX * boid.relX + boid.relY * boid.relY);
    if (distance < 0.001) {
        return false;
    }
    
    var dirToBoidX = boid.relX / distance;
    var dirToBoidY = boid.relY / distance;
    
    // Calculate relative velocity (boid velocity - predator velocity)
    var relVelX = boid.velX - predator.velX;
    var relVelY = boid.velY - predator.velY;
    
    // Dot product: negative means moving toward each other
    var velocityDotProduct = relVelX * dirToBoidX + relVelY * dirToBoidY;
    
    return velocityDotProduct < 0;
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