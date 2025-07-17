/**
 * Simulation Utilities Module
 * 
 * This module provides utilities for simulation management, performance monitoring,
 * and policy handling.
 */

/**
 * Performance monitor for tracking simulation performance
 * @constructor
 */
function PerformanceMonitor() {
    this.frameCount = 0;
    this.lastTime = performance.now();
    this.fps = 0;
    this.frameTime = 0;
    this.updateInterval = 60; // Update every 60 frames
    this.frameTimeHistory = [];
    this.maxHistorySize = 100;
}

/**
 * Update performance metrics
 */
PerformanceMonitor.prototype.update = function() {
    this.frameCount++;
    var currentTime = performance.now();
    this.frameTime = currentTime - this.lastTime;
    
    // Add to history
    this.frameTimeHistory.push(this.frameTime);
    if (this.frameTimeHistory.length > this.maxHistorySize) {
        this.frameTimeHistory.shift();
    }
    
    if (this.frameCount % this.updateInterval === 0) {
        this.fps = 1000 / this.frameTime;
    }
    
    this.lastTime = currentTime;
};

/**
 * Get current performance metrics
 * @returns {Object} Performance metrics
 */
PerformanceMonitor.prototype.getMetrics = function() {
    return {
        fps: Math.round(this.fps),
        frameTime: Math.round(this.frameTime * 100) / 100,
        frameCount: this.frameCount,
        averageFrameTime: this.getAverageFrameTime(),
        minFrameTime: Math.min.apply(Math, this.frameTimeHistory),
        maxFrameTime: Math.max.apply(Math, this.frameTimeHistory)
    };
};

/**
 * Get average frame time over history
 * @returns {number} Average frame time
 */
PerformanceMonitor.prototype.getAverageFrameTime = function() {
    if (this.frameTimeHistory.length === 0) return 0;
    
    var sum = 0;
    for (var i = 0; i < this.frameTimeHistory.length; i++) {
        sum += this.frameTimeHistory[i];
    }
    
    return Math.round((sum / this.frameTimeHistory.length) * 100) / 100;
};

/**
 * Reset performance metrics
 */
PerformanceMonitor.prototype.reset = function() {
    this.frameCount = 0;
    this.lastTime = performance.now();
    this.fps = 0;
    this.frameTime = 0;
    this.frameTimeHistory = [];
};

/**
 * Create policy based on type
 * @param {string} policyType - Type of policy ('closest_pursuit', 'transformer', etc.)
 * @param {Object} options - Policy options
 * @returns {Object} Policy instance
 */
function createPolicy(policyType, options) {
    options = options || {};
    
    switch (policyType) {
        case 'closest_pursuit':
            if (typeof createClosestPursuitPolicy === 'function') {
                return createClosestPursuitPolicy();
            } else {
                throw new Error('Closest pursuit policy not available');
            }
            
        case 'transformer':
            if (options.transformerParams) {
                if (typeof createTransformerPolicy === 'function') {
                    return createTransformerPolicy(options.transformerParams);
                } else {
                    throw new Error('Transformer policy not available');
                }
            } else if (typeof window !== 'undefined' && window.TRANSFORMER_PARAMS) {
                if (typeof createTransformerPolicyFromGlobal === 'function') {
                    return createTransformerPolicyFromGlobal();
                } else {
                    throw new Error('Transformer policy not available');
                }
            } else {
                console.warn('No transformer parameters available, falling back to closest pursuit');
                return createPolicy('closest_pursuit');
            }
            
        default:
            console.warn('Unknown policy type:', policyType, 'falling back to closest pursuit');
            return createPolicy('closest_pursuit');
    }
}

/**
 * Simulation statistics tracker
 * @constructor
 */
function SimulationStats() {
    this.totalFrames = 0;
    this.totalCaught = 0;
    this.startTime = Date.now();
    this.pausedTime = 0;
    this.lastPauseStart = null;
    this.boidCounts = [];
    this.catchEvents = [];
}

/**
 * Update simulation statistics
 * @param {number} boidCount - Current number of boids
 * @param {number} caughtCount - Number of boids caught this frame
 */
SimulationStats.prototype.update = function(boidCount, caughtCount) {
    this.totalFrames++;
    this.totalCaught += caughtCount;
    
    // Track boid count history
    this.boidCounts.push(boidCount);
    if (this.boidCounts.length > 1000) {
        this.boidCounts.shift();
    }
    
    // Track catch events
    if (caughtCount > 0) {
        this.catchEvents.push({
            frame: this.totalFrames,
            count: caughtCount,
            timestamp: Date.now()
        });
    }
};

/**
 * Start pause tracking
 */
SimulationStats.prototype.startPause = function() {
    this.lastPauseStart = Date.now();
};

/**
 * End pause tracking
 */
SimulationStats.prototype.endPause = function() {
    if (this.lastPauseStart) {
        this.pausedTime += Date.now() - this.lastPauseStart;
        this.lastPauseStart = null;
    }
};

/**
 * Get simulation statistics
 * @returns {Object} Simulation statistics
 */
SimulationStats.prototype.getStats = function() {
    var currentTime = Date.now();
    var runTime = currentTime - this.startTime - this.pausedTime;
    
    return {
        totalFrames: this.totalFrames,
        totalCaught: this.totalCaught,
        runTime: runTime,
        averageFPS: this.totalFrames / (runTime / 1000),
        catchRate: this.totalCaught / Math.max(1, this.totalFrames),
        currentBoidCount: this.boidCounts.length > 0 ? this.boidCounts[this.boidCounts.length - 1] : 0,
        catchEvents: this.catchEvents.length
    };
};

/**
 * Reset simulation statistics
 */
SimulationStats.prototype.reset = function() {
    this.totalFrames = 0;
    this.totalCaught = 0;
    this.startTime = Date.now();
    this.pausedTime = 0;
    this.lastPauseStart = null;
    this.boidCounts = [];
    this.catchEvents = [];
};

/**
 * Animation frame manager
 * @constructor
 */
function AnimationManager() {
    this.animationId = null;
    this.isRunning = false;
    this.callback = null;
    this.targetFPS = 60;
    this.frameInterval = 1000 / this.targetFPS;
    this.lastFrameTime = 0;
}

/**
 * Start animation loop
 * @param {Function} callback - Function to call each frame
 */
AnimationManager.prototype.start = function(callback) {
    if (this.isRunning) return;
    
    this.callback = callback;
    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.animate();
};

/**
 * Stop animation loop
 */
AnimationManager.prototype.stop = function() {
    this.isRunning = false;
    if (this.animationId) {
        cancelAnimationFrame(this.animationId);
        this.animationId = null;
    }
};

/**
 * Set target FPS
 * @param {number} fps - Target frames per second
 */
AnimationManager.prototype.setTargetFPS = function(fps) {
    this.targetFPS = fps;
    this.frameInterval = 1000 / fps;
};

/**
 * Animation loop
 */
AnimationManager.prototype.animate = function() {
    if (!this.isRunning) return;
    
    var currentTime = performance.now();
    var deltaTime = currentTime - this.lastFrameTime;
    
    if (deltaTime >= this.frameInterval) {
        if (this.callback) {
            this.callback(deltaTime);
        }
        this.lastFrameTime = currentTime;
    }
    
    this.animationId = requestAnimationFrame(this.animate.bind(this));
};

/**
 * Check if animation is running
 * @returns {boolean} True if running
 */
AnimationManager.prototype.isAnimating = function() {
    return this.isRunning;
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PerformanceMonitor: PerformanceMonitor,
        createPolicy: createPolicy,
        SimulationStats: SimulationStats,
        AnimationManager: AnimationManager
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.PerformanceMonitor = PerformanceMonitor;
    window.createPolicy = createPolicy;
    window.SimulationStats = SimulationStats;
    window.AnimationManager = AnimationManager;
} 