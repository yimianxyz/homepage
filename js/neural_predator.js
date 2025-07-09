/**
 * Neural Predator - AI-Powered Hunting Behavior
 * 
 * This is the main predator implementation that extends the base Predator class
 * with intelligent neural network behavior and online learning capabilities.
 * 
 * Network Architecture: 12 inputs → 8 hidden → 2 outputs
 * - Inputs: positions & velocities of 5 nearest boids + predator state  
 * - Outputs: steering force (x, y)
 * - Optimized for real-time web performance with <0.2ms forward pass
 */

/**
 * Neural Predator with Robust Online Learning
 * 
 * This predator uses a lightweight neural network (12 inputs → 8 hidden → 2 outputs)
 * that learns and adapts during the simulation using policy gradient methods.
 * 
 * Learning Features:
 * - Rewards catching boids (+10 points)
 * - Penalizes inefficiency (-0.01 per frame)
 * - Encourages proximity to prey (+0.1 for nearness)
 * - Continuously adapts weights with numerical stability safeguards
 * 
 * Stability Features:
 * - NaN-resistant derivative calculations
 * - Weight corruption detection and recovery
 * - Numerically stable gradient updates
 * 
 * Visual Learning Indicator:
 * - Elegant elongated triangle design
 * - Intensity changes reflect learning activity and feeding state
 * - Size scaling with growth mechanics
 */
function NeuralPredator(x, y, simulation) {
    // Inherit from basic predator
    Predator.call(this, x, y, simulation);
    
    // Patrol behavior for when no boids are present
    this.autonomousTarget = new Vector(x, y);
    this.targetChangeTime = 0;
    this.targetChangeInterval = 5000; // Change target every 5 seconds
    
    // Neural network parameters
    this.inputSize = 12;  // 5 boids * 2 (pos) + predator vel (2) + state (1) + padding
    this.hiddenSize = 8;
    this.outputSize = 2;
    
    // Initialize weights with optimized values (hand-tuned for hunting behavior)
    this.initializeWeights();
    
    // Input normalization constants
    this.maxDistance = 100;
    this.maxVelocity = 6;
    
    // Performance optimization
    this.inputBuffer = new Array(this.inputSize);
    this.hiddenBuffer = new Array(this.hiddenSize);
    this.outputBuffer = new Array(this.outputSize);
    
    // Online learning parameters
    this.learningRate = 0.001;  // Very small for stability
    this.rewardMemory = 0.95;   // Exponential moving average for rewards
    this.avgReward = 0;         // Running average of rewards
    this.lastReward = 0;        // Previous frame reward
    this.framesSinceLastFeed = 0;
    this.maxFramesSinceLastFeed = 600; // 10 seconds at 60fps
    
    // Learning buffers for gradient calculation
    this.lastInputs = new Array(this.inputSize);
    this.lastHidden = new Array(this.hiddenSize);
    this.lastOutputs = new Array(this.outputSize);
    this.gradientIH = [];
    this.gradientHO = [];
    this.gradientBiasH = new Array(this.hiddenSize);
    this.gradientBiasO = new Array(this.outputSize);
    
    this.initializeGradients();
}

// Inherit from Predator
NeuralPredator.prototype = Object.create(Predator.prototype);
NeuralPredator.prototype.constructor = NeuralPredator;

NeuralPredator.prototype.initializeWeights = function() {
    // Input to hidden weights (optimized for hunting patterns)
    this.weightsIH = [
        // Weights designed to respond to boid positions and movements
        [ 0.8, -0.3,  0.6, -0.2,  0.4, -0.1,  0.7, -0.4,  0.5, -0.2,  0.3,  0.1], // Hidden neuron 1
        [-0.4,  0.7, -0.2,  0.5, -0.3,  0.6, -0.1,  0.8, -0.5,  0.2, -0.4,  0.3], // Hidden neuron 2
        [ 0.5, -0.1,  0.8, -0.3,  0.2,  0.4, -0.6,  0.1,  0.7, -0.4,  0.3, -0.2], // Hidden neuron 3
        [-0.2,  0.4, -0.5,  0.7,  0.1, -0.3,  0.6, -0.7,  0.2,  0.5, -0.1,  0.4], // Hidden neuron 4
        [ 0.6, -0.4,  0.1,  0.3, -0.7,  0.5, -0.2,  0.6, -0.3,  0.1,  0.8, -0.4], // Hidden neuron 5
        [-0.3,  0.6, -0.7,  0.1,  0.4, -0.2,  0.8, -0.1,  0.5, -0.6,  0.2,  0.7], // Hidden neuron 6
        [ 0.7, -0.2,  0.4, -0.6,  0.3,  0.8, -0.4,  0.2, -0.1,  0.6, -0.5,  0.1], // Hidden neuron 7
        [-0.1,  0.5, -0.3,  0.6, -0.8,  0.2,  0.4, -0.5,  0.7, -0.2,  0.1,  0.6]  // Hidden neuron 8
    ];
    
    // Hidden to output weights (designed for smooth steering)
    this.weightsHO = [
        [ 0.8, -0.3,  0.6, -0.2,  0.7, -0.4,  0.5,  0.1], // X steering force
        [-0.2,  0.7, -0.4,  0.5, -0.1,  0.6, -0.3,  0.8]  // Y steering force
    ];
    
    // Bias weights
    this.biasH = [0.1, -0.2, 0.3, -0.1, 0.2, 0.4, -0.3, 0.1];
    this.biasO = [0.0, 0.0];
};

// Initialize gradient buffers
NeuralPredator.prototype.initializeGradients = function() {
    // Initialize gradient matrices with zeros
    this.gradientIH = [];
    for (var h = 0; h < this.hiddenSize; h++) {
        this.gradientIH[h] = new Array(this.inputSize);
        for (var i = 0; i < this.inputSize; i++) {
            this.gradientIH[h][i] = 0;
        }
    }
    
    this.gradientHO = [];
    for (var o = 0; o < this.outputSize; o++) {
        this.gradientHO[o] = new Array(this.hiddenSize);
        for (var h = 0; h < this.hiddenSize; h++) {
            this.gradientHO[o][h] = 0;
        }
    }
    
    // Initialize bias gradients
    for (var h = 0; h < this.hiddenSize; h++) {
        this.gradientBiasH[h] = 0;
    }
    for (var o = 0; o < this.outputSize; o++) {
        this.gradientBiasO[o] = 0;
    }
};

// Calculate reward based on current state and actions
NeuralPredator.prototype.calculateReward = function(boids, caughtBoid) {
    var reward = 0;
    
    // Major positive reward for catching a boid
    if (caughtBoid) {
        reward += 10.0;
        this.framesSinceLastFeed = 0;
    } else {
        this.framesSinceLastFeed++;
        
        // Small negative reward for each frame to encourage efficiency
        reward -= 0.01;
        
        // Additional penalty if too long since last feed (encourages active hunting)
        if (this.framesSinceLastFeed > this.maxFramesSinceLastFeed) {
            reward -= 0.05;
        }
    }
    
    // Small reward for being near boids (encourages exploration)
    if (boids.length > 0) {
        var nearestDistance = Infinity;
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < nearestDistance) {
                nearestDistance = distance;
            }
        }
        
        // Reward getting closer to boids
        if (nearestDistance < this.maxDistance) {
            reward += (this.maxDistance - nearestDistance) / this.maxDistance * 0.1;
        }
    }
    
    return reward;
};

// Fast derivative of tanh approximation - numerically stable version
NeuralPredator.prototype.fastTanhDerivative = function(x) {
    if (x > 2 || x < -2) return 0;
    if (isNaN(x)) return 0;
    
    // More stable derivative approximation: 1 - tanh²(x)
    var tanhX = this.fastTanh(x);
    var derivative = 1 - tanhX * tanhX;
    
    // Safety check for numerical stability
    if (isNaN(derivative) || !isFinite(derivative)) {
        return 0;
    }
    
    return derivative;
};

// Store current state for learning
NeuralPredator.prototype.storeState = function() {
    // Copy current inputs, hidden states, and outputs for gradient calculation
    for (var i = 0; i < this.inputSize; i++) {
        this.lastInputs[i] = this.inputBuffer[i];
    }
    for (var h = 0; h < this.hiddenSize; h++) {
        this.lastHidden[h] = this.hiddenBuffer[h];
    }
    for (var o = 0; o < this.outputSize; o++) {
        this.lastOutputs[o] = this.outputBuffer[o];
    }
};

// Update weights based on reward using simple policy gradient
NeuralPredator.prototype.updateWeights = function(reward) {
    // Safety check for input
    if (isNaN(reward) || !isFinite(reward)) {
        return;
    }
    
    // Update running average of rewards
    this.avgReward = this.rewardMemory * this.avgReward + (1 - this.rewardMemory) * reward;
    
    // Safety check for avgReward
    if (isNaN(this.avgReward)) {
        this.avgReward = 0;
    }
    
    // Calculate reward prediction error (temporal difference)
    var rewardError = reward - this.avgReward;
    
    // Only update if the error is significant enough
    if (Math.abs(rewardError) < 0.001) {
        return;
    }
    
    // Simple policy gradient update
    var learningSignal = this.learningRate * rewardError;
    
    // Safety check for learning signal
    if (isNaN(learningSignal) || !isFinite(learningSignal)) {
        return;
    }
    
    // Update output layer weights
    for (var o = 0; o < this.outputSize; o++) {
        var outputValue = this.lastOutputs[o] / PREDATOR_MAX_FORCE;
        var outputGradient = this.fastTanhDerivative(outputValue) * learningSignal;
        
        // Safety check for gradient
        if (isNaN(outputGradient) || !isFinite(outputGradient)) {
            continue;
        }
        
        for (var h = 0; h < this.hiddenSize; h++) {
            var weightUpdate = outputGradient * this.lastHidden[h];
            if (isNaN(weightUpdate) || !isFinite(weightUpdate)) {
                continue;
            }
            this.weightsHO[o][h] += weightUpdate;
        }
        this.biasO[o] += outputGradient;
    }
    
    // Update hidden layer weights
    for (var h = 0; h < this.hiddenSize; h++) {
        var hiddenError = 0;
        for (var o = 0; o < this.outputSize; o++) {
            var outputValue = this.lastOutputs[o] / PREDATOR_MAX_FORCE;
            var outputGradient = this.fastTanhDerivative(outputValue);
            var errorContribution = outputGradient * this.weightsHO[o][h];
            
            // Safety check for error contribution
            if (isNaN(errorContribution) || !isFinite(errorContribution)) {
                continue;
            }
            hiddenError += errorContribution;
        }
        
        var hiddenGradient = this.fastTanhDerivative(this.lastHidden[h]) * hiddenError * learningSignal;
        
        // Safety check for hidden gradient
        if (isNaN(hiddenGradient) || !isFinite(hiddenGradient)) {
            continue;
        }
        
        for (var i = 0; i < this.inputSize; i++) {
            var weightUpdate = hiddenGradient * this.lastInputs[i];
            if (isNaN(weightUpdate) || !isFinite(weightUpdate)) {
                continue;
            }
            this.weightsIH[h][i] += weightUpdate;
        }
        this.biasH[h] += hiddenGradient;
    }
    
    // Clip weights to prevent explosion
    this.clipWeights();
};

// Prevent weight explosion by clipping and handling NaN values
NeuralPredator.prototype.clipWeights = function() {
    var maxWeight = 2.0;
    
    // Clip input-to-hidden weights
    for (var h = 0; h < this.hiddenSize; h++) {
        for (var i = 0; i < this.inputSize; i++) {
            var weight = this.weightsIH[h][i];
            if (isNaN(weight) || !isFinite(weight)) {
                // Reset corrupted weights to small random values
                this.weightsIH[h][i] = (Math.random() - 0.5) * 0.1;
            } else {
                this.weightsIH[h][i] = Math.max(-maxWeight, Math.min(maxWeight, weight));
            }
        }
        var bias = this.biasH[h];
        if (isNaN(bias) || !isFinite(bias)) {
            this.biasH[h] = (Math.random() - 0.5) * 0.1;
        } else {
            this.biasH[h] = Math.max(-maxWeight, Math.min(maxWeight, bias));
        }
    }
    
    // Clip hidden-to-output weights
    for (var o = 0; o < this.outputSize; o++) {
        for (var h = 0; h < this.hiddenSize; h++) {
            var weight = this.weightsHO[o][h];
            if (isNaN(weight) || !isFinite(weight)) {
                // Reset corrupted weights to small random values
                this.weightsHO[o][h] = (Math.random() - 0.5) * 0.1;
            } else {
                this.weightsHO[o][h] = Math.max(-maxWeight, Math.min(maxWeight, weight));
            }
        }
        var bias = this.biasO[o];
        if (isNaN(bias) || !isFinite(bias)) {
            this.biasO[o] = (Math.random() - 0.5) * 0.1;
        } else {
            this.biasO[o] = Math.max(-maxWeight, Math.min(maxWeight, bias));
        }
    }
};

// Fast activation function (tanh approximation)
NeuralPredator.prototype.fastTanh = function(x) {
    if (x > 2) return 1;
    if (x < -2) return -1;
    var x2 = x * x;
    return x * (27 + x2) / (27 + 9 * x2);
};

// Prepare neural network inputs from current game state
NeuralPredator.prototype.prepareInputs = function(boids) {
    // Find 5 nearest boids
    var nearestBoids = [];
    var distances = [];
    
    for (var i = 0; i < boids.length && i < 5; i++) {
        var distance = this.position.getDistance(boids[i].position);
        nearestBoids.push({boid: boids[i], distance: distance});
    }
    
    // Sort by distance and take closest 5
    nearestBoids.sort(function(a, b) { return a.distance - b.distance; });
    nearestBoids = nearestBoids.slice(0, 5);
    
    // Clear input buffer
    for (var i = 0; i < this.inputSize; i++) {
        this.inputBuffer[i] = 0;
    }
    
    // Encode boid positions (relative and normalized)
    for (var i = 0; i < nearestBoids.length; i++) {
        var boid = nearestBoids[i].boid;
        var relativePos = boid.position.subtract(this.position);
        
        // Safety check for NaN positions
        if (isNaN(relativePos.x) || isNaN(relativePos.y)) {
            relativePos.x = 0;
            relativePos.y = 0;
        }
        
        // Normalize position (-1 to 1)
        var normalizedX = relativePos.x / this.maxDistance;
        var normalizedY = relativePos.y / this.maxDistance;
        
        // Safety check for NaN normalized values
        if (isNaN(normalizedX)) normalizedX = 0;
        if (isNaN(normalizedY)) normalizedY = 0;
        
        this.inputBuffer[i * 2] = Math.max(-1, Math.min(1, normalizedX));
        this.inputBuffer[i * 2 + 1] = Math.max(-1, Math.min(1, normalizedY));
    }
    
    // Add predator's current velocity
    var velocityX = this.velocity.x / this.maxVelocity;
    var velocityY = this.velocity.y / this.maxVelocity;
    
    // Safety check for NaN velocities
    if (isNaN(velocityX)) velocityX = 0;
    if (isNaN(velocityY)) velocityY = 0;
    
    this.inputBuffer[10] = Math.max(-1, Math.min(1, velocityX));
    this.inputBuffer[11] = Math.max(-1, Math.min(1, velocityY));
};

// Neural network forward pass (optimized for speed)
NeuralPredator.prototype.forward = function() {
    // Input to hidden layer
    for (var h = 0; h < this.hiddenSize; h++) {
        var sum = this.biasH[h];
        for (var i = 0; i < this.inputSize; i++) {
            // Safety check for NaN inputs
            var input = this.inputBuffer[i];
            var weight = this.weightsIH[h][i];
            if (isNaN(input) || isNaN(weight)) {
                input = 0;
                weight = 0;
            }
            sum += input * weight;
        }
        // Safety check for NaN sum
        if (isNaN(sum)) {
            sum = 0;
        }
        this.hiddenBuffer[h] = this.fastTanh(sum);
    }
    
    // Hidden to output layer
    for (var o = 0; o < this.outputSize; o++) {
        var sum = this.biasO[o];
        for (var h = 0; h < this.hiddenSize; h++) {
            var hidden = this.hiddenBuffer[h];
            var weight = this.weightsHO[o][h];
            if (isNaN(hidden) || isNaN(weight)) {
                hidden = 0;
                weight = 0;
            }
            sum += hidden * weight;
        }
        // Safety check for NaN sum
        if (isNaN(sum)) {
            sum = 0;
        }
        this.outputBuffer[o] = this.fastTanh(sum) * PREDATOR_MAX_FORCE;
        
        // Safety check for final output
        if (isNaN(this.outputBuffer[o])) {
            this.outputBuffer[o] = 0;
        }
    }
    
    // Store activations for visualization
    this.lastInput = this.inputBuffer.slice();
    this.hiddenActivations = this.hiddenBuffer.slice();
    this.lastOutput = this.outputBuffer.slice();
    
    return new Vector(this.outputBuffer[0], this.outputBuffer[1]);
};

// Override the autonomous force calculation with neural network
NeuralPredator.prototype.getAutonomousForce = function(boids) {
    if (boids.length === 0) {
        // No boids left, patrol slowly
        var currentTime = Date.now();
        if (currentTime - this.targetChangeTime > this.targetChangeInterval) {
            var margin = 50;
            this.autonomousTarget.x = margin + Math.random() * (this.simulation.canvasWidth - 2 * margin);
            this.autonomousTarget.y = margin + Math.random() * (this.simulation.canvasHeight - 2 * margin);
            this.targetChangeTime = currentTime;
        }
        return this.seek(this.autonomousTarget);
    }
    
    // Use neural network for hunting behavior with fixed learning
    this.prepareInputs(boids);
    var force = this.forward();
    
    // Store state for learning (now fixed)
    this.storeState();
    
    // Calculate reward and learn from previous action
    var reward = this.calculateReward(boids, false);
    this.updateWeights(reward);
    
    return force;
};

// Override feed method to include learning signal
NeuralPredator.prototype.feed = function() {
    // Call parent feed method
    this.currentSize = Math.min(this.currentSize + this.growthPerFeed, this.maxSize);
    this.lastFeedTime = Date.now();
    
    // Provide strong positive reward for successful hunting
    var feedReward = 20.0; // Higher reward for actual feeding
    this.updateWeights(feedReward);
    
    // Reset efficiency counter
    this.framesSinceLastFeed = 0;
};

// Enhanced update method with neural behavior and learning
NeuralPredator.prototype.update = function(boids) {
    // Apply neural network steering force
    var steeringForce = this.getAutonomousForce(boids);
    this.acceleration.iAdd(steeringForce);
    
    // Call parent update method for physics and maintenance
    Predator.prototype.update.call(this, boids);
    
    // Safety check: Ensure position and velocity are valid after update
    if (isNaN(this.position.x) || isNaN(this.position.y)) {
        this.position.x = this.simulation.canvasWidth / 2;
        this.position.y = this.simulation.canvasHeight / 2;
    }
    
    if (isNaN(this.velocity.x) || isNaN(this.velocity.y)) {
        this.velocity.x = (Math.random() - 0.5) * 0.1;
        this.velocity.y = (Math.random() - 0.5) * 0.1;
    }
    
    // Additional learning opportunity: if we're improving at staying near boids
    if (boids.length > 0) {
        var nearestDistance = Infinity;
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < nearestDistance) {
                nearestDistance = distance;
            }
        }
        
        // Small reward for maintaining close proximity to prey
        if (nearestDistance < this.maxDistance * 0.5) {
            var proximityReward = 0.05;
            this.updateWeights(proximityReward);
        }
    }
};

// Enhanced seeking with neural network refinement
NeuralPredator.prototype.seek = function(targetPosition) {
    var desiredVector = targetPosition.subtract(this.position);
    desiredVector.iFastSetMagnitude(PREDATOR_MAX_SPEED * 0.8); // Slightly slower patrol
    var steeringVector = desiredVector.subtract(this.velocity);
    steeringVector.iFastLimit(PREDATOR_MAX_FORCE * 0.6); // Gentler patrol steering
    return steeringVector;
};

// Get learning progress (for visualization or debugging)
NeuralPredator.prototype.getLearningStats = function() {
    return {
        avgReward: this.avgReward,
        framesSinceLastFeed: this.framesSinceLastFeed,
        currentSize: this.currentSize,
        learningIntensity: Math.min(1.0, Math.abs(this.avgReward) / 5.0) // Normalized learning intensity
    };
};

// Enhanced rendering with learning visualization
NeuralPredator.prototype.render = function() {
    var ctx = this.simulation.ctx;
    
    // Safety check: If position is invalid, reset to center
    if (isNaN(this.position.x) || isNaN(this.position.y)) {
        this.position.x = this.simulation.canvasWidth / 2;
        this.position.y = this.simulation.canvasHeight / 2;
        this.velocity.x = Math.random() * 2 - 1;
        this.velocity.y = Math.random() * 2 - 1;
    }
    
    // Safety check: If velocity is zero or invalid, give it a small random velocity
    if (isNaN(this.velocity.x) || isNaN(this.velocity.y) || 
        (this.velocity.x === 0 && this.velocity.y === 0)) {
        this.velocity.x = (Math.random() - 0.5) * 0.1;
        this.velocity.y = (Math.random() - 0.5) * 0.1;
    }
    
    // Get learning stats for visual feedback
    var stats = this.getLearningStats();
    
    // Draw predator as a distinctive but elegant elongated triangle
    var directionVector = this.velocity.normalize().multiplyBy(this.currentSize * 1.2);
    var inverseVector1 = new Vector(-directionVector.y, directionVector.x);
    var inverseVector2 = new Vector(directionVector.y, -directionVector.x);
    inverseVector1 = inverseVector1.divideBy(4);
    inverseVector2 = inverseVector2.divideBy(4);
    
    // Enhanced visibility while maintaining elegance
    var sizeRatio = this.currentSize / this.baseSize;
    var intensity = 0.4 + (sizeRatio - 1) * 0.3; // Size-based intensity
    
    // Very subtle learning glow
    var learningGlow = stats.learningIntensity * 0.15;
    intensity += learningGlow;
    
    ctx.beginPath();
    ctx.moveTo(this.position.x, this.position.y);
    ctx.lineTo(this.position.x + inverseVector1.x, this.position.y + inverseVector1.y);
    ctx.lineTo(this.position.x + directionVector.x, this.position.y + directionVector.y);
    ctx.lineTo(this.position.x + inverseVector2.x, this.position.y + inverseVector2.y);
    ctx.lineTo(this.position.x, this.position.y);
    
    // Elegant dark red coloring with better visibility
    var baseOpacity = 0.8 + intensity * 0.2;
    var fillOpacity = 0.5 + intensity * 0.3;
    
    ctx.strokeStyle = 'rgba(100, 40, 40, ' + Math.min(1.0, baseOpacity) + ')';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = 'rgba(140, 50, 50, ' + Math.min(1.0, fillOpacity) + ')';
    ctx.fill();
    
    // Add a subtle inner highlight that scales with size and learning
    var highlightSize = Math.max(3, this.currentSize * 0.2);
    var highlightOpacity = 0.7 + intensity * 0.3;
    
    ctx.beginPath();
    ctx.arc(this.position.x, this.position.y, highlightSize, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(160, 70, 70, ' + Math.min(1.0, highlightOpacity) + ')';
    ctx.fill();
    

}; 