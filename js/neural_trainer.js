/**
 * Neural Predator Training System
 * 
 * This file contains the complete training implementation for the neural predator.
 * It includes online learning, reward calculation, gradient computation, and 
 * parameter export functionality.
 * 
 * Features:
 * - Policy gradient learning with numerical stability
 * - Real-time training visualization
 * - Parameter export to parameters.js format
 * - Training progress tracking and statistics
 * - Robust error handling and recovery
 */

/**
 * Training-enabled Neural Predator
 * 
 * This extends the base Predator class with full training capabilities.
 * Unlike the prediction-only version, this includes learning algorithms.
 */
function TrainingNeuralPredator(x, y, simulation) {
    // Inherit from basic predator
    Predator.call(this, x, y, simulation);
    
    // Patrol behavior for when no boids are present
    this.autonomousTarget = new Vector(x, y);
    this.targetChangeTime = 0;
    this.targetChangeInterval = 5000; // Change target every 5 seconds
    
    // Load initial parameters from parameters.js
    this.loadParameters();
    
    // Performance optimization - pre-allocate buffers
    this.inputBuffer = new Array(this.inputSize);
    this.hiddenBuffer = new Array(this.hiddenSize);
    this.outputBuffer = new Array(this.outputSize);
    
    // Training parameters
    this.learningRate = 0.01;  // Increased learning rate
    this.rewardMemory = 0.9;   // Reduced memory for faster adaptation
    this.avgReward = 0;
    this.framesSinceLastFeed = 0;
    this.maxFramesSinceLastFeed = 600;
    
    // Learning buffers for gradient calculation
    this.lastInputs = new Array(this.inputSize);
    this.lastHidden = new Array(this.hiddenSize);
    this.lastOutputs = new Array(this.outputSize);
    
    // Initialize gradient buffers
    this.initializeGradients();
    
    // Training statistics
    this.totalReward = 0;
    this.episodeReward = 0;
    this.episodeFrames = 0;
    this.boidsEaten = 0;
}

// Inherit from Predator
TrainingNeuralPredator.prototype = Object.create(Predator.prototype);
TrainingNeuralPredator.prototype.constructor = TrainingNeuralPredator;

// Helper function to deep clone parameters
function cloneNeuralParams() {
    if (typeof window.NEURAL_PARAMS === 'undefined') {
        console.error('Neural parameters not loaded!');
        return null;
    }
    
    return {
        inputSize: window.NEURAL_PARAMS.inputSize,
        hiddenSize: window.NEURAL_PARAMS.hiddenSize,
        outputSize: window.NEURAL_PARAMS.outputSize,
        weightsIH: window.NEURAL_PARAMS.weightsIH.map(function(row) { return row.slice(); }),
        weightsHO: window.NEURAL_PARAMS.weightsHO.map(function(row) { return row.slice(); }),
        biasH: window.NEURAL_PARAMS.biasH.slice(),
        biasO: window.NEURAL_PARAMS.biasO.slice(),
        maxDistance: window.NEURAL_PARAMS.maxDistance,
        maxVelocity: window.NEURAL_PARAMS.maxVelocity,
        version: window.NEURAL_PARAMS.version,
        trained: window.NEURAL_PARAMS.trained,
        description: window.NEURAL_PARAMS.description
    };
}

// Helper function to export parameters as JS code
function exportNeuralParams(params) {
    var code = '/**\n * Neural Network Parameters\n * \n * Pure data file containing trained weights and biases for the neural predator.\n * \n * Network Architecture: 22 inputs → 12 hidden → 2 outputs\n * - Inputs: positions & velocities of 5 nearest boids + predator velocity\n * - Hidden: 12 neurons with tanh activation\n * - Outputs: steering force (x, y)\n */\n\n';
    
    code += 'window.NEURAL_PARAMS = {\n';
    code += '    // Network architecture\n';
    code += '    inputSize: ' + params.inputSize + ',\n';
    code += '    hiddenSize: ' + params.hiddenSize + ',\n';
    code += '    outputSize: ' + params.outputSize + ',\n';
    code += '    \n';
    
    code += '    // Input to hidden layer weights (' + params.hiddenSize + 'x' + params.inputSize + ' matrix)\n';
    code += '    weightsIH: [\n';
    for (var i = 0; i < params.weightsIH.length; i++) {
        code += '        // Hidden neuron ' + (i + 1) + '\n';
        code += '        [' + params.weightsIH[i].map(function(w) { return w.toFixed(3); }).join(', ') + ']';
        if (i < params.weightsIH.length - 1) code += ',';
        code += '\n';
    }
    code += '    ],\n';
    code += '    \n';
    
    code += '    // Hidden to output layer weights (' + params.outputSize + 'x' + params.hiddenSize + ' matrix)\n';
    code += '    weightsHO: [\n';
    for (var i = 0; i < params.weightsHO.length; i++) {
        code += '        // ' + (i === 0 ? 'X' : 'Y') + ' steering force output\n';
        code += '        [' + params.weightsHO[i].map(function(w) { return w.toFixed(3); }).join(', ') + ']';
        if (i < params.weightsHO.length - 1) code += ',';
        code += '\n';
    }
    code += '    ],\n';
    code += '    \n';
    
    code += '    // Hidden layer biases (' + params.hiddenSize + ' values)\n';
    code += '    biasH: [' + params.biasH.map(function(b) { return b.toFixed(3); }).join(', ') + '],\n';
    code += '    \n';
    
    code += '    // Output layer biases (' + params.outputSize + ' values)\n';
    code += '    biasO: [' + params.biasO.map(function(b) { return b.toFixed(3); }).join(', ') + '],\n';
    code += '    \n';
    
    code += '    // Normalization constants\n';
    code += '    maxDistance: ' + params.maxDistance + ',\n';
    code += '    maxVelocity: ' + params.maxVelocity + ',\n';
    code += '    \n';
    
    code += '    // Version info for parameter management\n';
    code += '    version: "' + params.version + '",\n';
    code += '    trained: ' + params.trained + ',\n';
    code += '    description: "' + params.description + '"\n';
    code += '};';
    
    return code;
}

// Load parameters from parameters.js
TrainingNeuralPredator.prototype.loadParameters = function() {
    if (typeof window.NEURAL_PARAMS === 'undefined') {
        console.error('Neural parameters not loaded!');
        return;
    }
    
    this.inputSize = window.NEURAL_PARAMS.inputSize;
    this.hiddenSize = window.NEURAL_PARAMS.hiddenSize;
    this.outputSize = window.NEURAL_PARAMS.outputSize;
    this.maxDistance = window.NEURAL_PARAMS.maxDistance;
    this.maxVelocity = window.NEURAL_PARAMS.maxVelocity;
    
    // Deep copy parameters for training
    this.weightsIH = window.NEURAL_PARAMS.weightsIH.map(function(row) { return row.slice(); });
    this.weightsHO = window.NEURAL_PARAMS.weightsHO.map(function(row) { return row.slice(); });
    this.biasH = window.NEURAL_PARAMS.biasH.slice();
    this.biasO = window.NEURAL_PARAMS.biasO.slice();
};

// Initialize gradient buffers
TrainingNeuralPredator.prototype.initializeGradients = function() {
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
    
    this.gradientBiasH = new Array(this.hiddenSize).fill(0);
    this.gradientBiasO = new Array(this.outputSize).fill(0);
};

// Fast activation function (tanh approximation)
TrainingNeuralPredator.prototype.fastTanh = function(x) {
    if (x > 2) return 1;
    if (x < -2) return -1;
    var x2 = x * x;
    return x * (27 + x2) / (27 + 9 * x2);
};

// Fast derivative of tanh
TrainingNeuralPredator.prototype.fastTanhDerivative = function(x) {
    if (x > 2 || x < -2) return 0;
    if (isNaN(x)) return 0;
    
    var tanhX = this.fastTanh(x);
    var derivative = 1 - tanhX * tanhX;
    
    if (isNaN(derivative) || !isFinite(derivative)) {
        return 0;
    }
    
    return derivative;
};

// Calculate reward based on current state
TrainingNeuralPredator.prototype.calculateReward = function(boids, caughtBoid) {
    var reward = 0;
    
    // Major reward for catching boids
    if (caughtBoid) {
        reward += 50.0; // Increased catch reward to match feeding reward
        this.framesSinceLastFeed = 0;
        this.boidsEaten++;
    } else {
        this.framesSinceLastFeed++;
        reward -= 0.1; // Increased penalty for inefficiency
        
        if (this.framesSinceLastFeed > this.maxFramesSinceLastFeed) {
            reward -= 0.5; // Increased penalty for too long without food
        }
    }
    
    // Reward for being near boids
    if (boids.length > 0) {
        var nearestDistance = Infinity;
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < nearestDistance) {
                nearestDistance = distance;
            }
        }
        
        if (nearestDistance < this.maxDistance) {
            reward += (this.maxDistance - nearestDistance) / this.maxDistance * 1.0; // Increased proximity reward
        }
    }
    
    return reward;
};

// Prepare inputs for neural network
TrainingNeuralPredator.prototype.prepareInputs = function(boids) {
    var nearestBoids = [];
    
    for (var i = 0; i < boids.length && i < 5; i++) {
        var distance = this.position.getDistance(boids[i].position);
        nearestBoids.push({boid: boids[i], distance: distance});
    }
    
    nearestBoids.sort(function(a, b) { return a.distance - b.distance; });
    nearestBoids = nearestBoids.slice(0, 5);
    
    // Clear input buffer
    for (var i = 0; i < this.inputSize; i++) {
        this.inputBuffer[i] = 0;
    }
    
    // Encode boid positions AND velocities (4 values per boid)
    for (var i = 0; i < nearestBoids.length; i++) {
        var boid = nearestBoids[i].boid;
        var relativePos = boid.position.subtract(this.position);
        
        if (isNaN(relativePos.x) || isNaN(relativePos.y)) {
            relativePos.x = 0;
            relativePos.y = 0;
        }
        
        // Normalize relative position (-1 to 1)
        var normalizedPosX = relativePos.x / this.maxDistance;
        var normalizedPosY = relativePos.y / this.maxDistance;
        
        if (isNaN(normalizedPosX)) normalizedPosX = 0;
        if (isNaN(normalizedPosY)) normalizedPosY = 0;
        
        // Normalize boid velocity (-1 to 1)
        var normalizedVelX = boid.velocity.x / this.maxVelocity;
        var normalizedVelY = boid.velocity.y / this.maxVelocity;
        
        if (isNaN(normalizedVelX)) normalizedVelX = 0;
        if (isNaN(normalizedVelY)) normalizedVelY = 0;
        
        // Store position and velocity for each boid (4 values per boid)
        var baseIndex = i * 4;
        this.inputBuffer[baseIndex] = Math.max(-1, Math.min(1, normalizedPosX));     // Position X
        this.inputBuffer[baseIndex + 1] = Math.max(-1, Math.min(1, normalizedPosY)); // Position Y
        this.inputBuffer[baseIndex + 2] = Math.max(-1, Math.min(1, normalizedVelX)); // Velocity X
        this.inputBuffer[baseIndex + 3] = Math.max(-1, Math.min(1, normalizedVelY)); // Velocity Y
    }
    
    // Add predator's current velocity (last 2 inputs)
    var predatorVelX = this.velocity.x / this.maxVelocity;
    var predatorVelY = this.velocity.y / this.maxVelocity;
    
    if (isNaN(predatorVelX)) predatorVelX = 0;
    if (isNaN(predatorVelY)) predatorVelY = 0;
    
    this.inputBuffer[20] = Math.max(-1, Math.min(1, predatorVelX));
    this.inputBuffer[21] = Math.max(-1, Math.min(1, predatorVelY));
};

// Forward pass through neural network
TrainingNeuralPredator.prototype.forward = function() {
    // Input to hidden layer
    for (var h = 0; h < this.hiddenSize; h++) {
        var sum = this.biasH[h];
        for (var i = 0; i < this.inputSize; i++) {
            var input = this.inputBuffer[i];
            var weight = this.weightsIH[h][i];
            if (isNaN(input) || isNaN(weight)) {
                input = 0;
                weight = 0;
            }
            sum += input * weight;
        }
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
        if (isNaN(sum)) {
            sum = 0;
        }
        this.outputBuffer[o] = this.fastTanh(sum) * PREDATOR_MAX_FORCE;
        
        if (isNaN(this.outputBuffer[o])) {
            this.outputBuffer[o] = 0;
        }
    }
    
    // Store for visualization
    this.lastInput = this.inputBuffer.slice();
    this.hiddenActivations = this.hiddenBuffer.slice();
    this.lastOutput = this.outputBuffer.slice();
    
    return new Vector(this.outputBuffer[0], this.outputBuffer[1]);
};

// Store current state for learning
TrainingNeuralPredator.prototype.storeState = function() {
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

// Update weights using policy gradient
TrainingNeuralPredator.prototype.updateWeights = function(reward) {
    if (isNaN(reward) || !isFinite(reward)) {
        return;
    }
    
    this.avgReward = this.rewardMemory * this.avgReward + (1 - this.rewardMemory) * reward;
    
    if (isNaN(this.avgReward)) {
        this.avgReward = 0;
    }
    
    var rewardError = reward - this.avgReward;
    
    // Remove the threshold that was preventing learning
    // Small reward errors are still valuable for learning
    
    var learningSignal = this.learningRate * rewardError;
    
    if (isNaN(learningSignal) || !isFinite(learningSignal)) {
        return;
    }
    
    // Debug logging (only occasionally to avoid spam)
    if (Math.random() < 0.01) {
        console.log('Training update:', {
            reward: reward.toFixed(4),
            avgReward: this.avgReward.toFixed(4),
            rewardError: rewardError.toFixed(4),
            learningSignal: learningSignal.toFixed(6),
            learningRate: this.learningRate.toFixed(4)
        });
    }
    
    // Update output layer weights
    for (var o = 0; o < this.outputSize; o++) {
        var outputValue = this.lastOutputs[o] / PREDATOR_MAX_FORCE;
        var outputGradient = this.fastTanhDerivative(outputValue) * learningSignal;
        
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
            
            if (isNaN(errorContribution) || !isFinite(errorContribution)) {
                continue;
            }
            hiddenError += errorContribution;
        }
        
        var hiddenGradient = this.fastTanhDerivative(this.lastHidden[h]) * hiddenError * learningSignal;
        
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
    
    this.clipWeights();
};

// Clip weights to prevent explosion
TrainingNeuralPredator.prototype.clipWeights = function() {
    var maxWeight = 2.0;
    
    for (var h = 0; h < this.hiddenSize; h++) {
        for (var i = 0; i < this.inputSize; i++) {
            var weight = this.weightsIH[h][i];
            if (isNaN(weight) || !isFinite(weight)) {
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
    
    for (var o = 0; o < this.outputSize; o++) {
        for (var h = 0; h < this.hiddenSize; h++) {
            var weight = this.weightsHO[o][h];
            if (isNaN(weight) || !isFinite(weight)) {
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

// Get autonomous force with learning
TrainingNeuralPredator.prototype.getAutonomousForce = function(boids) {
    if (boids.length === 0) {
        var currentTime = Date.now();
        if (currentTime - this.targetChangeTime > this.targetChangeInterval) {
            var margin = 50;
            this.autonomousTarget.x = margin + Math.random() * (this.simulation.canvasWidth - 2 * margin);
            this.autonomousTarget.y = margin + Math.random() * (this.simulation.canvasHeight - 2 * margin);
            this.targetChangeTime = currentTime;
        }
        return this.seek(this.autonomousTarget);
    }
    
    this.prepareInputs(boids);
    var force = this.forward();
    this.storeState();
    
    var reward = this.calculateReward(boids, false);
    this.updateWeights(reward);
    
    this.episodeReward += reward;
    this.episodeFrames++;
    
    return force;
};

// Override feed method with learning
TrainingNeuralPredator.prototype.feed = function() {
    this.currentSize = Math.min(this.currentSize + this.growthPerFeed, this.maxSize);
    this.lastFeedTime = Date.now();
    
    var feedReward = 50.0; // Increased feeding reward for better learning signal
    this.updateWeights(feedReward);
    this.episodeReward += feedReward;
    this.framesSinceLastFeed = 0;
};

// Override update method with learning
TrainingNeuralPredator.prototype.update = function(boids) {
    var steeringForce = this.getAutonomousForce(boids);
    this.acceleration.iAdd(steeringForce);
    
    Predator.prototype.update.call(this, boids);
    
    if (isNaN(this.position.x) || isNaN(this.position.y)) {
        this.position.x = this.simulation.canvasWidth / 2;
        this.position.y = this.simulation.canvasHeight / 2;
    }
    
    if (isNaN(this.velocity.x) || isNaN(this.velocity.y)) {
        this.velocity.x = (Math.random() - 0.5) * 0.1;
        this.velocity.y = (Math.random() - 0.5) * 0.1;
    }
    
    if (boids.length > 0) {
        var nearestDistance = Infinity;
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < nearestDistance) {
                nearestDistance = distance;
            }
        }
        
        if (nearestDistance < this.maxDistance * 0.5) {
            var proximityReward = 0.5; // Increased proximity reward
            this.updateWeights(proximityReward);
            this.episodeReward += proximityReward;
        }
    }
};

// Get training statistics
TrainingNeuralPredator.prototype.getStats = function() {
    return {
        avgReward: this.avgReward,
        episodeReward: this.episodeReward,
        episodeFrames: this.episodeFrames,
        framesSinceLastFeed: this.framesSinceLastFeed,
        currentSize: this.currentSize,
        boidsEaten: this.boidsEaten,
        efficiency: this.episodeFrames > 0 ? (this.boidsEaten / this.episodeFrames * 100) : 0
    };
};

// Reset episode statistics
TrainingNeuralPredator.prototype.resetEpisode = function() {
    this.episodeReward = 0;
    this.episodeFrames = 0;
    this.boidsEaten = 0;
    this.framesSinceLastFeed = 0;
    this.currentSize = this.baseSize;
};

// Export current parameters
TrainingNeuralPredator.prototype.exportParameters = function() {
    var params = {
        inputSize: this.inputSize,
        hiddenSize: this.hiddenSize,
        outputSize: this.outputSize,
        weightsIH: this.weightsIH.map(function(row) { return row.slice(); }),
        weightsHO: this.weightsHO.map(function(row) { return row.slice(); }),
        biasH: this.biasH.slice(),
        biasO: this.biasO.slice(),
        maxDistance: this.maxDistance,
        maxVelocity: this.maxVelocity,
        version: "1.0.0",
        trained: true,
        description: "Trained neural network parameters"
    };
    
    return exportNeuralParams(params);
};

// Render with training visualization
TrainingNeuralPredator.prototype.render = function() {
    var ctx = this.simulation.ctx;
    
    if (isNaN(this.position.x) || isNaN(this.position.y)) {
        this.position.x = this.simulation.canvasWidth / 2;
        this.position.y = this.simulation.canvasHeight / 2;
        this.velocity.x = Math.random() * 2 - 1;
        this.velocity.y = Math.random() * 2 - 1;
    }
    
    if (isNaN(this.velocity.x) || isNaN(this.velocity.y) || 
        (this.velocity.x === 0 && this.velocity.y === 0)) {
        this.velocity.x = (Math.random() - 0.5) * 0.1;
        this.velocity.y = (Math.random() - 0.5) * 0.1;
    }
    
    var directionVector = this.velocity.normalize().multiplyBy(this.currentSize * 1.2);
    var inverseVector1 = new Vector(-directionVector.y, directionVector.x);
    var inverseVector2 = new Vector(directionVector.y, -directionVector.x);
    inverseVector1 = inverseVector1.divideBy(4);
    inverseVector2 = inverseVector2.divideBy(4);
    
    var sizeRatio = this.currentSize / this.baseSize;
    var intensity = 0.4 + (sizeRatio - 1) * 0.3;
    
    ctx.beginPath();
    ctx.moveTo(this.position.x, this.position.y);
    ctx.lineTo(this.position.x + inverseVector1.x, this.position.y + inverseVector1.y);
    ctx.lineTo(this.position.x + directionVector.x, this.position.y + directionVector.y);
    ctx.lineTo(this.position.x + inverseVector2.x, this.position.y + inverseVector2.y);
    ctx.lineTo(this.position.x, this.position.y);
    
    var baseOpacity = 0.8 + intensity * 0.2;
    var fillOpacity = 0.5 + intensity * 0.3;
    
    ctx.strokeStyle = 'rgba(100, 40, 40, ' + Math.min(1.0, baseOpacity) + ')';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = 'rgba(140, 50, 50, ' + Math.min(1.0, fillOpacity) + ')';
    ctx.fill();
    
    var highlightSize = Math.max(3, this.currentSize * 0.2);
    var highlightOpacity = 0.7 + intensity * 0.3;
    
    ctx.beginPath();
    ctx.arc(this.position.x, this.position.y, highlightSize, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(160, 70, 70, ' + Math.min(1.0, highlightOpacity) + ')';
    ctx.fill();
};

/**
 * Training Controller
 * 
 * Manages the training process, UI updates, and simulation control
 */
function NeuralTrainer() {
    this.canvas = document.getElementById('boids-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.simulation = null;
    this.isTraining = false;
    this.currentEpisode = 0;
    this.maxEpisodes = 100;
    this.episodeLength = 1800; // frames per episode
    this.episodeFrame = 0;
    this.simulationSpeed = 1.0;
    this.showSimulation = true;
    this.neuralViz = null;
    
    // Performance tracking
    this.episodeStartTime = 0;
    this.episodeStartFrames = 0;
    this.completionTimes = []; // Store completion times for each episode
    this.bestCompletionTime = Infinity;
    this.averageCompletionTime = 0;
    this.totalBoidsAtStart = 0;
    
    this.initializeUI();
    this.initializeSimulation();
    this.initializeNeuralViz();
    this.startRenderLoop();
}

NeuralTrainer.prototype.initializeUI = function() {
    var self = this;
    
    // Training controls
    document.getElementById('start-training').addEventListener('click', function() {
        self.startTraining();
    });
    
    document.getElementById('stop-training').addEventListener('click', function() {
        self.stopTraining();
    });
    
    document.getElementById('reset-training').addEventListener('click', function() {
        self.resetNetwork();
    });
    
    document.getElementById('load-params').addEventListener('click', function() {
        self.loadParameters();
    });
    
    document.getElementById('restart-simulation').addEventListener('click', function() {
        self.restartSimulation();
    });
    
    // Export controls
    document.getElementById('export-params').addEventListener('click', function() {
        self.exportParameters();
    });
    
    document.getElementById('copy-params').addEventListener('click', function() {
        self.copyToClipboard();
    });
    
    // Simulation visibility toggle
    document.getElementById('viz-toggle').addEventListener('click', function(e) {
        self.toggleSimulationVisibility();
    });
    
    // Settings
    document.getElementById('learning-rate').addEventListener('input', function(e) {
        if (self.simulation && self.simulation.predator) {
            self.simulation.predator.learningRate = parseFloat(e.target.value);
        }
    });
    
    document.getElementById('max-episodes').addEventListener('input', function(e) {
        self.maxEpisodes = parseInt(e.target.value);
    });
    
    document.getElementById('episode-length').addEventListener('input', function(e) {
        self.episodeLength = parseInt(e.target.value);
    });
    
    // Simulation speed slider
    document.getElementById('sim-speed').addEventListener('input', function(e) {
        var speed = parseFloat(e.target.value);
        self.updateSimulationSpeed(speed);
        // Update number input to match slider
        document.getElementById('speed-input').value = speed;
    });
    
    // Simulation speed number input
    document.getElementById('speed-input').addEventListener('input', function(e) {
        var speed = parseFloat(e.target.value);
        if (speed < 0.1) speed = 0.1; // Minimum speed
        if (speed > 10000) speed = 10000; // Maximum speed
        
        self.updateSimulationSpeed(speed);
        
        // Update slider if speed is within slider range
        var slider = document.getElementById('sim-speed');
        if (speed <= parseFloat(slider.max)) {
            slider.value = speed;
        } else {
            // If speed exceeds slider max, set slider to max
            slider.value = slider.max;
        }
    });
    
    document.getElementById('boid-count').addEventListener('input', function(e) {
        self.setBoidCount(parseInt(e.target.value));
    });
    
    // Resize canvas
    window.addEventListener('resize', function() {
        self.resizeCanvas();
    });
    
    this.resizeCanvas();
};

NeuralTrainer.prototype.resizeCanvas = function() {
    var rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
    
    // Also resize neural viz canvas
    var neuralCanvas = document.getElementById('neural-viz-canvas');
    if (neuralCanvas) {
        neuralCanvas.width = rect.width;
        neuralCanvas.height = rect.height;
    }
    
    if (this.simulation) {
        this.simulation.canvasWidth = this.canvas.width;
        this.simulation.canvasHeight = this.canvas.height;
    }
    
    // Notify neural visualization of resize
    if (this.neuralViz && this.neuralViz.resize) {
        this.neuralViz.resize();
    }
};

NeuralTrainer.prototype.initializeSimulation = function() {
    this.simulation = new Simulation('boids-canvas');
    this.simulation.ctx = this.ctx;
    this.simulation.canvasWidth = this.canvas.width;
    this.simulation.canvasHeight = this.canvas.height;
    
    // Initialize without predator (we'll create our own training predator)
    this.simulation.initialize(false, true);
    
    // Create training predator
    var predator_x = this.simulation.canvasWidth / 2;
    var predator_y = this.simulation.canvasHeight / 2;
    this.simulation.predator = new TrainingNeuralPredator(predator_x, predator_y, this.simulation);
    
    // Apply UI parameters to the predator
    this.applyUIParameters();
    
    // Connect predator to neural visualization
    if (this.neuralViz && typeof connectNeuralViz === 'function') {
        connectNeuralViz(this.simulation.predator);
    }
    
    console.log('Training simulation initialized');
};

NeuralTrainer.prototype.initializeNeuralViz = function() {
    try {
        if (typeof NeuralVisualization !== 'undefined') {
            this.neuralViz = new NeuralVisualization('neural-viz-canvas');
            console.log('Neural visualization initialized');
        } else {
            console.warn('Neural visualization not available');
        }
    } catch (error) {
        console.error('Neural visualization initialization error:', error);
    }
};

// Apply UI parameters to the training predator
NeuralTrainer.prototype.applyUIParameters = function() {
    if (this.simulation && this.simulation.predator) {
        // Apply learning rate from UI
        var learningRateInput = document.getElementById('learning-rate');
        if (learningRateInput) {
            this.simulation.predator.learningRate = parseFloat(learningRateInput.value);
            console.log('Applied learning rate:', this.simulation.predator.learningRate);
        }
        
        // Apply episode length from UI
        var episodeLengthInput = document.getElementById('episode-length');
        if (episodeLengthInput) {
            this.episodeLength = parseInt(episodeLengthInput.value);
            console.log('Applied episode length:', this.episodeLength);
        }
        
        // Apply max episodes from UI
        var maxEpisodesInput = document.getElementById('max-episodes');
        if (maxEpisodesInput) {
            this.maxEpisodes = parseInt(maxEpisodesInput.value);
            console.log('Applied max episodes:', this.maxEpisodes);
        }
        
        // Force immediate weight update to test if learning is working
        console.log('Predator created with parameters:', {
            learningRate: this.simulation.predator.learningRate,
            rewardMemory: this.simulation.predator.rewardMemory,
            avgReward: this.simulation.predator.avgReward
        });
    }
};

NeuralTrainer.prototype.toggleSimulationVisibility = function() {
    this.showSimulation = !this.showSimulation;
    var simulationPanel = document.getElementById('simulation-panel');
    var toggle = document.getElementById('viz-toggle');
    
    if (this.showSimulation) {
        simulationPanel.classList.remove('hidden-viz');
        toggle.classList.add('active');
    } else {
        simulationPanel.classList.add('hidden-viz');
        toggle.classList.remove('active');
    }
    
    console.log('Simulation visibility:', this.showSimulation ? 'shown' : 'hidden');
};

NeuralTrainer.prototype.updateSimulationSpeed = function(speed) {
    this.simulationSpeed = speed;
    
    // Update display with appropriate formatting and color coding
    var displayElement = document.getElementById('speed-display');
    var displayText;
    
    if (speed >= 100) {
        displayText = speed.toFixed(0) + 'x'; // No decimals for very high speeds
        displayElement.style.color = '#dc3545'; // Red for extreme speeds
        displayElement.style.fontWeight = 'bold';
    } else if (speed >= 50) {
        displayText = speed.toFixed(1) + 'x'; // One decimal for high speeds
        displayElement.style.color = '#fd7e14'; // Orange for high speeds
        displayElement.style.fontWeight = 'bold';
    } else if (speed >= 10) {
        displayText = speed.toFixed(1) + 'x'; // One decimal for medium speeds
        displayElement.style.color = '#ffc107'; // Yellow for medium speeds
        displayElement.style.fontWeight = '500';
    } else {
        displayText = speed.toFixed(1) + 'x'; // One decimal for normal speeds
        displayElement.style.color = '#495057'; // Normal color for normal speeds
        displayElement.style.fontWeight = '500';
    }
    
    displayElement.textContent = displayText;
    
    // Log performance warnings for extreme speeds
    if (speed >= 500) {
        console.warn('Extreme simulation speed (' + speed + 'x) - May cause performance issues or instability');
    } else if (speed >= 100) {
        console.log('High simulation speed (' + speed + 'x) - Monitor performance');
    } else {
        console.log('Simulation speed updated to:', speed + 'x');
    }
};

NeuralTrainer.prototype.startTraining = function() {
    this.isTraining = true;
    this.currentEpisode = 0;
    this.episodeFrame = 0;
    
    // Apply UI parameters before starting training
    this.applyUIParameters();
    
    // Reset performance tracking
    this.completionTimes = [];
    this.bestCompletionTime = Infinity;
    this.averageCompletionTime = 0;
    
    // Start episode timing
    this.startEpisodeTimer();
    
    document.getElementById('start-training').disabled = true;
    document.getElementById('stop-training').disabled = false;
    document.getElementById('status-text').textContent = 'Training...';
    document.getElementById('status-dot').className = 'status-dot training';
    
    console.log('Started training for', this.maxEpisodes, 'episodes with learning rate:', this.simulation.predator.learningRate);
    
    // Log initial weights to compare against final weights
    this.logSampleWeights('Initial weights at training start');
};

NeuralTrainer.prototype.stopTraining = function() {
    this.isTraining = false;
    
    document.getElementById('start-training').disabled = false;
    document.getElementById('stop-training').disabled = true;
    document.getElementById('status-text').textContent = 'Stopped';
    document.getElementById('status-dot').className = 'status-dot stopped';
    
    console.log('Training stopped');
};

NeuralTrainer.prototype.resetNetwork = function() {
    if (this.simulation && this.simulation.predator) {
        console.log('🔄 Resetting neural network to random parameters...');
        
        // Log weights before reset
        this.logSampleWeights('BEFORE reset');
        
        // Reset episode statistics
        this.simulation.predator.resetEpisode();
        this.currentEpisode = 0;
        this.episodeFrame = 0;
        
        // Randomize ALL network weights and biases
        this.randomizeWeights();
        
        // Reset training state
        this.simulation.predator.avgReward = 0;
        this.simulation.predator.episodeReward = 0;
        this.simulation.predator.episodeFrames = 0;
        this.simulation.predator.framesSinceLastFeed = 0;
        
        document.getElementById('status-text').textContent = 'Reset (Random)';
        document.getElementById('status-dot').className = 'status-dot';
        // Reset performance metrics
        this.completionTimes = [];
        this.bestCompletionTime = Infinity;
        this.averageCompletionTime = 0;
        this.episodeStartTime = 0;
        
        console.log('✓ Network reset with randomized parameters');
        this.logSampleWeights('AFTER reset (randomized)');
        
        // Show a clear comparison message
        console.log('🎯 Network has been reset! All weights are now random. Start training to see them change.');
    }
};

// Randomize network weights to reset the neural network
NeuralTrainer.prototype.randomizeWeights = function() {
    if (this.simulation && this.simulation.predator) {
        var predator = this.simulation.predator;
        
        console.log('Randomizing all neural network weights and biases...');
        
        // Randomize input-to-hidden weights
        var ihCount = 0;
        for (var h = 0; h < predator.hiddenSize; h++) {
            for (var i = 0; i < predator.inputSize; i++) {
                predator.weightsIH[h][i] = (Math.random() - 0.5) * 2.0;
                ihCount++;
            }
        }
        
        // Randomize hidden-to-output weights
        var hoCount = 0;
        for (var o = 0; o < predator.outputSize; o++) {
            for (var h = 0; h < predator.hiddenSize; h++) {
                predator.weightsHO[o][h] = (Math.random() - 0.5) * 2.0;
                hoCount++;
            }
        }
        
        // Randomize hidden layer biases
        for (var h = 0; h < predator.hiddenSize; h++) {
            predator.biasH[h] = (Math.random() - 0.5) * 2.0;
        }
        
        // Randomize output layer biases
        for (var o = 0; o < predator.outputSize; o++) {
            predator.biasO[o] = (Math.random() - 0.5) * 2.0;
        }
        
        console.log('Randomized weights:', {
            'Input-to-Hidden': ihCount + ' weights',
            'Hidden-to-Output': hoCount + ' weights', 
            'Hidden biases': predator.hiddenSize + ' values',
            'Output biases': predator.outputSize + ' values'
        });
        
        console.log('All neural network parameters have been randomized (range: -1.0 to +1.0)');
    }
};

NeuralTrainer.prototype.loadParameters = function() {
    if (this.simulation && this.simulation.predator) {
        console.log('Loading original parameters from parameters.js...');
        this.logSampleWeights('Before loading parameters');
        
        this.simulation.predator.loadParameters();
        
        // Reset training state when loading original parameters
        this.simulation.predator.avgReward = 0;
        this.simulation.predator.episodeReward = 0;
        this.simulation.predator.episodeFrames = 0;
        this.simulation.predator.framesSinceLastFeed = 0;
        
        document.getElementById('status-text').textContent = 'Loaded Original';
        document.getElementById('status-dot').className = 'status-dot';
        console.log('✓ Parameters reloaded from parameters.js');
        this.logSampleWeights('After loading parameters');
    }
};

NeuralTrainer.prototype.restartSimulation = function() {
    // During training, preserve the predator and its learned weights
    var preservedPredator = null;
    if (this.isTraining && this.simulation && this.simulation.predator) {
        preservedPredator = this.simulation.predator;
        console.log('Preserving predator weights during episode restart');
    }
    
    this.initializeSimulation();
    
    // Restore the preserved predator if we were training
    if (preservedPredator) {
        this.simulation.predator = preservedPredator;
        // Reset predator position but keep learned weights
        this.simulation.predator.position.x = this.simulation.canvasWidth / 2;
        this.simulation.predator.position.y = this.simulation.canvasHeight / 2;
        this.simulation.predator.velocity.x = (Math.random() - 0.5) * 2;
        this.simulation.predator.velocity.y = (Math.random() - 0.5) * 2;
        this.simulation.predator.currentSize = this.simulation.predator.baseSize;
        console.log('Predator weights preserved during training');
    }
    
    this.episodeFrame = 0;
    console.log('Simulation restarted');
};

NeuralTrainer.prototype.setBoidCount = function(count) {
    if (this.simulation) {
        // Update global NUM_BOIDS variable used by simulation
        window.NUM_BOIDS = count;
        NUM_BOIDS = count;
        this.restartSimulation();
    }
};

NeuralTrainer.prototype.exportParameters = function() {
    if (this.simulation && this.simulation.predator) {
        // Log sample weights to verify they changed
        this.logSampleWeights('Before export');
        
        var exportedCode = this.simulation.predator.exportParameters();
        document.getElementById('export-output').value = exportedCode;
        console.log('Parameters exported');
    }
};

// Log sample weights for debugging
NeuralTrainer.prototype.logSampleWeights = function(label) {
    if (this.simulation && this.simulation.predator) {
        var predator = this.simulation.predator;
        console.log(label + ' - Sample weights:');
        console.log('  weightsIH[0][0]:', predator.weightsIH[0][0].toFixed(6));
        console.log('  weightsIH[0][1]:', predator.weightsIH[0][1].toFixed(6));
        console.log('  weightsHO[0][0]:', predator.weightsHO[0][0].toFixed(6));
        console.log('  weightsHO[0][1]:', predator.weightsHO[0][1].toFixed(6));
        console.log('  biasH[0]:', predator.biasH[0].toFixed(6));
        console.log('  biasO[0]:', predator.biasO[0].toFixed(6));
    }
};

// Start episode timer for performance tracking
NeuralTrainer.prototype.startEpisodeTimer = function() {
    this.episodeStartTime = Date.now();
    this.episodeStartFrames = this.episodeFrame;
    if (this.simulation && this.simulation.boids) {
        this.totalBoidsAtStart = this.simulation.boids.length;
    }
    console.log('Episode', this.currentEpisode + 1, 'started with', this.totalBoidsAtStart, 'boids');
};

// Calculate completion time (real-time equivalent)
NeuralTrainer.prototype.calculateCompletionTime = function() {
    var realTimeElapsed = (Date.now() - this.episodeStartTime) / 1000; // seconds
    var frameTimeElapsed = (this.episodeFrame - this.episodeStartFrames) / 60; // assuming 60 FPS base
    
    // Calculate real-time equivalent (accounting for simulation speed)
    var realTimeEquivalent = frameTimeElapsed;
    
    return {
        realTime: realTimeElapsed,
        equivalent: realTimeEquivalent
    };
};

// Record episode completion and update metrics
NeuralTrainer.prototype.recordEpisodeCompletion = function(reason) {
    var timingData = this.calculateCompletionTime();
    var completionTime = timingData.equivalent;
    
    if (reason === 'all_boids_eaten') {
        // Only count successful completions
        this.completionTimes.push(completionTime);
        
        // Update best time
        if (completionTime < this.bestCompletionTime) {
            this.bestCompletionTime = completionTime;
            console.log('🎯 NEW BEST TIME!', completionTime.toFixed(1) + 's');
        }
        
        // Calculate average time (last 10 episodes)
        var recentTimes = this.completionTimes.slice(-10);
        this.averageCompletionTime = recentTimes.reduce(function(sum, time) { 
            return sum + time; 
        }, 0) / recentTimes.length;
        
        console.log('✅ Episode', this.currentEpisode + 1, 'completed in', completionTime.toFixed(1) + 's');
        console.log('   📊 Best:', this.bestCompletionTime.toFixed(1) + 's', 'Avg (last 10):', this.averageCompletionTime.toFixed(1) + 's');
        
        // Show progress towards the goal
        if (this.completionTimes.length >= 2) {
            var firstTime = this.completionTimes[0];
            var improvement = ((firstTime - completionTime) / firstTime) * 100;
            if (improvement > 0) {
                console.log('   📈 Improvement from first episode:', '+' + improvement.toFixed(1) + '%');
            }
        }
    } else {
        console.log('⏱️ Episode', this.currentEpisode + 1, 'ended by timeout after', completionTime.toFixed(1) + 's');
    }
};

NeuralTrainer.prototype.copyToClipboard = function() {
    var textarea = document.getElementById('export-output');
    textarea.select();
    document.execCommand('copy');
    console.log('Parameters copied to clipboard');
};

NeuralTrainer.prototype.updateUI = function() {
    if (this.simulation && this.simulation.predator) {
        var stats = this.simulation.predator.getStats();
        
        document.getElementById('episode-count').textContent = this.currentEpisode;
        document.getElementById('avg-reward').textContent = stats.avgReward.toFixed(3);
        document.getElementById('boids-caught').textContent = stats.boidsEaten;
        document.getElementById('efficiency').textContent = stats.efficiency.toFixed(1) + '%';
        
        // Update performance metrics
        this.updatePerformanceMetrics();
        
        if (this.isTraining) {
            var progress = (this.currentEpisode / this.maxEpisodes) * 100;
            document.getElementById('training-progress').style.width = progress + '%';
        }
    }
};

// Update performance metrics in the UI
NeuralTrainer.prototype.updatePerformanceMetrics = function() {
    // Current episode time
    if (this.isTraining && this.episodeStartTime > 0) {
        var currentTime = this.calculateCompletionTime().equivalent;
        document.getElementById('current-time').textContent = currentTime.toFixed(1) + 's';
    } else {
        document.getElementById('current-time').textContent = '--';
    }
    
    // Boids remaining in current episode
    if (this.simulation && this.simulation.boids) {
        var remaining = this.simulation.boids.length;
        var total = this.totalBoidsAtStart || remaining;
        document.getElementById('boids-remaining').textContent = remaining + '/' + total;
    } else {
        document.getElementById('boids-remaining').textContent = '--';
    }
    
    // Best time
    if (this.bestCompletionTime !== Infinity) {
        var bestTimeElement = document.getElementById('best-time');
        bestTimeElement.textContent = this.bestCompletionTime.toFixed(1) + 's';
        bestTimeElement.style.color = '#28a745'; // Green for best time
        bestTimeElement.style.fontWeight = 'bold';
    } else {
        var bestTimeElement = document.getElementById('best-time');
        bestTimeElement.textContent = '--';
        bestTimeElement.style.color = '#495057'; // Default color
        bestTimeElement.style.fontWeight = '600';
    }
    
    // Average time
    if (this.averageCompletionTime > 0) {
        document.getElementById('avg-time').textContent = this.averageCompletionTime.toFixed(1) + 's';
    } else {
        document.getElementById('avg-time').textContent = '--';
    }
    
    // Improvement percentage
    if (this.completionTimes.length >= 2) {
        var firstTime = this.completionTimes[0];
        var currentAvg = this.averageCompletionTime;
        var improvement = ((firstTime - currentAvg) / firstTime) * 100;
        
        var improvementElement = document.getElementById('improvement');
        if (improvement > 0) {
            improvementElement.textContent = '+' + improvement.toFixed(1) + '%';
            improvementElement.style.color = '#28a745'; // Green for improvement
        } else {
            improvementElement.textContent = improvement.toFixed(1) + '%';
            improvementElement.style.color = '#dc3545'; // Red for regression
        }
    } else {
        var improvementElement = document.getElementById('improvement');
        improvementElement.textContent = '--';
        improvementElement.style.color = '#495057'; // Default color
    }
};

NeuralTrainer.prototype.update = function() {
    if (this.simulation) {
        // Update simulation based on speed - ALL simulation logic happens here
        for (var i = 0; i < this.simulationSpeed; i++) {
            // Update boids (flocking calculation AND movement)
            this.simulation.tick(); // Calculate flocking forces
            
            // Apply boid movement (this was missing!)
            for (var j = 0; j < this.simulation.boids.length; j++) {
                this.simulation.boids[j].update(); // Apply forces and move boids
            }
            
            // Update predator (neural network decision making and movement)
            if (this.simulation.predator) {
                this.simulation.predator.update(this.simulation.boids);
                
                // Handle predator-prey interactions
                var caughtBoids = this.simulation.predator.checkForPrey(this.simulation.boids);
                for (var j = caughtBoids.length - 1; j >= 0; j--) {
                    this.simulation.boids.splice(caughtBoids[j], 1);
                    
                    // Trigger feeding behavior and learning
                    this.simulation.predator.feed(); // This will give the feeding reward
                    
                    // Also give catch reward through the normal reward system
                    if (this.simulation.predator.calculateReward) {
                        var catchReward = this.simulation.predator.calculateReward(this.simulation.boids, true);
                        this.simulation.predator.updateWeights(catchReward);
                    }
                }
            }
            
            // Training episode management (only if training is active)
            if (this.isTraining) {
                this.episodeFrame++;
                
                // Check if episode is complete
                var episodeComplete = false;
                var completionReason = '';
                
                if (this.simulation.boids.length === 0) {
                    episodeComplete = true;
                    completionReason = 'all_boids_eaten';
                } else if (this.episodeFrame >= this.episodeLength) {
                    episodeComplete = true;
                    completionReason = 'timeout';
                }
                
                if (episodeComplete) {
                    // Record performance metrics before moving to next episode
                    this.recordEpisodeCompletion(completionReason);
                    
                    this.currentEpisode++;
                    this.episodeFrame = 0;
                    
                    if (this.simulation.predator) {
                        this.simulation.predator.resetEpisode();
                    }
                    
                    // Check if training is complete
                    if (this.currentEpisode >= this.maxEpisodes) {
                        this.stopTraining();
                        document.getElementById('status-text').textContent = 'Complete';
                        document.getElementById('status-dot').className = 'status-dot running';
                        this.exportParameters();
                    } else {
                        // Restart simulation for next episode
                        this.restartSimulation();
                        // Start timing for new episode
                        this.startEpisodeTimer();
                    }
                    
                    // Break out of speed loop when episode ends to avoid multiple episode transitions
                    break;
                }
            }
        }
    }
};

NeuralTrainer.prototype.render = function() {
    if (this.simulation && this.showSimulation) {
        // Only handle visual rendering here - all simulation logic is in update()
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Render boids (visual only - no logic updates)
        for (var i = 0; i < this.simulation.boids.length; i++) {
            this.simulation.boids[i].render();
        }
        
        // Render predator (visual only - no logic updates)
        if (this.simulation.predator) {
            this.simulation.predator.render();
        }
    }
};

NeuralTrainer.prototype.startRenderLoop = function() {
    var self = this;
    
    function loop() {
        self.update();
        self.render();
        self.updateUI();
        requestAnimationFrame(loop);
    }
    
    loop();
}; 