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
    
    // Position tracking removed (edge penalty system removed)
    
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

// Calculate shortest distance considering wraparound (toroidal world)
TrainingNeuralPredator.prototype.calculateWrappedDistance = function(targetPos, sourcePos) {
    var canvasWidth = this.simulation.canvasWidth;
    var canvasHeight = this.simulation.canvasHeight;
    
    // Calculate direct distances
    var directX = targetPos.x - sourcePos.x;
    var directY = targetPos.y - sourcePos.y;
    
    // Calculate wrapped distances
    var wrappedX, wrappedY;
    
    if (directX > 0) {
        // Target is to the right, check if going left through wrap is shorter
        wrappedX = directX - canvasWidth;
    } else {
        // Target is to the left, check if going right through wrap is shorter  
        wrappedX = directX + canvasWidth;
    }
    
    if (directY > 0) {
        // Target is below, check if going up through wrap is shorter
        wrappedY = directY - canvasHeight;
    } else {
        // Target is above, check if going down through wrap is shorter
        wrappedY = directY + canvasHeight;
    }
    
    // Choose shortest distance in each dimension
    var shortestX = Math.abs(directX) < Math.abs(wrappedX) ? directX : wrappedX;
    var shortestY = Math.abs(directY) < Math.abs(wrappedY) ? directY : wrappedY;
    
    return new Vector(shortestX, shortestY);
};

// Note: Edge penalty system removed to prevent device-dependent training interference
// The neural network should learn general hunting strategies that work across all screen sizes

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
    
    // Reward for being near boids (using wrapped distance)
    if (boids.length > 0) {
        var nearestDistance = Infinity;
        for (var i = 0; i < boids.length; i++) {
            var wrappedVector = this.calculateWrappedDistance(boids[i].position, this.position);
            var wrappedDistance = Math.sqrt(wrappedVector.x * wrappedVector.x + wrappedVector.y * wrappedVector.y);
            if (wrappedDistance < nearestDistance) {
                nearestDistance = wrappedDistance;
            }
        }
        
        if (nearestDistance < this.maxDistance) {
            reward += (this.maxDistance - nearestDistance) / this.maxDistance * 1.0; // Increased proximity reward
        }
    }
    
    // Note: Edge penalty removed to prevent device-dependent training interference
    // The neural network should learn optimal hunting strategies across all screen sizes
    
    return reward;
};

// Prepare inputs for neural network
TrainingNeuralPredator.prototype.prepareInputs = function(boids) {
    var nearestBoids = [];
    
    for (var i = 0; i < boids.length; i++) {
        var wrappedVector = this.calculateWrappedDistance(boids[i].position, this.position);
        var wrappedDistance = Math.sqrt(wrappedVector.x * wrappedVector.x + wrappedVector.y * wrappedVector.y);
        var boidSpeed = Math.sqrt(boids[i].velocity.x * boids[i].velocity.x + boids[i].velocity.y * boids[i].velocity.y);
        nearestBoids.push({
            boid: boids[i], 
            distance: wrappedDistance,
            speed: boidSpeed
        });
    }
    
    // Sort by wrapped distance and take closest available boids
    nearestBoids.sort(function(a, b) { return a.distance - b.distance; });
    
    // If we have fewer than 5 boids, find the slowest one to replicate
    var targetBoids = [];
    var slowestBoid = null;
    var minSpeed = Infinity;
    
    for (var i = 0; i < Math.min(5, nearestBoids.length); i++) {
        targetBoids.push(nearestBoids[i]);
        if (nearestBoids[i].speed < minSpeed) {
            minSpeed = nearestBoids[i].speed;
            slowestBoid = nearestBoids[i];
        }
    }
    
    // Fill remaining slots with the slowest boid (easiest target)
    while (targetBoids.length < 5 && slowestBoid !== null) {
        targetBoids.push(slowestBoid);
    }
    
    // Clear input buffer
    for (var i = 0; i < this.inputSize; i++) {
        this.inputBuffer[i] = 0;
    }
    
    // Encode boid positions AND velocities (4 values per boid)
    for (var i = 0; i < targetBoids.length && i < 5; i++) {
        var boidData = targetBoids[i];
        var boid = boidData.boid;
        
        // Calculate shortest wrapped distance (considering screen edges)
        var relativePos = this.calculateWrappedDistance(boid.position, this.position);
        
        if (isNaN(relativePos.x) || isNaN(relativePos.y)) {
            relativePos.x = 0;
            relativePos.y = 0;
        }
        
        // Normalize relative position based on actual screen size
        var screenNormX = this.simulation.canvasWidth / 2;  // Half screen width for normalization
        var screenNormY = this.simulation.canvasHeight / 2; // Half screen height for normalization
        var normalizedPosX = relativePos.x / screenNormX;
        var normalizedPosY = relativePos.y / screenNormY;
        
        if (isNaN(normalizedPosX)) normalizedPosX = 0;
        if (isNaN(normalizedPosY)) normalizedPosY = 0;
        
        // Normalize boid velocity based on screen size
        var screenVelNormX = this.simulation.canvasWidth / 100;  // Screen-relative velocity normalization
        var screenVelNormY = this.simulation.canvasHeight / 100; // Screen-relative velocity normalization
        var normalizedVelX = boid.velocity.x / screenVelNormX;
        var normalizedVelY = boid.velocity.y / screenVelNormY;
        
        if (isNaN(normalizedVelX)) normalizedVelX = 0;
        if (isNaN(normalizedVelY)) normalizedVelY = 0;
        
        // Store position and velocity for each boid (4 values per boid)
        var baseIndex = i * 4;
        this.inputBuffer[baseIndex] = normalizedPosX;     // Position X (screen-normalized)
        this.inputBuffer[baseIndex + 1] = normalizedPosY; // Position Y (screen-normalized)
        this.inputBuffer[baseIndex + 2] = normalizedVelX; // Velocity X (screen-normalized)
        this.inputBuffer[baseIndex + 3] = normalizedVelY; // Velocity Y (screen-normalized)
    }
    
    // Add predator's current velocity (last 2 inputs)
    var screenVelNormX = this.simulation.canvasWidth / 100;  // Screen-relative velocity normalization
    var screenVelNormY = this.simulation.canvasHeight / 100; // Screen-relative velocity normalization
    var predatorVelX = this.velocity.x / screenVelNormX;
    var predatorVelY = this.velocity.y / screenVelNormY;
    
    if (isNaN(predatorVelX)) predatorVelX = 0;
    if (isNaN(predatorVelY)) predatorVelY = 0;
    
    this.inputBuffer[20] = predatorVelX;
    this.inputBuffer[21] = predatorVelY;
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
            var wrappedVector = this.calculateWrappedDistance(boids[i].position, this.position);
            var wrappedDistance = Math.sqrt(wrappedVector.x * wrappedVector.x + wrappedVector.y * wrappedVector.y);
            if (wrappedDistance < nearestDistance) {
                nearestDistance = wrappedDistance;
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
    
    // Position tracking no longer needed (edge penalty removed)
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
    this.completionThreshold = 0; // Episode complete when this many boids remain
    
    // Neural network visualization
    this.showNeuralViz = true;
    this.neuralVizFrequency = 10; // Update every N frames
    this.neuralVizFrame = 0;
    
    // Fractional speed handling for speeds < 1
    this.simulationFrameAccumulator = 0;
    
    // Performance tracking
    this.episodeStartTime = 0;
    this.episodeStartFrames = 0;
    this.completionTimes = []; // Store completion times for each episode
    this.bestCompletionTime = Infinity;
    this.averageCompletionTime = 0;
    this.totalBoidsAtStart = 0;
    
    // Chart for performance visualization
    this.performanceChart = null;
    
    this.initializeUI();
    this.initializeSimulation();
    this.initializeNeuralViz();
    this.initializeChart();
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
    
    // Neural visualization toggle
    document.getElementById('neural-viz-toggle').addEventListener('click', function(e) {
        self.toggleNeuralVisualization();
    });
    
    // Neural visualization frequency
    document.getElementById('neural-viz-frequency').addEventListener('change', function(e) {
        self.neuralVizFrequency = parseInt(e.target.value);
        console.log('Neural viz frequency set to:', self.neuralVizFrequency);
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
    
    // Edge penalty control removed - was causing device-dependent training interference
    
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
    if (this.neuralViz && this.neuralViz.setPredator) {
        this.neuralViz.setPredator(this.simulation.predator);
        console.log('Predator connected to neural visualization');
    } else if (typeof connectNeuralViz === 'function') {
        connectNeuralViz(this.simulation.predator);
    }
    
    console.log('Training simulation initialized');
};

NeuralTrainer.prototype.initializeNeuralViz = function() {
    try {
        if (typeof NeuralVisualization !== 'undefined') {
            this.neuralViz = new NeuralVisualization('neural-viz-canvas');
            if (this.neuralViz && this.neuralViz.canvas) {
                console.log('Neural visualization initialized');
            } else {
                console.warn('Neural visualization canvas not found');
                this.neuralViz = null;
            }
        } else {
            console.warn('Neural visualization not available');
            this.neuralViz = null;
        }
    } catch (error) {
        console.error('Neural visualization initialization error:', error);
        this.neuralViz = null;
    }
};

// Initialize performance chart
NeuralTrainer.prototype.initializeChart = function() {
    try {
        if (typeof Chart !== 'undefined') {
            var ctx = document.getElementById('performance-chart').getContext('2d');
            this.performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [], // Episode numbers
                    datasets: [{
                        label: 'Completion Time (seconds)',
                        data: [], // Completion times
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        pointRadius: 3,
                        pointHoverRadius: 5
                    }, {
                        label: 'Best Time',
                        data: [], // Best time running history
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 2,
                        pointHoverRadius: 4,
                        borderDash: [5, 5]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Episode'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time (seconds)'
                            },
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    var label = context.dataset.label + ': ';
                                    if (context.dataset.label === 'Outliers') {
                                        // For outliers, show the actual value, not the displayed position
                                        var actualValue = context.chart.trainer.completionTimes[context.dataIndex];
                                        label += actualValue.toFixed(1) + 's (outlier)';
                                    } else {
                                        label += context.parsed.y.toFixed(1) + 's';
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    },
                    animation: {
                        duration: 300
                    }
                }
            });
            
            // Store reference to trainer for tooltip access
            this.performanceChart.trainer = this;
            
            console.log('Performance chart initialized');
        } else {
            console.warn('Chart.js not available');
        }
    } catch (error) {
        console.error('Chart initialization error:', error);
    }
};

// Calculate robust Y-axis bounds that minimize outlier impact
NeuralTrainer.prototype.calculateChartBounds = function() {
    if (this.completionTimes.length === 0) {
        return { min: 0, max: 100 };
    }
    
    // Sort completion times to find percentiles
    var sortedTimes = this.completionTimes.slice().sort(function(a, b) { return a - b; });
    var length = sortedTimes.length;
    
    // Calculate key percentiles
    var q1Index = Math.floor(length * 0.25);
    var q3Index = Math.floor(length * 0.75);
    var p95Index = Math.floor(length * 0.95);
    
    var q1 = sortedTimes[q1Index] || sortedTimes[0];
    var q3 = sortedTimes[q3Index] || sortedTimes[length - 1];
    var p95 = sortedTimes[p95Index] || sortedTimes[length - 1];
    var median = sortedTimes[Math.floor(length / 2)];
    
    // Calculate interquartile range for outlier detection
    var iqr = q3 - q1;
    var outlierThreshold = q3 + 1.5 * iqr;
    
    // Use 95th percentile as upper bound, but ensure it's reasonable
    var upperBound = Math.max(p95, median * 2); // At least 2x median
    
    // If 95th percentile is much higher than normal range, use a more conservative bound
    if (upperBound > outlierThreshold && length > 5) {
        upperBound = Math.max(outlierThreshold, median * 1.5);
    }
    
    // Add some margin for visual comfort
    var margin = upperBound * 0.1;
    var yMax = upperBound + margin;
    
    // Ensure minimum range for very consistent performance
    if (yMax < 10) {
        yMax = 10;
    }
    
    return {
        min: 0,
        max: yMax,
        outlierThreshold: outlierThreshold
    };
};

// Update performance chart with new data
NeuralTrainer.prototype.updateChart = function() {
    if (this.performanceChart && this.completionTimes.length > 0) {
        var chart = this.performanceChart;
        var labels = [];
        var bestTimes = [];
        var runningBest = Infinity;
        
        // Create labels (episode numbers) and calculate running best times
        for (var i = 0; i < this.completionTimes.length; i++) {
            labels.push(i + 1);
            if (this.completionTimes[i] < runningBest) {
                runningBest = this.completionTimes[i];
            }
            bestTimes.push(runningBest);
        }
        
        // Calculate smart Y-axis bounds
        var bounds = this.calculateChartBounds();
        
        // Separate normal data from outliers for visual treatment
        var normalData = [];
        var outlierData = [];
        var outlierCount = 0;
        
        for (var i = 0; i < this.completionTimes.length; i++) {
            var time = this.completionTimes[i];
            if (time > bounds.outlierThreshold) {
                // Outlier: show at the top of visible range but mark differently
                normalData.push(null);
                outlierData.push(bounds.max * 0.95); // Show near top of chart
                outlierCount++;
            } else {
                // Normal data
                normalData.push(time);
                outlierData.push(null);
            }
        }
        
        // Log outlier handling for user awareness
        if (outlierCount > 0 && this.completionTimes.length > 5) {
            console.log('📊 Chart: Detected', outlierCount, 'outlier(s) out of', this.completionTimes.length, 'episodes');
            console.log('   Chart Y-axis capped at', bounds.max.toFixed(1) + 's for better readability');
        }
        
        // Update chart data
        chart.data.labels = labels;
        chart.data.datasets[0].data = normalData; // Normal completion times
        chart.data.datasets[1].data = bestTimes; // Running best times
        
        // Add outlier dataset if needed
        if (chart.data.datasets.length < 3) {
            chart.data.datasets.push({
                label: 'Outliers',
                data: outlierData,
                borderColor: '#dc3545',
                backgroundColor: '#dc3545',
                borderWidth: 0,
                fill: false,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointStyle: 'triangle',
                showLine: false
            });
        } else {
            chart.data.datasets[2].data = outlierData;
        }
        
        // Update Y-axis scale
        chart.options.scales.y.max = bounds.max;
        
        // Update chart
        chart.update('none'); // Use 'none' for faster updates during training
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
        
        // Edge penalty removed - was causing device-dependent training interference
        
        // Apply max episodes from UI
        var maxEpisodesInput = document.getElementById('max-episodes');
        if (maxEpisodesInput) {
            this.maxEpisodes = parseInt(maxEpisodesInput.value);
            console.log('Applied max episodes:', this.maxEpisodes);
        }
        
        // Apply completion threshold from UI
        var completionThresholdInput = document.getElementById('completion-threshold');
        if (completionThresholdInput) {
            this.completionThreshold = parseInt(completionThresholdInput.value);
            console.log('Applied completion threshold:', this.completionThreshold);
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

NeuralTrainer.prototype.toggleNeuralVisualization = function() {
    this.showNeuralViz = !this.showNeuralViz;
    var toggle = document.getElementById('neural-viz-toggle');
    var neuralCanvas = document.getElementById('neural-viz-canvas');
    
    if (this.showNeuralViz) {
        toggle.classList.add('active');
        if (neuralCanvas) {
            neuralCanvas.style.opacity = '0.15';
        }
        // Reconnect predator to neural visualization when enabled
        if (this.neuralViz && this.neuralViz.setPredator && this.simulation && this.simulation.predator) {
            this.neuralViz.setPredator(this.simulation.predator);
        }
    } else {
        toggle.classList.remove('active');
        if (neuralCanvas) {
            neuralCanvas.style.opacity = '0';
        }
        // Disconnect predator from neural visualization when disabled
        if (this.neuralViz && this.neuralViz.setPredator) {
            this.neuralViz.setPredator(null);
        }
        // Clear neural data display
        document.getElementById('avg-input').textContent = '--';
        document.getElementById('avg-hidden').textContent = '--';
        document.getElementById('output-x').textContent = '--';
        document.getElementById('output-y').textContent = '--';
        document.getElementById('input-values').innerHTML = '<div style="color: #6c757d;">Neural visualization disabled</div>';
    }
    
    console.log('Neural visualization:', this.showNeuralViz ? 'enabled' : 'disabled');
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
    if (speed >= 1000) {
        console.warn('Ultra-extreme simulation speed (' + speed + 'x) - Training only mode, disable visualization for stability');
    } else if (speed >= 500) {
        console.warn('Extreme simulation speed (' + speed + 'x) - May cause performance issues, recommend disabling visualization');
    } else if (speed >= 100) {
        console.log('High simulation speed (' + speed + 'x) - Monitor performance, consider disabling visualization');
    } else if (speed < 1) {
        console.log('Slow motion speed (' + speed + 'x) - Using frame accumulation for smooth slow motion');
    } else {
        console.log('Simulation speed updated to:', speed + 'x');
    }
};

NeuralTrainer.prototype.startTraining = function() {
    this.isTraining = true;
    this.currentEpisode = 0;
    this.episodeFrame = 0;
    
    // Reset simulation speed accumulator
    this.simulationFrameAccumulator = 0;
    
    // Apply UI parameters before starting training
    this.applyUIParameters();
    
    // Reset performance tracking
    this.completionTimes = [];
    this.bestCompletionTime = Infinity;
    this.averageCompletionTime = 0;
    
            // Clear the chart
        if (this.performanceChart) {
            this.performanceChart.data.labels = [];
            this.performanceChart.data.datasets[0].data = [];
            this.performanceChart.data.datasets[1].data = [];
            if (this.performanceChart.data.datasets.length > 2) {
                this.performanceChart.data.datasets[2].data = [];
            }
            this.performanceChart.update('none');
        }
    
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
        
        // Clear the chart
        if (this.performanceChart) {
            this.performanceChart.data.labels = [];
            this.performanceChart.data.datasets[0].data = [];
            this.performanceChart.data.datasets[1].data = [];
            if (this.performanceChart.data.datasets.length > 2) {
                this.performanceChart.data.datasets[2].data = [];
            }
            this.performanceChart.update('none');
        }
        
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
        
        // Reconnect to neural visualization
        if (this.neuralViz && this.neuralViz.setPredator && this.showNeuralViz) {
            this.neuralViz.setPredator(this.simulation.predator);
        }
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
    
    if (reason === 'target_reached') {
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
        
        var boidsRemaining = this.simulation.boids.length;
        var completionMessage = boidsRemaining === 0 ? 'all boids eaten' : 
                               boidsRemaining + ' boids remaining (threshold: ' + this.completionThreshold + ')';
        
        console.log('✅ Episode', this.currentEpisode + 1, 'completed in', completionTime.toFixed(1) + 's', '(' + completionMessage + ')');
        console.log('   📊 Best:', this.bestCompletionTime.toFixed(1) + 's', 'Avg (last 10):', this.averageCompletionTime.toFixed(1) + 's');
        
        // Show progress towards the goal
        if (this.completionTimes.length >= 2) {
            var firstTime = this.completionTimes[0];
            var improvement = ((firstTime - completionTime) / firstTime) * 100;
            if (improvement > 0) {
                console.log('   📈 Improvement from first episode:', '+' + improvement.toFixed(1) + '%');
            }
        }
        
        // Update the performance chart
        this.updateChart();
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
        
        // Update neural network visualization
        this.updateNeuralVisualization();
        
        if (this.isTraining) {
            var progress = (this.currentEpisode / this.maxEpisodes) * 100;
            document.getElementById('training-progress').style.width = progress + '%';
        }
    }
};

// Update neural network visualization display
NeuralTrainer.prototype.updateNeuralVisualization = function() {
    if (!this.showNeuralViz || !this.simulation || !this.simulation.predator) {
        return;
    }
    
    // Only update at specified frequency for performance
    this.neuralVizFrame++;
    if (this.neuralVizFrame % this.neuralVizFrequency !== 0) {
        return;
    }
    
    var predator = this.simulation.predator;
    
    // Update neural network state display
    if (predator.lastInput && predator.hiddenActivations && predator.lastOutput) {
        // Calculate average input activation
        var avgInput = 0;
        for (var i = 0; i < predator.lastInput.length; i++) {
            avgInput += Math.abs(predator.lastInput[i]);
        }
        avgInput /= predator.lastInput.length;
        
        // Calculate average hidden activation
        var avgHidden = 0;
        for (var i = 0; i < predator.hiddenActivations.length; i++) {
            avgHidden += Math.abs(predator.hiddenActivations[i]);
        }
        avgHidden /= predator.hiddenActivations.length;
        
        // Update UI
        document.getElementById('avg-input').textContent = avgInput.toFixed(3);
        document.getElementById('avg-hidden').textContent = avgHidden.toFixed(3);
        document.getElementById('output-x').textContent = predator.lastOutput[0].toFixed(2);
        document.getElementById('output-y').textContent = predator.lastOutput[1].toFixed(2);
        
        // Update detailed input values (only if neural viz is enabled)
        if (this.showNeuralViz) {
            this.updateInputValuesDisplay(predator.lastInput);
        }
    }
    
    // Neural visualization updates automatically once predator is connected
    // No need to call update methods - it handles itself through animation loop
};

// Update the detailed input values display
NeuralTrainer.prototype.updateInputValuesDisplay = function(inputs) {
    var inputContainer = document.getElementById('input-values');
    if (!inputContainer || !inputs) return;
    
    var html = '';
    
    // Group inputs by type for better readability
    var inputLabels = [
        'Boid 1: pos_x', 'Boid 1: pos_y', 'Boid 1: vel_x', 'Boid 1: vel_y',
        'Boid 2: pos_x', 'Boid 2: pos_y', 'Boid 2: vel_x', 'Boid 2: vel_y',
        'Boid 3: pos_x', 'Boid 3: pos_y', 'Boid 3: vel_x', 'Boid 3: vel_y',
        'Boid 4: pos_x', 'Boid 4: pos_y', 'Boid 4: vel_x', 'Boid 4: vel_y',
        'Boid 5: pos_x', 'Boid 5: pos_y', 'Boid 5: vel_x', 'Boid 5: vel_y',
        'Pred: vel_x', 'Pred: vel_y'
    ];
    
    for (var i = 0; i < inputs.length; i++) {
        var value = inputs[i];
        var label = inputLabels[i] || 'Input ' + i;
        var color = '#212529';
        
        // Color coding based on value (screen-normalized positions and velocities)
        var isPosition = label.includes('pos_');
        var threshold1 = isPosition ? 0.2 : 0.2;   // Near-zero threshold  
        var threshold2 = isPosition ? 0.5 : 0.5;   // Medium threshold  
        var threshold3 = isPosition ? 1.0 : 1.0;   // High threshold
        
        if (Math.abs(value) < threshold1) {
            color = '#6c757d'; // Gray for near-zero
        } else if (Math.abs(value) > threshold3) {
            color = '#dc3545'; // Red for high values
        } else if (Math.abs(value) > threshold2) {
            color = '#fd7e14'; // Orange for medium values
        }
        
        html += '<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">';
        html += '<span style="color: ' + color + ';">' + label + ':</span>';
        html += '<span style="color: ' + color + '; font-weight: bold;">' + value.toFixed(3) + '</span>';
        html += '</div>';
        
        // Add separator between boids
        if (i % 4 === 3 && i < inputs.length - 3) {
            html += '<hr style="margin: 5px 0; border: none; border-top: 1px solid #dee2e6;">';
        }
    }
    
    inputContainer.innerHTML = html;
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
        var displayText = remaining + '/' + total;
        
        // Add threshold indicator if it's set
        if (this.completionThreshold > 0) {
            displayText += ' (→' + this.completionThreshold + ')';
        }
        
        document.getElementById('boids-remaining').textContent = displayText;
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
        // Handle both fast (>= 1x) and slow (< 1x) simulation speeds
        this.simulationFrameAccumulator += this.simulationSpeed;
        
        // Run simulation steps based on accumulated time
        var simulationStepsToRun = 0;
        
        if (this.simulationSpeed >= 1) {
            // For speeds >= 1x, run multiple steps per frame
            simulationStepsToRun = Math.floor(this.simulationFrameAccumulator);
            this.simulationFrameAccumulator = 0; // Reset accumulator for high speeds
        } else {
            // For speeds < 1x, use frame accumulation
            while (this.simulationFrameAccumulator >= 1) {
                simulationStepsToRun++;
                this.simulationFrameAccumulator -= 1;
            }
        }
        
        // Run the calculated number of simulation steps
        for (var i = 0; i < simulationStepsToRun; i++) {
            // Update boids (flocking calculation AND movement)
            this.simulation.tick(); // Calculate flocking forces
            
            // Apply boid movement
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
                
                if (this.simulation.boids.length <= this.completionThreshold) {
                    episodeComplete = true;
                    completionReason = 'target_reached';
                } else if (this.episodeFrame >= this.episodeLength) {
                    episodeComplete = true;
                    completionReason = 'timeout';
                }
                
                if (episodeComplete) {
                    // Give completion bonus if target was reached
                    if (completionReason === 'target_reached' && this.simulation.predator) {
                        var boidsRemaining = this.simulation.boids.length;
                        var completionBonus = 25.0; // Base completion bonus
                        
                        // Extra bonus for getting closer to 0 boids remaining
                        if (boidsRemaining === 0) {
                            completionBonus += 25.0; // Total 50 for perfect completion
                        } else if (boidsRemaining <= 2) {
                            completionBonus += 15.0; // Total 40 for near-perfect
                        } else if (boidsRemaining <= 5) {
                            completionBonus += 10.0; // Total 35 for good completion
                        }
                        
                        this.simulation.predator.updateWeights(completionBonus);
                        this.simulation.predator.episodeReward += completionBonus;
                        
                        console.log('🎉 Completion bonus:', completionBonus.toFixed(1), 'for reaching threshold with', boidsRemaining, 'boids remaining');
                    }
                    
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