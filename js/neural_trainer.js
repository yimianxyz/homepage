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
    this.learningRate = 0.001;
    this.rewardMemory = 0.95;
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
    this.weightsIH = window.NEURAL_PARAMS.weightsIH.map(row => row.slice());
    this.weightsHO = window.NEURAL_PARAMS.weightsHO.map(row => row.slice());
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
        reward += 10.0;
        this.framesSinceLastFeed = 0;
        this.boidsEaten++;
    } else {
        this.framesSinceLastFeed++;
        reward -= 0.01; // Small penalty for inefficiency
        
        if (this.framesSinceLastFeed > this.maxFramesSinceLastFeed) {
            reward -= 0.05;
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
            reward += (this.maxDistance - nearestDistance) / this.maxDistance * 0.1;
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
    
    // Encode boid positions
    for (var i = 0; i < nearestBoids.length; i++) {
        var boid = nearestBoids[i].boid;
        var relativePos = boid.position.subtract(this.position);
        
        if (isNaN(relativePos.x) || isNaN(relativePos.y)) {
            relativePos.x = 0;
            relativePos.y = 0;
        }
        
        var normalizedX = relativePos.x / this.maxDistance;
        var normalizedY = relativePos.y / this.maxDistance;
        
        if (isNaN(normalizedX)) normalizedX = 0;
        if (isNaN(normalizedY)) normalizedY = 0;
        
        this.inputBuffer[i * 2] = Math.max(-1, Math.min(1, normalizedX));
        this.inputBuffer[i * 2 + 1] = Math.max(-1, Math.min(1, normalizedY));
    }
    
    // Add predator velocity
    var velocityX = this.velocity.x / this.maxVelocity;
    var velocityY = this.velocity.y / this.maxVelocity;
    
    if (isNaN(velocityX)) velocityX = 0;
    if (isNaN(velocityY)) velocityY = 0;
    
    this.inputBuffer[10] = Math.max(-1, Math.min(1, velocityX));
    this.inputBuffer[11] = Math.max(-1, Math.min(1, velocityY));
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
    
    if (Math.abs(rewardError) < 0.001) {
        return;
    }
    
    var learningSignal = this.learningRate * rewardError;
    
    if (isNaN(learningSignal) || !isFinite(learningSignal)) {
        return;
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
    
    var feedReward = 20.0;
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
            var proximityReward = 0.05;
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
        weightsIH: this.weightsIH.map(row => row.slice()),
        weightsHO: this.weightsHO.map(row => row.slice()),
        biasH: this.biasH.slice(),
        biasO: this.biasO.slice(),
        maxDistance: this.maxDistance,
        maxVelocity: this.maxVelocity,
        version: "1.0.0",
        trained: true,
        description: "Trained neural network parameters"
    };
    
    return window.exportNeuralParams(params);
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
    
    document.getElementById('sim-speed').addEventListener('input', function(e) {
        self.simulationSpeed = parseFloat(e.target.value);
        document.getElementById('speed-display').textContent = self.simulationSpeed.toFixed(1) + 'x';
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

NeuralTrainer.prototype.startTraining = function() {
    this.isTraining = true;
    this.currentEpisode = 0;
    this.episodeFrame = 0;
    
    document.getElementById('start-training').disabled = true;
    document.getElementById('stop-training').disabled = false;
    document.getElementById('status-text').textContent = 'Training...';
    document.getElementById('status-dot').className = 'status-dot training';
    
    console.log('Started training for', this.maxEpisodes, 'episodes');
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
        this.simulation.predator.loadParameters();
        this.simulation.predator.resetEpisode();
        this.currentEpisode = 0;
        this.episodeFrame = 0;
        
        document.getElementById('status-text').textContent = 'Reset';
        document.getElementById('status-dot').className = 'status-dot';
        console.log('Network reset to initial parameters');
    }
};

NeuralTrainer.prototype.loadParameters = function() {
    if (this.simulation && this.simulation.predator) {
        this.simulation.predator.loadParameters();
        console.log('Parameters reloaded from parameters.js');
    }
};

NeuralTrainer.prototype.restartSimulation = function() {
    this.initializeSimulation();
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
        var exportedCode = this.simulation.predator.exportParameters();
        document.getElementById('export-output').value = exportedCode;
        console.log('Parameters exported');
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
        
        if (this.isTraining) {
            var progress = (this.currentEpisode / this.maxEpisodes) * 100;
            document.getElementById('training-progress').style.width = progress + '%';
        }
    }
};

NeuralTrainer.prototype.update = function() {
    if (this.simulation) {
        // Update simulation based on speed
        for (var i = 0; i < this.simulationSpeed; i++) {
            this.simulation.tick();
            
            if (this.isTraining) {
                this.episodeFrame++;
                
                // Check if episode is complete
                if (this.episodeFrame >= this.episodeLength || this.simulation.boids.length === 0) {
                    this.currentEpisode++;
                    this.episodeFrame = 0;
                    
                    if (this.simulation.predator) {
                        this.simulation.predator.resetEpisode();
                    }
                    
                    // Restart simulation for next episode
                    this.restartSimulation();
                    
                    // Check if training is complete
                    if (this.currentEpisode >= this.maxEpisodes) {
                        this.stopTraining();
                        document.getElementById('status-text').textContent = 'Complete';
                        document.getElementById('status-dot').className = 'status-dot running';
                        this.exportParameters();
                    }
                }
            }
        }
        
        // Handle predator-prey interactions
        if (this.simulation.predator) {
            var caughtBoids = this.simulation.predator.checkForPrey(this.simulation.boids);
            for (var i = caughtBoids.length - 1; i >= 0; i--) {
                this.simulation.boids.splice(caughtBoids[i], 1);
                
                if (this.simulation.predator.calculateReward) {
                    var catchReward = this.simulation.predator.calculateReward(this.simulation.boids, true);
                    this.simulation.predator.updateWeights(catchReward);
                }
            }
        }
    }
};

NeuralTrainer.prototype.render = function() {
    if (this.simulation) {
        // Only render visuals if simulation is visible
        if (this.showSimulation) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Render boids
            for (var i = 0; i < this.simulation.boids.length; i++) {
                this.simulation.boids[i].run(this.simulation.boids);
            }
            
            // Render predator
            if (this.simulation.predator) {
                this.simulation.predator.render();
            }
        }
        
        // Always update predator logic (even when not visible)
        if (this.simulation.predator) {
            this.simulation.predator.update(this.simulation.boids);
        }
        
        // Update boid logic (even when not visible)
        if (!this.showSimulation) {
            for (var i = 0; i < this.simulation.boids.length; i++) {
                this.simulation.boids[i].flock(this.simulation.boids);
                this.simulation.boids[i].update();
            }
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