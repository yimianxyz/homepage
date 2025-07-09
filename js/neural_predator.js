/**
 * Neural Predator - AI-Powered Hunting Behavior
 * 
 * This is the main predator implementation that extends the base Predator class
 * with intelligent neural network behavior for prediction only.
 * 
 * Network Architecture: 22 inputs → 12 hidden → 2 outputs
 * - Inputs: positions & velocities of 5 nearest boids + predator velocity
 * - Outputs: steering force (x, y)
 * - Enhanced with boid velocity data for predictive hunting
 * - Optimized for real-time web performance with <0.3ms forward pass
 * 
 * Prediction Features:
 * - Loads pre-trained weights from parameters.js
 * - Fast forward pass for real-time hunting behavior
 * - No online learning - stable and consistent behavior
 * - Elegant visual design with size scaling
 */
function NeuralPredator(x, y, simulation) {
    // Inherit from basic predator
    Predator.call(this, x, y, simulation);
    
    // Patrol behavior for when no boids are present
    this.autonomousTarget = new Vector(x, y);
    this.targetChangeTime = 0;
    this.targetChangeInterval = 5000; // Change target every 5 seconds
    
    // Load neural network parameters from parameters.js
    this.loadParameters();
    
    // Performance optimization - pre-allocate buffers
    this.inputBuffer = new Array(this.inputSize);
    this.hiddenBuffer = new Array(this.hiddenSize);
    this.outputBuffer = new Array(this.outputSize);
}

// Inherit from Predator
NeuralPredator.prototype = Object.create(Predator.prototype);
NeuralPredator.prototype.constructor = NeuralPredator;

// Load neural network parameters from parameters.js
NeuralPredator.prototype.loadParameters = function() {
    if (typeof window.NEURAL_PARAMS === 'undefined') {
        console.error('Neural parameters not loaded! Make sure parameters.js is included.');
        // Fallback to default parameters
        this.inputSize = 12;
        this.hiddenSize = 8;
        this.outputSize = 2;
        this.maxDistance = 100;
        this.maxVelocity = 6;
        this.weightsIH = []; // Will cause errors, but prevents crash
        this.weightsHO = [];
        this.biasH = [];
        this.biasO = [];
        return;
    }
    
    // Load network architecture
    this.inputSize = window.NEURAL_PARAMS.inputSize;
    this.hiddenSize = window.NEURAL_PARAMS.hiddenSize;
    this.outputSize = window.NEURAL_PARAMS.outputSize;
    
    // Load normalization constants
    this.maxDistance = window.NEURAL_PARAMS.maxDistance;
    this.maxVelocity = window.NEURAL_PARAMS.maxVelocity;
    
    // Load weights and biases (deep copy to prevent modification)
    this.weightsIH = window.NEURAL_PARAMS.weightsIH.map(function(row) { return row.slice(); });
    this.weightsHO = window.NEURAL_PARAMS.weightsHO.map(function(row) { return row.slice(); });
    this.biasH = window.NEURAL_PARAMS.biasH.slice();
    this.biasO = window.NEURAL_PARAMS.biasO.slice();
    
    console.log('Neural predator loaded parameters version:', window.NEURAL_PARAMS.version);
};







// Fast activation function (tanh approximation)
NeuralPredator.prototype.fastTanh = function(x) {
    if (x > 2) return 1;
    if (x < -2) return -1;
    var x2 = x * x;
    return x * (27 + x2) / (27 + 9 * x2);
};

// Calculate shortest distance considering wraparound (toroidal world)
NeuralPredator.prototype.calculateWrappedDistance = function(targetPos, sourcePos) {
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

// Prepare neural network inputs from current game state
NeuralPredator.prototype.prepareInputs = function(boids) {
    // Find 5 nearest boids using wrapped distance
    var nearestBoids = [];
    
    for (var i = 0; i < boids.length; i++) {
        var wrappedVector = this.calculateWrappedDistance(boids[i].position, this.position);
        var wrappedDistance = Math.sqrt(wrappedVector.x * wrappedVector.x + wrappedVector.y * wrappedVector.y);
        nearestBoids.push({boid: boids[i], distance: wrappedDistance});
    }
    
    // Sort by wrapped distance and take closest 5
    nearestBoids.sort(function(a, b) { return a.distance - b.distance; });
    nearestBoids = nearestBoids.slice(0, 5);
    
    // Clear input buffer
    for (var i = 0; i < this.inputSize; i++) {
        this.inputBuffer[i] = 0;
    }
    
    // Encode boid positions AND velocities (4 values per boid)
    for (var i = 0; i < nearestBoids.length; i++) {
        var boid = nearestBoids[i].boid;
        
        // Calculate shortest wrapped distance (considering screen edges)
        var relativePos = this.calculateWrappedDistance(boid.position, this.position);
        
        // Safety check for NaN positions
        if (isNaN(relativePos.x) || isNaN(relativePos.y)) {
            relativePos.x = 0;
            relativePos.y = 0;
        }
        
        // Normalize relative position (-1 to 1)
        var normalizedPosX = relativePos.x / this.maxDistance;
        var normalizedPosY = relativePos.y / this.maxDistance;
        
        // Safety check for NaN normalized values
        if (isNaN(normalizedPosX)) normalizedPosX = 0;
        if (isNaN(normalizedPosY)) normalizedPosY = 0;
        
        // Normalize boid velocity (-1 to 1)
        var normalizedVelX = boid.velocity.x / this.maxVelocity;
        var normalizedVelY = boid.velocity.y / this.maxVelocity;
        
        // Safety check for NaN velocities
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
    
    // Safety check for NaN velocities
    if (isNaN(predatorVelX)) predatorVelX = 0;
    if (isNaN(predatorVelY)) predatorVelY = 0;
    
    this.inputBuffer[20] = Math.max(-1, Math.min(1, predatorVelX));
    this.inputBuffer[21] = Math.max(-1, Math.min(1, predatorVelY));
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
    
    // Use neural network for hunting behavior (prediction only)
    this.prepareInputs(boids);
    var force = this.forward();
    
    return force;
};

// Override feed method (prediction only - no learning)
NeuralPredator.prototype.feed = function() {
    // Call parent feed method
    this.currentSize = Math.min(this.currentSize + this.growthPerFeed, this.maxSize);
    this.lastFeedTime = Date.now();
};

// Enhanced update method with neural behavior (prediction only)
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
};

// Enhanced seeking with neural network refinement
NeuralPredator.prototype.seek = function(targetPosition) {
    var desiredVector = targetPosition.subtract(this.position);
    desiredVector.iFastSetMagnitude(PREDATOR_MAX_SPEED * 0.8); // Slightly slower patrol
    var steeringVector = desiredVector.subtract(this.velocity);
    steeringVector.iFastLimit(PREDATOR_MAX_FORCE * 0.6); // Gentler patrol steering
    return steeringVector;
};



// Enhanced rendering with size-based visualization
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
    
    // Draw predator as a distinctive but elegant elongated triangle
    var directionVector = this.velocity.normalize().multiplyBy(this.currentSize * 1.2);
    var inverseVector1 = new Vector(-directionVector.y, directionVector.x);
    var inverseVector2 = new Vector(directionVector.y, -directionVector.x);
    inverseVector1 = inverseVector1.divideBy(4);
    inverseVector2 = inverseVector2.divideBy(4);
    
    // Enhanced visibility while maintaining elegance
    var sizeRatio = this.currentSize / this.baseSize;
    var intensity = 0.4 + (sizeRatio - 1) * 0.3; // Size-based intensity
    
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
    
    // Add a subtle inner highlight that scales with size
    var highlightSize = Math.max(3, this.currentSize * 0.2);
    var highlightOpacity = 0.7 + intensity * 0.3;
    
    ctx.beginPath();
    ctx.arc(this.position.x, this.position.y, highlightSize, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(160, 70, 70, ' + Math.min(1.0, highlightOpacity) + ')';
    ctx.fill();
}; 