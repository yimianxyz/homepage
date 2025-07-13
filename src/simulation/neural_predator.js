/**
 * Neural Predator - Game entity that uses modular neural components
 * 
 * Combines:
 * - NeuralNetwork (pure neural math)
 * - InputProcessor (game state -> neural inputs)
 * - ActionProcessor (neural outputs -> game actions)
 * 
 * Updated for new encoding system: 204 inputs → 64 hidden1 → 32 hidden2 → 2 outputs
 */
function NeuralPredator(x, y, simulation) {
    // Inherit from basic predator
    Predator.call(this, x, y, simulation);
    
    // Create modular components with new architecture
    this.neuralNetwork = new NeuralNetwork(204, 64, 32, 2);
    this.inputProcessor = new InputProcessor();
    this.actionProcessor = new ActionProcessor();
    
    // Load pre-trained weights if available and store result
    this.modelLoadResult = this.neuralNetwork.loadParameters();
    
    // If loading failed, the network will use random initialization
    if (!this.modelLoadResult.success && this.modelLoadResult.fallbackReason) {
        console.warn("Neural Network: " + this.modelLoadResult.message + " - " + this.modelLoadResult.fallbackReason);
    }
    
    // Simple patrol behavior when no boids
    this.autonomousTarget = new Vector(x, y);
    this.targetChangeTime = 0;
    this.targetChangeInterval = window.SIMULATION_CONSTANTS.TARGET_CHANGE_INTERVAL;
}

// Inherit from Predator
NeuralPredator.prototype = Object.create(Predator.prototype);
NeuralPredator.prototype.constructor = NeuralPredator;

/**
 * Get steering force using modular neural components
 */
NeuralPredator.prototype.getAutonomousForce = function(boids) {
    if (boids.length === 0) {
        // No boids - simple patrol behavior
        var currentTime = Date.now();
        if (currentTime - this.targetChangeTime > this.targetChangeInterval) {
            var margin = 50;
            this.autonomousTarget.x = margin + Math.random() * (this.simulation.canvasWidth - 2 * margin);
            this.autonomousTarget.y = margin + Math.random() * (this.simulation.canvasHeight - 2 * margin);
            this.targetChangeTime = currentTime;
        }
        return this.seek(this.autonomousTarget);
    }
    
    // Step 1: Convert game state to neural inputs
    var inputs = this.inputProcessor.processInputs(
        boids,
        { x: this.position.x, y: this.position.y },
        { x: this.velocity.x, y: this.velocity.y },
        this.simulation.canvasWidth,
        this.simulation.canvasHeight
    );
    
    // Step 2: Get neural network prediction
    var neuralOutputs = this.neuralNetwork.forward(inputs);
    
    // Step 3: Convert neural outputs to game actions
    var actions = this.actionProcessor.processAction(neuralOutputs);
    
    return new Vector(actions[0], actions[1]);
};

/**
 * Update predator behavior
 */
NeuralPredator.prototype.update = function(boids) {
    // Get neural network steering force
    var steeringForce = this.getAutonomousForce(boids);
    this.acceleration.iAdd(steeringForce);
    
    // Call parent update for physics
    Predator.prototype.update.call(this, boids);
};

/**
 * Simple seeking behavior for patrol
 */
NeuralPredator.prototype.seek = function(targetPosition) {
    var desiredVector = targetPosition.subtract(this.position);
    desiredVector.iFastSetMagnitude(window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED);
    var steeringVector = desiredVector.subtract(this.velocity);
    steeringVector.iFastLimit(window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * 30); // Scaled for patrol
    return steeringVector;
};

 