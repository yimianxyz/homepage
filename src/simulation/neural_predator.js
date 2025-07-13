/**
 * Neural Predator - Transformer-based game entity
 * 
 * Architecture:
 * - TransformerEncoder with d_model=48, n_heads=4, n_layers=3
 * - Token sequence: [CLS] + [CTX] + Predator + Boids
 * - GEGLU feed-forward networks
 * - Multi-head self-attention for entity interactions
 */
function NeuralPredator(x, y, simulation) {
    // Inherit from basic predator
    Predator.call(this, x, y, simulation);
    
    // Create transformer-based components
    this.transformerEncoder = new TransformerEncoder();
    this.inputProcessor = new InputProcessor();
    this.actionProcessor = new ActionProcessor();
    
    // Simple patrol behavior when no boids
    this.autonomousTarget = new Vector(x, y);
    this.targetChangeTime = 0;
    this.targetChangeInterval = window.SIMULATION_CONSTANTS.TARGET_CHANGE_INTERVAL;
    
    console.log("Initialized Transformer-based Neural Predator");
}

// Inherit from Predator
NeuralPredator.prototype = Object.create(Predator.prototype);
NeuralPredator.prototype.constructor = NeuralPredator;

/**
 * Get steering force using transformer encoder
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
    
    // Step 1: Convert game state to structured inputs
    var structuredInputs = this.inputProcessor.processInputs(
        boids,
        { x: this.position.x, y: this.position.y },
        { x: this.velocity.x, y: this.velocity.y },
        this.simulation.canvasWidth,
        this.simulation.canvasHeight
    );
    
    // Step 2: Process through transformer encoder
    var neuralOutputs = this.transformerEncoder.forward(structuredInputs);
    
    // Step 3: Convert neural outputs to game actions
    var actions = this.actionProcessor.processAction(neuralOutputs);
    
    return new Vector(actions[0], actions[1]);
};

/**
 * Update predator behavior
 */
NeuralPredator.prototype.update = function(boids) {
    // Get transformer-based steering force
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

 