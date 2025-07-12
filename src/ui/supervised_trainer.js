// Supervised Learning Trainer
function SupervisedTrainer() {
    // Call parent constructor
    BaseTrainer.call(this);
    
    // Supervised learning specific components
    this.teacherPolicy = new TeacherPolicy();
    this.supervisedLearning = new SupervisedLearning();
    this.currentLoss = 0;
    
    this.initialize();
}

// Inherit from BaseTrainer
SupervisedTrainer.prototype = Object.create(BaseTrainer.prototype);
SupervisedTrainer.prototype.constructor = SupervisedTrainer;

// Override network reset to include supervised learning specific cleanup
SupervisedTrainer.prototype.onNetworkReset = function() {
    this.supervisedLearning.reset();
    this.currentLoss = 0;
    this.resetLossChart();
};

// Override network load to include supervised learning specific cleanup
SupervisedTrainer.prototype.onNetworkLoad = function() {
    this.supervisedLearning.reset();
    this.currentLoss = 0;
    this.resetLossChart();
};

// Override training step with supervised learning specific logic
SupervisedTrainer.prototype.updateTrainingStep = function() {
    if (!this.isTraining) return;
    
    var inputs = this.inputProcessor.processInputs(
        this.simulation.boids,
        { x: this.simulation.predator.position.x, y: this.simulation.predator.position.y },
        { x: this.simulation.predator.velocity.x, y: this.simulation.predator.velocity.y },
        this.simulation.canvasWidth,
        this.simulation.canvasHeight
    );
    
    var neuralOutputs = this.neuralNetwork.forward(inputs);
    var neuralAction = this.actionProcessor.processAction(neuralOutputs);
    var teacherAction = this.teacherPolicy.getAction(inputs);
    
    var currentLoss = this.supervisedLearning.collectTrainingData(inputs, neuralAction, teacherAction);
    this.currentLoss = currentLoss;
    
    // Update loss chart with current loss
    if (this.frameCount % 10 === 0) {
        this.addLossDataPoint(currentLoss);
    }
    
    if (this.frameCount % 10 === 0) {
        this.supervisedLearning.trainBatch(this.neuralNetwork);
        this.updateDisplay();
    }
};



SupervisedTrainer.prototype.updateDisplay = function() {
    var stats = this.supervisedLearning.getStatistics();
    
    document.getElementById('epochs').textContent = stats.epochs;
    document.getElementById('current-loss').textContent = (this.currentLoss || 0).toFixed(3);
    document.getElementById('avg-loss').textContent = stats.averageLoss.toFixed(3);
    document.getElementById('buffer-size').textContent = stats.bufferSize;
    
    // Progress: higher when loss is lower (inverse relationship)
    // Assume max useful loss is around 2.0, so progress = (2.0 - loss) / 2.0 * 100
    var maxLoss = 2.0;
    var currentAvgLoss = Math.min(stats.averageLoss, maxLoss);
    var progress = Math.max(0, (maxLoss - currentAvgLoss) / maxLoss * 100);
    document.getElementById('training-progress').style.width = progress + '%';
};

// Initialize trainer when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (!isCanvasSupported()) {
        alert('Canvas not supported. Please use a modern browser.');
        return;
    }
    
    window.trainer = new SupervisedTrainer();
}); 