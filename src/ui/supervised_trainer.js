// Supervised Learning Trainer - Transformer Architecture
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
    
    // Reset transformer encoder parameters
    if (this.simulation && this.simulation.predator && this.simulation.predator.transformerEncoder) {
        this.simulation.predator.transformerEncoder.reset();
        console.log("Reset transformer encoder parameters");
    }
};

// Override network load to load transformer parameters from model.js
SupervisedTrainer.prototype.onNetworkLoad = function() {
    this.supervisedLearning.reset();
    this.currentLoss = 0;
    this.resetLossChart();
    
    // Try to load transformer parameters
    if (this.simulation && this.simulation.predator && this.simulation.predator.transformerEncoder) {
        var loadResult = this.simulation.predator.transformerEncoder.loadParameters();
        if (loadResult.success) {
            console.log("✅ " + loadResult.message);
            this.showLoadSuccessMessage("Transformer model loaded successfully!");
        } else {
            console.warn("⚠️ " + loadResult.message + " - " + loadResult.fallbackReason);
            this.showLoadWarningMessage("Failed to load transformer model: " + loadResult.fallbackReason);
        }
    }
};

// Show load success message
SupervisedTrainer.prototype.showLoadSuccessMessage = function(message) {
    var messageDiv = document.createElement('div');
    messageDiv.style.cssText = 'background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; margin: 10px 0; border-radius: 4px; font-size: 12px;';
    messageDiv.textContent = '✅ ' + message;
    
    var controlHeader = document.querySelector('.control-header');
    if (controlHeader && controlHeader.nextSibling) {
        controlHeader.parentNode.insertBefore(messageDiv, controlHeader.nextSibling);
        setTimeout(function() { messageDiv.remove(); }, 3000);
    }
};

// Show load warning message
SupervisedTrainer.prototype.showLoadWarningMessage = function(message) {
    var messageDiv = document.createElement('div');
    messageDiv.style.cssText = 'background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; margin: 10px 0; border-radius: 4px; font-size: 12px;';
    messageDiv.textContent = '⚠️ ' + message;
    
    var controlHeader = document.querySelector('.control-header');
    if (controlHeader && controlHeader.nextSibling) {
        controlHeader.parentNode.insertBefore(messageDiv, controlHeader.nextSibling);
        setTimeout(function() { messageDiv.remove(); }, 5000);
    }
};

// Override training step with supervised learning specific logic
SupervisedTrainer.prototype.updateTrainingStep = function() {
    if (!this.isTraining) return;
    
    // Get structured inputs using new transformer-ready format
    var structuredInputs = this.inputProcessor.processInputs(
        this.simulation.boids,
        { x: this.simulation.predator.position.x, y: this.simulation.predator.position.y },
        { x: this.simulation.predator.velocity.x, y: this.simulation.predator.velocity.y },
        this.simulation.canvasWidth,
        this.simulation.canvasHeight
    );
    
    // Process through transformer encoder
    var neuralOutputs = this.simulation.predator.transformerEncoder.forward(structuredInputs);
    var neuralAction = this.actionProcessor.processAction(neuralOutputs);
    
    // Get teacher action based on structured inputs
    var teacherAction = this.teacherPolicy.getAction(structuredInputs);
    
    var currentLoss = this.supervisedLearning.collectTrainingData(structuredInputs, neuralAction, teacherAction);
    this.currentLoss = currentLoss;
    
    // Update loss chart with current loss
    if (this.frameCount % 10 === 0) {
        this.addLossDataPoint(currentLoss);
    }
    
    // TODO: Implement transformer backpropagation for supervised learning
    if (this.frameCount % 10 === 0) {
        // For now, just update display
        this.updateDisplay();
    }
};

SupervisedTrainer.prototype.updateDisplay = function() {
    var stats = this.supervisedLearning.getStatistics();
    
    document.getElementById('epochs').textContent = stats.epochs || 0;
    document.getElementById('current-loss').textContent = (this.currentLoss || 0).toFixed(3);
    document.getElementById('avg-loss').textContent = (stats.averageLoss || 0).toFixed(3);
    document.getElementById('buffer-size').textContent = stats.bufferSize || 0;
    
    // Progress: higher when loss is lower (inverse relationship)
    // Assume max useful loss is around 2.0, so progress = (2.0 - loss) / 2.0 * 100
    var maxLoss = 2.0;
    var currentAvgLoss = Math.min(stats.averageLoss || maxLoss, maxLoss);
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