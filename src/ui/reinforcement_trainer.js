// Reinforcement Learning Trainer - Transformer Architecture
function SimpleRLTrainer() {
    // Call parent constructor
    BaseTrainer.call(this);
    
    // Reinforcement learning specific components
    this.reinforcementLearning = new ReinforcementLearning();
    
    // Simple episode tracking
    this.episodeStartFrame = 0;
    this.episodeActive = false;
    
    this.initialize();
}

// Inherit from BaseTrainer
SimpleRLTrainer.prototype = Object.create(BaseTrainer.prototype);
SimpleRLTrainer.prototype.constructor = SimpleRLTrainer;

// Override start training to handle episode state
SimpleRLTrainer.prototype.startTraining = function() {
    BaseTrainer.prototype.startTraining.call(this);
    // No special RL-specific start logic needed currently
};

// Override stop training to handle episode state
SimpleRLTrainer.prototype.stopTraining = function() {
    BaseTrainer.prototype.stopTraining.call(this);
    this.episodeActive = false;
};

// Override network reset to include RL specific cleanup
SimpleRLTrainer.prototype.onNetworkReset = function() {
    this.reinforcementLearning.reset();
    this.episodeActive = false;
    
    // Reset transformer encoder parameters
    if (this.simulation && this.simulation.predator && this.simulation.predator.transformerEncoder) {
        this.simulation.predator.transformerEncoder.reset();
        console.log("Reset transformer encoder parameters");
    }
};

// Override network load to load transformer parameters from model.js
SimpleRLTrainer.prototype.onNetworkLoad = function() {
    this.reinforcementLearning.reset();
    this.episodeActive = false;
    
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
SimpleRLTrainer.prototype.showLoadSuccessMessage = function(message) {
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
SimpleRLTrainer.prototype.showLoadWarningMessage = function(message) {
    var messageDiv = document.createElement('div');
    messageDiv.style.cssText = 'background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; margin: 10px 0; border-radius: 4px; font-size: 12px;';
    messageDiv.textContent = '⚠️ ' + message;
    
    var controlHeader = document.querySelector('.control-header');
    if (controlHeader && controlHeader.nextSibling) {
        controlHeader.parentNode.insertBefore(messageDiv, controlHeader.nextSibling);
        setTimeout(function() { messageDiv.remove(); }, 5000);
    }
};

// Override training step with reinforcement learning specific logic
SimpleRLTrainer.prototype.updateTrainingStep = function() {
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
    
    this.reinforcementLearning.storeExperience(structuredInputs, neuralAction);
    
    if (!this.episodeActive) {
        this.startEpisode();
    }
    
    if (this.shouldCompleteEpisode()) {
        this.completeEpisode();
    }
};

SimpleRLTrainer.prototype.startEpisode = function() {
    this.episodeActive = true;
    this.episodeStartFrame = this.frameCount;
};

SimpleRLTrainer.prototype.shouldCompleteEpisode = function() {
    if (!this.episodeActive) return false;
    
    // Episode completes only when predator catches enough boids (success-only)
    if (this.simulation.boids.length <= 20) return true;
    
    return false;
};

SimpleRLTrainer.prototype.completeEpisode = function() {
    if (!this.episodeActive) return;
    
    var completionFrames = this.frameCount - this.episodeStartFrame;
    var isComplete = this.simulation.boids.length <= 20;
    // Only complete episodes are success episodes now (no timeout)
    var reward = this.reinforcementLearning.calculateReward(isComplete, completionFrames);
    
    // TODO: Implement transformer policy gradient backpropagation
    // this.reinforcementLearning.applyPolicyGradient(this.transformerEncoder, reward);
    
    this.reinforcementLearning.updateEpisodeStats(reward, completionFrames, isComplete);
    
    this.episodeActive = false;
    this.reinforcementLearning.resetEpisode();
    this.updateDisplay();
    
    // Restart simulation for next episode
    this.initializeSimulation();
};

SimpleRLTrainer.prototype.updateDisplay = function() {
    var stats = this.reinforcementLearning.getStatistics();
    
    document.getElementById('episodes').textContent = stats.episodeCount || 0;
    document.getElementById('last-reward').textContent = Math.round(stats.episodeReward || 0);
    document.getElementById('avg-reward').textContent = Math.round(stats.averageReward || 0);
    
    var progress = Math.min(((stats.averageReward || 0) / 100) * 100, 100);
    document.getElementById('progress').textContent = Math.round(progress) + '%';
    document.getElementById('training-progress').style.width = progress + '%';
};

// Initialize trainer when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (!isCanvasSupported()) {
        alert('Canvas not supported. Please use a modern browser.');
        return;
    }
    
    window.trainer = new SimpleRLTrainer();
}); 