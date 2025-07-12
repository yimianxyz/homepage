// Simple Reinforcement Learning Trainer
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
};

// Override network load to include RL specific cleanup
SimpleRLTrainer.prototype.onNetworkLoad = function() {
    this.reinforcementLearning.reset();
    this.episodeActive = false;
};

// Override training step with reinforcement learning specific logic
SimpleRLTrainer.prototype.updateTrainingStep = function() {
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
    
    this.reinforcementLearning.storeExperience(inputs, neuralAction);
    
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
    
    this.reinforcementLearning.applyPolicyGradient(this.neuralNetwork, reward);
    this.reinforcementLearning.updateEpisodeStats(reward, completionFrames, isComplete);
    
    this.episodeActive = false;
    this.reinforcementLearning.resetEpisode();
    this.updateDisplay();
    
    // Restart simulation for next episode
    this.initializeSimulation();
};



SimpleRLTrainer.prototype.updateDisplay = function() {
    var stats = this.reinforcementLearning.getStatistics();
    
    document.getElementById('episodes').textContent = stats.episodeCount;
    document.getElementById('last-reward').textContent = Math.round(stats.episodeReward);
    document.getElementById('avg-reward').textContent = Math.round(stats.averageReward);
    
    var progress = Math.min((stats.averageReward / 100) * 100, 100);
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