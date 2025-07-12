/**
 * Reinforcement Learning - Simple policy gradient training
 */

function ReinforcementLearning() {
    this.learningRate = 0.001;
    
    this.episodeCount = 0;
    this.episodeReward = 0;
    
    this.stateHistory = [];
    this.actionHistory = [];
    this.maxHistorySize = 1000;
    
    this.averageReward = 0;
}

ReinforcementLearning.prototype.storeExperience = function(state, action) {
    this.stateHistory.push(state.slice());
    this.actionHistory.push(action.slice());
    
    if (this.stateHistory.length > this.maxHistorySize) {
        this.stateHistory.shift();
        this.actionHistory.shift();
    }
};

ReinforcementLearning.prototype.calculateReward = function(episodeComplete, completionFrames) {
    // All episodes are now success episodes (no timeout)
    // Reward based on efficiency: faster completion = higher reward
    return Math.max(1000 - completionFrames, 10);
};

ReinforcementLearning.prototype.applyPolicyGradient = function(policyNetwork, reward) {
    if (this.stateHistory.length === 0) {
        return;
    }
    
    // Simple policy gradient - use reward directly
    for (var i = 0; i < this.stateHistory.length; i++) {
        var state = this.stateHistory[i];
        policyNetwork.forward(state);
        policyNetwork.policyGradientBackprop(reward, 0, this.learningRate);
    }
    
    this.stateHistory = [];
    this.actionHistory = [];
};

ReinforcementLearning.prototype.updateEpisodeStats = function(reward, completionFrames, isComplete) {
    this.episodeCount++;
    this.episodeReward = reward;
    this.averageReward = this.averageReward * 0.9 + reward * 0.1;
};

ReinforcementLearning.prototype.resetEpisode = function() {
    this.episodeReward = 0;
};

ReinforcementLearning.prototype.getStatistics = function() {
    return {
        episodeCount: this.episodeCount,
        episodeReward: this.episodeReward,
        averageReward: this.averageReward
    };
};

ReinforcementLearning.prototype.reset = function() {
    this.episodeCount = 0;
    this.episodeReward = 0;
    this.stateHistory = [];
    this.actionHistory = [];
    this.averageReward = 0;
}; 