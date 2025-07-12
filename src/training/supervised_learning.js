/**
 * Supervised Learning - MSE loss-based teacher policy imitation
 */

function SupervisedLearning() {
    this.learningRate = 0.01;
    this.batchSize = 32;
    this.trainingBuffer = [];
    this.maxBufferSize = 1000;
    this.epochs = 0;
    this.averageLoss = 0;
}

SupervisedLearning.prototype.collectTrainingData = function(inputs, neuralAction, teacherAction) {
    var loss = this.calculateMSELoss(neuralAction, teacherAction);
    
    var sample = {
        inputs: inputs.slice(),
        neuralAction: neuralAction.slice(),
        teacherAction: teacherAction.slice(),
        loss: loss
    };
    
    this.trainingBuffer.push(sample);
    
    if (this.trainingBuffer.length > this.maxBufferSize) {
        this.trainingBuffer.shift();
    }
    
    return loss;
};

SupervisedLearning.prototype.calculateMSELoss = function(neuralAction, teacherAction) {
    var sumSquaredError = 0;
    for (var i = 0; i < neuralAction.length; i++) {
        var error = teacherAction[i] - neuralAction[i];
        sumSquaredError += error * error;
    }
    return sumSquaredError / neuralAction.length;
};

SupervisedLearning.prototype.trainBatch = function(policyNetwork) {
    if (this.trainingBuffer.length < this.batchSize) {
        return null;
    }
    
    var batchSize = Math.min(this.batchSize, this.trainingBuffer.length);
    var totalLoss = 0;
    var batchSamples = [];
    
    for (var i = 0; i < batchSize; i++) {
        var randomIndex = Math.floor(Math.random() * this.trainingBuffer.length);
        batchSamples.push(this.trainingBuffer[randomIndex]);
    }
    
    for (var i = 0; i < batchSamples.length; i++) {
        var sample = batchSamples[i];
        var networkOutput = policyNetwork.forward(sample.inputs);
        
        // Direct supervision: train network to output teacher action
        policyNetwork.supervisedBackprop(sample.teacherAction, this.learningRate);
        
        totalLoss += sample.loss;
    }
    
    var avgLoss = totalLoss / batchSize;
    this.averageLoss = this.averageLoss * 0.9 + avgLoss * 0.1;
    this.epochs++;
    
    return {
        epoch: this.epochs,
        avgLoss: avgLoss,
        runningAvgLoss: this.averageLoss
    };
};

SupervisedLearning.prototype.getStatistics = function() {
    return {
        epochs: this.epochs,
        bufferSize: this.trainingBuffer.length,
        averageLoss: this.averageLoss
    };
};

SupervisedLearning.prototype.reset = function() {
    this.trainingBuffer = [];
    this.epochs = 0;
    this.averageLoss = 0;
}; 