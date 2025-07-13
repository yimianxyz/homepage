/**
 * Pure Neural Network - Core neural network mathematics
 * 4-layer architecture: Input → Hidden1 → Hidden2 → Output
 * Updated for new encoding system: 204 inputs → 64 hidden1 → 32 hidden2 → 2 outputs
 */
function NeuralNetwork(inputSize, hidden1Size, hidden2Size, outputSize) {
    // Use new architecture parameters from config, with fallbacks
    this.inputSize = inputSize || window.SIMULATION_CONSTANTS.NEURAL_INPUT_SIZE || 204;
    this.hidden1Size = hidden1Size || 64;
    this.hidden2Size = hidden2Size || 32;
    this.outputSize = outputSize || 2;
    
    this.initializeWeights();
    
    this.lastInputs = null;
    this.lastHidden1 = null;
    this.lastHidden2 = null;
    this.lastOutput = null;
}

NeuralNetwork.prototype.initializeWeights = function() {
    // Input to Hidden1 weights
    this.weightsIH1 = [];
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        this.weightsIH1[h1] = [];
        for (var i = 0; i < this.inputSize; i++) {
            this.weightsIH1[h1][i] = (Math.random() - 0.5) * 0.5;
        }
    }
    
    // Hidden1 to Hidden2 weights
    this.weightsH1H2 = [];
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        this.weightsH1H2[h2] = [];
        for (var h1 = 0; h1 < this.hidden1Size; h1++) {
            this.weightsH1H2[h2][h1] = (Math.random() - 0.5) * 0.5;
        }
    }
    
    // Hidden2 to Output weights
    this.weightsH2O = [];
    for (var o = 0; o < this.outputSize; o++) {
        this.weightsH2O[o] = [];
        for (var h2 = 0; h2 < this.hidden2Size; h2++) {
            this.weightsH2O[o][h2] = (Math.random() - 0.5) * 0.5;
        }
    }
    
    // Hidden1 biases
    this.biasH1 = [];
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        this.biasH1[h1] = (Math.random() - 0.5) * 0.1;
    }
    
    // Hidden2 biases
    this.biasH2 = [];
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        this.biasH2[h2] = (Math.random() - 0.5) * 0.1;
    }
    
    // Output biases
    this.biasO = [];
    for (var o = 0; o < this.outputSize; o++) {
        this.biasO[o] = (Math.random() - 0.5) * 0.1;
    }
};

NeuralNetwork.prototype.forward = function(inputs) {
    this.lastInputs = inputs.slice();
    
    // Forward pass through Hidden1
    this.lastHidden1 = [];
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        var activation = this.biasH1[h1];
        for (var i = 0; i < this.inputSize; i++) {
            activation += inputs[i] * this.weightsIH1[h1][i];
        }
        this.lastHidden1[h1] = Math.tanh(activation);
    }
    
    // Forward pass through Hidden2
    this.lastHidden2 = [];
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        var activation = this.biasH2[h2];
        for (var h1 = 0; h1 < this.hidden1Size; h1++) {
            activation += this.lastHidden1[h1] * this.weightsH1H2[h2][h1];
        }
        this.lastHidden2[h2] = Math.tanh(activation);
    }
    
    // Forward pass through Output
    this.lastOutput = [];
    for (var o = 0; o < this.outputSize; o++) {
        var activation = this.biasO[o];
        for (var h2 = 0; h2 < this.hidden2Size; h2++) {
            activation += this.lastHidden2[h2] * this.weightsH2O[o][h2];
        }
        this.lastOutput[o] = Math.tanh(activation);
    }
    
    return this.lastOutput.slice();
};

NeuralNetwork.prototype.supervisedBackprop = function(targetOutput, learningRate) {
    if (!this.lastInputs || !this.lastHidden1 || !this.lastHidden2 || !this.lastOutput) {
        return;
    }
    
    // Calculate output errors
    var outputErrors = [];
    for (var o = 0; o < this.outputSize; o++) {
        outputErrors[o] = targetOutput[o] - this.lastOutput[o];
    }
    
    // Calculate output gradients
    var outputGradients = [];
    for (var o = 0; o < this.outputSize; o++) {
        outputGradients[o] = outputErrors[o] * (1 - this.lastOutput[o] * this.lastOutput[o]);
    }
    
    // Update Hidden2 to Output weights and biases
    for (var o = 0; o < this.outputSize; o++) {
        for (var h2 = 0; h2 < this.hidden2Size; h2++) {
            this.weightsH2O[o][h2] += learningRate * outputGradients[o] * this.lastHidden2[h2];
        }
        this.biasO[o] += learningRate * outputGradients[o];
    }
    
    // Calculate Hidden2 errors
    var hidden2Errors = [];
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        hidden2Errors[h2] = 0;
        for (var o = 0; o < this.outputSize; o++) {
            hidden2Errors[h2] += outputGradients[o] * this.weightsH2O[o][h2];
        }
    }
    
    // Calculate Hidden2 gradients
    var hidden2Gradients = [];
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        hidden2Gradients[h2] = hidden2Errors[h2] * (1 - this.lastHidden2[h2] * this.lastHidden2[h2]);
    }
    
    // Update Hidden1 to Hidden2 weights and biases
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        for (var h1 = 0; h1 < this.hidden1Size; h1++) {
            this.weightsH1H2[h2][h1] += learningRate * hidden2Gradients[h2] * this.lastHidden1[h1];
        }
        this.biasH2[h2] += learningRate * hidden2Gradients[h2];
    }
    
    // Calculate Hidden1 errors
    var hidden1Errors = [];
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        hidden1Errors[h1] = 0;
        for (var h2 = 0; h2 < this.hidden2Size; h2++) {
            hidden1Errors[h1] += hidden2Gradients[h2] * this.weightsH1H2[h2][h1];
        }
    }
    
    // Calculate Hidden1 gradients
    var hidden1Gradients = [];
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        hidden1Gradients[h1] = hidden1Errors[h1] * (1 - this.lastHidden1[h1] * this.lastHidden1[h1]);
    }
    
    // Update Input to Hidden1 weights and biases
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        for (var i = 0; i < this.inputSize; i++) {
            this.weightsIH1[h1][i] += learningRate * hidden1Gradients[h1] * this.lastInputs[i];
        }
        this.biasH1[h1] += learningRate * hidden1Gradients[h1];
    }
};

NeuralNetwork.prototype.policyGradientBackprop = function(reward, baseline, learningRate) {
    if (!this.lastInputs || !this.lastHidden1 || !this.lastHidden2 || !this.lastOutput) {
        return;
    }
    
    var advantage = reward - baseline;
    
    // Update Hidden2 to Output weights and biases
    for (var o = 0; o < this.outputSize; o++) {
        var outputGradient = advantage * (1 - this.lastOutput[o] * this.lastOutput[o]);
        
        for (var h2 = 0; h2 < this.hidden2Size; h2++) {
            this.weightsH2O[o][h2] += learningRate * outputGradient * this.lastHidden2[h2];
        }
        this.biasO[o] += learningRate * outputGradient;
    }
    
    // Update Hidden1 to Hidden2 weights and biases
    for (var h2 = 0; h2 < this.hidden2Size; h2++) {
        var hidden2Error = 0;
        for (var o = 0; o < this.outputSize; o++) {
            hidden2Error += advantage * (1 - this.lastOutput[o] * this.lastOutput[o]) * this.weightsH2O[o][h2];
        }
        
        var hidden2Gradient = hidden2Error * (1 - this.lastHidden2[h2] * this.lastHidden2[h2]);
        
        for (var h1 = 0; h1 < this.hidden1Size; h1++) {
            this.weightsH1H2[h2][h1] += learningRate * hidden2Gradient * this.lastHidden1[h1];
        }
        this.biasH2[h2] += learningRate * hidden2Gradient;
    }
    
    // Update Input to Hidden1 weights and biases
    for (var h1 = 0; h1 < this.hidden1Size; h1++) {
        var hidden1Error = 0;
        for (var h2 = 0; h2 < this.hidden2Size; h2++) {
            var hidden2Error = 0;
            for (var o = 0; o < this.outputSize; o++) {
                hidden2Error += advantage * (1 - this.lastOutput[o] * this.lastOutput[o]) * this.weightsH2O[o][h2];
            }
            hidden1Error += hidden2Error * (1 - this.lastHidden2[h2] * this.lastHidden2[h2]) * this.weightsH1H2[h2][h1];
        }
        
        var hidden1Gradient = hidden1Error * (1 - this.lastHidden1[h1] * this.lastHidden1[h1]);
        
        for (var i = 0; i < this.inputSize; i++) {
            this.weightsIH1[h1][i] += learningRate * hidden1Gradient * this.lastInputs[i];
        }
        this.biasH1[h1] += learningRate * hidden1Gradient;
    }
};

NeuralNetwork.prototype.loadParameters = function(params) {
    var loadResult = {
        success: false,
        message: "Using random initialization",
        fallbackReason: null
    };
    
    if (!params) {
        if (typeof window.NEURAL_PARAMS !== 'undefined') {
            params = window.NEURAL_PARAMS;
        } else {
            loadResult.fallbackReason = "No model parameters found";
            return loadResult;
        }
    }
    
    // Check if parameters exist and are compatible with current architecture
    var hasValidWeights = params.weightsIH1 && params.weightsH1H2 && params.weightsH2O;
    var hasValidBiases = params.biasH1 && params.biasH2 && params.biasO;
    
    if (!hasValidWeights || !hasValidBiases) {
        loadResult.fallbackReason = "Model parameters are null or missing";
        return loadResult;
    }
    
    // Check architecture compatibility
    var expectedInputSize = params.inputSize || this.inputSize;
    var expectedHidden1Size = params.hidden1Size || this.hidden1Size;
    var expectedHidden2Size = params.hidden2Size || this.hidden2Size;
    var expectedOutputSize = params.outputSize || this.outputSize;
    
    // Validate weight matrix dimensions
    var inputWeightsValid = params.weightsIH1.length === expectedHidden1Size && 
                           params.weightsIH1[0] && params.weightsIH1[0].length === expectedInputSize;
    var hidden1WeightsValid = params.weightsH1H2.length === expectedHidden2Size && 
                             params.weightsH1H2[0] && params.weightsH1H2[0].length === expectedHidden1Size;
    var outputWeightsValid = params.weightsH2O.length === expectedOutputSize && 
                            params.weightsH2O[0] && params.weightsH2O[0].length === expectedHidden2Size;
    
    // Validate bias vector dimensions
    var hidden1BiasValid = params.biasH1.length === expectedHidden1Size;
    var hidden2BiasValid = params.biasH2.length === expectedHidden2Size;
    var outputBiasValid = params.biasO.length === expectedOutputSize;
    
    if (!inputWeightsValid) {
        loadResult.fallbackReason = `Input weights incompatible: expected ${expectedHidden1Size}×${expectedInputSize}, got ${params.weightsIH1.length}×${params.weightsIH1[0] ? params.weightsIH1[0].length : 'undefined'}`;
        return loadResult;
    }
    
    if (!hidden1WeightsValid) {
        loadResult.fallbackReason = `Hidden1 weights incompatible: expected ${expectedHidden2Size}×${expectedHidden1Size}, got ${params.weightsH1H2.length}×${params.weightsH1H2[0] ? params.weightsH1H2[0].length : 'undefined'}`;
        return loadResult;
    }
    
    if (!outputWeightsValid) {
        loadResult.fallbackReason = `Output weights incompatible: expected ${expectedOutputSize}×${expectedHidden2Size}, got ${params.weightsH2O.length}×${params.weightsH2O[0] ? params.weightsH2O[0].length : 'undefined'}`;
        return loadResult;
    }
    
    if (!hidden1BiasValid || !hidden2BiasValid || !outputBiasValid) {
        loadResult.fallbackReason = `Bias dimensions incompatible: expected [${expectedHidden1Size}, ${expectedHidden2Size}, ${expectedOutputSize}], got [${params.biasH1.length}, ${params.biasH2.length}, ${params.biasO.length}]`;
        return loadResult;
    }
    
    // Architecture compatibility check
    if (expectedInputSize !== this.inputSize || expectedHidden1Size !== this.hidden1Size || 
        expectedHidden2Size !== this.hidden2Size || expectedOutputSize !== this.outputSize) {
        loadResult.fallbackReason = `Architecture mismatch: model is ${expectedInputSize}→${expectedHidden1Size}→${expectedHidden2Size}→${expectedOutputSize}, current is ${this.inputSize}→${this.hidden1Size}→${this.hidden2Size}→${this.outputSize}`;
        return loadResult;
    }
    
    // If we get here, parameters are valid - load them
    try {
        this.inputSize = expectedInputSize;
        this.hidden1Size = expectedHidden1Size;
        this.hidden2Size = expectedHidden2Size;
        this.outputSize = expectedOutputSize;
        
        this.weightsIH1 = params.weightsIH1.map(function(row) { return row.slice(); });
        this.weightsH1H2 = params.weightsH1H2.map(function(row) { return row.slice(); });
        this.weightsH2O = params.weightsH2O.map(function(row) { return row.slice(); });
        this.biasH1 = params.biasH1.slice();
        this.biasH2 = params.biasH2.slice();
        this.biasO = params.biasO.slice();
        
        loadResult.success = true;
        loadResult.message = "Successfully loaded pre-trained model";
        loadResult.fallbackReason = null;
        
    } catch (error) {
        loadResult.fallbackReason = "Error loading parameters: " + error.message;
    }
    
    return loadResult;
};

NeuralNetwork.prototype.reset = function() {
    this.initializeWeights();
    this.lastInputs = null;
    this.lastHidden1 = null;
    this.lastHidden2 = null;
    this.lastOutput = null;
}; 