/**
 * Neural Network Parameters
 * 
 * This file contains the trained weights and biases for the neural predator.
 * These parameters are loaded by both the prediction system (index.html) and
 * the training system (training.html).
 * 
 * Network Architecture: 12 inputs → 8 hidden → 2 outputs
 * - Inputs: positions & velocities of 5 nearest boids + predator state
 * - Hidden: 8 neurons with tanh activation
 * - Outputs: steering force (x, y)
 */

// Global neural network parameters
window.NEURAL_PARAMS = {
    // Network architecture
    inputSize: 12,
    hiddenSize: 8,
    outputSize: 2,
    
    // Input to hidden layer weights (8x12 matrix)
    weightsIH: [
        // Hidden neuron 1 - responds to boid positions and movements
        [ 0.8, -0.3,  0.6, -0.2,  0.4, -0.1,  0.7, -0.4,  0.5, -0.2,  0.3,  0.1],
        // Hidden neuron 2
        [-0.4,  0.7, -0.2,  0.5, -0.3,  0.6, -0.1,  0.8, -0.5,  0.2, -0.4,  0.3],
        // Hidden neuron 3
        [ 0.5, -0.1,  0.8, -0.3,  0.2,  0.4, -0.6,  0.1,  0.7, -0.4,  0.3, -0.2],
        // Hidden neuron 4
        [-0.2,  0.4, -0.5,  0.7,  0.1, -0.3,  0.6, -0.7,  0.2,  0.5, -0.1,  0.4],
        // Hidden neuron 5
        [ 0.6, -0.4,  0.1,  0.3, -0.7,  0.5, -0.2,  0.6, -0.3,  0.1,  0.8, -0.4],
        // Hidden neuron 6
        [-0.3,  0.6, -0.7,  0.1,  0.4, -0.2,  0.8, -0.1,  0.5, -0.6,  0.2,  0.7],
        // Hidden neuron 7
        [ 0.7, -0.2,  0.4, -0.6,  0.3,  0.8, -0.4,  0.2, -0.1,  0.6, -0.5,  0.1],
        // Hidden neuron 8
        [-0.1,  0.5, -0.3,  0.6, -0.8,  0.2,  0.4, -0.5,  0.7, -0.2,  0.1,  0.6]
    ],
    
    // Hidden to output layer weights (2x8 matrix)
    weightsHO: [
        // X steering force output
        [ 0.8, -0.3,  0.6, -0.2,  0.7, -0.4,  0.5,  0.1],
        // Y steering force output
        [-0.2,  0.7, -0.4,  0.5, -0.1,  0.6, -0.3,  0.8]
    ],
    
    // Hidden layer biases (8 values)
    biasH: [0.1, -0.2, 0.3, -0.1, 0.2, 0.4, -0.3, 0.1],
    
    // Output layer biases (2 values)
    biasO: [0.0, 0.0],
    
    // Normalization constants
    maxDistance: 100,
    maxVelocity: 6,
    
    // Version info for parameter management
    version: "1.0.0",
    trained: false,
    description: "Initial hand-tuned weights for hunting behavior"
};

// Helper function to deep clone parameters (for training)
window.cloneNeuralParams = function() {
    return {
        inputSize: window.NEURAL_PARAMS.inputSize,
        hiddenSize: window.NEURAL_PARAMS.hiddenSize,
        outputSize: window.NEURAL_PARAMS.outputSize,
        weightsIH: window.NEURAL_PARAMS.weightsIH.map(row => row.slice()),
        weightsHO: window.NEURAL_PARAMS.weightsHO.map(row => row.slice()),
        biasH: window.NEURAL_PARAMS.biasH.slice(),
        biasO: window.NEURAL_PARAMS.biasO.slice(),
        maxDistance: window.NEURAL_PARAMS.maxDistance,
        maxVelocity: window.NEURAL_PARAMS.maxVelocity,
        version: window.NEURAL_PARAMS.version,
        trained: window.NEURAL_PARAMS.trained,
        description: window.NEURAL_PARAMS.description
    };
};

// Helper function to export parameters as JS code
window.exportNeuralParams = function(params) {
    var code = '/**\n * Neural Network Parameters\n * \n * This file contains the trained weights and biases for the neural predator.\n * These parameters are loaded by both the prediction system (index.html) and\n * the training system (training.html).\n * \n * Network Architecture: 12 inputs → 8 hidden → 2 outputs\n * - Inputs: positions & velocities of 5 nearest boids + predator state\n * - Hidden: 8 neurons with tanh activation\n * - Outputs: steering force (x, y)\n */\n\n';
    
    code += '// Global neural network parameters\n';
    code += 'window.NEURAL_PARAMS = {\n';
    code += '    // Network architecture\n';
    code += '    inputSize: ' + params.inputSize + ',\n';
    code += '    hiddenSize: ' + params.hiddenSize + ',\n';
    code += '    outputSize: ' + params.outputSize + ',\n';
    code += '    \n';
    
    code += '    // Input to hidden layer weights (' + params.hiddenSize + 'x' + params.inputSize + ' matrix)\n';
    code += '    weightsIH: [\n';
    for (var i = 0; i < params.weightsIH.length; i++) {
        code += '        // Hidden neuron ' + (i + 1) + '\n';
        code += '        [' + params.weightsIH[i].map(w => w.toFixed(3)).join(', ') + ']';
        if (i < params.weightsIH.length - 1) code += ',';
        code += '\n';
    }
    code += '    ],\n';
    code += '    \n';
    
    code += '    // Hidden to output layer weights (' + params.outputSize + 'x' + params.hiddenSize + ' matrix)\n';
    code += '    weightsHO: [\n';
    for (var i = 0; i < params.weightsHO.length; i++) {
        code += '        // ' + (i === 0 ? 'X' : 'Y') + ' steering force output\n';
        code += '        [' + params.weightsHO[i].map(w => w.toFixed(3)).join(', ') + ']';
        if (i < params.weightsHO.length - 1) code += ',';
        code += '\n';
    }
    code += '    ],\n';
    code += '    \n';
    
    code += '    // Hidden layer biases (' + params.hiddenSize + ' values)\n';
    code += '    biasH: [' + params.biasH.map(b => b.toFixed(3)).join(', ') + '],\n';
    code += '    \n';
    
    code += '    // Output layer biases (' + params.outputSize + ' values)\n';
    code += '    biasO: [' + params.biasO.map(b => b.toFixed(3)).join(', ') + '],\n';
    code += '    \n';
    
    code += '    // Normalization constants\n';
    code += '    maxDistance: ' + params.maxDistance + ',\n';
    code += '    maxVelocity: ' + params.maxVelocity + ',\n';
    code += '    \n';
    
    code += '    // Version info for parameter management\n';
    code += '    version: "' + params.version + '",\n';
    code += '    trained: ' + params.trained + ',\n';
    code += '    description: "' + params.description + '"\n';
    code += '};\n\n';
    
    code += '// Helper function to deep clone parameters (for training)\n';
    code += 'window.cloneNeuralParams = function() {\n';
    code += '    return {\n';
    code += '        inputSize: window.NEURAL_PARAMS.inputSize,\n';
    code += '        hiddenSize: window.NEURAL_PARAMS.hiddenSize,\n';
    code += '        outputSize: window.NEURAL_PARAMS.outputSize,\n';
    code += '        weightsIH: window.NEURAL_PARAMS.weightsIH.map(row => row.slice()),\n';
    code += '        weightsHO: window.NEURAL_PARAMS.weightsHO.map(row => row.slice()),\n';
    code += '        biasH: window.NEURAL_PARAMS.biasH.slice(),\n';
    code += '        biasO: window.NEURAL_PARAMS.biasO.slice(),\n';
    code += '        maxDistance: window.NEURAL_PARAMS.maxDistance,\n';
    code += '        maxVelocity: window.NEURAL_PARAMS.maxVelocity,\n';
    code += '        version: window.NEURAL_PARAMS.version,\n';
    code += '        trained: window.NEURAL_PARAMS.trained,\n';
    code += '        description: window.NEURAL_PARAMS.description\n';
    code += '    };\n';
    code += '};\n\n';
    
    code += '// Helper function to export parameters as JS code\n';
    code += 'window.exportNeuralParams = function(params) {\n';
    code += '    // ... (same export function as this one)\n';
    code += '    return "Parameter export function - copy from working file";\n';
    code += '};';
    
    return code;
}; 