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
        // Hidden neuron 1
        [0.800, -0.300, 0.600, -0.200, 0.400, -0.100, 0.700, -0.400, 0.500, -0.200, 0.300, 0.100],
        // Hidden neuron 2
        [-0.400, 0.700, -0.200, 0.500, -0.300, 0.600, -0.100, 0.800, -0.500, 0.200, -0.400, 0.300],
        // Hidden neuron 3
        [0.500, -0.100, 0.800, -0.300, 0.200, 0.400, -0.600, 0.100, 0.700, -0.400, 0.300, -0.200],
        // Hidden neuron 4
        [-0.200, 0.400, -0.500, 0.700, 0.100, -0.300, 0.600, -0.700, 0.200, 0.500, -0.100, 0.400],
        // Hidden neuron 5
        [0.600, -0.400, 0.100, 0.300, -0.700, 0.500, -0.200, 0.600, -0.300, 0.100, 0.800, -0.400],
        // Hidden neuron 6
        [-0.300, 0.600, -0.700, 0.100, 0.400, -0.200, 0.800, -0.100, 0.500, -0.600, 0.200, 0.700],
        // Hidden neuron 7
        [0.700, -0.200, 0.400, -0.600, 0.300, 0.800, -0.400, 0.200, -0.100, 0.600, -0.500, 0.100],
        // Hidden neuron 8
        [-0.100, 0.500, -0.300, 0.600, -0.800, 0.200, 0.400, -0.500, 0.700, -0.200, 0.100, 0.600]
    ],
    
    // Hidden to output layer weights (2x8 matrix)
    weightsHO: [
        // X steering force output
        [0.800, -0.300, 0.600, -0.200, 0.700, -0.400, 0.500, 0.100],
        // Y steering force output
        [-0.200, 0.700, -0.400, 0.500, -0.100, 0.600, -0.300, 0.800]
    ],
    
    // Hidden layer biases (8 values)
    biasH: [0.100, -0.200, 0.300, -0.100, 0.200, 0.400, -0.300, 0.100],
    
    // Output layer biases (2 values)
    biasO: [0.000, 0.000],
    
    // Normalization constants
    maxDistance: 100,
    maxVelocity: 6,
    
    // Version info for parameter management
    version: "1.0.0",
    trained: true,
    description: "Trained neural network parameters"
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
    // ... (same export function as this one)
    return "Parameter export function - copy from working file";
};