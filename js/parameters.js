/**
 * Neural Network Parameters
 * 
 * Pure data file containing trained weights and biases for the neural predator.
 * 
 * Network Architecture: 12 inputs → 8 hidden → 2 outputs
 * - Inputs: positions & velocities of 5 nearest boids + predator state
 * - Hidden: 8 neurons with tanh activation
 * - Outputs: steering force (x, y)
 */

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