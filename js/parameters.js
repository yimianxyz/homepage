/**
 * Neural Network Parameters
 * 
 * Pure data file containing trained weights and biases for the neural predator.
 * 
 * Network Architecture: 22 inputs → 12 hidden → 2 outputs
 * - Inputs: positions & velocities of 5 nearest boids + predator velocity (screen-size normalized)
 * - Hidden: 12 neurons with tanh activation
 * - Outputs: steering force (x, y) with separate X/Y screen-size scaling for consistent relative effects
 * - Uses toroidal distance calculation for wraparound-aware hunting
 * - Simplified mechanics with no cooldowns and multi-catch capability for easier neural network learning
 * - Designed for device-independent training with consistent input/output normalization and screen-scaled speed
 */

window.NEURAL_PARAMS = {
    // Network architecture
    inputSize: 22,
    hiddenSize: 12,
    outputSize: 2,
    
    // Input to hidden layer weights (12x22 matrix)
    weightsIH: [
        // Hidden neuron 1
        [0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.15, -0.05, 0.1, -0.1, 0.05, 0.1, -0.1, 0.1, 0.05, -0.1, 0.1, -0.1],
        // Hidden neuron 2
        [-0.1, 0.2, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.05, 0.15, -0.1, 0.1, -0.1, 0.05, 0.1, -0.1, 0.1, 0.05, -0.1, 0.1],
        // Hidden neuron 3
        [0.15, -0.05, 0.1, -0.1, 0.1, 0.1, -0.2, 0.05, 0.2, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.2, -0.1],
        // Hidden neuron 4
        [-0.1, 0.1, -0.15, 0.2, 0.05, -0.1, 0.1, -0.2, 0.1, 0.1, -0.05, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.05, 0.1],
        // Hidden neuron 5
        [0.2, -0.1, 0.05, 0.1, -0.2, 0.1, -0.1, 0.15, -0.1, 0.05, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
        // Hidden neuron 6
        [-0.1, 0.15, -0.2, 0.05, 0.1, -0.1, 0.1, -0.05, 0.1, -0.2, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
        // Hidden neuron 7
        [0.1, -0.1, 0.1, -0.15, 0.1, 0.2, -0.1, 0.1, -0.05, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2],
        // Hidden neuron 8
        [-0.05, 0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.1, 0.15, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1],
        // Hidden neuron 9
        [0.1, -0.2, 0.1, 0.05, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.15, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1],
        // Hidden neuron 10
        [-0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.2, -0.05, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.15, 0.1, 0.1, -0.1],
        // Hidden neuron 11
        [0.1, -0.1, 0.15, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, 0.05, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1],
        // Hidden neuron 12
        [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, -0.05, 0.1, 0.1, -0.1, 0.1, 0.1, -0.15, 0.1]
    ],
    
    // Hidden to output layer weights (2x12 matrix)
    weightsHO: [
        // X steering force output
        [0.3, -0.2, 0.4, -0.1, 0.3, -0.2, 0.2, 0.1, -0.3, 0.2, 0.4, -0.1],
        // Y steering force output
        [-0.1, 0.4, -0.2, 0.3, -0.1, 0.3, -0.2, 0.4, 0.1, -0.2, 0.3, 0.2]
    ],
    
    // Hidden layer biases (12 values)
    biasH: [0.05, -0.1, 0.1, -0.05, 0.1, 0.2, -0.1, 0.05, 0.1, -0.1, 0.05, 0.1],
    
    // Output layer biases (2 values)
    biasO: [0.000, 0.000],
    
    // Normalization constants
    maxDistance: 5000,  // Covers all modern screens including 4K and ultrawide displays
    maxVelocity: 6,
    
    // Version info for parameter management
    version: "2.9.0",
    trained: false,
    description: "Complete screen-adaptive normalization for both positions and velocities - consistent spatial understanding across all screen sizes"
};