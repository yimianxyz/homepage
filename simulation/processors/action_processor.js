/**
 * Action Processor - Neural output to game actions
 * 
 * This is a data conversion layer that converts neural network outputs to game forces.
 * This MUST match exactly with the Python implementation.
 */
function ActionProcessor() {
    // Scale neural outputs to match actual predator force limits
    this.forceScale = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE;
}

/**
 * Convert neural network outputs to game actions
 * @param {Array} neuralOutputs - Neural network outputs [x, y] in [-1, 1] range
 * @returns {Array} Action forces [forceX, forceY] in game units
 */
ActionProcessor.prototype.processAction = function(neuralOutputs) {
    return [
        neuralOutputs[0] * this.forceScale,
        neuralOutputs[1] * this.forceScale
    ];
}; 