/**
 * Action Processor - Neural output to game actions
 * 
 * Converts neural network outputs to game forces.
 * Simple but maintains architectural consistency with InputProcessor.
 */
function ActionProcessor() {
    // Scale neural outputs to match actual predator force limits
    this.forceScale = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * window.SIMULATION_CONSTANTS.PREDATOR_FORCE_SCALE;
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