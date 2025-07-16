/**
 * Action Processor - Universal policy output interface
 * 
 * This is a data conversion layer that converts policy outputs to game actions.
 * Works with any policy type (neural networks, rule-based, human, etc.).
 * This MUST match exactly with the Python implementation.
 */
function ActionProcessor() {
    // Scale neural outputs to match actual predator force limits
    this.forceScale = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE;
}

/**
 * Convert policy outputs to game actions
 * 
 * This universal interface works with any policy type:
 * - Neural networks: normalized outputs [-1, 1]
 * - Rule-based policies: computed steering forces
 * - Human policies: input-based actions
 * - Hybrid policies: combined outputs
 * 
 * @param {Array} policyOutputs - Policy outputs [x, y] in [-1, 1] range
 * @returns {Array} Action forces [forceX, forceY] in game units
 */
ActionProcessor.prototype.processAction = function(policyOutputs) {
    return [
        policyOutputs[0] * this.forceScale,
        policyOutputs[1] * this.forceScale
    ];
}; 