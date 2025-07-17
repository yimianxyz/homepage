/**
 * Universal Transformer Policy
 * 
 * This provides a universal transformer policy that can work with any trained model
 * by accepting transformer parameters during initialization. It provides the same
 * interface as other policies like closest_pursuit_policy.
 * 
 * Dependencies (must be loaded in order):
 * - src/config/constants.js
 * - policy/transformer/utils/transformer_encoder.js
 * - policy/transformer/utils/transformer_policy_wrapper.js
 * - policy/transformer/transformer_policy.js (this file)
 * 
 * Usage:
 * - Load model parameters (e.g., from model.js)
 * - Create policy: var policy = createTransformerPolicy(transformerParams);
 * - Use like any other policy: policy.getAction(structured_inputs);
 */

/**
 * Transformer Policy Constructor
 * 
 * @param {Object} transformerParams - The transformer model parameters
 * @constructor
 */
function TransformerPolicy(transformerParams) {
    // Validate parameters
    if (!transformerParams) {
        throw new Error("TransformerPolicy requires transformer parameters. Please provide trained model parameters.");
    }
    
    // Validate dependencies
    if (typeof window === 'undefined') {
        throw new Error("TransformerPolicy requires browser environment (window object).");
    }
    
    if (!window.SIMULATION_CONSTANTS) {
        throw new Error("TransformerPolicy requires window.SIMULATION_CONSTANTS. Please load src/config/constants.js first.");
    }
    
    if (!window.createTransformerPolicyWrapper) {
        throw new Error("TransformerPolicy requires TransformerPolicyWrapper. Please load policy/transformer/utils/transformer_policy_wrapper.js first.");
    }
    
    // Create the policy wrapper with provided parameters
    this.policyWrapper = createTransformerPolicyWithModel(transformerParams);
    
    // Policy metadata
    this.policyType = 'transformer';
    this.modelInfo = this.policyWrapper.getModelInfo();
    this.transformerParams = transformerParams;
    
    console.log("Created TransformerPolicy:");
    console.log("  Type:", this.policyType);
    console.log("  Model loaded:", this.modelInfo.modelLoaded);
    console.log("  Parameters:", this.modelInfo.parameterCount.toLocaleString());
    console.log("  Architecture:", this.modelInfo.architecture);
}

/**
 * Get normalized policy outputs (compatible with ActionProcessor)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input
 * - context: {canvasWidth: number, canvasHeight: number}
 * - predator: {velX: number, velY: number}
 * - boids: [{relX: number, relY: number, velX: number, velY: number}, ...]
 * 
 * @returns {Array} Normalized policy outputs [x, y] in [-1, 1] range
 */
TransformerPolicy.prototype.getAction = function(structured_inputs) {
    return this.policyWrapper.getAction(structured_inputs);
};

/**
 * Get normalized action (deprecated - use getAction instead)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input
 * @returns {Array} Normalized policy outputs [x, y] in [-1, 1] range
 */
TransformerPolicy.prototype.getNormalizedAction = function(structured_inputs) {
    return this.getAction(structured_inputs);
};

/**
 * Get model information
 * 
 * @returns {Object} Model information
 */
TransformerPolicy.prototype.getModelInfo = function() {
    return this.policyWrapper.getModelInfo();
};

/**
 * Get the transformer parameters used by this policy
 * 
 * @returns {Object} Transformer parameters
 */
TransformerPolicy.prototype.getTransformerParams = function() {
    return this.transformerParams;
};

/**
 * Create transformer policy with parameters
 * 
 * @param {Object} transformerParams - The transformer model parameters
 * @returns {TransformerPolicy} Policy instance
 */
function createTransformerPolicy(transformerParams) {
    var policy = new TransformerPolicy(transformerParams);
    
    console.log("TransformerPolicy created successfully:");
    console.log("  Ready for use with StateManager");
    console.log("  Compatible with ActionProcessor");
    console.log("  Model loaded:", policy.modelInfo.modelLoaded);
    
    return policy;
}

/**
 * Create transformer policy with comprehensive validation
 * 
 * @param {Object} transformerParams - The transformer model parameters
 * @returns {TransformerPolicy} Policy instance
 */
function createTransformerPolicyWithValidation(transformerParams) {
    // Check dependencies
    var missingDeps = [];
    
    if (typeof window === 'undefined') {
        missingDeps.push("Browser environment (window object)");
    }
    
    if (!window.SIMULATION_CONSTANTS) {
        missingDeps.push("window.SIMULATION_CONSTANTS (load src/config/constants.js)");
    }
    
    if (!window.TransformerEncoder) {
        missingDeps.push("TransformerEncoder (load policy/transformer/utils/transformer_encoder.js)");
    }
    
    if (!window.createTransformerPolicyWrapper) {
        missingDeps.push("TransformerPolicyWrapper (load policy/transformer/utils/transformer_policy_wrapper.js)");
    }
    
    if (!transformerParams) {
        missingDeps.push("Transformer parameters (provide trained model parameters)");
    }
    
    if (missingDeps.length > 0) {
        throw new Error("Missing dependencies for TransformerPolicy:\n- " + missingDeps.join("\n- "));
    }
    
    return createTransformerPolicy(transformerParams);
}

/**
 * Create transformer policy from global window.TRANSFORMER_PARAMS
 * (for backward compatibility with existing code)
 * 
 * @returns {TransformerPolicy} Policy instance
 */
function createTransformerPolicyFromGlobal() {
    if (!window.TRANSFORMER_PARAMS) {
        throw new Error("createTransformerPolicyFromGlobal requires window.TRANSFORMER_PARAMS to be loaded. Please load a model.js file first.");
    }
    
    return createTransformerPolicy(window.TRANSFORMER_PARAMS);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TransformerPolicy: TransformerPolicy,
        createTransformerPolicy: createTransformerPolicy,
        createTransformerPolicyWithValidation: createTransformerPolicyWithValidation,
        createTransformerPolicyFromGlobal: createTransformerPolicyFromGlobal
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.TransformerPolicy = TransformerPolicy;
    window.createTransformerPolicy = createTransformerPolicy;
    window.createTransformerPolicyWithValidation = createTransformerPolicyWithValidation;
    window.createTransformerPolicyFromGlobal = createTransformerPolicyFromGlobal;
} 