/**
 * Transformer Policy Wrapper - Universal wrapper for transformer-based policies
 * 
 * This wrapper provides a universal policy interface that can work with any trained
 * transformer model from a model.js file. It handles loading different model parameters
 * and provides consistent inference interface for the simulation framework.
 * 
 * Interface:
 * - Input: structured_inputs (same format as universal policy input)
 * - Output: normalized policy outputs [x, y] in [-1, 1] range
 * - Compatible with ActionProcessor for scaling to game forces
 * 
 * Usage:
 * - Can load from global TRANSFORMER_PARAMS (from model.js)
 * - Can load from custom parameter objects
 * - Can work with different trained models without code changes
 */

/**
 * Transformer Policy Wrapper - works with any trained transformer model
 * @param {Object} modelParams - Model parameters object
 * @constructor
 */
function TransformerPolicyWrapper(modelParams) {
    if (!modelParams) {
        throw new Error("TransformerPolicyWrapper requires model parameters");
    }
    
    // Initialize transformer encoder with model parameters
    this.transformer = new TransformerEncoder(modelParams);
    
    // Policy metadata
    this.policyType = 'transformer';
    this.architecture = {
        d_model: this.transformer.d_model,
        n_heads: this.transformer.n_heads,
        n_layers: this.transformer.n_layers,
        ffn_hidden: this.transformer.ffn_hidden
    };
    
    console.log("Created TransformerPolicyWrapper:");
    console.log("  Architecture:", this.architecture);
    console.log("  Model load result:", this.transformer.modelLoadResult);
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
TransformerPolicyWrapper.prototype.getAction = function(structured_inputs) {
    try {
        // Validate input structure
        if (!this.validateInputs(structured_inputs)) {
            console.warn("TransformerPolicyWrapper: Invalid input structure, returning zero action");
            return [0.0, 0.0];
        }
        
        // Forward pass through transformer
        var outputs = this.transformer.forward(structured_inputs);
        
        // Ensure outputs are in [-1, 1] range (transformer already applies tanh)
        var clampedOutputs = [
            Math.max(-1, Math.min(1, outputs[0])),
            Math.max(-1, Math.min(1, outputs[1]))
        ];
        
        return clampedOutputs;
        
    } catch (error) {
        console.error("TransformerPolicyWrapper: Error during forward pass:", error);
        return [0.0, 0.0];
    }
};



/**
 * Validate structured inputs
 * 
 * @param {Object} structured_inputs - Inputs to validate
 * @returns {boolean} True if valid
 */
TransformerPolicyWrapper.prototype.validateInputs = function(structured_inputs) {
    // Check required top-level structure
    if (!structured_inputs || typeof structured_inputs !== 'object') {
        return false;
    }
    
    // Check context
    if (!structured_inputs.context || 
        typeof structured_inputs.context.canvasWidth !== 'number' ||
        typeof structured_inputs.context.canvasHeight !== 'number') {
        return false;
    }
    
    // Check predator
    if (!structured_inputs.predator ||
        typeof structured_inputs.predator.velX !== 'number' ||
        typeof structured_inputs.predator.velY !== 'number') {
        return false;
    }
    
    // Check boids (can be empty array)
    if (!Array.isArray(structured_inputs.boids)) {
        return false;
    }
    
    // Validate each boid
    for (var i = 0; i < structured_inputs.boids.length; i++) {
        var boid = structured_inputs.boids[i];
        if (!boid || 
            typeof boid.relX !== 'number' ||
            typeof boid.relY !== 'number' ||
            typeof boid.velX !== 'number' ||
            typeof boid.velY !== 'number') {
            return false;
        }
    }
    
    return true;
};

/**
 * Get model information
 * 
 * @returns {Object} Model information
 */
TransformerPolicyWrapper.prototype.getModelInfo = function() {
    return {
        policyType: this.policyType,
        architecture: this.architecture,
        modelLoaded: true,
        loadResult: this.transformer.modelLoadResult,
        parameterCount: this.getParameterCount()
    };
};

/**
 * Get approximate parameter count
 * 
 * @returns {number} Approximate number of parameters
 */
TransformerPolicyWrapper.prototype.getParameterCount = function() {
    var d_model = this.architecture.d_model;
    var n_heads = this.architecture.n_heads;
    var n_layers = this.architecture.n_layers;
    var ffn_hidden = this.architecture.ffn_hidden;
    
    // Approximate calculation
    var embedding_params = d_model * 4; // 4 type embeddings
    var projection_params = (2 + 4 + 4) * d_model; // ctx, predator, boid projections
    var layer_params = n_layers * (
        d_model * 3 * d_model + // QKV projection
        d_model * d_model +     // attention output
        d_model * ffn_hidden * 3 + // GEGLU (gate, up, down)
        d_model * 4             // layer norm parameters
    );
    var output_params = d_model * 2;
    
    return embedding_params + projection_params + layer_params + output_params;
};



/**
 * Get normalized action (deprecated - use getAction instead)
 * 
 * @param {Object} structured_inputs - Same format as universal policy input
 * @returns {Array} Normalized policy outputs [x, y] in [-1, 1] range
 */
TransformerPolicyWrapper.prototype.getNormalizedAction = function(structured_inputs) {
    return this.getAction(structured_inputs);
};

/**
 * Create transformer policy wrapper instance
 * 
 * @param {Object} modelParams - Model parameters from model.js
 * @returns {TransformerPolicyWrapper} Policy wrapper instance
 */
function createTransformerPolicyWrapper(modelParams) {
    var policy = new TransformerPolicyWrapper(modelParams);
    
    console.log("TransformerPolicyWrapper created:");
    console.log("  Policy type:", policy.policyType);
    console.log("  Parameters:", policy.getParameterCount().toLocaleString());
    console.log("  Architecture:", policy.architecture.d_model + "×" + 
               policy.architecture.n_heads + "×" + 
               policy.architecture.n_layers + "×" + 
               policy.architecture.ffn_hidden);
    
    return policy;
}

/**
 * Create transformer policy wrapper from global TRANSFORMER_PARAMS
 * 
 * @returns {TransformerPolicyWrapper} Policy wrapper instance
 */
function createTransformerPolicyFromGlobal() {
    if (!window.TRANSFORMER_PARAMS) {
        throw new Error("No global TRANSFORMER_PARAMS found");
    }
    return createTransformerPolicyWrapper(window.TRANSFORMER_PARAMS);
}

/**
 * Create transformer policy wrapper with custom model parameters
 * 
 * @param {Object} customParams - Custom model parameters object
 * @returns {TransformerPolicyWrapper} Policy wrapper instance
 */
function createTransformerPolicyWithModel(customParams) {
    return createTransformerPolicyWrapper(customParams);
}

// Export to global scope for browser use
if (typeof window !== 'undefined') {
    window.TransformerPolicyWrapper = TransformerPolicyWrapper;
    window.createTransformerPolicyWrapper = createTransformerPolicyWrapper;
    window.createTransformerPolicyFromGlobal = createTransformerPolicyFromGlobal;
    window.createTransformerPolicyWithModel = createTransformerPolicyWithModel;
} else if (typeof global !== 'undefined') {
    // For Node.js environment
    global.window = global.window || {};
    global.window.TransformerPolicyWrapper = TransformerPolicyWrapper;
    global.window.createTransformerPolicyWrapper = createTransformerPolicyWrapper;
    global.window.createTransformerPolicyFromGlobal = createTransformerPolicyFromGlobal;
    global.window.createTransformerPolicyWithModel = createTransformerPolicyWithModel;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TransformerPolicyWrapper: TransformerPolicyWrapper,
        createTransformerPolicyWrapper: createTransformerPolicyWrapper,
        createTransformerPolicyFromGlobal: createTransformerPolicyFromGlobal,
        createTransformerPolicyWithModel: createTransformerPolicyWithModel
    };
} 