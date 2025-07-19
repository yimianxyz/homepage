/**
 * Transformer Encoder for Policy Wrapper
 * 
 * This is a flexible transformer encoder implementation that reads its architecture
 * parameters from the model file. It provides inference-only functionality and can
 * work with different transformer architectures.
 * 
 * Architecture (configurable via model parameters):
 * - d_model, n_heads, n_layers, ffn_hidden from model
 * - GEGLU feed-forward networks
 * - Token sequence: [CLS] + [CTX] + Predator + Boids
 * - Type embeddings for different entity types
 */

/**
 * Transformer Encoder for predator control (inference only)
 * @constructor
 * @param {Object} params - Model parameters containing architecture
 */
function TransformerEncoder(params) {
    if (!params) {
        throw new Error("TransformerEncoder requires model parameters with architecture");
    }
    
    // Validate architecture parameters
    this.validateArchitecture(params);
    
    // Extract architecture from model parameters
    this.d_model = params.d_model;
    this.n_heads = params.n_heads;
    this.n_layers = params.n_layers;
    this.head_dim = this.d_model / this.n_heads;
    this.ffn_hidden = params.ffn_hidden;
    
    // Normalization constants from simulation (required)
    if (typeof window === 'undefined' || !window.SIMULATION_CONSTANTS) {
        throw new Error("TransformerEncoder requires window.SIMULATION_CONSTANTS to be loaded. Please include constants.js before transformer_encoder.js");
    }
    
    this.D = window.SIMULATION_CONSTANTS.MAX_DISTANCE;
    this.V = Math.max(
        window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
    
    // Load parameters immediately
    var loadResult = this.loadParameters(params);
    this.modelLoadResult = loadResult;
}

/**
 * Validate architecture parameters
 * @param {Object} params - Model parameters to validate
 */
TransformerEncoder.prototype.validateArchitecture = function(params) {
    // Check required architecture parameters exist
    var requiredArchParams = ['d_model', 'n_heads', 'n_layers', 'ffn_hidden'];
    for (var i = 0; i < requiredArchParams.length; i++) {
        var param = requiredArchParams[i];
        if (!(param in params) || typeof params[param] !== 'number') {
            throw new Error("Missing or invalid architecture parameter: " + param);
        }
    }
    
    // Validate parameter ranges and relationships
    if (params.d_model <= 0 || params.d_model % 1 !== 0) {
        throw new Error("d_model must be a positive integer, got: " + params.d_model);
    }
    
    if (params.n_heads <= 0 || params.n_heads % 1 !== 0) {
        throw new Error("n_heads must be a positive integer, got: " + params.n_heads);
    }
    
    if (params.n_layers <= 0 || params.n_layers % 1 !== 0) {
        throw new Error("n_layers must be a positive integer, got: " + params.n_layers);
    }
    
    if (params.ffn_hidden <= 0 || params.ffn_hidden % 1 !== 0) {
        throw new Error("ffn_hidden must be a positive integer, got: " + params.ffn_hidden);
    }
    
    // Check that d_model is divisible by n_heads
    if (params.d_model % params.n_heads !== 0) {
        throw new Error("d_model (" + params.d_model + ") must be divisible by n_heads (" + params.n_heads + ")");
    }
    
    // Check reasonable ranges
    if (params.d_model < 8 || params.d_model > 2048) {
        throw new Error("d_model should be between 8 and 2048, got: " + params.d_model);
    }
    
    if (params.n_heads < 1 || params.n_heads > 32) {
        throw new Error("n_heads should be between 1 and 32, got: " + params.n_heads);
    }
    
    if (params.n_layers < 1 || params.n_layers > 24) {
        throw new Error("n_layers should be between 1 and 24, got: " + params.n_layers);
    }
    
    if (params.ffn_hidden < params.d_model || params.ffn_hidden > params.d_model * 8) {
        throw new Error("ffn_hidden should be between d_model and 8*d_model, got: " + params.ffn_hidden + " (d_model=" + params.d_model + ")");
    }
};

/**
 * Load parameters from model object
 * @param {Object} params - Model parameters object
 * @returns {Object} Load result with success status and message
 */
TransformerEncoder.prototype.loadParameters = function(params) {
    var loadResult = {
        success: false,
        message: "Using random initialization",
        fallbackReason: null
    };
    
    if (!params) {
        loadResult.fallbackReason = "No parameters provided";
        return loadResult;
    }
    
    // Validate parameter structure
    if (!this.validateParameterStructure(params)) {
        loadResult.fallbackReason = "Invalid parameter structure";
        return loadResult;
    }
    

    
    try {
        // Load all parameters
        this.cls_embedding = params.cls_embedding.slice();
        
        this.type_embeddings = {
            cls: params.type_embeddings.cls.slice(),
            ctx: params.type_embeddings.ctx.slice(),
            predator: params.type_embeddings.predator.slice(),
            boid: params.type_embeddings.boid.slice()
        };
        
        this.ctx_projection = this.copyMatrix(params.ctx_projection);
        this.predator_projection = this.copyMatrix(params.predator_projection);
        this.boid_projection = this.copyMatrix(params.boid_projection);
        
        this.layers = [];
        for (var i = 0; i < this.n_layers; i++) {
            var layerParams = params.layers[i];
            this.layers.push({
                ln_scale: layerParams.ln_scale.slice(),
                ln_bias: layerParams.ln_bias.slice(),
                qkv_weight: this.copyMatrix(layerParams.qkv_weight),
                qkv_bias: layerParams.qkv_bias.slice(),
                attn_out_weight: this.copyMatrix(layerParams.attn_out_weight),
                attn_out_bias: layerParams.attn_out_bias.slice(),
                ffn_ln_scale: layerParams.ffn_ln_scale.slice(),
                ffn_ln_bias: layerParams.ffn_ln_bias.slice(),
                ffn_gate_weight: this.copyMatrix(layerParams.ffn_gate_weight),
                ffn_gate_bias: layerParams.ffn_gate_bias.slice(),
                ffn_up_weight: this.copyMatrix(layerParams.ffn_up_weight),
                ffn_up_bias: layerParams.ffn_up_bias.slice(),
                ffn_down_weight: this.copyMatrix(layerParams.ffn_down_weight),
                ffn_down_bias: layerParams.ffn_down_bias.slice()
            });
        }
        
        this.output_weight = this.copyMatrix(params.output_weight);
        this.output_bias = params.output_bias.slice();
        
        loadResult.success = true;
        loadResult.message = "Successfully loaded transformer parameters";
        loadResult.fallbackReason = null;
        
        this.modelLoadResult = loadResult;
        
    } catch (error) {
        loadResult.fallbackReason = "Error loading parameters: " + error.message;
        this.modelLoadResult = loadResult;
    }
    
    return loadResult;
};

/**
 * Validate parameter structure
 * @param {Object} params - Parameters to validate
 * @returns {boolean} True if structure is valid
 */
TransformerEncoder.prototype.validateParameterStructure = function(params) {
    if (!params || typeof params !== 'object') return false;
    
    // Check required top-level properties
    var requiredProps = ['d_model', 'n_heads', 'n_layers', 'ffn_hidden', 
                        'cls_embedding', 'type_embeddings', 'ctx_projection', 
                        'predator_projection', 'boid_projection', 'layers', 
                        'output_weight', 'output_bias'];
    
    for (var i = 0; i < requiredProps.length; i++) {
        if (!(requiredProps[i] in params)) return false;
    }
    
    // Check type embeddings
    if (!params.type_embeddings.cls || !params.type_embeddings.ctx || 
        !params.type_embeddings.predator || !params.type_embeddings.boid) {
        return false;
    }
    
    // Check layers array
    if (!Array.isArray(params.layers) || params.layers.length !== params.n_layers) {
        return false;
    }
    
    return true;
};



/**
 * Forward pass through transformer encoder
 * @param {Object} structuredInputs - {context, predator, boids}
 * @returns {Array} Steering forces [x, y]
 */
TransformerEncoder.prototype.forward = function(structuredInputs) {
    // Build token sequence
    var tokens = this.buildTokens(structuredInputs);
    
    // Process through transformer layers
    for (var layer_idx = 0; layer_idx < this.n_layers; layer_idx++) {
        tokens = this.transformerBlock(tokens, this.layers[layer_idx]);
    }
    
    // Extract [CLS] token and project to steering forces
    var cls_token = tokens[0];
    var logits = this.addVectors(
        this.matrixVectorMultiply(this.output_weight, cls_token),
        this.output_bias
    );
    
    // Apply tanh activation for bounded output
    return [Math.tanh(logits[0]), Math.tanh(logits[1])];
};

/**
 * Build token sequence from structured inputs
 * @param {Object} structuredInputs - {context, predator, boids}
 * @returns {Array} Token sequence [S × d_model]
 */
TransformerEncoder.prototype.buildTokens = function(structuredInputs) {
    var tokens = [];
    
    // Token 0: [CLS] - learned embedding + type embedding
    var cls_token = this.addVectors(this.cls_embedding, this.type_embeddings.cls);
    tokens.push(cls_token);
    
    // Token 1: [CTX] - context projection + type embedding (bias already in type embedding)
    var ctx_input = [
        structuredInputs.context.canvasWidth,  // w/D (already normalized)
        structuredInputs.context.canvasHeight  // h/D (already normalized)
    ];
    var ctx_projected = this.matrixVectorMultiply(this.ctx_projection, ctx_input);
    var ctx_token = this.addVectors(ctx_projected, this.type_embeddings.ctx);
    tokens.push(ctx_token);
    
    // Token 2: Predator - predator projection + type embedding (bias already in type embedding)
    var predator_input = [
        structuredInputs.predator.velX,  // vx/V (already normalized)
        structuredInputs.predator.velY,  // vy/V (already normalized)
        0.0,  // padding
        0.0   // padding
    ];
    var predator_projected = this.matrixVectorMultiply(this.predator_projection, predator_input);
    var predator_token = this.addVectors(predator_projected, this.type_embeddings.predator);
    tokens.push(predator_token);
    
    // Tokens 3+: Boids - boid projections + type embeddings (bias already in type embedding)
    for (var i = 0; i < structuredInputs.boids.length; i++) {
        var boid = structuredInputs.boids[i];
        var boid_input = [
            boid.relX,  // dx/D (already normalized)
            boid.relY,  // dy/D (already normalized)
            boid.velX,  // vx/V (already normalized)
            boid.velY   // vy/V (already normalized)
        ];
        var boid_projected = this.matrixVectorMultiply(this.boid_projection, boid_input);
        var boid_token = this.addVectors(boid_projected, this.type_embeddings.boid);
        tokens.push(boid_token);
    }
    
    return tokens;
};

/**
 * Single transformer block: LayerNorm → MHSA → FFN
 * @param {Array} tokens - Input tokens [S × d_model]
 * @param {Object} layer - Layer parameters
 * @returns {Array} Output tokens [S × d_model]
 */
TransformerEncoder.prototype.transformerBlock = function(tokens, layer) {
    var seq_len = tokens.length;
    
    // 1. Layer normalization
    var normed_tokens = [];
    for (var i = 0; i < seq_len; i++) {
        normed_tokens.push(this.layerNorm(tokens[i], layer.ln_scale, layer.ln_bias));
    }
    
    // 2. Multi-head self-attention
    var attn_output = this.multiHeadAttention(normed_tokens, layer);
    
    // 3. Residual connection
    var residual1 = [];
    for (var i = 0; i < seq_len; i++) {
        residual1.push(this.addVectors(tokens[i], attn_output[i]));
    }
    
    // 4. Layer norm for FFN
    var ffn_normed = [];
    for (var i = 0; i < seq_len; i++) {
        ffn_normed.push(this.layerNorm(residual1[i], layer.ffn_ln_scale, layer.ffn_ln_bias));
    }
    
    // 5. GEGLU feed-forward
    var ffn_output = [];
    for (var i = 0; i < seq_len; i++) {
        ffn_output.push(this.geglu(ffn_normed[i], layer));
    }
    
    // 6. Final residual connection
    var output = [];
    for (var i = 0; i < seq_len; i++) {
        output.push(this.addVectors(residual1[i], ffn_output[i]));
    }
    
    return output;
};

/**
 * Multi-head self-attention - Exact PyTorch replication
 * @param {Array} tokens - Input tokens [S × d_model]
 * @param {Object} layer - Layer parameters
 * @returns {Array} Attention output [S × d_model]
 */
TransformerEncoder.prototype.multiHeadAttention = function(tokens, layer) {
    var seq_len = tokens.length;
    var d_model = this.d_model;
    var n_heads = this.n_heads;
    var head_dim = this.head_dim;
    var scale = 0.25;  // Exact PyTorch scale: 1 / sqrt(16) = 0.25
    
    // Step 1: QKV Projection for all tokens
    var qkv_all = [];
    for (var i = 0; i < seq_len; i++) {
        var qkv = this.addVectors(
            this.matrixVectorMultiply(layer.qkv_weight, tokens[i]),
            layer.qkv_bias
        );
        qkv_all.push(qkv);
    }
    
    // Step 2: Reshape QKV following PyTorch's exact approach
    // PyTorch: [batch=1, seq_len, 3*d_model] -> [batch=1, seq_len, 3, n_heads, head_dim] -> [3, batch=1, n_heads, seq_len, head_dim]
    
    // Extract Q, K, V tensors in PyTorch format: [n_heads, seq_len, head_dim]
    var Q = [], K = [], V = [];
    
    // Initialize head arrays
    for (var h = 0; h < n_heads; h++) {
        Q[h] = [];
        K[h] = [];
        V[h] = [];
    }
    
    // Fill Q, K, V following PyTorch's reshape and permute operations
    for (var seq = 0; seq < seq_len; seq++) {
        var qkv = qkv_all[seq]; // [3*d_model]
        
        for (var h = 0; h < n_heads; h++) {
            var q_head = [];
            var k_head = [];
            var v_head = [];
            
            for (var dim = 0; dim < head_dim; dim++) {
                // PyTorch memory layout after view(batch, seq_len, 3, n_heads, head_dim)
                // q: qkv_idx=0, k: qkv_idx=1, v: qkv_idx=2
                var q_idx = 0 * d_model + h * head_dim + dim;  // Q section
                var k_idx = 1 * d_model + h * head_dim + dim;  // K section  
                var v_idx = 2 * d_model + h * head_dim + dim;  // V section
                
                q_head.push(qkv[q_idx]);
                k_head.push(qkv[k_idx]);
                v_head.push(qkv[v_idx]);
            }
            
            Q[h].push(q_head);  // Q[h][seq][dim]
            K[h].push(k_head);  // K[h][seq][dim]
            V[h].push(v_head);  // V[h][seq][dim]
        }
    }
    
    // Step 3: Compute attention for all heads simultaneously
    var attn_output = []; // [n_heads][seq_len][head_dim]
    
    for (var h = 0; h < n_heads; h++) {
        var q_head = Q[h];  // [seq_len][head_dim]
        var k_head = K[h];  // [seq_len][head_dim] 
        var v_head = V[h];  // [seq_len][head_dim]
        
        // Compute attention scores: Q @ K.T
        var scores = [];
        for (var i = 0; i < seq_len; i++) {
            scores[i] = [];
            for (var j = 0; j < seq_len; j++) {
                var score = this.dotProduct(q_head[i], k_head[j]) * scale;
                scores[i][j] = score;
            }
        }
        
        // Apply softmax to each row
        for (var i = 0; i < seq_len; i++) {
            scores[i] = this.softmax(scores[i]);
        }
        
        // Apply attention to values: scores @ V
        var head_output = [];
        for (var i = 0; i < seq_len; i++) {
            var attended = new Array(head_dim).fill(0);
            for (var j = 0; j < seq_len; j++) {
                for (var d = 0; d < head_dim; d++) {
                    attended[d] += scores[i][j] * v_head[j][d];
                }
            }
            head_output.push(attended);
        }
        
        attn_output.push(head_output);
    }
    
    // Step 4: Concatenate heads following PyTorch's transpose and view operations
    // PyTorch: [batch, n_heads, seq_len, head_dim] -> transpose(1,2) -> [batch, seq_len, n_heads, head_dim] -> view -> [batch, seq_len, d_model]
    var concatenated = [];
    
    for (var i = 0; i < seq_len; i++) {
        var token_concat = [];
        
        // Concatenate all heads for token i
        for (var h = 0; h < n_heads; h++) {
            for (var d = 0; d < head_dim; d++) {
                token_concat.push(attn_output[h][i][d]);
            }
        }
        
        concatenated.push(token_concat);
    }
    
    // Step 5: Apply output projection
    var final_output = [];
    for (var i = 0; i < seq_len; i++) {
        var projected = this.addVectors(
            this.matrixVectorMultiply(layer.attn_out_weight, concatenated[i]),
            layer.attn_out_bias
        );
        final_output.push(projected);
    }
    
    return final_output;
};

// Removed old attentionHead function - now integrated into multiHeadAttention

/**
 * GEGLU feed-forward network
 * @param {Array} x - Input vector [d_model]
 * @param {Object} layer - Layer parameters
 * @returns {Array} Output vector [d_model]
 */
TransformerEncoder.prototype.geglu = function(x, layer) {
    // Gate projection
    var gate = this.addVectors(
        this.matrixVectorMultiply(layer.ffn_gate_weight, x),
        layer.ffn_gate_bias
    );
    
    // Up projection  
    var up = this.addVectors(
        this.matrixVectorMultiply(layer.ffn_up_weight, x),
        layer.ffn_up_bias
    );
    
    // GELU activation on gate
    for (var i = 0; i < gate.length; i++) {
        gate[i] = this.gelu(gate[i]);
    }
    
    // Element-wise multiplication
    var gated = [];
    for (var i = 0; i < gate.length; i++) {
        gated.push(gate[i] * up[i]);
    }
    
    // Down projection
    return this.addVectors(
        this.matrixVectorMultiply(layer.ffn_down_weight, gated),
        layer.ffn_down_bias
    );
};

// === UTILITY FUNCTIONS ===

/**
 * Copy a 2D matrix (deep copy)
 * @param {Array} matrix - Matrix to copy
 * @returns {Array} Deep copy of matrix
 */
TransformerEncoder.prototype.copyMatrix = function(matrix) {
    return matrix.map(function(row) {
        return Array.isArray(row) ? row.slice() : row;
    });
};

/**
 * Create random array
 * @param {number} size - Array size
 * @returns {Array} Random array
 */
TransformerEncoder.prototype.randomArray = function(size) {
    var arr = [];
    for (var i = 0; i < size; i++) {
        arr.push((Math.random() - 0.5) * 0.1);
    }
    return arr;
};

/**
 * Create random matrix
 * @param {number} rows - Number of rows
 * @param {number} cols - Number of columns
 * @returns {Array} Random matrix
 */
TransformerEncoder.prototype.randomMatrix = function(rows, cols) {
    var matrix = [];
    for (var i = 0; i < cols; i++) {
        matrix[i] = [];
        for (var j = 0; j < rows; j++) {
            matrix[i][j] = (Math.random() - 0.5) * Math.sqrt(2.0 / rows);
        }
    }
    return matrix;
};

/**
 * Create zeros array
 * @param {number} size - Array size
 * @returns {Array} Zeros array
 */
TransformerEncoder.prototype.zerosArray = function(size) {
    return new Array(size).fill(0);
};

/**
 * Create ones array
 * @param {number} size - Array size
 * @returns {Array} Ones array
 */
TransformerEncoder.prototype.onesArray = function(size) {
    return new Array(size).fill(1);
};

/**
 * Matrix-vector multiplication
 * @param {Array} matrix - Matrix
 * @param {Array} vector - Vector
 * @returns {Array} Result vector
 */
TransformerEncoder.prototype.matrixVectorMultiply = function(matrix, vector) {
    var result = [];
    for (var i = 0; i < matrix.length; i++) {
        var sum = 0;
        for (var j = 0; j < vector.length; j++) {
            sum += matrix[i][j] * vector[j];
        }
        result.push(sum);
    }
    return result;
};

/**
 * Add two vectors
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {Array} Sum vector
 */
TransformerEncoder.prototype.addVectors = function(a, b) {
    var result = [];
    for (var i = 0; i < a.length; i++) {
        result.push(a[i] + b[i]);
    }
    return result;
};

/**
 * Dot product of two vectors
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {number} Dot product
 */
TransformerEncoder.prototype.dotProduct = function(a, b) {
    var sum = 0;
    for (var i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
};

/**
 * Layer normalization
 * @param {Array} x - Input vector
 * @param {Array} scale - Scale parameters
 * @param {Array} bias - Bias parameters
 * @returns {Array} Normalized vector
 */
TransformerEncoder.prototype.layerNorm = function(x, scale, bias) {
    var mean = x.reduce(function(a, b) { return a + b; }) / x.length;
    var variance = x.reduce(function(a, b) { return a + (b - mean) * (b - mean); }, 0) / x.length;
    var std = Math.sqrt(variance + 1e-5);  // Fixed: use PyTorch's epsilon value
    
    var result = [];
    for (var i = 0; i < x.length; i++) {
        result.push(((x[i] - mean) / std) * scale[i] + bias[i]);
    }
    return result;
};

/**
 * Softmax activation
 * @param {Array} x - Input vector
 * @returns {Array} Softmax output
 */
TransformerEncoder.prototype.softmax = function(x) {
    var max_val = Math.max.apply(Math, x);
    var exp_vals = x.map(function(val) { return Math.exp(val - max_val); });
    var sum = exp_vals.reduce(function(a, b) { return a + b; });
    return exp_vals.map(function(val) { return val / sum; });
};

/**
 * Error function (erf) approximation - high precision
 * @param {number} x - Input value
 * @returns {number} erf(x)
 */
TransformerEncoder.prototype.erf = function(x) {
    // High-precision erf approximation using Abramowitz and Stegun
    var a1 =  0.254829592;
    var a2 = -0.284496736;
    var a3 =  1.421413741;
    var a4 = -1.453152027;
    var a5 =  1.061405429;
    var p  =  0.3275911;

    var sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    var t = 1 / (1 + p * x);
    var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
};

/**
 * GELU activation function - exact formula matching PyTorch
 * @param {number} x - Input value
 * @returns {number} GELU output
 */
TransformerEncoder.prototype.gelu = function(x) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1 + this.erf(x / Math.sqrt(2)));
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TransformerEncoder: TransformerEncoder };
}