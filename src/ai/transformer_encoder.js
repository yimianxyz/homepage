/**
 * Transformer Encoder - Vision Transformer style architecture for predator control
 * 
 * Architecture:
 * - d_model = 48, n_heads = 4, n_layers = 3
 * - GEGLU feed-forward networks (96 hidden → 48)
 * - Token sequence: [CLS] + [CTX] + Predator + Boids
 * - Type embeddings for different entity types
 */
function TransformerEncoder() {
    // Model hyperparameters
    this.d_model = 48;
    this.n_heads = 4;
    this.n_layers = 3;
    this.head_dim = this.d_model / this.n_heads; // 12
    this.ffn_hidden = 96;
    
    // Normalization constants from simulation
    this.D = window.SIMULATION_CONSTANTS.MAX_DISTANCE;
    this.V = Math.max(
        window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
    
    // Try to load pre-trained parameters, fallback to random initialization
    this.modelLoadResult = this.loadParameters();
    if (!this.modelLoadResult.success) {
        this.initializeParameters();
        console.warn("Transformer: " + this.modelLoadResult.message + " - " + this.modelLoadResult.fallbackReason);
    } else {
        console.log("Transformer: " + this.modelLoadResult.message);
    }
}

/**
 * Load parameters from model.js or global TRANSFORMER_PARAMS
 * @param {Object} params - Optional parameters object
 * @returns {Object} Load result with success status and message
 */
TransformerEncoder.prototype.loadParameters = function(params) {
    var loadResult = {
        success: false,
        message: "Using random initialization",
        fallbackReason: null
    };
    
    if (!params) {
        if (typeof window.TRANSFORMER_PARAMS !== 'undefined') {
            params = window.TRANSFORMER_PARAMS;
        } else {
            loadResult.fallbackReason = "No transformer parameters found in model.js";
            return loadResult;
        }
    }
    
    // Validate parameter structure
    if (!this.validateParameterStructure(params)) {
        loadResult.fallbackReason = "Invalid transformer parameter structure";
        return loadResult;
    }
    
    // Check architecture compatibility
    if (params.d_model !== this.d_model || 
        params.n_heads !== this.n_heads || 
        params.n_layers !== this.n_layers ||
        params.ffn_hidden !== this.ffn_hidden) {
        loadResult.fallbackReason = `Architecture mismatch: model is ${params.d_model}×${params.n_heads}×${params.n_layers}×${params.ffn_hidden}, current is ${this.d_model}×${this.n_heads}×${this.n_layers}×${this.ffn_hidden}`;
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
        loadResult.message = "Successfully loaded pre-trained transformer";
        loadResult.fallbackReason = null;
        
    } catch (error) {
        loadResult.fallbackReason = "Error loading parameters: " + error.message;
    }
    
    return loadResult;
};

/**
 * Validate parameter structure for transformer
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
 * Get current parameters for saving
 * @returns {Object} Complete parameter object
 */
TransformerEncoder.prototype.getParameters = function() {
    return {
        // Architecture info
        d_model: this.d_model,
        n_heads: this.n_heads,
        n_layers: this.n_layers,
        ffn_hidden: this.ffn_hidden,
        
        // Embeddings
        cls_embedding: this.cls_embedding.slice(),
        type_embeddings: {
            cls: this.type_embeddings.cls.slice(),
            ctx: this.type_embeddings.ctx.slice(),
            predator: this.type_embeddings.predator.slice(),
            boid: this.type_embeddings.boid.slice()
        },
        
        // Input projections
        ctx_projection: this.copyMatrix(this.ctx_projection),
        predator_projection: this.copyMatrix(this.predator_projection),
        boid_projection: this.copyMatrix(this.boid_projection),
        
        // Transformer layers
        layers: this.layers.map(function(layer) {
            return {
                ln_scale: layer.ln_scale.slice(),
                ln_bias: layer.ln_bias.slice(),
                qkv_weight: this.copyMatrix(layer.qkv_weight),
                qkv_bias: layer.qkv_bias.slice(),
                attn_out_weight: this.copyMatrix(layer.attn_out_weight),
                attn_out_bias: layer.attn_out_bias.slice(),
                ffn_ln_scale: layer.ffn_ln_scale.slice(),
                ffn_ln_bias: layer.ffn_ln_bias.slice(),
                ffn_gate_weight: this.copyMatrix(layer.ffn_gate_weight),
                ffn_gate_bias: layer.ffn_gate_bias.slice(),
                ffn_up_weight: this.copyMatrix(layer.ffn_up_weight),
                ffn_up_bias: layer.ffn_up_bias.slice(),
                ffn_down_weight: this.copyMatrix(layer.ffn_down_weight),
                ffn_down_bias: layer.ffn_down_bias.slice()
            };
        }.bind(this)),
        
        // Output projection
        output_weight: this.copyMatrix(this.output_weight),
        output_bias: this.output_bias.slice()
    };
};

/**
 * Reset all parameters to random initialization
 */
TransformerEncoder.prototype.reset = function() {
    this.initializeParameters();
    console.log("Transformer parameters reset to random initialization");
};

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

TransformerEncoder.prototype.initializeParameters = function() {
    // [CLS] token - learned embedding
    this.cls_embedding = this.randomArray(this.d_model);
    
    // Type embeddings for different entity types
    this.type_embeddings = {
        cls: this.randomArray(this.d_model),      // type_id 0
        ctx: this.randomArray(this.d_model),      // type_id 1  
        predator: this.randomArray(this.d_model), // type_id 2
        boid: this.randomArray(this.d_model)      // type_id 3
    };
    
    // Input projection layers
    this.ctx_projection = this.randomMatrix(2, this.d_model);     // [w/D, h/D] → 48D
    this.predator_projection = this.randomMatrix(4, this.d_model); // [vx/V, vy/V, 0, 0] → 48D
    this.boid_projection = this.randomMatrix(4, this.d_model);     // [dx/D, dy/D, dvx/V, dvy/V] → 48D
    
    // Transformer layers
    this.layers = [];
    for (var i = 0; i < this.n_layers; i++) {
        this.layers.push({
            // Layer normalization parameters
            ln_scale: this.onesArray(this.d_model),
            ln_bias: this.zerosArray(this.d_model),
            
            // Fused QKV projection (3 × 48 × 48)
            qkv_weight: this.randomMatrix(this.d_model, 3 * this.d_model),
            qkv_bias: this.zerosArray(3 * this.d_model),
            
            // Output projection after attention
            attn_out_weight: this.randomMatrix(this.d_model, this.d_model),
            attn_out_bias: this.zerosArray(this.d_model),
            
            // GEGLU feed-forward
            ffn_ln_scale: this.onesArray(this.d_model),
            ffn_ln_bias: this.zerosArray(this.d_model),
            ffn_gate_weight: this.randomMatrix(this.d_model, this.ffn_hidden),
            ffn_gate_bias: this.zerosArray(this.ffn_hidden),
            ffn_up_weight: this.randomMatrix(this.d_model, this.ffn_hidden),
            ffn_up_bias: this.zerosArray(this.ffn_hidden),
            ffn_down_weight: this.randomMatrix(this.ffn_hidden, this.d_model),
            ffn_down_bias: this.zerosArray(this.d_model)
        });
    }
    
    // Final output projection: [CLS] token → steering forces
    this.output_weight = this.randomMatrix(this.d_model, 2);
    this.output_bias = this.zerosArray(2);
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
    
    // Token 1: [CTX] - context projection + type embedding
    var ctx_input = [
        structuredInputs.context.canvasWidth,  // w/D (already normalized)
        structuredInputs.context.canvasHeight  // h/D (already normalized)
    ];
    var ctx_projected = this.matrixVectorMultiply(this.ctx_projection, ctx_input);
    var ctx_token = this.addVectors(ctx_projected, this.type_embeddings.ctx);
    tokens.push(ctx_token);
    
    // Token 2: Predator - predator projection + type embedding
    var predator_input = [
        structuredInputs.predator.velX,  // vx/V (already normalized)
        structuredInputs.predator.velY,  // vy/V (already normalized)
        0.0,  // padding
        0.0   // padding
    ];
    var predator_projected = this.matrixVectorMultiply(this.predator_projection, predator_input);
    var predator_token = this.addVectors(predator_projected, this.type_embeddings.predator);
    tokens.push(predator_token);
    
    // Tokens 3+: Boids - boid projections + type embeddings
    for (var i = 0; i < structuredInputs.boids.length; i++) {
        var boid = structuredInputs.boids[i];
        var boid_input = [
            boid.relX,  // dx/D (already normalized)
            boid.relY,  // dy/D (already normalized)
            boid.velX,  // dvx/V (already normalized)
            boid.velY   // dvy/V (already normalized)
        ];
        var boid_projected = this.matrixVectorMultiply(this.boid_projection, boid_input);
        var boid_token = this.addVectors(boid_projected, this.type_embeddings.boid);
        tokens.push(boid_token);
    }
    
    return tokens;
};

/**
 * Forward pass through transformer encoder
 * @param {Object} structuredInputs - {context, predator, boids}
 * @returns {Array} Steering forces [x, y]
 */
TransformerEncoder.prototype.forward = function(structuredInputs) {
    // Build token sequence
    var tokens = this.buildTokens(structuredInputs);
    var seq_len = tokens.length;
    
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
 * Multi-head self-attention
 * @param {Array} tokens - Input tokens [S × d_model]
 * @param {Object} layer - Layer parameters
 * @returns {Array} Attention output [S × d_model]
 */
TransformerEncoder.prototype.multiHeadAttention = function(tokens, layer) {
    var seq_len = tokens.length;
    
    // Compute QKV for all tokens
    var Q = [], K = [], V = [];
    for (var i = 0; i < seq_len; i++) {
        var qkv = this.addVectors(
            this.matrixVectorMultiply(layer.qkv_weight, tokens[i]),
            layer.qkv_bias
        );
        
        // Split into Q, K, V
        Q.push(qkv.slice(0, this.d_model));
        K.push(qkv.slice(this.d_model, 2 * this.d_model));
        V.push(qkv.slice(2 * this.d_model, 3 * this.d_model));
    }
    
    // Process each attention head
    var head_outputs = [];
    for (var head = 0; head < this.n_heads; head++) {
        var head_output = this.attentionHead(Q, K, V, head);
        head_outputs.push(head_output);
    }
    
    // Concatenate heads and project
    var concat_output = [];
    for (var i = 0; i < seq_len; i++) {
        var concat_token = [];
        for (var head = 0; head < this.n_heads; head++) {
            concat_token = concat_token.concat(head_outputs[head][i]);
        }
        
        var projected = this.addVectors(
            this.matrixVectorMultiply(layer.attn_out_weight, concat_token),
            layer.attn_out_bias
        );
        concat_output.push(projected);
    }
    
    return concat_output;
};

/**
 * Single attention head computation
 * @param {Array} Q - Query matrices [S × d_model]
 * @param {Array} K - Key matrices [S × d_model]
 * @param {Array} V - Value matrices [S × d_model]
 * @param {number} head - Head index
 * @returns {Array} Head output [S × head_dim]
 */
TransformerEncoder.prototype.attentionHead = function(Q, K, V, head) {
    var seq_len = Q.length;
    var start_idx = head * this.head_dim;
    var end_idx = start_idx + this.head_dim;
    
    // Extract head-specific Q, K, V
    var q_head = [], k_head = [], v_head = [];
    for (var i = 0; i < seq_len; i++) {
        q_head.push(Q[i].slice(start_idx, end_idx));
        k_head.push(K[i].slice(start_idx, end_idx));
        v_head.push(V[i].slice(start_idx, end_idx));
    }
    
    // Compute attention scores
    var scores = [];
    var scale = 1.0 / Math.sqrt(this.head_dim);
    
    for (var i = 0; i < seq_len; i++) {
        scores[i] = [];
        for (var j = 0; j < seq_len; j++) {
            scores[i][j] = this.dotProduct(q_head[i], k_head[j]) * scale;
        }
        
        // Softmax
        scores[i] = this.softmax(scores[i]);
    }
    
    // Apply attention to values
    var output = [];
    for (var i = 0; i < seq_len; i++) {
        var attended = new Array(this.head_dim).fill(0);
        for (var j = 0; j < seq_len; j++) {
            for (var k = 0; k < this.head_dim; k++) {
                attended[k] += scores[i][j] * v_head[j][k];
            }
        }
        output.push(attended);
    }
    
    return output;
};

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

// Utility functions
TransformerEncoder.prototype.randomArray = function(size) {
    var arr = [];
    for (var i = 0; i < size; i++) {
        arr.push((Math.random() - 0.5) * 0.1);
    }
    return arr;
};

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

TransformerEncoder.prototype.zerosArray = function(size) {
    return new Array(size).fill(0);
};

TransformerEncoder.prototype.onesArray = function(size) {
    return new Array(size).fill(1);
};

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

TransformerEncoder.prototype.addVectors = function(a, b) {
    var result = [];
    for (var i = 0; i < a.length; i++) {
        result.push(a[i] + b[i]);
    }
    return result;
};

TransformerEncoder.prototype.dotProduct = function(a, b) {
    var sum = 0;
    for (var i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
};

TransformerEncoder.prototype.layerNorm = function(x, scale, bias) {
    var mean = x.reduce(function(a, b) { return a + b; }) / x.length;
    var variance = x.reduce(function(a, b) { return a + (b - mean) * (b - mean); }, 0) / x.length;
    var std = Math.sqrt(variance + 1e-6);
    
    var result = [];
    for (var i = 0; i < x.length; i++) {
        result.push(((x[i] - mean) / std) * scale[i] + bias[i]);
    }
    return result;
};

TransformerEncoder.prototype.softmax = function(x) {
    var max_val = Math.max.apply(Math, x);
    var exp_vals = x.map(function(val) { return Math.exp(val - max_val); });
    var sum = exp_vals.reduce(function(a, b) { return a + b; });
    return exp_vals.map(function(val) { return val / sum; });
};

TransformerEncoder.prototype.gelu = function(x) {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}; 