// Complete transformer forward pass test
console.log("Testing complete transformer forward pass...");

global.window = {
    SIMULATION_CONSTANTS: {
        MAX_DISTANCE: 800,
        BOID_MAX_SPEED: 3.5,
        PREDATOR_MAX_SPEED: 2,
        PREDATOR_MAX_FORCE: 0.5,
        PREDATOR_FORCE_SCALE: 30
    }
};

require('./src/config/model.js');
const params = global.window.TRANSFORMER_PARAMS;

// Implement essential transformer operations
function matrixVectorMultiply(matrix, vector) {
    const result = [];
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        for (let j = 0; j < vector.length; j++) {
            sum += matrix[i][j] * vector[j];
        }
        result.push(sum);
    }
    return result;
}

function addVectors(a, b) {
    return a.map((val, i) => val + b[i]);
}

function layerNorm(x, scale, bias) {
    const mean = x.reduce((a, b) => a + b) / x.length;
    const variance = x.reduce((a, b) => a + (b - mean) * (b - mean), 0) / x.length;
    const std = Math.sqrt(variance + 1e-6);
    
    return x.map((val, i) => ((val - mean) / std) * scale[i] + bias[i]);
}

function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function softmax(x) {
    const maxVal = Math.max(...x);
    const expVals = x.map(val => Math.exp(val - maxVal));
    const sum = expVals.reduce((a, b) => a + b);
    return expVals.map(val => val / sum);
}

function dotProduct(a, b) {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Simple forward pass
function simpleForward(structuredInputs) {
    // Build tokens
    const tokens = [];
    
    // Token 0: [CLS]
    const cls_token = addVectors(params.cls_embedding, params.type_embeddings.cls);
    tokens.push(cls_token);
    
    // Token 1: [CTX] 
    const ctx_input = [
        structuredInputs.context.canvasWidth,
        structuredInputs.context.canvasHeight
    ];
    const ctx_projected = matrixVectorMultiply(params.ctx_projection, ctx_input);
    const ctx_token = addVectors(ctx_projected, params.type_embeddings.ctx);
    tokens.push(ctx_token);
    
    // Token 2: Predator
    const predator_input = [
        structuredInputs.predator.velX,
        structuredInputs.predator.velY,
        0.0,
        0.0
    ];
    const predator_projected = matrixVectorMultiply(params.predator_projection, predator_input);
    const predator_token = addVectors(predator_projected, params.type_embeddings.predator);
    tokens.push(predator_token);
    
    // Boid tokens
    for (const boid of structuredInputs.boids) {
        const boid_input = [boid.relX, boid.relY, boid.velX, boid.velY];
        const boid_projected = matrixVectorMultiply(params.boid_projection, boid_input);
        const boid_token = addVectors(boid_projected, params.type_embeddings.boid);
        tokens.push(boid_token);
    }
    
    console.log(`Built ${tokens.length} tokens`);
    
    // Simple transformer processing (just first layer for testing)
    const layer = params.layers[0];
    
    // Layer norm
    const normed_tokens = tokens.map(token => layerNorm(token, layer.ln_scale, layer.ln_bias));
    
    // Simple self-attention (simplified version)
    const seq_len = tokens.length;
    const d_model = 48;
    const n_heads = 4;
    const head_dim = d_model / n_heads;
    
    // Just use first head for simplicity
    const qkv_output = normed_tokens.map(token => 
        addVectors(matrixVectorMultiply(layer.qkv_weight, token), layer.qkv_bias)
    );
    
    // Extract Q, K, V for first head
    const Q = qkv_output.map(qkv => qkv.slice(0, head_dim));
    const K = qkv_output.map(qkv => qkv.slice(d_model, d_model + head_dim));
    const V = qkv_output.map(qkv => qkv.slice(2 * d_model, 2 * d_model + head_dim));
    
    // Compute attention for first token only (CLS)
    const q = Q[0];
    const scores = K.map(k => dotProduct(q, k) / Math.sqrt(head_dim));
    const attn_weights = softmax(scores);
    
    // Apply attention to values
    const attended = new Array(head_dim).fill(0);
    for (let i = 0; i < seq_len; i++) {
        for (let j = 0; j < head_dim; j++) {
            attended[j] += attn_weights[i] * V[i][j];
        }
    }
    
    // For simplicity, just use the attended output as the CLS token
    // (skipping full multi-head concat and FFN)
    const cls_output = attended.concat(new Array(d_model - head_dim).fill(0)).slice(0, d_model);
    
    // Final output projection
    const logits = addVectors(
        matrixVectorMultiply(params.output_weight, cls_output),
        params.output_bias
    );
    
    // Apply tanh
    const final_output = [Math.tanh(logits[0]), Math.tanh(logits[1])];
    
    return final_output;
}

// Test with realistic inputs
const structuredInputs = {
    context: { canvasWidth: 1.0, canvasHeight: 0.75 },
    predator: { velX: 0.1, velY: 0.1 },
    boids: [
        { relX: -0.1, relY: -0.1, velX: 0.2, velY: 0.0 },
        { relX: 0.2, relY: 0.1, velX: -0.1, velY: 0.3 },
        { relX: 0.0, relY: 0.3, velX: 0.0, velY: -0.2 }
    ]
};

console.log("\nTest inputs:");
console.log("Context:", structuredInputs.context);
console.log("Predator:", structuredInputs.predator);
console.log("Boids:", structuredInputs.boids.length);

console.log("\nRunning forward pass...");
try {
    const output = simpleForward(structuredInputs);
    console.log("✅ Forward pass successful!");
    console.log("Raw output:", output);
    
    // Convert to actions
    const forceScale = global.window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * global.window.SIMULATION_CONSTANTS.PREDATOR_FORCE_SCALE;
    const actions = [output[0] * forceScale, output[1] * forceScale];
    console.log("Actions:", actions);
    
    const magnitude = Math.sqrt(actions[0] * actions[0] + actions[1] * actions[1]);
    console.log("Action magnitude:", magnitude);
    
    if (magnitude < 1e-6) {
        console.log("❌ PROBLEM: Action magnitude too small! Predator won't move.");
    } else if (magnitude > 100) {
        console.log("❌ PROBLEM: Action magnitude too large! Predator movement unstable.");
    } else {
        console.log("✅ Action magnitude looks reasonable");
    }
    
} catch (error) {
    console.log("❌ Forward pass failed:", error.message);
    console.log(error.stack);
}

// Test multiple scenarios
console.log("\n=== Testing multiple scenarios ===");
for (let i = 0; i < 5; i++) {
    const testInputs = {
        context: { canvasWidth: 1.0, canvasHeight: 0.75 },
        predator: { 
            velX: (Math.random() - 0.5) * 0.4, 
            velY: (Math.random() - 0.5) * 0.4 
        },
        boids: [
            { 
                relX: (Math.random() - 0.5) * 0.5, 
                relY: (Math.random() - 0.5) * 0.5, 
                velX: (Math.random() - 0.5) * 0.8, 
                velY: (Math.random() - 0.5) * 0.8 
            }
        ]
    };
    
    try {
        const output = simpleForward(testInputs);
        const forceScale = global.window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * global.window.SIMULATION_CONSTANTS.PREDATOR_FORCE_SCALE;
        const actions = [output[0] * forceScale, output[1] * forceScale];
        const magnitude = Math.sqrt(actions[0] * actions[0] + actions[1] * actions[1]);
        
        console.log(`Test ${i+1}: output=[${output[0].toFixed(4)}, ${output[1].toFixed(4)}], magnitude=${magnitude.toFixed(4)}`);
    } catch (error) {
        console.log(`Test ${i+1}: FAILED - ${error.message}`);
    }
}

