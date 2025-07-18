/**
 * Example Transformer Model - Clean Architecture Design
 * 
 * This demonstrates the new model structure where architecture parameters
 * are clearly defined at the top level, making it easy to create different
 * transformer variants without changing the encoder code.
 */

window.TRANSFORMER_PARAMS = {
    // === ARCHITECTURE PARAMETERS ===
    // These define the transformer structure and must be at the top level
    d_model: 48,        // Model dimension
    n_heads: 4,         // Number of attention heads  
    n_layers: 3,        // Number of transformer layers
    ffn_hidden: 96,     // Feed-forward hidden dimension
    
    // === MODEL PARAMETERS ===
    // All the trained weights and biases
    
    // CLS token embedding [d_model]
    cls_embedding: [
        -0.1234, 0.5678, -0.9012, 0.3456, 0.7890, -0.2345, 0.6789, -0.0123,
        0.4567, -0.8901, 0.2345, 0.6789, -0.0123, 0.4567, -0.8901, 0.2345,
        0.6789, -0.0123, 0.4567, -0.8901, 0.2345, 0.6789, -0.0123, 0.4567,
        -0.8901, 0.2345, 0.6789, -0.0123, 0.4567, -0.8901, 0.2345, 0.6789,
        -0.0123, 0.4567, -0.8901, 0.2345, 0.6789, -0.0123, 0.4567, -0.8901,
        0.2345, 0.6789, -0.0123, 0.4567, -0.8901, 0.2345, 0.6789, -0.0123
    ],
    
    // Type embeddings for different entity types
    type_embeddings: {
        cls: [/* 48 values */],
        ctx: [/* 48 values */], 
        predator: [/* 48 values */],
        boid: [/* 48 values */]
    },
    
    // Input projection matrices
    ctx_projection: [/* [48, 2] matrix */],
    predator_projection: [/* [48, 4] matrix */], 
    boid_projection: [/* [48, 4] matrix */],
    
    // Transformer layers (n_layers entries)
    layers: [
        {
            // Layer 0 parameters
            ln_scale: [/* 48 values */],
            ln_bias: [/* 48 values */],
            qkv_weight: [/* [144, 48] matrix */],
            qkv_bias: [/* 144 values */],
            attn_out_weight: [/* [48, 48] matrix */],
            attn_out_bias: [/* 48 values */],
            ffn_ln_scale: [/* 48 values */],
            ffn_ln_bias: [/* 48 values */],
            ffn_gate_weight: [/* [96, 48] matrix */],
            ffn_gate_bias: [/* 96 values */],
            ffn_up_weight: [/* [96, 48] matrix */],
            ffn_up_bias: [/* 96 values */],
            ffn_down_weight: [/* [48, 96] matrix */],
            ffn_down_bias: [/* 48 values */]
        },
        {
            // Layer 1 parameters (same structure)
        },
        {
            // Layer 2 parameters (same structure)
        }
    ],
    
    // Output projection
    output_weight: [/* [2, 48] matrix */],
    output_bias: [/* 2 values */]
};

console.log("Loaded example transformer model:");
console.log("  Architecture: " + window.TRANSFORMER_PARAMS.d_model + "×" + 
           window.TRANSFORMER_PARAMS.n_heads + "×" + 
           window.TRANSFORMER_PARAMS.n_layers + "×" + 
           window.TRANSFORMER_PARAMS.ffn_hidden); 