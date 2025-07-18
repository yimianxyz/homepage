/**
 * Small Transformer Model Example - Different Architecture
 * 
 * This demonstrates a smaller transformer with different architecture parameters.
 * Shows how the new design can support various transformer sizes without
 * changing the encoder code.
 */

window.TRANSFORMER_PARAMS = {
    // === ARCHITECTURE PARAMETERS ===
    // Different architecture: smaller model for faster inference
    d_model: 32,        // Smaller model dimension
    n_heads: 2,         // Fewer attention heads
    n_layers: 2,        // Fewer layers
    ffn_hidden: 64,     // Smaller feed-forward hidden dimension
    
    // === MODEL PARAMETERS ===
    // All parameters sized according to the architecture above
    
    // CLS token embedding [32]
    cls_embedding: [
        -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
        -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85,
        0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75, -0.85
    ],
    
    // Type embeddings for different entity types [32 each]
    type_embeddings: {
        cls: new Array(32).fill(0).map((_, i) => Math.sin(i * 0.1)),
        ctx: new Array(32).fill(0).map((_, i) => Math.cos(i * 0.1)),
        predator: new Array(32).fill(0).map((_, i) => Math.sin(i * 0.2)),
        boid: new Array(32).fill(0).map((_, i) => Math.cos(i * 0.2))
    },
    
    // Input projection matrices
    ctx_projection: new Array(32).fill(0).map(() => 
        new Array(2).fill(0).map(() => (Math.random() - 0.5) * 0.2)
    ),
    predator_projection: new Array(32).fill(0).map(() => 
        new Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.2)
    ),
    boid_projection: new Array(32).fill(0).map(() => 
        new Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.2)
    ),
    
    // Transformer layers (2 layers for this small model)
    layers: [
        {
            // Layer 0 parameters
            ln_scale: new Array(32).fill(1.0),
            ln_bias: new Array(32).fill(0.0),
            qkv_weight: new Array(96).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            qkv_bias: new Array(96).fill(0.0),
            attn_out_weight: new Array(32).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            attn_out_bias: new Array(32).fill(0.0),
            ffn_ln_scale: new Array(32).fill(1.0),
            ffn_ln_bias: new Array(32).fill(0.0),
            ffn_gate_weight: new Array(64).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            ffn_gate_bias: new Array(64).fill(0.0),
            ffn_up_weight: new Array(64).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            ffn_up_bias: new Array(64).fill(0.0),
            ffn_down_weight: new Array(32).fill(0).map(() => 
                new Array(64).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            ffn_down_bias: new Array(32).fill(0.0)
        },
        {
            // Layer 1 parameters (same structure as layer 0)
            ln_scale: new Array(32).fill(1.0),
            ln_bias: new Array(32).fill(0.0),
            qkv_weight: new Array(96).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            qkv_bias: new Array(96).fill(0.0),
            attn_out_weight: new Array(32).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            attn_out_bias: new Array(32).fill(0.0),
            ffn_ln_scale: new Array(32).fill(1.0),
            ffn_ln_bias: new Array(32).fill(0.0),
            ffn_gate_weight: new Array(64).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            ffn_gate_bias: new Array(64).fill(0.0),
            ffn_up_weight: new Array(64).fill(0).map(() => 
                new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            ffn_up_bias: new Array(64).fill(0.0),
            ffn_down_weight: new Array(32).fill(0).map(() => 
                new Array(64).fill(0).map(() => (Math.random() - 0.5) * 0.1)
            ),
            ffn_down_bias: new Array(32).fill(0.0)
        }
    ],
    
    // Output projection [2, 32]
    output_weight: new Array(2).fill(0).map(() => 
        new Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.2)
    ),
    output_bias: [0.0, 0.0]
};

console.log("Loaded small transformer model:");
console.log("  Architecture: " + window.TRANSFORMER_PARAMS.d_model + "×" + 
           window.TRANSFORMER_PARAMS.n_heads + "×" + 
           window.TRANSFORMER_PARAMS.n_layers + "×" + 
           window.TRANSFORMER_PARAMS.ffn_hidden);
console.log("  Total parameters: ~" + 
           Math.round((window.TRANSFORMER_PARAMS.d_model * window.TRANSFORMER_PARAMS.d_model * 
                      window.TRANSFORMER_PARAMS.n_layers * 6) / 1000) + "k"); 