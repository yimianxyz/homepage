// Simple Node.js test for transformer debugging
console.log("Testing transformer export...");

// Mock window object and constants
global.window = {
    SIMULATION_CONSTANTS: {
        MAX_DISTANCE: 800,
        BOID_MAX_SPEED: 3.5,
        PREDATOR_MAX_SPEED: 2,
        PREDATOR_MAX_FORCE: 0.5,
        PREDATOR_FORCE_SCALE: 30
    }
};

// Load the exported model
require('./src/config/model.js');

console.log("Model loaded. Checking structure...");
console.log("TRANSFORMER_PARAMS exists:", typeof global.window.TRANSFORMER_PARAMS !== 'undefined');

if (global.window.TRANSFORMER_PARAMS) {
    const params = global.window.TRANSFORMER_PARAMS;
    console.log("Architecture:", {
        d_model: params.d_model,
        n_heads: params.n_heads, 
        n_layers: params.n_layers,
        ffn_hidden: params.ffn_hidden
    });
    
    console.log("cls_embedding length:", params.cls_embedding.length);
    console.log("output_weight shape:", [params.output_weight.length, params.output_weight[0].length]);
    console.log("output_bias:", params.output_bias);
    
    // Check for NaN or extreme values
    let hasNaN = false;
    let maxVal = -Infinity;
    let minVal = Infinity;
    
    function checkArray(arr, name) {
        if (Array.isArray(arr)) {
            for (let i = 0; i < arr.length; i++) {
                if (Array.isArray(arr[i])) {
                    checkArray(arr[i], name + "[" + i + "]");
                } else {
                    const val = arr[i];
                    if (isNaN(val)) {
                        hasNaN = true;
                        console.log("Found NaN in", name + "[" + i + "]");
                    }
                    maxVal = Math.max(maxVal, val);
                    minVal = Math.min(minVal, val);
                }
            }
        }
    }
    
    console.log("Checking for NaN values...");
    checkArray(params.cls_embedding, "cls_embedding");
    checkArray(params.output_weight, "output_weight");
    checkArray(params.output_bias, "output_bias");
    
    console.log("Has NaN values:", hasNaN);
    console.log("Value range:", minVal, "to", maxVal);
}
