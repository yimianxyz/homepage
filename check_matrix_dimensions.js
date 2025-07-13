// Check matrix dimensions in detail
console.log("Checking matrix dimensions...");

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

console.log("\n=== DETAILED PARAMETER ANALYSIS ===");

console.log("\n1. Architecture:");
console.log(`d_model: ${params.d_model}`);
console.log(`n_heads: ${params.n_heads}`);
console.log(`n_layers: ${params.n_layers}`);
console.log(`ffn_hidden: ${params.ffn_hidden}`);

console.log("\n2. Embeddings:");
console.log(`cls_embedding: [${params.cls_embedding.length}]`);
console.log(`type_embeddings.cls: [${params.type_embeddings.cls.length}]`);
console.log(`type_embeddings.ctx: [${params.type_embeddings.ctx.length}]`);
console.log(`type_embeddings.predator: [${params.type_embeddings.predator.length}]`);
console.log(`type_embeddings.boid: [${params.type_embeddings.boid.length}]`);

console.log("\n3. Input projections:");
console.log(`ctx_projection: [${params.ctx_projection.length}, ${params.ctx_projection[0]?.length}]`);
console.log(`predator_projection: [${params.predator_projection.length}, ${params.predator_projection[0]?.length}]`);
console.log(`boid_projection: [${params.boid_projection.length}, ${params.boid_projection[0]?.length}]`);

console.log("\n4. Output projection:");
console.log(`output_weight: [${params.output_weight.length}, ${params.output_weight[0]?.length}]`);
console.log(`output_bias: [${params.output_bias.length}]`);

console.log("\n5. First layer details:");
const layer0 = params.layers[0];
console.log(`ln_scale: [${layer0.ln_scale.length}]`);
console.log(`ln_bias: [${layer0.ln_bias.length}]`);
console.log(`qkv_weight: [${layer0.qkv_weight.length}, ${layer0.qkv_weight[0]?.length}]`);
console.log(`qkv_bias: [${layer0.qkv_bias.length}]`);
console.log(`attn_out_weight: [${layer0.attn_out_weight.length}, ${layer0.attn_out_weight[0]?.length}]`);
console.log(`attn_out_bias: [${layer0.attn_out_bias.length}]`);
console.log(`ffn_ln_scale: [${layer0.ffn_ln_scale.length}]`);
console.log(`ffn_ln_bias: [${layer0.ffn_ln_bias.length}]`);
console.log(`ffn_gate_weight: [${layer0.ffn_gate_weight.length}, ${layer0.ffn_gate_weight[0]?.length}]`);
console.log(`ffn_gate_bias: [${layer0.ffn_gate_bias.length}]`);
console.log(`ffn_up_weight: [${layer0.ffn_up_weight.length}, ${layer0.ffn_up_weight[0]?.length}]`);
console.log(`ffn_up_bias: [${layer0.ffn_up_bias.length}]`);
console.log(`ffn_down_weight: [${layer0.ffn_down_weight.length}, ${layer0.ffn_down_weight[0]?.length}]`);
console.log(`ffn_down_bias: [${layer0.ffn_down_bias.length}]`);

console.log("\n6. Expected vs Actual:");
console.log("Expected input projection shapes:");
console.log("- ctx_projection: [2, 48] -> maps 2D context to 48D");
console.log("- predator_projection: [4, 48] -> maps 4D predator to 48D");  
console.log("- boid_projection: [4, 48] -> maps 4D boid to 48D");

console.log("\nExpected layer shapes:");
console.log("- qkv_weight: [48, 144] -> maps 48D to 3x48D (Q,K,V)");
console.log("- attn_out_weight: [48, 48] -> attention output projection");
console.log("- ffn_gate_weight: [48, 96] -> FFN gate projection");
console.log("- ffn_up_weight: [48, 96] -> FFN up projection");
console.log("- ffn_down_weight: [96, 48] -> FFN down projection");

console.log("\nExpected output shape:");
console.log("- output_weight: [48, 2] -> maps 48D cls token to 2D steering");

console.log("\n7. Checking for dimension mismatches...");
let issues = [];

if (params.ctx_projection.length !== 2 || params.ctx_projection[0].length !== 48) {
    issues.push(`ctx_projection wrong shape: [${params.ctx_projection.length}, ${params.ctx_projection[0]?.length}] != [2, 48]`);
}

if (params.predator_projection.length !== 4 || params.predator_projection[0].length !== 48) {
    issues.push(`predator_projection wrong shape: [${params.predator_projection.length}, ${params.predator_projection[0]?.length}] != [4, 48]`);
}

if (params.boid_projection.length !== 4 || params.boid_projection[0].length !== 48) {
    issues.push(`boid_projection wrong shape: [${params.boid_projection.length}, ${params.boid_projection[0]?.length}] != [4, 48]`);
}

if (layer0.qkv_weight.length !== 48 || layer0.qkv_weight[0].length !== 144) {
    issues.push(`qkv_weight wrong shape: [${layer0.qkv_weight.length}, ${layer0.qkv_weight[0]?.length}] != [48, 144]`);
}

if (params.output_weight.length !== 48 || params.output_weight[0].length !== 2) {
    issues.push(`output_weight wrong shape: [${params.output_weight.length}, ${params.output_weight[0]?.length}] != [48, 2]`);
}

if (issues.length > 0) {
    console.log("\n❌ DIMENSION ISSUES FOUND:");
    issues.forEach(issue => console.log("- " + issue));
} else {
    console.log("\n✅ All dimensions look correct");
}

