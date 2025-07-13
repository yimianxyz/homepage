// Full transformer test with realistic inputs
console.log("Full transformer test...");

// Mock window and Vector
global.window = {
    SIMULATION_CONSTANTS: {
        MAX_DISTANCE: 800,
        BOID_MAX_SPEED: 3.5,
        PREDATOR_MAX_SPEED: 2,
        PREDATOR_MAX_FORCE: 0.5,
        PREDATOR_FORCE_SCALE: 30
    }
};

function Vector(x, y) {
    this.x = x || 0;
    this.y = y || 0;
}

// Load model
require('./src/config/model.js');

// Mock input processor
function InputProcessor() {
    this.maxDistance = global.window.SIMULATION_CONSTANTS.MAX_DISTANCE;
    this.unifiedMaxVelocity = Math.max(
        global.window.SIMULATION_CONSTANTS.BOID_MAX_SPEED,
        global.window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED
    );
}

InputProcessor.prototype.processInputs = function(boids, predatorPos, predatorVel, canvasWidth, canvasHeight) {
    var context = {
        canvasWidth: canvasWidth / this.maxDistance,
        canvasHeight: canvasHeight / this.maxDistance
    };
    
    var predator = {
        velX: Math.max(-1, Math.min(1, predatorVel.x / this.unifiedMaxVelocity)),
        velY: Math.max(-1, Math.min(1, predatorVel.y / this.unifiedMaxVelocity))
    };
    
    var boidArray = [];
    for (var i = 0; i < boids.length; i++) {
        var boid = boids[i];
        var dx = boid.position.x - predatorPos.x;
        var dy = boid.position.y - predatorPos.y;
        
        // Handle wrapping
        if (Math.abs(dx) > canvasWidth / 2) {
            dx = dx > 0 ? dx - canvasWidth : dx + canvasWidth;
        }
        if (Math.abs(dy) > canvasHeight / 2) {
            dy = dy > 0 ? dy - canvasHeight : dy + canvasHeight;
        }
        
        boidArray.push({
            relX: dx / this.maxDistance,
            relY: dy / this.maxDistance,
            velX: Math.max(-1, Math.min(1, boid.velocity.x / this.unifiedMaxVelocity)),
            velY: Math.max(-1, Math.min(1, boid.velocity.y / this.unifiedMaxVelocity))
        });
    }
    
    return {
        context: context,
        predator: predator,
        boids: boidArray
    };
};

// Mock action processor
function ActionProcessor() {
    this.forceScale = global.window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE * global.window.SIMULATION_CONSTANTS.PREDATOR_FORCE_SCALE;
}

ActionProcessor.prototype.processAction = function(neuralOutputs) {
    return [
        neuralOutputs[0] * this.forceScale,
        neuralOutputs[1] * this.forceScale
    ];
};

// Test with realistic data
const inputProcessor = new InputProcessor();
const actionProcessor = new ActionProcessor();

// Create test scenario
const testBoids = [
    { position: {x: 100, y: 100}, velocity: {x: 1, y: 0} },
    { position: {x: 200, y: 150}, velocity: {x: -1, y: 1} },
    { position: {x: 300, y: 200}, velocity: {x: 0, y: -1} }
];

const predatorPos = {x: 150, y: 150};
const predatorVel = {x: 0.5, y: 0.5};
const canvasWidth = 800;
const canvasHeight = 600;

console.log("Test scenario:");
console.log("- Boids:", testBoids.length);
console.log("- Predator pos:", predatorPos);
console.log("- Predator vel:", predatorVel);

const structuredInputs = inputProcessor.processInputs(testBoids, predatorPos, predatorVel, canvasWidth, canvasHeight);

console.log("\nStructured inputs:");
console.log("- Context:", structuredInputs.context);
console.log("- Predator:", structuredInputs.predator);
console.log("- Boids count:", structuredInputs.boids.length);
console.log("- First boid:", structuredInputs.boids[0]);

// Now I need to implement a minimal transformer forward pass
// Let's just check if the parameters are being loaded correctly first
const params = global.window.TRANSFORMER_PARAMS;
console.log("\nModel architecture validation:");
console.log("- Architecture matches:", params.d_model === 48 && params.n_heads === 4);
console.log("- Has required components:", !!(params.cls_embedding && params.type_embeddings && params.layers && params.output_weight));
console.log("- Layer count:", params.layers.length);

// Check a few key parameter shapes
console.log("\nParameter shapes:");
console.log("- cls_embedding:", params.cls_embedding.length);
console.log("- ctx_projection:", [params.ctx_projection.length, params.ctx_projection[0]?.length]);
console.log("- output_weight:", [params.output_weight.length, params.output_weight[0]?.length]);

// Test if we can do a simple matrix multiplication
function matmul(matrix, vector) {
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

console.log("\nTesting matrix operations:");
try {
    const testInput = [structuredInputs.context.canvasWidth, structuredInputs.context.canvasHeight];
    console.log("- Test input:", testInput);
    const projected = matmul(params.ctx_projection, testInput);
    console.log("- Projection result length:", projected.length);
    console.log("- Projection result range:", [Math.min(...projected), Math.max(...projected)]);
    console.log("✅ Matrix multiplication works");
} catch (error) {
    console.log("❌ Matrix multiplication failed:", error.message);
}

