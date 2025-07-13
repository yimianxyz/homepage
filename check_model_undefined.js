// Check exported model for undefined values
console.log("Checking model for undefined values...");

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

function checkForUndefined(obj, path = '') {
    let undefinedCount = 0;
    
    if (Array.isArray(obj)) {
        for (let i = 0; i < obj.length; i++) {
            if (obj[i] === undefined) {
                console.log(`❌ UNDEFINED at ${path}[${i}]`);
                undefinedCount++;
            } else if (typeof obj[i] === 'object') {
                undefinedCount += checkForUndefined(obj[i], `${path}[${i}]`);
            }
        }
    } else if (typeof obj === 'object' && obj !== null) {
        for (const key in obj) {
            if (obj[key] === undefined) {
                console.log(`❌ UNDEFINED at ${path}.${key}`);
                undefinedCount++;
            } else if (typeof obj[key] === 'object') {
                undefinedCount += checkForUndefined(obj[key], `${path}.${key}`);
            }
        }
    }
    
    return undefinedCount;
}

console.log("Checking all model parameters for undefined values...");
const undefinedCount = checkForUndefined(params, 'TRANSFORMER_PARAMS');

console.log(`\nTotal undefined values found: ${undefinedCount}`);

if (undefinedCount === 0) {
    console.log("✅ No undefined values found in model");
} else {
    console.log("❌ Model contains undefined values!");
}

// Specifically check critical matrices
console.log("\nChecking critical matrices:");

console.log("output_weight shape:", [params.output_weight.length, params.output_weight[0]?.length]);
console.log("output_weight[0] sample:", params.output_weight[0]?.slice(0, 5));

// Check for undefined in output_weight specifically
let outputUndefined = 0;
for (let i = 0; i < params.output_weight.length; i++) {
    for (let j = 0; j < params.output_weight[i].length; j++) {
        if (params.output_weight[i][j] === undefined) {
            console.log(`❌ output_weight[${i}][${j}] is undefined`);
            outputUndefined++;
        }
    }
}

console.log(`output_weight undefined count: ${outputUndefined}`);

