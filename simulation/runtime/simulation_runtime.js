/**
 * Pure Simulation Runtime - Core simulation logic abstracted for exact JS/Python matching
 * 
 * This module contains the pure simulation runtime that takes current states and actions,
 * and returns the next states. It's designed to be identical line-by-line between
 * JavaScript and Python implementations.
 * 
 * Interface:
 * - Input: boids_states, predator_state, predator_action, canvas_width, canvas_height
 * - Output: {boids_states, predator_state, caught_boids}
 */

// Constants imported from centralized config/constants.js

// === VECTOR MATH UTILITIES ===

function vectorAdd(v1, v2) {
    // Add two vectors
    return {x: v1.x + v2.x, y: v1.y + v2.y};
}

function vectorSubtract(v1, v2) {
    // Subtract two vectors
    return {x: v1.x - v2.x, y: v1.y - v2.y};
}

function vectorMultiply(v, scalar) {
    // Multiply vector by scalar
    return {x: v.x * scalar, y: v.y * scalar};
}

function vectorDivide(v, scalar) {
    // Divide vector by scalar
    return {x: v.x / scalar, y: v.y / scalar};
}

function vectorDistance(v1, v2) {
    // Calculate distance between two vectors
    var dx = v1.x - v2.x;
    var dy = v1.y - v2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function vectorFastMagnitude(v) {
    // Fast magnitude approximation - matches Python exactly
    var alpha = 0.96;
    var beta = 0.398;
    var absX = Math.abs(v.x);
    var absY = Math.abs(v.y);
    return Math.max(absX, absY) * alpha + Math.min(absX, absY) * beta;
}

function vectorNormalize(v) {
    // Normalize vector using exact magnitude
    var magnitude = vectorDistance(v, {x: 0, y: 0});
    if (magnitude > 0) {
        return vectorDivide(v, magnitude);
    } else {
        return {x: 0, y: 0};
    }
}

function vectorFastNormalize(v) {
    // Fast normalize using fast magnitude
    var fastMag = vectorFastMagnitude(v);
    if (fastMag > 0) {
        return vectorDivide(v, fastMag);
    } else {
        return {x: 0, y: 0};
    }
}

function vectorFastSetMagnitude(v, magnitude) {
    // Set magnitude using fast approximation
    var fastMag = vectorFastMagnitude(v);
    if (fastMag > 0) {
        var scale = magnitude / fastMag;
        return vectorMultiply(v, scale);
    } else {
        return {x: 0, y: 0};
    }
}

function vectorFastLimit(v, maxMagnitude) {
    // Limit magnitude using fast approximation
    var fastMag = vectorFastMagnitude(v);
    if (fastMag > maxMagnitude) {
        var scale = maxMagnitude / fastMag;
        return vectorMultiply(v, scale);
    } else {
        return v;
    }
}

// === BOID BEHAVIOR FUNCTIONS ===

function boidGetCohesionVector(boidIndex, boidsStates) {
    // Calculate cohesion force - move towards average position of neighbors
    var currentBoid = boidsStates[boidIndex];
    var totalPosition = {x: 0, y: 0};
    var neighborCount = 0;
    
    for (var i = 0; i < boidsStates.length; i++) {
        if (i === boidIndex) {
            continue;
        }
        
        var distance = vectorDistance(currentBoid.position, boidsStates[i].position) + window.SIMULATION_CONSTANTS.EPSILON;
        if (distance <= window.SIMULATION_CONSTANTS.BOID_NEIGHBOR_DISTANCE) {
            totalPosition = vectorAdd(totalPosition, boidsStates[i].position);
            neighborCount++;
        }
    }
    
    if (neighborCount > 0) {
        var averagePosition = vectorDivide(totalPosition, neighborCount);
        return boidSeek(currentBoid, averagePosition);
    } else {
        return {x: 0, y: 0};
    }
}

function boidSeek(boidState, targetPosition) {
    // Seek towards target position
    var desiredVector = vectorSubtract(targetPosition, boidState.position);
    desiredVector = vectorFastSetMagnitude(desiredVector, window.SIMULATION_CONSTANTS.BOID_MAX_SPEED);
    var steeringVector = vectorSubtract(desiredVector, boidState.velocity);
    steeringVector = vectorFastLimit(steeringVector, window.SIMULATION_CONSTANTS.BOID_MAX_FORCE);
    return steeringVector;
}

function boidGetSeparationVector(boidIndex, boidsStates) {
    // Calculate separation force - avoid crowding neighbors
    var currentBoid = boidsStates[boidIndex];
    var steeringVector = {x: 0, y: 0};
    var neighborCount = 0;
    
    for (var i = 0; i < boidsStates.length; i++) {
        if (i === boidIndex) {
            continue;
        }
        
        var distance = vectorDistance(currentBoid.position, boidsStates[i].position) + window.SIMULATION_CONSTANTS.EPSILON;
        if (distance > 0 && distance < window.SIMULATION_CONSTANTS.BOID_DESIRED_SEPARATION) {
            var deltaVector = vectorSubtract(currentBoid.position, boidsStates[i].position);
            deltaVector = vectorNormalize(deltaVector);
            deltaVector = vectorDivide(deltaVector, distance);
            steeringVector = vectorAdd(steeringVector, deltaVector);
            neighborCount++;
        }
    }
    
    if (neighborCount > 0) {
        var averageSteeringVector = vectorDivide(steeringVector, neighborCount);
        averageSteeringVector = vectorFastSetMagnitude(averageSteeringVector, window.SIMULATION_CONSTANTS.BOID_MAX_SPEED);
        averageSteeringVector = vectorSubtract(averageSteeringVector, currentBoid.velocity);
        averageSteeringVector = vectorFastLimit(averageSteeringVector, window.SIMULATION_CONSTANTS.BOID_MAX_FORCE);
        return averageSteeringVector;
    } else {
        return {x: 0, y: 0};
    }
}

function boidGetAlignmentVector(boidIndex, boidsStates) {
    // Calculate alignment force - steer towards average heading of neighbors
    var currentBoid = boidsStates[boidIndex];
    var perceivedFlockVelocity = {x: 0, y: 0};
    var neighborCount = 0;
    
    for (var i = 0; i < boidsStates.length; i++) {
        if (i === boidIndex) {
            continue;
        }
        
        var distance = vectorDistance(currentBoid.position, boidsStates[i].position) + window.SIMULATION_CONSTANTS.EPSILON;
        if (distance > 0 && distance < window.SIMULATION_CONSTANTS.BOID_NEIGHBOR_DISTANCE) {
            perceivedFlockVelocity = vectorAdd(perceivedFlockVelocity, boidsStates[i].velocity);
            neighborCount++;
        }
    }
    
    if (neighborCount > 0) {
        var averageVelocity = vectorDivide(perceivedFlockVelocity, neighborCount);
        averageVelocity = vectorFastSetMagnitude(averageVelocity, window.SIMULATION_CONSTANTS.BOID_MAX_SPEED);
        var steeringVector = vectorSubtract(averageVelocity, currentBoid.velocity);
        steeringVector = vectorFastLimit(steeringVector, window.SIMULATION_CONSTANTS.BOID_MAX_FORCE);
        return steeringVector;
    } else {
        return {x: 0, y: 0};
    }
}

function boidGetPredatorAvoidanceVector(boidState, predatorState) {
    // Calculate predator avoidance force
    var distance = vectorDistance(boidState.position, predatorState.position) + window.SIMULATION_CONSTANTS.EPSILON;
    if (distance > 0 && distance < window.SIMULATION_CONSTANTS.PREDATOR_RANGE) {
        var avoidanceVector = vectorSubtract(boidState.position, predatorState.position);
        avoidanceVector = vectorFastNormalize(avoidanceVector);
        
        var avoidanceStrength = (window.SIMULATION_CONSTANTS.PREDATOR_RANGE - distance) / window.SIMULATION_CONSTANTS.PREDATOR_RANGE;
        avoidanceVector = vectorMultiply(avoidanceVector, avoidanceStrength * window.SIMULATION_CONSTANTS.PREDATOR_TURN_FACTOR);
        avoidanceVector = vectorFastLimit(avoidanceVector, window.SIMULATION_CONSTANTS.BOID_MAX_FORCE * 1.5);
        
        return avoidanceVector;
    }
    
    return {x: 0, y: 0};
}

function boidBound(position, canvasWidth, canvasHeight) {
    // Handle boundary wrapping for boid
    var result = {x: position.x, y: position.y};
    
    if (result.x > canvasWidth + window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET) {
        result.x = -window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET;
    }
    if (result.x < -window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET) {
        result.x = canvasWidth + window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET;
    }
    if (result.y > canvasHeight + window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET) {
        result.y = -window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET;
    }
    if (result.y < -window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET) {
        result.y = canvasHeight + window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET;
    }
    
    return result;
}

// === PREDATOR BEHAVIOR FUNCTIONS ===

function predatorBound(position, canvasWidth, canvasHeight) {
    // Handle boundary wrapping for predator
    var result = {x: position.x, y: position.y};
    
    if (result.x > canvasWidth + window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET) {
        result.x = -window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET;
    }
    if (result.x < -window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET) {
        result.x = canvasWidth + window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET;
    }
    if (result.y > canvasHeight + window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET) {
        result.y = -window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET;
    }
    if (result.y < -window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET) {
        result.y = canvasHeight + window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET;
    }
    
    return result;
}

function predatorCheckForPrey(predatorState, boidsStates) {
    // Check for caught boids and return their indices
    var caughtBoids = [];
    var catchRadius = window.SIMULATION_CONSTANTS.PREDATOR_SIZE * 0.7;
    
    for (var i = 0; i < boidsStates.length; i++) {
        var distance = vectorDistance(predatorState.position, boidsStates[i].position);
        if (distance < catchRadius) {
            caughtBoids.push(i);
        }
    }
    
    return caughtBoids;
}

// === MAIN SIMULATION RUNTIME ===

function simulationStep(boidsStates, predatorState, predatorAction, canvasWidth, canvasHeight) {
    /**
     * Pure simulation runtime - compute next state from current state and action
     * 
     * Args:
     *     boidsStates: Array of boid states [{position: {x, y}, velocity: {x, y}}]
     *     predatorState: Predator state {position: {x, y}, velocity: {x, y}}
     *     predatorAction: Predator action {force_x, force_y}
     *     canvasWidth: Canvas width
     *     canvasHeight: Canvas height
     * 
     * Returns:
     *     {
     *         boids_states: [...],      // Updated boids states
     *         predator_state: {...},    // Updated predator state
     *         caught_boids: [...]       // Indices of caught boids
     *     }
     */
    // Deep copy states to avoid mutation
    var newBoidsStates = [];
    for (var i = 0; i < boidsStates.length; i++) {
        newBoidsStates.push({
            id: boidsStates[i].id,  // Preserve boid ID
            position: {x: boidsStates[i].position.x, y: boidsStates[i].position.y},
            velocity: {x: boidsStates[i].velocity.x, y: boidsStates[i].velocity.y}
        });
    }
    
    var newPredatorState = {
        position: {x: predatorState.position.x, y: predatorState.position.y},
        velocity: {x: predatorState.velocity.x, y: predatorState.velocity.y}
    };
    
    // Step 1: Update all boids
    for (var i = 0; i < newBoidsStates.length; i++) {
        var boidState = newBoidsStates[i];
        
        // Calculate flocking forces
        var cohesionVector = boidGetCohesionVector(i, newBoidsStates);
        var separationVector = boidGetSeparationVector(i, newBoidsStates);
        var alignmentVector = boidGetAlignmentVector(i, newBoidsStates);
        
        // Apply multipliers
        separationVector = vectorMultiply(separationVector, window.SIMULATION_CONSTANTS.SEPARATION_MULTIPLIER);
        cohesionVector = vectorMultiply(cohesionVector, window.SIMULATION_CONSTANTS.COHESION_MULTIPLIER);
        alignmentVector = vectorMultiply(alignmentVector, window.SIMULATION_CONSTANTS.ALIGNMENT_MULTIPLIER);
        
        // Combine forces
        var acceleration = vectorAdd(cohesionVector, separationVector);
        acceleration = vectorAdd(acceleration, alignmentVector);
        
        // Add predator avoidance
        var predatorAvoidance = boidGetPredatorAvoidanceVector(boidState, newPredatorState);
        acceleration = vectorAdd(acceleration, predatorAvoidance);
        
        // Update physics
        boidState.velocity = vectorAdd(boidState.velocity, acceleration);
        boidState.velocity = vectorFastLimit(boidState.velocity, window.SIMULATION_CONSTANTS.BOID_MAX_SPEED);
        boidState.position = vectorAdd(boidState.position, boidState.velocity);
        boidState.position = boidBound(boidState.position, canvasWidth, canvasHeight);
    }
    
    // Step 2: Update predator
    // Apply predator action to acceleration
    var predatorAcceleration = {x: predatorAction.force_x, y: predatorAction.force_y};
    
    // Update predator physics
    newPredatorState.velocity = vectorAdd(newPredatorState.velocity, predatorAcceleration);
    newPredatorState.velocity = vectorFastLimit(newPredatorState.velocity, window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED);
    newPredatorState.position = vectorAdd(newPredatorState.position, newPredatorState.velocity);
    newPredatorState.position = predatorBound(newPredatorState.position, canvasWidth, canvasHeight);
    
    // Step 3: Check for caught boids
    var caughtBoids = predatorCheckForPrey(newPredatorState, newBoidsStates);
    
    return {
        boids_states: newBoidsStates,
        predator_state: newPredatorState,
        caught_boids: caughtBoids
    };
} 