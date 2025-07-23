/**
 * State Manager - Simple state management for simulation
 * 
 * This provides a simple interface that bridges between policy and runtime,
 * handling input/output conversion and state management. 
 * This MUST match exactly with the Python implementation.
 */

/**
 * Simple state manager that bridges policy and runtime
 * @constructor
 */
function StateManager() {
    // Initialize processors
    this.inputProcessor = new InputProcessor();
    this.actionProcessor = new ActionProcessor();
    
    // State variables
    this.currentState = null;
    this.policy = null;
}

/**
 * Initialize state manager with initial state and policy
 * 
 * @param {Object} initialState - Initial state dict with structure:
 *   {
 *       boids_states: [
 *           {
 *               position: {x: float, y: float},
 *               velocity: {x: float, y: float}
 *           },
 *           ...
 *       ],
 *       predator_state: {
 *           position: {x: float, y: float},
 *           velocity: {x: float, y: float}
 *       },
 *       canvas_width: float,
 *       canvas_height: float
 *   }
 * @param {Object} policy - Policy object with getAction method
 */
StateManager.prototype.init = function(initialState, policy) {
    // Validate input structure
    var requiredKeys = ['boids_states', 'predator_state', 'canvas_width', 'canvas_height'];
    for (var i = 0; i < requiredKeys.length; i++) {
        if (!(requiredKeys[i] in initialState)) {
            throw new Error("Missing required key: " + requiredKeys[i]);
        }
    }
    
    // Validate boids_states structure
    if (!Array.isArray(initialState.boids_states)) {
        throw new Error("boids_states must be an array");
    }
    
    for (var i = 0; i < initialState.boids_states.length; i++) {
        var boidState = initialState.boids_states[i];
        if (typeof boidState !== 'object' || boidState === null) {
            throw new Error("boids_states[" + i + "] must be an object");
        }
        if (!('position' in boidState) || !('velocity' in boidState)) {
            throw new Error("boids_states[" + i + "] must have 'position' and 'velocity' keys");
        }
    }
    
    // Validate predator_state structure
    if (typeof initialState.predator_state !== 'object' || initialState.predator_state === null) {
        throw new Error("predator_state must be an object");
    }
    if (!('position' in initialState.predator_state) || !('velocity' in initialState.predator_state)) {
        throw new Error("predator_state must have 'position' and 'velocity' keys");
    }
    
    // Deep copy the initial state to avoid mutation
    this.currentState = this._deepCopy(initialState);
    
    // Store policy
    this.policy = policy;
};

/**
 * Run one simulation step using policy and return updated state
 * 
 * @returns {Object} Updated state dict with same structure as input
 */
StateManager.prototype.step = function() {
    if (this.currentState === null || this.policy === null) {
        throw new Error("State manager not initialized. Call init() first.");
    }
    
    // Convert current state to structured format for policy
    var structuredInputs = this._convertStateToStructuredInputs(this.currentState);
    
    // Get policy action
    var policyOutputs = this.policy.getAction(structuredInputs);
    
    // Convert policy outputs to game actions
    var actions = this.actionProcessor.processAction(policyOutputs);
    var predatorAction = {
        force_x: actions[0],
        force_y: actions[1]
    };
    
    // Run simulation step
    var stepResult = simulationStep(
        this.currentState.boids_states,
        this.currentState.predator_state,
        predatorAction,
        this.currentState.canvas_width,
        this.currentState.canvas_height
    );
    
    // Remove caught boids from the state
    var caughtBoidsIndices = stepResult.caught_boids;
    var newBoidsStates = stepResult.boids_states;
    
    // Convert caught boid indices to IDs before removing them
    var caughtBoidIds = [];
    for (var i = 0; i < caughtBoidsIndices.length; i++) {
        var boidIndex = caughtBoidsIndices[i];
        if (boidIndex < newBoidsStates.length) {
            caughtBoidIds.push(newBoidsStates[boidIndex].id);
        }
    }
    
    // Remove caught boids in reverse order to maintain indices
    for (var i = caughtBoidsIndices.length - 1; i >= 0; i--) {
        newBoidsStates.splice(caughtBoidsIndices[i], 1);
    }
    
    // Update current state
    this.currentState = {
        boids_states: newBoidsStates,
        predator_state: stepResult.predator_state,
        canvas_width: this.currentState.canvas_width,
        canvas_height: this.currentState.canvas_height
    };
    
    // Return state with caught boid IDs (not indices)
    var result = this.getState();
    result.caught_boids = caughtBoidIds;  // Add caught boid IDs to the result
    return result;
};

/**
 * Get current state without running a step
 * 
 * @returns {Object} Current state dict (deep copy to prevent mutation)
 */
StateManager.prototype.getState = function() {
    if (this.currentState === null) {
        throw new Error("State manager not initialized. Call init() first.");
    }
    
    return this._deepCopy(this.currentState);
};

/**
 * Convert raw state to structured inputs for policy
 * 
 * @param {Object} state - Raw state dict
 * @returns {Object} Structured inputs dict for policy
 * @private
 */
StateManager.prototype._convertStateToStructuredInputs = function(state) {
    // Convert boids states to the format expected by InputProcessor
    var boidsForProcessor = [];
    for (var i = 0; i < state.boids_states.length; i++) {
        boidsForProcessor.push({
            id: state.boids_states[i].id,  // Include boid ID
            position: state.boids_states[i].position,
            velocity: state.boids_states[i].velocity
        });
    }
    
    // Use InputProcessor to convert to structured format
    var structuredInputs = this.inputProcessor.processInputs(
        boidsForProcessor,
        state.predator_state.position,
        state.predator_state.velocity,
        state.canvas_width,
        state.canvas_height
    );
    
    return structuredInputs;
};

/**
 * Deep copy object to prevent mutation
 * 
 * @param {Object} obj - Object to copy
 * @returns {Object} Deep copy of object
 * @private
 */
StateManager.prototype._deepCopy = function(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    
    if (obj instanceof Array) {
        var copy = [];
        for (var i = 0; i < obj.length; i++) {
            copy[i] = this._deepCopy(obj[i]);
        }
        return copy;
    }
    
    var copy = {};
    for (var key in obj) {
        if (obj.hasOwnProperty(key)) {
            copy[key] = this._deepCopy(obj[key]);
        }
    }
    return copy;
}; 