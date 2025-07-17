/**
 * Rendering Utilities Module
 * 
 * This module provides rendering functions for boids, predators, and debug visualizations.
 * Handles all visual representation of simulation entities.
 */

/**
 * Render a boid on the canvas
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} boidState - Boid state with position and velocity
 * @param {number} renderSize - Size of the boid
 * @param {string} strokeColor - Stroke color (optional)
 * @param {string} fillColor - Fill color (optional)
 */
function renderBoid(ctx, boidState, renderSize, strokeColor, fillColor) {
    var position = boidState.position;
    var velocity = boidState.velocity;
    
    // Calculate direction vector
    var magnitude = Math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    if (magnitude === 0) return;
    
    var directionX = (velocity.x / magnitude) * renderSize;
    var directionY = (velocity.y / magnitude) * renderSize;
    
    // Calculate perpendicular vectors for triangle
    var inverse1X = -directionY / 3;
    var inverse1Y = directionX / 3;
    var inverse2X = directionY / 3;
    var inverse2Y = -directionX / 3;

    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
    ctx.lineTo(position.x + inverse1X, position.y + inverse1Y);
    ctx.lineTo(position.x + directionX, position.y + directionY);
    ctx.lineTo(position.x + inverse2X, position.y + inverse2Y);
    ctx.lineTo(position.x, position.y);
    
    ctx.strokeStyle = strokeColor || 'rgba(0, 0, 0, 0.3)';
    ctx.stroke();
    ctx.fillStyle = fillColor || 'rgba(0, 0, 0, 0.3)';
    ctx.fill();
}

/**
 * Render a predator on the canvas
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} predatorState - Predator state with position and velocity
 * @param {number} predatorSize - Size of the predator
 * @param {string} strokeColor - Stroke color (optional)
 * @param {string} fillColor - Fill color (optional)
 */
function renderPredator(ctx, predatorState, predatorSize, strokeColor, fillColor) {
    var position = predatorState.position;
    var velocity = predatorState.velocity;
    
    // Calculate direction vector
    var magnitude = Math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    if (magnitude === 0) return;
    
    var directionX = (velocity.x / magnitude) * predatorSize * 1.2;
    var directionY = (velocity.y / magnitude) * predatorSize * 1.2;
    
    // Calculate perpendicular vectors for elongated triangle
    var inverse1X = -directionY / 4;
    var inverse1Y = directionX / 4;
    var inverse2X = directionY / 4;
    var inverse2Y = -directionX / 4;
    
    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
    ctx.lineTo(position.x + inverse1X, position.y + inverse1Y);
    ctx.lineTo(position.x + directionX, position.y + directionY);
    ctx.lineTo(position.x + inverse2X, position.y + inverse2Y);
    ctx.lineTo(position.x, position.y);
    
    // Distinctive dark red coloring
    ctx.strokeStyle = strokeColor || 'rgba(80, 30, 30, 0.7)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = fillColor || 'rgba(120, 40, 40, 0.4)';
    ctx.fill();
    
    // Add inner highlight
    ctx.beginPath();
    ctx.arc(position.x, position.y, Math.max(2, predatorSize * 0.15), 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(160, 60, 60, 0.3)';
    ctx.fill();
}

/**
 * Render debug information for a boid
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} boidState - Boid state with position and velocity
 * @param {number} index - Boid index
 * @param {boolean} showVelocity - Whether to show velocity vector
 * @param {boolean} showIndex - Whether to show boid index
 */
function renderBoidDebug(ctx, boidState, index, showVelocity, showIndex) {
    var position = boidState.position;
    var velocity = boidState.velocity;
    
    // Show velocity vector
    if (showVelocity) {
        ctx.beginPath();
        ctx.moveTo(position.x, position.y);
        ctx.lineTo(position.x + velocity.x * 10, position.y + velocity.y * 10);
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    // Show boid index
    if (showIndex) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(index.toString(), position.x, position.y - 10);
    }
}

/**
 * Render debug information for predator
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} predatorState - Predator state with position and velocity
 * @param {boolean} showVelocity - Whether to show velocity vector
 * @param {boolean} showRange - Whether to show predator range
 */
function renderPredatorDebug(ctx, predatorState, showVelocity, showRange) {
    var position = predatorState.position;
    var velocity = predatorState.velocity;
    
    // Show velocity vector
    if (showVelocity) {
        ctx.beginPath();
        ctx.moveTo(position.x, position.y);
        ctx.lineTo(position.x + velocity.x * 20, position.y + velocity.y * 20);
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Show predator range
    if (showRange) {
        ctx.beginPath();
        ctx.arc(position.x, position.y, window.SIMULATION_CONSTANTS.PREDATOR_RANGE, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

/**
 * Render neighbor connections for a boid
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} boidState - Boid state with position and velocity
 * @param {Array} neighbors - Array of neighbor boids
 * @param {string} lineColor - Connection line color
 */
function renderBoidNeighbors(ctx, boidState, neighbors, lineColor) {
    var position = boidState.position;
    
    ctx.strokeStyle = lineColor || 'rgba(0, 0, 255, 0.2)';
    ctx.lineWidth = 1;
    
    for (var i = 0; i < neighbors.length; i++) {
        var neighbor = neighbors[i];
        ctx.beginPath();
        ctx.moveTo(position.x, position.y);
        ctx.lineTo(neighbor.position.x, neighbor.position.y);
        ctx.stroke();
    }
}

/**
 * Render force vectors for debugging
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} position - Position to render from
 * @param {Object} force - Force vector
 * @param {string} color - Force vector color
 * @param {number} scale - Scale factor for visualization
 */
function renderForceVector(ctx, position, force, color, scale) {
    scale = scale || 50;
    
    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
    ctx.lineTo(position.x + force.x * scale, position.y + force.y * scale);
    ctx.strokeStyle = color || 'rgba(255, 255, 0, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Add arrow head
    var angle = Math.atan2(force.y, force.x);
    var headLength = 8;
    
    ctx.beginPath();
    ctx.moveTo(position.x + force.x * scale, position.y + force.y * scale);
    ctx.lineTo(
        position.x + force.x * scale - headLength * Math.cos(angle - Math.PI / 6),
        position.y + force.y * scale - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.moveTo(position.x + force.x * scale, position.y + force.y * scale);
    ctx.lineTo(
        position.x + force.x * scale - headLength * Math.cos(angle + Math.PI / 6),
        position.y + force.y * scale - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.stroke();
}

/**
 * Render spatial grid for debugging
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} cellSize - Size of grid cells
 * @param {string} color - Grid color
 */
function renderSpatialGrid(ctx, cellSize, color) {
    var canvasWidth = ctx.canvas.width;
    var canvasHeight = ctx.canvas.height;
    
    ctx.strokeStyle = color || 'rgba(128, 128, 128, 0.3)';
    ctx.lineWidth = 1;
    
    // Vertical lines
    for (var x = 0; x <= canvasWidth; x += cellSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
        ctx.stroke();
    }
    
    // Horizontal lines
    for (var y = 0; y <= canvasHeight; y += cellSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        renderBoid: renderBoid,
        renderPredator: renderPredator,
        renderBoidDebug: renderBoidDebug,
        renderPredatorDebug: renderPredatorDebug,
        renderBoidNeighbors: renderBoidNeighbors,
        renderForceVector: renderForceVector,
        renderSpatialGrid: renderSpatialGrid
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.renderBoid = renderBoid;
    window.renderPredator = renderPredator;
    window.renderBoidDebug = renderBoidDebug;
    window.renderPredatorDebug = renderPredatorDebug;
    window.renderBoidNeighbors = renderBoidNeighbors;
    window.renderForceVector = renderForceVector;
    window.renderSpatialGrid = renderSpatialGrid;
} 