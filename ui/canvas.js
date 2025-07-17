/**
 * Canvas Utilities Module
 * 
 * This module provides canvas-related utilities for simulation interfaces.
 * Handles canvas initialization, resizing, and basic operations.
 */

/**
 * Resize canvas to match window dimensions
 * @param {string} canvasId - ID of the canvas element
 */
function resizeCanvas(canvasId) {
    var canvas = document.getElementById(canvasId);
    if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
}

/**
 * Check if canvas is supported by browser
 * @returns {boolean} True if canvas is supported
 */
function isCanvasSupported() {
    var canvas = document.createElement('canvas');
    return !!(canvas.getContext && canvas.getContext('2d'));
}

/**
 * Clear canvas with optional background color
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {string} color - Optional background color
 */
function clearCanvas(ctx, color) {
    if (color) {
        ctx.fillStyle = color;
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    } else {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
}

/**
 * Get canvas dimensions
 * @param {string} canvasId - ID of the canvas element
 * @returns {Object} Canvas dimensions {width, height}
 */
function getCanvasDimensions(canvasId) {
    var canvas = document.getElementById(canvasId);
    if (canvas) {
        return {
            width: canvas.width,
            height: canvas.height
        };
    }
    return { width: 0, height: 0 };
}

/**
 * Set canvas dimensions
 * @param {string} canvasId - ID of the canvas element
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 */
function setCanvasDimensions(canvasId, width, height) {
    var canvas = document.getElementById(canvasId);
    if (canvas) {
        canvas.width = width;
        canvas.height = height;
    }
}

/**
 * Initialize canvas with proper settings
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} options - Canvas initialization options
 * @returns {CanvasRenderingContext2D} Canvas context
 */
function initializeCanvas(canvasId, options) {
    options = options || {};
    
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        throw new Error('Canvas element not found: ' + canvasId);
    }
    
    var ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Canvas context not available');
    }
    
    // Set default canvas size
    if (options.autoResize !== false) {
        resizeCanvas(canvasId);
    }
    
    // Set canvas properties
    if (options.imageSmoothingEnabled !== undefined) {
        ctx.imageSmoothingEnabled = options.imageSmoothingEnabled;
    }
    
    return ctx;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        resizeCanvas: resizeCanvas,
        isCanvasSupported: isCanvasSupported,
        clearCanvas: clearCanvas,
        getCanvasDimensions: getCanvasDimensions,
        setCanvasDimensions: setCanvasDimensions,
        initializeCanvas: initializeCanvas
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.resizeCanvas = resizeCanvas;
    window.isCanvasSupported = isCanvasSupported;
    window.clearCanvas = clearCanvas;
    window.getCanvasDimensions = getCanvasDimensions;
    window.setCanvasDimensions = setCanvasDimensions;
    window.initializeCanvas = initializeCanvas;
} 