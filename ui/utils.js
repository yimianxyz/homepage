/**
 * General Utilities Module
 * 
 * This module provides general utility functions that can be used across
 * different parts of the simulation interfaces.
 */

/**
 * Clamp value between min and max
 * @param {number} value - Value to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Clamped value
 */
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

/**
 * Linear interpolation between two values
 * @param {number} a - Start value
 * @param {number} b - End value
 * @param {number} t - Interpolation factor (0-1)
 * @returns {number} Interpolated value
 */
function lerp(a, b, t) {
    return a + (b - a) * t;
}

/**
 * Convert degrees to radians
 * @param {number} degrees - Degrees
 * @returns {number} Radians
 */
function degreesToRadians(degrees) {
    return degrees * Math.PI / 180;
}

/**
 * Convert radians to degrees
 * @param {number} radians - Radians
 * @returns {number} Degrees
 */
function radiansToDegrees(radians) {
    return radians * 180 / Math.PI;
}

/**
 * Safe console logging with timestamp
 * @param {string} level - Log level ('log', 'warn', 'error')
 * @param {...*} args - Arguments to log
 */
function safeLog(level) {
    if (typeof console !== 'undefined' && console[level]) {
        var timestamp = new Date().toISOString();
        var args = Array.prototype.slice.call(arguments, 1);
        console[level].apply(console, ['[' + timestamp + ']'].concat(args));
    }
}

/**
 * Create a debounced function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    var timeout;
    return function() {
        var context = this;
        var args = arguments;
        var later = function() {
            timeout = null;
            func.apply(context, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Create a throttled function
 * @param {Function} func - Function to throttle
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Throttled function
 */
function throttle(func, wait) {
    var lastTime = 0;
    return function() {
        var now = Date.now();
        if (now - lastTime >= wait) {
            lastTime = now;
            func.apply(this, arguments);
        }
    };
}

/**
 * Generate a random number within a range
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Random number
 */
function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * Generate a random integer within a range
 * @param {number} min - Minimum value (inclusive)
 * @param {number} max - Maximum value (exclusive)
 * @returns {number} Random integer
 */
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

/**
 * Check if a value is within a range
 * @param {number} value - Value to check
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {boolean} True if within range
 */
function inRange(value, min, max) {
    return value >= min && value <= max;
}

/**
 * Map a value from one range to another
 * @param {number} value - Value to map
 * @param {number} fromMin - Source range minimum
 * @param {number} fromMax - Source range maximum
 * @param {number} toMin - Target range minimum
 * @param {number} toMax - Target range maximum
 * @returns {number} Mapped value
 */
function mapRange(value, fromMin, fromMax, toMin, toMax) {
    var fromRange = fromMax - fromMin;
    var toRange = toMax - toMin;
    var normalizedValue = (value - fromMin) / fromRange;
    return toMin + normalizedValue * toRange;
}

/**
 * Format a number with specified decimal places
 * @param {number} value - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number
 */
function formatNumber(value, decimals) {
    decimals = decimals || 0;
    return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

/**
 * Calculate distance between two points
 * @param {Object} point1 - First point {x, y}
 * @param {Object} point2 - Second point {x, y}
 * @returns {number} Distance
 */
function distance(point1, point2) {
    var dx = point2.x - point1.x;
    var dy = point2.y - point1.y;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate angle between two points
 * @param {Object} point1 - First point {x, y}
 * @param {Object} point2 - Second point {x, y}
 * @returns {number} Angle in radians
 */
function angle(point1, point2) {
    return Math.atan2(point2.y - point1.y, point2.x - point1.x);
}

/**
 * Normalize an angle to be between 0 and 2π
 * @param {number} angle - Angle in radians
 * @returns {number} Normalized angle
 */
function normalizeAngle(angle) {
    while (angle < 0) angle += 2 * Math.PI;
    while (angle >= 2 * Math.PI) angle -= 2 * Math.PI;
    return angle;
}

/**
 * Check if two rectangles intersect
 * @param {Object} rect1 - First rectangle {x, y, width, height}
 * @param {Object} rect2 - Second rectangle {x, y, width, height}
 * @returns {boolean} True if intersecting
 */
function rectanglesIntersect(rect1, rect2) {
    return rect1.x < rect2.x + rect2.width &&
           rect1.x + rect1.width > rect2.x &&
           rect1.y < rect2.y + rect2.height &&
           rect1.y + rect1.height > rect2.y;
}

/**
 * Check if a point is inside a rectangle
 * @param {Object} point - Point {x, y}
 * @param {Object} rect - Rectangle {x, y, width, height}
 * @returns {boolean} True if point is inside
 */
function pointInRectangle(point, rect) {
    return point.x >= rect.x &&
           point.x <= rect.x + rect.width &&
           point.y >= rect.y &&
           point.y <= rect.y + rect.height;
}

/**
 * Generate a UUID v4
 * @returns {string} UUID
 */
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0;
        var v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

/**
 * Deep clone an object
 * @param {Object} obj - Object to clone
 * @returns {Object} Cloned object
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    
    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }
    
    if (obj instanceof Array) {
        var copy = [];
        for (var i = 0; i < obj.length; i++) {
            copy[i] = deepClone(obj[i]);
        }
        return copy;
    }
    
    var copy = {};
    for (var key in obj) {
        if (obj.hasOwnProperty(key)) {
            copy[key] = deepClone(obj[key]);
        }
    }
    return copy;
}

/**
 * Check if a value is a number
 * @param {*} value - Value to check
 * @returns {boolean} True if number
 */
function isNumber(value) {
    return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

/**
 * Check if a value is a valid object
 * @param {*} value - Value to check
 * @returns {boolean} True if valid object
 */
function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Convert camelCase to kebab-case
 * @param {string} str - camelCase string
 * @returns {string} kebab-case string
 */
function camelToKebab(str) {
    return str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();
}

/**
 * Convert kebab-case to camelCase
 * @param {string} str - kebab-case string
 * @returns {string} camelCase string
 */
function kebabToCamel(str) {
    return str.replace(/-([a-z])/g, function(match, letter) {
        return letter.toUpperCase();
    });
}

/**
 * Capitalize first letter of a string
 * @param {string} str - String to capitalize
 * @returns {string} Capitalized string
 */
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        clamp: clamp,
        lerp: lerp,
        degreesToRadians: degreesToRadians,
        radiansToDegrees: radiansToDegrees,
        safeLog: safeLog,
        debounce: debounce,
        throttle: throttle,
        randomRange: randomRange,
        randomInt: randomInt,
        inRange: inRange,
        mapRange: mapRange,
        formatNumber: formatNumber,
        distance: distance,
        angle: angle,
        normalizeAngle: normalizeAngle,
        rectanglesIntersect: rectanglesIntersect,
        pointInRectangle: pointInRectangle,
        generateUUID: generateUUID,
        deepClone: deepClone,
        isNumber: isNumber,
        isObject: isObject,
        camelToKebab: camelToKebab,
        kebabToCamel: kebabToCamel,
        capitalize: capitalize
    };
}

// Global registration for browser use
if (typeof window !== 'undefined') {
    window.clamp = clamp;
    window.lerp = lerp;
    window.degreesToRadians = degreesToRadians;
    window.radiansToDegrees = radiansToDegrees;
    window.safeLog = safeLog;
    window.debounce = debounce;
    window.throttle = throttle;
    window.randomRange = randomRange;
    window.randomInt = randomInt;
    window.inRange = inRange;
    window.mapRange = mapRange;
    window.formatNumber = formatNumber;
    window.distance = distance;
    window.angle = angle;
    window.normalizeAngle = normalizeAngle;
    window.rectanglesIntersect = rectanglesIntersect;
    window.pointInRectangle = pointInRectangle;
    window.generateUUID = generateUUID;
    window.deepClone = deepClone;
    window.isNumber = isNumber;
    window.isObject = isObject;
    window.camelToKebab = camelToKebab;
    window.kebabToCamel = kebabToCamel;
    window.capitalize = capitalize;
} 