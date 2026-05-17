// Predator settings optimized for subtle background presence
var PREDATOR_MAX_SPEED = 2.5;
var PREDATOR_MAX_FORCE = 0.05;
var PREDATOR_TURN_FACTOR = 0.3;
var PREDATOR_SIZE = 12;

// Dynamic predator range based on device capability
function getPredatorRange() {
    // Default values if not defined
    var mobileRange = (typeof PREDATOR_MOBILE_RANGE !== 'undefined') ? PREDATOR_MOBILE_RANGE : 60;
    var desktopRange = (typeof PREDATOR_DESKTOP_RANGE !== 'undefined') ? PREDATOR_DESKTOP_RANGE : 80;
    
    return (typeof isMobileDevice === 'function' && isMobileDevice()) ? 
           mobileRange : desktopRange;
}

var PREDATOR_RANGE = getPredatorRange();

function Predator(x, y, simulation) {
    this.position = new Vector(x, y);
    this.velocity = new Vector(Math.random() * 2 - 1, Math.random() * 2 - 1); // Start with some velocity
    this.acceleration = new Vector(0, 0);
    this.simulation = simulation;
    this.autonomousTarget = new Vector(x, y);
    this.targetChangeTime = 0;
    this.targetChangeInterval = 5000; // Change target every 5 seconds for smoother, less erratic movement
    
    // Growth and feeding mechanics
    this.baseSize = PREDATOR_SIZE;
    this.currentSize = this.baseSize;
    this.maxSize = this.baseSize * 1.8; // Maximum growth
    this.growthPerFeed = 1.2; // How much to grow per boid eaten
    this.decayRate = 0.002; // How fast size decays back to normal
    this.lastFeedTime = 0;
    this.feedCooldown = 100; // Minimum time between feeding (ms)
}

Predator.prototype = {
    
    seek: function(targetPosition) {
        var desiredVector = targetPosition.subtract(this.position);
        desiredVector.iFastSetMagnitude(PREDATOR_MAX_SPEED);
        var steeringVector = desiredVector.subtract(this.velocity);
        steeringVector.iFastLimit(PREDATOR_MAX_FORCE);
        return steeringVector;
    },
    
    // Autonomous movement - hunt boids or patrol randomly
    getAutonomousForce: function(boids) {
        var currentTime = Date.now();
        
        // Find nearest boid within hunting range
        var nearestBoid = null;
        var nearestDistance = Infinity;
        
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < PREDATOR_RANGE && distance < nearestDistance) {
                nearestDistance = distance;
                nearestBoid = boids[i];
            }
        }
        
        // If boid found, hunt it
        if (nearestBoid) {
            return this.seek(nearestBoid.position);
        }
        
        // Otherwise, patrol autonomously
        if (currentTime - this.targetChangeTime > this.targetChangeInterval) {
            // Generate new random target within canvas bounds
            var margin = 50;
            this.autonomousTarget.x = margin + Math.random() * (this.simulation.canvasWidth - 2 * margin);
            this.autonomousTarget.y = margin + Math.random() * (this.simulation.canvasHeight - 2 * margin);
            this.targetChangeTime = currentTime;
        }
        
        // Move towards autonomous target
        var distanceToTarget = this.position.getDistance(this.autonomousTarget);
        if (distanceToTarget < 30) {
            // Close enough to target, generate new one
            this.targetChangeTime = 0;
        }
        
        return this.seek(this.autonomousTarget);
    },
    
    // Wrap-around boundary handling (similar to boids)
    bound: function() {
        var BORDER_OFFSET = 20;
        
        if (this.position.x > this.simulation.canvasWidth + BORDER_OFFSET) {
            this.position.x = -BORDER_OFFSET;
        }
        if (this.position.x < -BORDER_OFFSET) {
            this.position.x = this.simulation.canvasWidth + BORDER_OFFSET;
        }
        if (this.position.y > this.simulation.canvasHeight + BORDER_OFFSET) {
            this.position.y = -BORDER_OFFSET;
        }
        if (this.position.y < -BORDER_OFFSET) {
            this.position.y = this.simulation.canvasHeight + BORDER_OFFSET;
        }
    },
    
    // Check for boid collisions and handle feeding
    checkForPrey: function(boids) {
        var currentTime = Date.now();
        if (currentTime - this.lastFeedTime < this.feedCooldown) {
            return []; // Still digesting
        }
        
        var caughtBoids = [];
        var catchRadius = this.currentSize * 0.7; // Catch radius scales with size
        
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < catchRadius) {
                caughtBoids.push(i);
                this.feed();
                break; // Only catch one boid at a time for smooth animation
            }
        }
        
        return caughtBoids;
    },
    
    // Handle feeding - grow predator
    feed: function() {
        this.currentSize = Math.min(this.currentSize + this.growthPerFeed, this.maxSize);
        this.lastFeedTime = Date.now();
    },
    
    // Gradually decay size back to normal
    decaySize: function() {
        if (this.currentSize > this.baseSize) {
            this.currentSize = Math.max(this.currentSize - this.decayRate, this.baseSize);
        }
    },

    update: function(boids) {
        // Reset acceleration
        this.acceleration.iMultiplyBy(0);
        
        // Apply autonomous steering force
        var autonomousForce = this.getAutonomousForce(boids);
        this.acceleration.iAdd(autonomousForce);
        
        // Update velocity and position
        this.velocity.iAdd(this.acceleration);
        this.velocity.iFastLimit(PREDATOR_MAX_SPEED);
        this.position.iAdd(this.velocity);
        
        // Handle boundaries
        this.bound();
        
        // Handle size decay
        this.decaySize();
    },
    
    render: function() {
        var ctx = this.simulation.ctx;
        
        // Draw predator as a distinctive but subtle elongated triangle (size scales with feeding)
        var directionVector = this.velocity.normalize().multiplyBy(this.currentSize * 1.2);
        var inverseVector1 = new Vector(-directionVector.y, directionVector.x);
        var inverseVector2 = new Vector(directionVector.y, -directionVector.x);
        inverseVector1 = inverseVector1.divideBy(4);
        inverseVector2 = inverseVector2.divideBy(4);
        
        // Subtle intensity based on size (well-fed predator is slightly more visible)
        var sizeRatio = this.currentSize / this.baseSize;
        var intensity = 0.3 + (sizeRatio - 1) * 0.2; // Subtle intensity increase when fed
        
        ctx.beginPath();
        ctx.moveTo(this.position.x, this.position.y);
        ctx.lineTo(this.position.x + inverseVector1.x, this.position.y + inverseVector1.y);
        ctx.lineTo(this.position.x + directionVector.x, this.position.y + directionVector.y);
        ctx.lineTo(this.position.x + inverseVector2.x, this.position.y + inverseVector2.y);
        ctx.lineTo(this.position.x, this.position.y);
        
        // Subtle but distinguishable dark red coloring that intensifies when fed
        ctx.strokeStyle = 'rgba(80, 30, 30, ' + (0.7 + intensity * 0.3) + ')';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = 'rgba(120, 40, 40, ' + (0.4 + intensity * 0.2) + ')';
        ctx.fill();
        
        // Add a subtle inner highlight that scales with size
        ctx.beginPath();
        ctx.arc(this.position.x, this.position.y, Math.max(2, this.currentSize * 0.15), 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(140, 60, 60, ' + (0.6 + intensity * 0.2) + ')';
        ctx.fill();
    }
}; 