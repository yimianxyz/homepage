/**
 * Base Predator Class
 * 
 * Provides core functionality for predator entities.
 * Extended by NeuralPredator for AI behavior.
 */

var PREDATOR_MAX_SPEED = window.SIMULATION_CONSTANTS.PREDATOR_MAX_SPEED;
var PREDATOR_MAX_FORCE = window.SIMULATION_CONSTANTS.PREDATOR_MAX_FORCE;
var PREDATOR_SIZE = window.SIMULATION_CONSTANTS.PREDATOR_SIZE;

function Predator(x, y, simulation) {
    this.position = new Vector(x, y);
    this.velocity = new Vector(Math.random() * 2 - 1, Math.random() * 2 - 1);
    this.acceleration = new Vector(0, 0);
    this.simulation = simulation;
    this.size = PREDATOR_SIZE;
}

Predator.prototype = {
    
    seek: function(targetPosition) {
        var desiredVector = targetPosition.subtract(this.position);
        desiredVector.iFastSetMagnitude(PREDATOR_MAX_SPEED);
        var steeringVector = desiredVector.subtract(this.velocity);
        steeringVector.iFastLimit(PREDATOR_MAX_FORCE);
        return steeringVector;
    },
    
    bound: function() {
        var BORDER_OFFSET = window.SIMULATION_CONSTANTS.PREDATOR_BORDER_OFFSET;
        
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
    
    checkForPrey: function(boids) {
        var caughtBoids = [];
        var catchRadius = this.size * 0.7;
        
        for (var i = 0; i < boids.length; i++) {
            var distance = this.position.getDistance(boids[i].position);
            if (distance < catchRadius) {
                caughtBoids.push(i);
            }
        }
        
        return caughtBoids;
    },

    update: function(boids) {
        this.velocity.iAdd(this.acceleration);
        this.velocity.iFastLimit(PREDATOR_MAX_SPEED);
        this.position.iAdd(this.velocity);
        
        this.bound();
        this.acceleration.iMultiplyBy(0);
    },
    
    render: function() {
        var ctx = this.simulation.ctx;
        
        // Draw predator as a distinctive but subtle elongated triangle (like main branch)
        var directionVector = this.velocity.normalize().multiplyBy(this.size * 1.2);
        var inverseVector1 = new Vector(-directionVector.y, directionVector.x);
        var inverseVector2 = new Vector(directionVector.y, -directionVector.x);
        inverseVector1 = inverseVector1.divideBy(4);
        inverseVector2 = inverseVector2.divideBy(4);
        
        ctx.beginPath();
        ctx.moveTo(this.position.x, this.position.y);
        ctx.lineTo(this.position.x + inverseVector1.x, this.position.y + inverseVector1.y);
        ctx.lineTo(this.position.x + directionVector.x, this.position.y + directionVector.y);
        ctx.lineTo(this.position.x + inverseVector2.x, this.position.y + inverseVector2.y);
        ctx.lineTo(this.position.x, this.position.y);
        
        // Subtle but distinguishable dark red coloring (like main branch)
        ctx.strokeStyle = 'rgba(80, 30, 30, 0.7)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = 'rgba(120, 40, 40, 0.4)';
        ctx.fill();
        
        // Add a subtle inner highlight
        ctx.beginPath();
        ctx.arc(this.position.x, this.position.y, Math.max(2, this.size * 0.15), 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(160, 60, 60, 0.3)';
        ctx.fill();
    }
}; 