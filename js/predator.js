// Predator settings optimized for subtle background presence
var PREDATOR_MAX_SPEED = 2.5;
var PREDATOR_MAX_FORCE = 0.05;
var PREDATOR_SIZE = 12;

function Predator(x, y, simulation) {
    this.position = new Vector(x, y);
    this.velocity = new Vector(simRandom() * 2 - 1, simRandom() * 2 - 1); // Start with some velocity
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

    // Autonomous movement. The steering policy is the trained neural
    // network in window.__predatorModel; only patrol-target bookkeeping
    // remains in plain JS. The simulation is gated on the weights being
    // loaded, so window.__predatorModel is always present here.
    //
    // Patrol target = centroid of the live boid swarm. The predator
    // heads toward where the boids actually are when none are in
    // hunting range, instead of wandering toward random canvas points.
    // Measured +39% catches/eval on the dev harness (z=3.55, held out
    // on a second seed set at z=3.59) — see dev/reports/autotarget_*.
    getAutonomousForce: function(boids) {
        var R = POLICY_R;
        var anyInRange = false;
        for (var i = 0; i < boids.length; i++) {
            if (this.position.getDistance(boids[i].position) < R) {
                anyInRange = true;
                break;
            }
        }
        if (!anyInRange && boids.length > 0) {
            // weighted_predicted patrol: density-weighted centroid + a short
            // lookahead of the density-weighted mean velocity. Pulls toward the
            // nearest dense cluster and anticipates where it's heading.
            // +1.77 catches over the plain flock centroid (JS 256-seed, z=3.02);
            // see dev/reports/weighted_predicted_patrol.md.
            var LOOKAHEAD = 5; // frames
            var wsum = 0, sx = 0, sy = 0, svx = 0, svy = 0;
            for (var i = 0; i < boids.length; i++) {
                var dx = boids[i].position.x - this.position.x;
                var dy = boids[i].position.y - this.position.y;
                var w = 1 / Math.sqrt(dx * dx + dy * dy + 1);
                wsum += w;
                sx += boids[i].position.x * w;
                sy += boids[i].position.y * w;
                svx += boids[i].velocity.x * w;
                svy += boids[i].velocity.y * w;
            }
            if (wsum > 0) {
                this.autonomousTarget.x = sx / wsum + LOOKAHEAD * svx / wsum;
                this.autonomousTarget.y = sy / wsum + LOOKAHEAD * svy / wsum;
            }
        }

        var features = buildPredatorFeatures(this.position, this.velocity, boids, this.autonomousTarget);
        var out = window.__predatorModel.forward(features);
        return new Vector(out[0], out[1]);
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
        var currentTime = simNow();
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
        this.lastFeedTime = simNow();
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

// Load the trained NN weights at page boot. The simulation start (in
// boids.js) awaits window.__predatorReady before constructing the
// Simulation, so the NN is guaranteed to be present from frame 1.
window.__predatorReady = fetch('js/predator_weights.json', { cache: 'no-cache' })
    .then(function (r) {
        if (!r.ok) throw new Error('predator weights fetch failed: ' + r.status);
        return r.json();
    })
    .then(function (json) {
        window.__predatorModel = PredatorNN.loadModel(json);
    });
