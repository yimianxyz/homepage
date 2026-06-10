// Predator settings optimized for subtle background presence
var PREDATOR_MAX_SPEED = 2.5;
var PREDATOR_MAX_FORCE = 0.05;
var PREDATOR_SIZE = 12;

// Evolved patrol-target params (the deployed policy). Found by GPU-batched
// evolutionary search (CEM over the sim_torch 'evolved' target, AlphaEvolve
// flavor) — beats the previous nearest_cluster patrol by ~+0.52 catches/eval
// (8.38 vs 7.86, 2048 fresh seeds, ~4 SE) on the GPU sim. See
// dev/reports/evolve_patrol_results.md and dev/sim_torch.py 'evolved' branch
// (this JS is a line-for-line port of that computation).
var EVOLVED_PATROL = {
    cluster_r: 178.09,   // neighbor radius for the density count
    dens_pow: 2.373,     // exponent on (neighbors+1) — how much density attracts
    reach_scale: 1515.0, // exp(-distPred/reach_scale) — slow predator prefers near
    sharp: 9.25,         // peakiness of the softmax-like weighting (->densest boid)
    lead_scale: 0.454,   // adaptive travel-time lead gain
    lead_max: 230.6,     // cap on lead distance
    nbhd: 0.461,         // blend toward the densest boid's neighborhood centroid
    momentum: 0.0        // frame-to-frame target smoothing (0 = off)
};

// Patrol target = the evolved heuristic. For each live boid, attractiveness =
// (localNeighbors+1)^dens_pow * exp(-distToPredator/reach_scale); normalize to
// the per-frame max and raise to `sharp` to get soft weights, then take the
// weighted centroid + mean velocity, blend in the densest boid's neighborhood
// centroid (nbhd), and lead forward by the predator's travel time (capped).
// Mirrors sim_torch's 'evolved' _update_auto_target exactly.
function computeEvolvedTarget(predPos, boids, opt, prevTarget) {
    var n = boids.length;
    if (n === 0) return null;
    var R2 = opt.cluster_r * opt.cluster_r;
    var px = predPos.x, py = predPos.y;
    var attract = new Array(n);
    var amax = 1e-12, bestIdx = 0, bestA = -1;
    for (var i = 0; i < n; i++) {
        var bxi = boids[i].position.x, byi = boids[i].position.y;
        var cnt = 0; // neighbors within cluster_r (includes self)
        for (var j = 0; j < n; j++) {
            var ex = boids[j].position.x - bxi, ey = boids[j].position.y - byi;
            if (ex * ex + ey * ey < R2) cnt++;
        }
        var ddx = bxi - px, ddy = byi - py;
        var dpred = Math.sqrt(ddx * ddx + ddy * ddy);
        var a = Math.pow(cnt + 1, opt.dens_pow) * Math.exp(-dpred / opt.reach_scale);
        attract[i] = a;
        if (a > amax) amax = a;
        if (a > bestA) { bestA = a; bestIdx = i; }
    }
    var wsum = 0, cx0 = 0, cy0 = 0, vx0 = 0, vy0 = 0;
    for (var k = 0; k < n; k++) {
        var w = Math.pow(attract[k] / amax, opt.sharp);
        wsum += w;
        cx0 += w * boids[k].position.x; cy0 += w * boids[k].position.y;
        vx0 += w * boids[k].velocity.x; vy0 += w * boids[k].velocity.y;
    }
    if (wsum < 1e-12) wsum = 1e-12;
    cx0 /= wsum; cy0 /= wsum; vx0 /= wsum; vy0 /= wsum;
    var nbhd = opt.nbhd || 0;
    if (nbhd > 0) {
        var bx = boids[bestIdx].position.x, by = boids[bestIdx].position.y;
        var nsum = 0, ncx = 0, ncy = 0, nvx = 0, nvy = 0;
        for (var m = 0; m < n; m++) {
            var gx = boids[m].position.x - bx, gy = boids[m].position.y - by;
            if (gx * gx + gy * gy < R2) {
                ncx += boids[m].position.x; ncy += boids[m].position.y;
                nvx += boids[m].velocity.x; nvy += boids[m].velocity.y; nsum++;
            }
        }
        if (nsum < 1e-12) nsum = 1e-12;
        ncx /= nsum; ncy /= nsum; nvx /= nsum; nvy /= nsum;
        cx0 = (1 - nbhd) * cx0 + nbhd * ncx;
        cy0 = (1 - nbhd) * cy0 + nbhd * ncy;
        vx0 = (1 - nbhd) * vx0 + nbhd * nvx;
        vy0 = (1 - nbhd) * vy0 + nbhd * nvy;
    }
    var dx2 = cx0 - px, dy2 = cy0 - py;
    var dcent = Math.sqrt(dx2 * dx2 + dy2 * dy2);
    var lead = dcent / PREDATOR_MAX_SPEED * opt.lead_scale;
    if (lead < 0) lead = 0;
    if (lead > opt.lead_max) lead = opt.lead_max;
    var tx = cx0 + lead * vx0, ty = cy0 + lead * vy0;
    var mom = opt.momentum || 0;
    if (mom > 0 && prevTarget) {
        tx = mom * prevTarget.x + (1 - mom) * tx;
        ty = mom * prevTarget.y + (1 - mom) * ty;
    }
    return { x: tx, y: ty };
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { computeEvolvedTarget: computeEvolvedTarget, EVOLVED_PATROL: EVOLVED_PATROL };
}

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

    // Autonomous movement, split by regime exactly as the production teacher
    // (dev/sim_torch.py) defines it:
    //   CHASE  (a boid within POLICY_R): analytic seek toward the nearest boid
    //          — fast-set the offset to MAX_SPEED, subtract velocity, fast-limit
    //          to MAX_FORCE. This is closed-form and exact (cos 1.0).
    //   PATROL (no boid in range): the 10,352-param radial net in
    //          window.__predatorModel, straight from the raw boid set. It does
    //          internally what the old computeEvolvedTarget (density-weighted
    //          cluster select) + 35-feat seek-net did in two stages — patrol
    //          cos_med 0.9878, matching the 129k-param transformer ceiling.
    // The simulation is gated on the weights being loaded (see boids.js), so
    // window.__predatorModel is always present here.
    getAutonomousForce: function(boids) {
        var model = window.__predatorModel;
        var n = boids.length;
        if (n === 0) return new Vector(0, 0);

        var px = this.position.x, py = this.position.y;
        var bestD2 = Infinity, nx = 0, ny = 0;
        for (var i = 0; i < n; i++) {
            var dx = boids[i].position.x - px, dy = boids[i].position.y - py;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; nx = dx; ny = dy; }
        }

        if (bestD2 < model.POLICY_R * model.POLICY_R) {
            // CHASE: analytic seek-nearest (matches teacher fast_set_magnitude +
            // fast_limit exactly).
            var desired = new Vector(nx, ny);
            desired.iFastSetMagnitude(PREDATOR_MAX_SPEED);
            var steer = desired.subtract(this.velocity);
            steer.iFastLimit(PREDATOR_MAX_FORCE);
            return steer;
        }

        // PATROL: radial set-net maps raw boids -> steering force directly.
        var out = model.forward(this.position, this.velocity, boids);
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
// Guarded so the file can also be require()'d in node (parity tests) where
// there is no window/fetch.
if (typeof window !== 'undefined' && typeof fetch !== 'undefined') {
    window.__predatorReady = fetch('js/predator_radial_weights.json', { cache: 'no-cache' })
        .then(function (r) {
            if (!r.ok) throw new Error('predator weights fetch failed: ' + r.status);
            return r.json();
        })
        .then(function (json) {
            window.__predatorModel = PredatorRadial.loadModel(json);
        });
}
