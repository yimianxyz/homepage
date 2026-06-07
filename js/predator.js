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

// --- catch test: do the predator's and a boid's drawn triangles overlap? ------
// Both are arrowheads built exactly as render() does: tip = center +
// heading*forward, base corners = center +/- perpendicular*half. The predator's
// forward/half grow with its (feeding) size; a boid uses render_size. Overlap is
// exact via the separating-axis theorem, with a cheap circle reject first. A
// touch counts as a catch; any gap does not.
function catchTriangle(cx, cy, vx, vy, fwd, half) {
    var m = Math.sqrt(vx * vx + vy * vy) || 1, ux = vx / m, uy = vy / m;
    return [cx + ux * fwd, cy + uy * fwd, cx - uy * half, cy + ux * half, cx + uy * half, cy - ux * half];
}
function catchAxisSeparates(A, C) {            // is any edge-normal of A a separating axis?
    for (var i = 0; i < 6; i += 2) {
        var nx = A[i + 1] - A[(i + 3) % 6], ny = A[(i + 2) % 6] - A[i];
        var a0 = Infinity, a1 = -Infinity, c0 = Infinity, c1 = -Infinity, d;
        for (var j = 0; j < 6; j += 2) { d = A[j] * nx + A[j + 1] * ny; if (d < a0) a0 = d; if (d > a1) a1 = d; }
        for (var k = 0; k < 6; k += 2) { d = C[k] * nx + C[k + 1] * ny; if (d < c0) c0 = d; if (d > c1) c1 = d; }
        if (a1 < c0 || c1 < a0) return true;
    }
    return false;
}
function predatorCatchesBoid(px, py, pvx, pvy, psize, bx, by, bvx, bvy) {
    var reach = psize * 1.2 + render_size;     // farthest vertex from either center
    var dx = bx - px, dy = by - py;
    if (dx * dx + dy * dy > reach * reach) return false;
    var P = catchTriangle(px, py, pvx, pvy, psize * 1.2, psize * 0.3);
    var B = catchTriangle(bx, by, bvx, bvy, render_size, render_size / 3);
    return !catchAxisSeparates(P, B) && !catchAxisSeparates(B, P);
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

    // The predator's policy: the cheap ballistic patrol (js/predator_cheap.js),
    // distilled from the receding-horizon planner teacher. Every few frames it
    // ballistically picks + rolls one candidate target; each frame it chases the
    // nearest boid within POLICY_R, else seeks that target. The page boot waits on
    // window.__predatorReady (the value-net load) before starting the sim, so
    // window.__cheap is always present here.
    getAutonomousForce: function(boids) {
        return window.__cheap.force(this, boids);
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
        var caughtBoids = [];
        for (var i = 0; i < boids.length; i++) {
            var b = boids[i];
            // Catch when the predator's drawn triangle overlaps the boid's.
            if (predatorCatchesBoid(this.position.x, this.position.y, this.velocity.x, this.velocity.y, this.currentSize,
                                    b.position.x, b.position.y, b.velocity.x, b.velocity.y)) {
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

// The page boot gate (window.__predatorReady) is owned by js/predator_cheap.js,
// which loads value_net.json before boids.js starts the simulation — so the cheap
// policy drives the predator from frame 1.
