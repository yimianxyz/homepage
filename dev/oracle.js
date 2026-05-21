// Headless Node port of the simulation. Loads the same js/ source files into
// a sandboxed VM context so the trace matches the live page bit-for-bit
// (same RNG, same Vector ops, same loop order). Exposes a step-by-step API
// and convenience trace collectors used by gen_dataset.js and eval.js.

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const spec = require('./policy_spec');

const JS_DIR = path.join(__dirname, '..', 'js');
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js'];

// Defaults match the desktop branch of js/simulation.js so the oracle
// produces the trajectories the NN is trained against.
const DEFAULT_OPTS = {
    width: 1680,
    height: 1680,
    numBoids: 120,
    refreshIntervalMs: 12,
    seed: 1,
    nnFn: null,            // optional: (features) => [ax, ay], replaces the rule
    forcePolicyR: true,    // override PREDATOR_RANGE to POLICY_R inside the VM
    autoTargetMode: 'random', // 'random' | 'nearest_boid' | 'flock_centroid' | 'farthest_in_K'
};

// Compute the patrol target the predator should aim at when no boid is in
// hunting range. The rule's original choice ('random') wanders to a random
// canvas point every 5 s — fine for visual feel, terrible for catch rate
// when boids cluster on the other side of the map. The boid-aware modes
// give the predator a smarter "go toward the flock" patrol behavior; we
// can A/B them against 'random' on the eval suite to find the structural
// best.
function computeAutoTarget(mode, predator, boids, defaultTarget, canvasWidth, canvasHeight, simRandomFn) {
    if (mode === 'nearest_boid') {
        // Aim at the nearest boid (regardless of distance). The predator
        // will pursue continuously rather than wander while waiting for one
        // to enter R. seek_auto_xy then == seek_boid_xy in patrol mode.
        let bestD2 = Infinity, bx = defaultTarget.x, by = defaultTarget.y;
        for (let i = 0; i < boids.length; i++) {
            const dx = boids[i].position.x - predator.position.x;
            const dy = boids[i].position.y - predator.position.y;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; bx = boids[i].position.x; by = boids[i].position.y; }
        }
        return { x: bx, y: by };
    }
    if (mode === 'flock_centroid') {
        // Aim at the centroid of all boids. Smoother than nearest_boid:
        // the predator doesn't oscillate when nearest flips between two
        // similar-distance boids.
        if (boids.length === 0) return defaultTarget;
        let sx = 0, sy = 0;
        for (let i = 0; i < boids.length; i++) {
            sx += boids[i].position.x;
            sy += boids[i].position.y;
        }
        return { x: sx / boids.length, y: sy / boids.length };
    }
    if (mode === 'farthest_in_K') {
        // Aim at the farthest of the K=4 nearest. Drags predator outward
        // when nearest cluster keeps escaping — pre-emptive coverage of
        // the secondary cluster. Speculative; included for the A/B sweep.
        const K = 4;
        const pairs = [];
        for (let i = 0; i < boids.length; i++) {
            const dx = boids[i].position.x - predator.position.x;
            const dy = boids[i].position.y - predator.position.y;
            pairs.push({ d2: dx * dx + dy * dy, x: boids[i].position.x, y: boids[i].position.y });
        }
        pairs.sort((a, b) => a.d2 - b.d2);
        const top = pairs.slice(0, Math.min(K, pairs.length));
        if (top.length === 0) return defaultTarget;
        const last = top[top.length - 1];
        return { x: last.x, y: last.y };
    }
    // 'random' — keep the default patrol target (regenerated externally).
    return defaultTarget;
}

function makeStubCtx() {
    // Canvas 2D API surface used by boid.render / predator.render. No-ops.
    const noop = function () {};
    return {
        beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, fill: noop,
        arc: noop, clearRect: noop, fillRect: noop, strokeRect: noop,
        save: noop, restore: noop, translate: noop, rotate: noop, scale: noop,
        set strokeStyle(v) {}, set fillStyle(v) {}, set lineWidth(v) {},
        get strokeStyle() { return ''; }, get fillStyle() { return ''; },
        get lineWidth() { return 1; },
    };
}

function buildSandbox(opts) {
    const ctx = makeStubCtx();
    // Stub fetch so predator.js's load-time `window.__predatorReady = fetch(...)`
    // doesn't throw. We never await this promise in headless mode; the oracle
    // overrides predator.getAutonomousForce to call the rule (or nnFn) directly.
    const stubFetch = function () {
        return new Promise(function () {}); // forever pending; never resolves
    };
    const win = {
        innerWidth: opts.width,
        innerHeight: opts.height,
        matchMedia: function () { return { matches: false, addEventListener: function () {} }; },
    };
    const sandbox = {
        // Browser shims the JS files reach for.
        navigator: { userAgent: 'NodeOracle' },
        window: win,
        document: {
            getElementById: function () {
                return {
                    getContext: function () { return ctx; },
                    width: opts.width,
                    height: opts.height,
                };
            },
            addEventListener: function () {},
        },
        fetch: stubFetch,
        // Browser-only viz that simulation.render calls; stub as no-op in
        // headless mode so we don't have to load activation_viz.js (which
        // depends on a loaded __predatorModel for normalization).
        renderActivationViz: function () {},
        // Math is already in vm context's global; just expose what we need.
        Math: Math,
        Date: Date,
        console: console,
    };
    sandbox.global = sandbox;
    return vm.createContext(sandbox);
}

function loadAllJs(sandbox) {
    for (const f of JS_FILES) {
        const code = fs.readFileSync(path.join(JS_DIR, f), 'utf8');
        vm.runInContext(code, sandbox, { filename: 'js/' + f });
    }
}

class Oracle {
    constructor(userOpts) {
        const opts = Object.assign({}, DEFAULT_OPTS, userOpts || {});
        this.opts = opts;

        const sandbox = buildSandbox(opts);
        loadAllJs(sandbox);
        this.sb = sandbox;

        // Seed BEFORE creating the simulation (boids/predator constructors
        // pull from simRandom).
        sandbox.setSimSeed(opts.seed, opts.refreshIntervalMs);

        if (opts.forcePolicyR) {
            // Hard-pin the predator's hunting range to POLICY_R regardless of
            // whatever device-detection ran inside js/predator.js.
            sandbox.PREDATOR_RANGE = spec.POLICY_R;
        }

        // Manually construct a Simulation, mimicking what canvas_init.js does.
        const Simulation = sandbox.Simulation;
        // Override NUM_BOIDS via re-eval so the predator population is
        // independent of (mock) device detection.
        sandbox.NUM_BOIDS = opts.numBoids;

        const sim = new Simulation('boids1');
        // The constructor pulled canvas dims from getElementById which we
        // stubbed with our opts; double-check:
        sim.canvasWidth = opts.width;
        sim.canvasHeight = opts.height;
        sim.initialize(false);
        sandbox.setFrameMs(opts.refreshIntervalMs);
        this.sim = sim;

        // Wrap getAutonomousForce so we always capture (features, output)
        // pairs. In rule mode the wrapper delegates to the original JS rule
        // (no reimpl drift). In NN mode it calls nnFn instead.
        const Vector = sandbox.Vector;
        const buildFeatures = spec.buildFeatures;
        const R = spec.POLICY_R;
        const predator = sim.predator;
        predator._lastFeatures = null;
        predator._lastOutput = null;

        const autoTargetMode = opts.autoTargetMode;
        if (opts.nnFn) {
            const nnFn = opts.nnFn;
            predator.getAutonomousForce = function (boids) {
                // External bookkeeping: in patrol mode, maintain autoTarget.
                // The "any boid in range" check is a tiny rule too, but it
                // operates only on autoTarget state, not on steering output.
                let anyInRange = false;
                for (let i = 0; i < boids.length; i++) {
                    if (this.position.getDistance(boids[i].position) < R) {
                        anyInRange = true;
                        break;
                    }
                }
                if (!anyInRange) {
                    if (autoTargetMode === 'random') {
                        const currentTime = sandbox.simNow();
                        if (currentTime - this.targetChangeTime > this.targetChangeInterval) {
                            const margin = 50;
                            this.autonomousTarget.x = margin + sandbox.simRandom() * (this.simulation.canvasWidth - 2 * margin);
                            this.autonomousTarget.y = margin + sandbox.simRandom() * (this.simulation.canvasHeight - 2 * margin);
                            this.targetChangeTime = currentTime;
                        }
                        if (this.position.getDistance(this.autonomousTarget) < 30) {
                            this.targetChangeTime = 0;
                        }
                    } else {
                        // Boid-aware modes: recompute autoTarget every patrol-mode
                        // frame to track the flock continuously.
                        const t = computeAutoTarget(
                            autoTargetMode, this, boids, this.autonomousTarget,
                            this.simulation.canvasWidth, this.simulation.canvasHeight,
                            sandbox.simRandom
                        );
                        this.autonomousTarget.x = t.x;
                        this.autonomousTarget.y = t.y;
                    }
                }
                const features = buildFeatures(this.position, this.velocity, boids, this.autonomousTarget);
                const out = nnFn(features);
                this._lastFeatures = features;
                this._lastOutput = [out[0], out[1]];
                return new Vector(out[0], out[1]);
            };
        } else {
            // Rule mode: run the analytical rule from policy_spec.rulePolicy
            // directly, not the live page's predator.getAutonomousForce
            // (which now calls the NN). Patrol-target bookkeeping is done
            // externally so the rule is purely a function of features.
            const rulePolicy = spec.rulePolicy;
            predator.getAutonomousForce = function (boids) {
                let anyInRange = false;
                for (let i = 0; i < boids.length; i++) {
                    if (this.position.getDistance(boids[i].position) < R) {
                        anyInRange = true;
                        break;
                    }
                }
                if (!anyInRange) {
                    if (autoTargetMode === 'random') {
                        const currentTime = sandbox.simNow();
                        if (currentTime - this.targetChangeTime > this.targetChangeInterval) {
                            const margin = 50;
                            this.autonomousTarget.x = margin + sandbox.simRandom() * (this.simulation.canvasWidth - 2 * margin);
                            this.autonomousTarget.y = margin + sandbox.simRandom() * (this.simulation.canvasHeight - 2 * margin);
                            this.targetChangeTime = currentTime;
                        }
                        if (this.position.getDistance(this.autonomousTarget) < 30) {
                            this.targetChangeTime = 0;
                        }
                    } else {
                        const t = computeAutoTarget(
                            autoTargetMode, this, boids, this.autonomousTarget,
                            this.simulation.canvasWidth, this.simulation.canvasHeight,
                            sandbox.simRandom
                        );
                        this.autonomousTarget.x = t.x;
                        this.autonomousTarget.y = t.y;
                    }
                }
                const features = buildFeatures(this.position, this.velocity, boids, this.autonomousTarget);
                const out = rulePolicy(features);
                this._lastFeatures = features;
                this._lastOutput = [out[0], out[1]];
                return new Vector(out[0], out[1]);
            };
        }

        // Initial sim.tick() that simulation.run() performs before its
        // setInterval. With no obstacles this only seeds boid.acceleration
        // values that get immediately overwritten in the next render, so we
        // skip it unless caller asks for full parity.
        if (opts.fullTick) sim.tick();

        this.frame = 0;
    }

    // Advance one frame, matching the body of the interval callback in
    // simulation.run(). With no obstacles in play, sim.tick() is purely
    // redundant work (render's boid.run overwrites the acceleration that
    // tick computes), so we skip it for a ~30-40% speedup. Pass
    // opts.fullTick=true to restore parity with the original loop ordering.
    step(record) {
        this.sb.simTick();
        if (this.opts.fullTick) this.sim.tick();
        this.sim.render();
        this.frame += 1;

        if (record) {
            const p = this.sim.predator;
            return {
                frame: this.frame,
                px: p.position.x, py: p.position.y,
                vx: p.velocity.x, vy: p.velocity.y,
                size: p.currentSize,
                boidCount: this.sim.boids.length,
                atx: p.autonomousTarget.x,
                aty: p.autonomousTarget.y,
            };
        }
        return null;
    }

    // Snapshot of the current decision branch and which boid (if any) the
    // rule would seek. Useful for eval; computed off the current state.
    currentRuleBranch() {
        const p = this.sim.predator;
        const boids = this.sim.boids;
        let bestIdx = -1;
        let bestD = Infinity;
        for (let i = 0; i < boids.length; i++) {
            const dx = boids[i].position.x - p.position.x;
            const dy = boids[i].position.y - p.position.y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < spec.POLICY_R && d < bestD) { bestD = d; bestIdx = i; }
        }
        return bestIdx === -1 ? { branch: 'patrol', idx: -1 }
                              : { branch: 'hunt', idx: bestIdx, dist: bestD };
    }

    currentFeatures() {
        const p = this.sim.predator;
        return spec.buildFeatures(p.position, p.velocity, this.sim.boids, p.autonomousTarget);
    }

    currentPredator() {
        const p = this.sim.predator;
        return {
            px: p.position.x, py: p.position.y,
            vx: p.velocity.x, vy: p.velocity.y,
            size: p.currentSize,
            atx: p.autonomousTarget.x, aty: p.autonomousTarget.y,
        };
    }

    currentBoidPositions() {
        return this.sim.boids.map(b => [b.position.x, b.position.y]);
    }
}

// Convenience: run an oracle for N frames returning a compact trace.
function runTrace(opts, frames) {
    const o = new Oracle(opts);
    const trace = {
        seed: o.opts.seed,
        frames: frames,
        px: new Float32Array(frames),
        py: new Float32Array(frames),
        vx: new Float32Array(frames),
        vy: new Float32Array(frames),
        size: new Float32Array(frames),
        boidCount: new Int32Array(frames),
        catches: [], // [{frame, boidIdxBefore, boidIdxAfter? we just log counts}]
    };
    let prevCount = o.sim.boids.length;
    for (let i = 0; i < frames; i++) {
        const s = o.step(true);
        trace.px[i] = s.px; trace.py[i] = s.py;
        trace.vx[i] = s.vx; trace.vy[i] = s.vy;
        trace.size[i] = s.size;
        trace.boidCount[i] = s.boidCount;
        if (s.boidCount < prevCount) {
            trace.catches.push({ frame: i, before: prevCount, after: s.boidCount });
            prevCount = s.boidCount;
        }
    }
    return trace;
}

module.exports = { Oracle, runTrace, DEFAULT_OPTS };
