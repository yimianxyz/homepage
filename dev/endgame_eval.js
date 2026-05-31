'use strict';
// ===========================================================================
// ENDGAME EVAL HARNESS  (north star for the "catch the last boid" work)
// ---------------------------------------------------------------------------
// Loads the REAL shipping sim (js/*.js from the dev worktree) into a VM
// sandbox and runs a faithful closed-loop endgame: start with N boids, measure
// time-to-extinction (TTE, frames) and catch rate over many seeds and canvas
// sizes. Lower TTE = better.
//
// The predator's getAutonomousForce can be OVERRIDDEN with a pluggable policy
// so we can A/B candidate endgame steering laws against the deployed radial
// net under identical physics / seeds / RNG.
//
//   node dev/endgame_eval.js --policy radial [--startN 1] [--seeds 32]
//        [--maxFrames 18000] [--size 1440x900] [--workers 8] [--params a,b,..]
//
// policies: radial (deployed, no override) | pursuit | pursuit_torus |
//           intercept | intercept_torus | param (CEM-able, see policyParam)
//
// Physics facts the policies exploit (verified from js/boid.js, js/predator.js):
//   - Toroidal world (period ~ W+20, H+20); predator & boids wrap.
//   - PREDATOR_MAX_SPEED 2.5, MAX_FORCE 0.05. Boid MAX_SPEED 6, no friction.
//   - Boid avoidance: only within 80px, force <=0.15/frame, points away.
//   - A lone boid with predator far drifts in a straight line (no flock forces).
//   - Catch radius ~ predator.currentSize*0.7 (~8.4px base). feedCooldown 100ms.
// ===========================================================================
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

const DEV_DIR = '/tmp/dev_wt';
const FILES = ['rng.js', 'predator_radial.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js'];
const WEIGHTS = 'js/predator_radial_weights.json';

const PMS = 2.5;    // PREDATOR_MAX_SPEED
const PMF = 0.05;   // PREDATOR_MAX_FORCE

function buildSandbox(W, H) {
    const noop = () => {};
    const ctx = { beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, fill: noop, arc: noop, clearRect: noop, fillRect: noop, strokeRect: noop, save: noop, restore: noop, translate: noop, rotate: noop, scale: noop, set strokeStyle(v) {}, set fillStyle(v) {}, set lineWidth(v) {}, get strokeStyle() { return ''; }, get fillStyle() { return ''; }, get lineWidth() { return 1; } };
    const s = {
        navigator: { userAgent: 'N' },
        window: { innerWidth: W, innerHeight: H, matchMedia: () => ({ matches: false, addEventListener: noop }) },
        document: { getElementById: () => ({ getContext: () => ctx, width: W, height: H }), addEventListener: noop },
        fetch: () => new Promise(() => {}), renderActivationViz: noop, Math: Math, Date: Date, console: console,
    };
    s.global = s;
    return vm.createContext(s);
}

// ---- vector helpers (plain Math.sqrt; final picks get JS-verified w/ real code) ----
function setMag(x, y, m) { const d = Math.sqrt(x * x + y * y) || 1e-12; const k = m / d; return [x * k, y * k]; }
function limit(x, y, m) { const d = Math.sqrt(x * x + y * y); if (d > m) { const k = m / d; return [x * k, y * k]; } return [x, y]; }
// analytic "seek" toward an offset (dx,dy) at max speed (mirrors chase law)
function seekStep(dx, dy, vx, vy) { const [sx, sy] = setMag(dx, dy, PMS); return limit(sx - vx, sy - vy, PMF); }
// toroidal shortest delta
function torusDelta(d, P) { if (d > P * 0.5) return d - P; if (d < -P * 0.5) return d + P; return d; }

// nearest boid by raw (non-toroidal) distance — matches deployed perception
function nearestRaw(px, py, boids) {
    let bd = Infinity, idx = -1;
    for (let i = 0; i < boids.length; i++) {
        const dx = boids[i].position.x - px, dy = boids[i].position.y - py, d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; idx = i; }
    }
    return idx;
}
function nearestTorus(px, py, boids, PX, PY) {
    let bd = Infinity, idx = -1, bdx = 0, bdy = 0;
    for (let i = 0; i < boids.length; i++) {
        const dx = torusDelta(boids[i].position.x - px, PX), dy = torusDelta(boids[i].position.y - py, PY), d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; idx = i; bdx = dx; bdy = dy; }
    }
    return { idx, dx: bdx, dy: bdy, d: Math.sqrt(bd) };
}

// ---- pluggable policies: (pred, boids, W,H, P) -> [fx,fy] ----
function makePolicy(name, params, PX, PY) {
    const p = params || [];
    if (name === 'pursuit') {
        return (pred, boids) => {
            const i = nearestRaw(pred.position.x, pred.position.y, boids);
            return seekStep(boids[i].position.x - pred.position.x, boids[i].position.y - pred.position.y, pred.velocity.x, pred.velocity.y);
        };
    }
    if (name === 'pursuit_torus') {
        return (pred, boids) => {
            const n = nearestTorus(pred.position.x, pred.position.y, boids, PX, PY);
            return seekStep(n.dx, n.dy, pred.velocity.x, pred.velocity.y);
        };
    }
    if (name === 'intercept') {
        const leadGain = p[0] !== undefined ? p[0] : 1.0, leadMax = p[1] !== undefined ? p[1] : 400;
        return (pred, boids) => {
            const i = nearestRaw(pred.position.x, pred.position.y, boids);
            const b = boids[i]; let dx = b.position.x - pred.position.x, dy = b.position.y - pred.position.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            let lead = dist / PMS * leadGain; if (lead > leadMax) lead = leadMax;
            dx = (b.position.x + b.velocity.x * lead) - pred.position.x;
            dy = (b.position.y + b.velocity.y * lead) - pred.position.y;
            return seekStep(dx, dy, pred.velocity.x, pred.velocity.y);
        };
    }
    if (name === 'intercept_torus') {
        const leadGain = p[0] !== undefined ? p[0] : 1.0, leadMax = p[1] !== undefined ? p[1] : 400;
        return (pred, boids) => {
            const n = nearestTorus(pred.position.x, pred.position.y, boids, PX, PY);
            const b = boids[n.idx];
            let lead = n.d / PMS * leadGain; if (lead > leadMax) lead = leadMax;
            const dx = n.dx + b.velocity.x * lead, dy = n.dy + b.velocity.y * lead;
            return seekStep(dx, dy, pred.velocity.x, pred.velocity.y);
        };
    }
    return null; // 'radial' => no override
}

function runSeed(sb, S, makeOverride, seed, maxFrames, startN, refreshMs, boidSpeed) {
    S.setSimSeed(seed, refreshMs);
    S.NUM_BOIDS = startN;
    const sim = new S.Simulation('boids1');
    sim.canvasWidth = sb.W; sim.canvasHeight = sb.H;
    sim.initialize(false);
    S.setFrameMs(refreshMs);
    // Optionally pre-pump boids to a given drift speed (simulates a survivor that
    // accumulated velocity during the hunt — no friction in the sim).
    if (boidSpeed && boidSpeed > 0) {
        for (const b of sim.boids) {
            const ang = S.simRandom() * 2 * Math.PI;
            b.velocity.x = Math.cos(ang) * boidSpeed;
            b.velocity.y = Math.sin(ang) * boidSpeed;
        }
    }
    if (makeOverride) {
        const PX = sb.W + 20, PY = sb.H + 20;
        const pol = makeOverride(PX, PY);
        sim.predator.getAutonomousForce = function (boids) {
            if (boids.length === 0) return new S.Vector(0, 0);
            const f = pol(sim.predator, boids, sb.W, sb.H);
            return new S.Vector(f[0], f[1]);
        };
    }
    for (let f = 0; f < maxFrames; f++) {
        S.simTick(); sim.render();
        if (sim.boids.length === 0) return f + 1;
    }
    return maxFrames;
}

function evalBranch(policy, params, seeds, maxFrames, startN, W, H, refreshMs, boidSpeed) {
    const S = buildSandbox(W, H); S.W = W; S.H = H;
    for (const f of FILES) vm.runInContext(fs.readFileSync(path.join(DEV_DIR, 'js', f), 'utf8'), S, { filename: f });
    const json = JSON.parse(fs.readFileSync(path.join(DEV_DIR, WEIGHTS), 'utf8'));
    S.window.__predatorModel = S.PredatorRadial.loadModel(json);
    const sb = { W, H };
    const makeOverride = policy === 'radial' ? null : (PX, PY) => makePolicy(policy, params, PX, PY);
    const ttes = [];
    for (const seed of seeds) ttes.push(runSeed(sb, S, makeOverride, seed, maxFrames, startN, refreshMs, boidSpeed));
    return ttes;
}

if (!isMainThread) {
    const { policy, params, seeds, maxFrames, startN, W, H, refreshMs, boidSpeed } = workerData;
    parentPort.postMessage(evalBranch(policy, params, seeds, maxFrames, startN, W, H, refreshMs, boidSpeed));
} else {
    const args = { policy: 'radial', params: null, seeds: 32, seedStart: 100, maxFrames: 18000, startN: 1, size: '1440x900', workers: 8, refreshMs: 12, boidSpeed: 0 };
    for (let i = 2; i < process.argv.length; i++) {
        const a = process.argv[i];
        if (a === '--policy') args.policy = process.argv[++i];
        else if (a === '--params') args.params = process.argv[++i].split(',').map(Number);
        else if (a === '--seeds') args.seeds = +process.argv[++i];
        else if (a === '--seedStart') args.seedStart = +process.argv[++i];
        else if (a === '--maxFrames') args.maxFrames = +process.argv[++i];
        else if (a === '--startN') args.startN = +process.argv[++i];
        else if (a === '--size') args.size = process.argv[++i];
        else if (a === '--workers') args.workers = +process.argv[++i];
        else if (a === '--boidSpeed') args.boidSpeed = +process.argv[++i];
    }
    const [W, H] = args.size.split('x').map(Number);
    const seeds = Array.from({ length: args.seeds }, (_, i) => args.seedStart + i);
    const nw = Math.min(args.workers, seeds.length);
    const chunk = Math.ceil(seeds.length / nw);
    const chunks = [];
    for (let i = 0; i < seeds.length; i += chunk) chunks.push(seeds.slice(i, i + chunk));
    const t0 = Date.now();
    Promise.all(chunks.map(cs => new Promise((res, rej) => {
        const w = new Worker(__filename, { workerData: { policy: args.policy, params: args.params, seeds: cs, maxFrames: args.maxFrames, startN: args.startN, W, H, refreshMs: args.refreshMs, boidSpeed: args.boidSpeed } });
        w.on('message', res); w.on('error', rej); w.on('exit', c => { if (c !== 0) rej(new Error('worker ' + c)); });
    }))).then(all => {
        const ttes = [].concat(...all);
        const n = ttes.length;
        const caught = ttes.filter(t => t < args.maxFrames);
        const mean = ttes.reduce((a, b) => a + b, 0) / n;
        const sorted = ttes.slice().sort((a, b) => a - b);
        const med = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
        const cm = caught.length ? caught.reduce((a, b) => a + b, 0) / caught.length : NaN;
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`policy=${args.policy}${args.params ? ' params=[' + args.params + ']' : ''} startN=${args.startN} size=${W}x${H} seeds=${n} | caught ${caught.length}/${n} | mean TTE=${mean.toFixed(0)} median=${med} | mean TTE(caught)=${isNaN(cm) ? '--' : cm.toFixed(0)} | ${elapsed}s`);
    }).catch(e => { console.error(e); process.exit(1); });
}
