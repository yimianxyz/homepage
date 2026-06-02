// Headless ground-truth driver for the in-browser planner teacher.
//
// The live page's planner runs its K-candidate x H-frame rollout in a Web
// Worker (js/predator_planner_worker.js) whose flat-typed-array sim mirrors the
// REAL boid.js + predator.js math EXACTLY — including the live page's
// DOUBLE-flock-per-frame (tick()+render()) two-pass dynamics (worker header
// lines 9,17-23; rolloutFlat lines 247-249). That worker also ships
// `evalClosedLoop` (lines 293-336): a ZERO-STALENESS closed-loop episode eval
// that re-plans from the CURRENT state every D frames with no main-thread lag —
// the JS mirror of dev/planner_probe.py. Its predator follows the committed
// target via analytic chase/seek (NOT the shipped NN), so controller='planner'
// is the lookahead teacher and controller='e3d' is the analytic patrol — the
// exact two policies in the 2x2 premise table.
//
// This driver loads the worker's pure functions into a VM sandbox (stubbing
// importScripts since vector/boid/predator are already loaded), builds each
// seed's frame-0 state from the REAL Simulation.initialize (so the snapshot is
// the live frame-0 state, RNG-order-faithful), and calls evalClosedLoop. The
// result is the GROUND-TRUTH two-pass catch count per seed — authoritative over
// sim_torch, and used to (a) fill the two-pass row of the premise table in real
// JS and (b) cross-validate sim_torch's two-pass numbers.
//
//   node dev/eval_planner_worker.js --js ./js --controller planner \
//        --seedStart 200000 --seeds 256 --frames 5000 --D 8 --K 16 --H 120
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(argv) {
    const a = { js: './js', weights: null, controller: 'planner',
                seedStart: 200000, seeds: 256, frames: 5000, D: 8, K: 16, H: 120,
                width: 1680, height: 1680, numBoids: 120, refreshIntervalMs: 12,
                POLICY_R: 80, out: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--js') a.js = argv[++i];
        else if (k === '--weights') a.weights = argv[++i];
        else if (k === '--controller') a.controller = argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--D') a.D = +argv[++i];
        else if (k === '--K') a.K = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--numBoids') a.numBoids = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
    }
    if (!a.weights) a.weights = path.join(a.js, 'predator_weights.json');
    return a;
}

function makeStubCtx() {
    const noop = function () {};
    return {
        beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, fill: noop,
        arc: noop, clearRect: noop, fillRect: noop, strokeRect: noop, closePath: noop,
        save: noop, restore: noop, translate: noop, rotate: noop, scale: noop,
        fillText: noop, setLineDash: noop, ellipse: noop, quadraticCurveTo: noop,
        bezierCurveTo: noop, createLinearGradient: function () { return { addColorStop: noop }; },
        set strokeStyle(v) {}, set fillStyle(v) {}, set lineWidth(v) {}, set globalAlpha(v) {},
        set font(v) {}, set lineCap(v) {}, set lineJoin(v) {}, set shadowBlur(v) {}, set shadowColor(v) {},
        get strokeStyle() { return ''; }, get fillStyle() { return ''; }, get lineWidth() { return 1; },
        get globalAlpha() { return 1; },
    };
}

// Same load order as eval_js_patrol.js so a sim can be initialized from a seed.
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'policy_features.js',
                  'predator_nn.js', 'predator.js', 'simulation.js'];

function buildContext(opt, weightsJson) {
    const ctx = makeStubCtx();
    const win = {
        innerWidth: opt.width, innerHeight: opt.height,
        matchMedia: function () { return { matches: false, addEventListener: function () {} }; },
        addEventListener: function () {},
    };
    const sandbox = {
        navigator: { userAgent: 'NodeJSEval' }, window: win,
        document: {
            getElementById: function () {
                return { getContext: function () { return ctx; }, width: opt.width, height: opt.height };
            },
            addEventListener: function () {},
        },
        fetch: function () { return new Promise(function () {}); },
        renderActivationViz: function () {},
        // Worker file calls importScripts(vector,boid,predator) — already loaded.
        importScripts: function () {},
        postMessage: function () {},
        Math: Math, Date: Date, console: console,
    };
    sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    for (const f of JS_FILES) {
        vm.runInContext(fs.readFileSync(path.join(opt.js, f), 'utf8'), context, { filename: 'js/' + f });
    }
    win.__predatorModel = sandbox.PredatorNN.loadModel(weightsJson);
    // Load the worker's pure functions (evalClosedLoop, plan, cfg, ...) into the
    // SAME context. Its trailing `onmessage = ...` just assigns a sandbox global.
    vm.runInContext(fs.readFileSync(path.join(opt.js, 'predator_planner_worker.js'), 'utf8'),
                    context, { filename: 'js/predator_planner_worker.js' });
    // Configure exactly as predator_planner.js's config message would.
    sandbox.cfg.K = opt.K; sandbox.cfg.H = opt.H; sandbox.cfg.POLICY_R = opt.POLICY_R;
    sandbox.cfg.W = opt.width; sandbox.cfg.Hc = opt.height;
    // Set PREDATOR_RANGE explicitly (desktop=80) rather than relying on the
    // boid.js getBoidPredatorRange() default resolving correctly by load order.
    sandbox.PREDATOR_RANGE = 80;
    return { sandbox, win };
}

function evalSeed(ctxObj, opt, seed) {
    const { sandbox } = ctxObj;
    sandbox.setSimSeed(seed, opt.refreshIntervalMs);
    sandbox.NUM_BOIDS = opt.numBoids;
    const sim = new sandbox.Simulation('boids1');
    sim.canvasWidth = opt.width;
    sim.canvasHeight = opt.height;
    sim.initialize(false);
    sandbox.setFrameMs(opt.refreshIntervalMs);
    // Build the frame-0 snapshot from the REAL initialized objects.
    const pred = sim.predator, boids = sim.boids, n = boids.length;
    const bx = new Float64Array(n), by = new Float64Array(n);
    const bvx = new Float64Array(n), bvy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        bx[i] = boids[i].position.x; by[i] = boids[i].position.y;
        bvx[i] = boids[i].velocity.x; bvy[i] = boids[i].velocity.y;
    }
    const snap = {
        bx: bx, by: by, bvx: bvx, bvy: bvy,
        px: pred.position.x, py: pred.position.y,
        pvx: pred.velocity.x, pvy: pred.velocity.y,
        psize: pred.currentSize, lastFeed: 0, nowMs: 0,
    };
    const res = sandbox.evalClosedLoop(snap, opt.frames, opt.D, opt.controller);
    return res.catches;
}

function main() {
    const opt = parseArgs(process.argv);
    const weightsJson = JSON.parse(fs.readFileSync(opt.weights, 'utf8'));
    // One sandbox reused across seeds (sim state is rebuilt per seed; the worker's
    // scratch buffers are overwritten each rollout). Fresh per seed is safer but
    // slower; reuse is fine because evalClosedLoop fully re-inits from the snapshot.
    const ctxObj = buildContext(opt, weightsJson);
    const per = [];
    for (let i = 0; i < opt.seeds; i++) {
        per.push(evalSeed(ctxObj, opt, opt.seedStart + i));
        if ((i + 1) % 16 === 0) process.stderr.write(`  ${i + 1}/${opt.seeds}\r`);
    }
    const n = per.length;
    const mean = per.reduce((s, v) => s + v, 0) / n;
    const sd = Math.sqrt(per.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (n - 1));
    const se = sd / Math.sqrt(n);
    const out = {
        controller: opt.controller, js: opt.js, seedStart: opt.seedStart, seeds: n,
        frames: opt.frames, D: opt.D, K: opt.K, H: opt.H, numBoids: opt.numBoids,
        two_pass: true, mean_catches: mean, sd: sd, se: se, per_seed: per,
    };
    console.log(JSON.stringify({ ...out, per_seed: undefined }));
    if (opt.out) fs.writeFileSync(opt.out, JSON.stringify(out));
}

main();
