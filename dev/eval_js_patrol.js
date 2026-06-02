// Faithful headless JS eval of the SHIPPED predator code path.
//
// Unlike dev/oracle.js (which overrides getAutonomousForce and reimplements
// the patrol target via dev/policy_spec.js), this driver loads the real
// browser js/ files into a VM sandbox and runs the ACTUAL
// predator.getAutonomousForce — same buildPredatorFeatures (js/policy_features.js),
// same NN forward (js/predator_nn.js + js/predator_weights.json), same boid
// flocking and catch detection as the live page. The only thing that varies
// between two runs is which js/predator.js is loaded, so the catch-rate delta
// it measures is exactly the patrol-policy change being deployed.
//
// Frame loop mirrors dev/oracle.js Oracle.step (simTick(); sim.render()),
// which is validated to match the live page bit-for-bit. Seeds use the same
// mulberry32 (js/rng.js) that dev/sim_torch.py replicates in fp64, so seed N
// here == seed N on the GPU sim.
//
//   node dev/eval_js_patrol.js --js ./js --weights ./js/predator_weights.json \
//        --seedStart 50000 --seeds 256 --frames 1500
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(argv) {
    const a = { js: './js', weights: null, seedStart: 50000, seeds: 256, frames: 1500,
                width: 1680, height: 1680, numBoids: 120, refreshIntervalMs: 12,
                twopass: false };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--js') a.js = argv[++i];
        else if (k === '--weights') a.weights = argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--numBoids') a.numBoids = +argv[++i];
        else if (k === '--twopass') a.twopass = true;
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

// Files loaded into the sandbox, in load order. policy_features + predator_nn
// must precede predator.js (predator.js references their globals at call time;
// loading them first also lets us attach the model before any frame runs).
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'policy_features.js',
                  'predator_nn.js', 'predator.js', 'simulation.js'];

function buildContext(opt, weightsJson) {
    const ctx = makeStubCtx();
    const win = {
        innerWidth: opt.width,
        innerHeight: opt.height,
        matchMedia: function () { return { matches: false, addEventListener: function () {} }; },
        addEventListener: function () {},
    };
    // Forever-pending fetch: main/predator.js calls fetch() unconditionally at
    // load; the branch version guards on `typeof fetch`. Either way we set
    // window.__predatorModel ourselves below, so this promise is never awaited.
    const stubFetch = function () { return new Promise(function () {}); };
    const sandbox = {
        navigator: { userAgent: 'NodeJSEval' },
        window: win,
        document: {
            getElementById: function () {
                return { getContext: function () { return ctx; }, width: opt.width, height: opt.height };
            },
            addEventListener: function () {},
        },
        fetch: stubFetch,
        renderActivationViz: function () {},
        Math: Math, Date: Date, console: console,
    };
    sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    for (const f of JS_FILES) {
        const code = fs.readFileSync(path.join(opt.js, f), 'utf8');
        vm.runInContext(code, context, { filename: 'js/' + f });
    }
    // Inject the real trained model exactly as the live page does post-fetch.
    win.__predatorModel = sandbox.PredatorNN.loadModel(weightsJson);
    return { sandbox, win };
}

function evalSeed(opt, weightsJson, seed) {
    const { sandbox, win } = buildContext(opt, weightsJson);
    sandbox.setSimSeed(seed, opt.refreshIntervalMs);
    sandbox.NUM_BOIDS = opt.numBoids;
    const Simulation = sandbox.Simulation;
    const sim = new Simulation('boids1');
    sim.canvasWidth = opt.width;
    sim.canvasHeight = opt.height;
    sim.initialize(false);
    sandbox.setFrameMs(opt.refreshIntervalMs);
    if (win.__predatorModel == null) throw new Error('model not loaded');
    const startBoids = sim.boids.length;
    for (let f = 0; f < opt.frames; f++) {
        sandbox.simTick();
        if (opt.twopass) sim.tick();   // live-page run() does tick() before render()
        sim.render();
    }
    return { seed, catches: sim.boidsEaten, startBoids, remaining: sim.boids.length };
}

function main() {
    const opt = parseArgs(process.argv);
    const weightsJson = JSON.parse(fs.readFileSync(opt.weights, 'utf8'));
    const per = [];
    for (let i = 0; i < opt.seeds; i++) {
        per.push(evalSeed(opt, weightsJson, opt.seedStart + i).catches);
        if ((i + 1) % 32 === 0) process.stderr.write(`  ${i + 1}/${opt.seeds}\r`);
    }
    const n = per.length;
    const mean = per.reduce((s, v) => s + v, 0) / n;
    const sd = Math.sqrt(per.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (n - 1));
    const se = sd / Math.sqrt(n);
    console.log(JSON.stringify({
        js: opt.js, weights: opt.weights, seedStart: opt.seedStart, seeds: n,
        frames: opt.frames, numBoids: opt.numBoids, twopass: opt.twopass,
        mean_catches: mean, sd: sd, se: se,
        per_seed: per,
    }));
}

main();
