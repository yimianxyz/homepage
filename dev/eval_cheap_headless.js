// Headless faithful eval of the CHEAP ballistic policy vs the planner + e3d, in
// the worker's two-pass closed-loop sim (predator_planner_worker.evalClosedLoop).
// Seeds use the live mulberry32 (rng.js) == sim_torch seed N. The decisive check
// is cheap vs planner WITHIN this sim: if cheap ~= planner >> e3d, the 1-rollout
// ballistic port reproduces the 16-rollout planner.
//
//   node dev/eval_cheap_headless.js --seedStart 300000 --seeds 64 --frames 1500
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(argv) {
    const a = { js: path.join(__dirname, '..', 'js'), seedStart: 300000, seeds: 64,
        frames: 1500, width: 1680, height: 1680, numBoids: 120, Hs: 60, D: 8,
        controllers: 'e3d,planner,cheap' };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--js') a.js = argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--Hs') a.Hs = +argv[++i];
        else if (k === '--controllers') a.controllers = argv[++i];
    }
    return a;
}

function stubCtx() {
    const noop = function () {};
    const o = {};
    ['beginPath', 'moveTo', 'lineTo', 'stroke', 'fill', 'arc', 'clearRect', 'fillRect',
        'strokeRect', 'closePath', 'save', 'restore', 'translate', 'rotate', 'scale',
        'fillText', 'setLineDash', 'ellipse', 'quadraticCurveTo', 'bezierCurveTo'].forEach(function (m) { o[m] = noop; });
    o.createLinearGradient = function () { return { addColorStop: noop }; };
    return o;
}

function buildSandbox(opt, valueNet, weightsJson) {
    const ctx = stubCtx();
    const win = { innerWidth: opt.width, innerHeight: opt.height,
        matchMedia: function () { return { matches: false, addEventListener: function () {} }; },
        addEventListener: function () {} };
    const sandbox = {
        navigator: { userAgent: 'NodeJSEval' }, window: win,
        document: { getElementById: function () { return { getContext: function () { return ctx; }, width: opt.width, height: opt.height }; }, addEventListener: function () {} },
        fetch: function () { return new Promise(function () {}); },
        renderActivationViz: function () {}, Math: Math, Date: Date, console: console,
    };
    sandbox.self = sandbox; sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    sandbox.importScripts = function () {
        for (let i = 0; i < arguments.length; i++) {
            const code = fs.readFileSync(path.join(opt.js, arguments[i]), 'utf8');
            vm.runInContext(code, context, { filename: arguments[i] });
        }
    };
    sandbox.postMessage = function () {};
    // live files (seeding + predator deps) then cheap_planner then the worker
    ['rng.js', 'vector.js', 'boid.js', 'policy_features.js', 'predator_nn.js',
        'predator.js', 'simulation.js', 'cheap_planner.js'].forEach(function (f) { sandbox.importScripts(f); });
    sandbox.importScripts('predator_planner_worker.js');   // its importScripts reload deps (idempotent)
    win.__predatorModel = sandbox.PredatorNN.loadModel(weightsJson);
    // configure worker globals via its message handler
    sandbox.onmessage({ data: { type: 'config', K: 16, H: 120, POLICY_R: 80, W: opt.width, Hc: opt.height, predRange: 80 } });
    sandbox.onmessage({ data: { type: 'cheapconfig', net: valueNet, Hs: opt.Hs } });
    return { sandbox };
}

function snapForSeed(sandbox, seed, opt) {
    sandbox.setSimSeed(seed, 12);
    sandbox.NUM_BOIDS = opt.numBoids;
    const sim = new sandbox.Simulation('boids1');
    sim.canvasWidth = opt.width; sim.canvasHeight = opt.height;
    sim.initialize(false);
    sandbox.setFrameMs(12);
    const bx = [], by = [], bvx = [], bvy = [];
    for (let i = 0; i < sim.boids.length; i++) {
        const b = sim.boids[i];
        bx.push(b.position.x); by.push(b.position.y); bvx.push(b.velocity.x); bvy.push(b.velocity.y);
    }
    const p = sim.predator;
    return { bx: bx, by: by, bvx: bvx, bvy: bvy, px: p.position.x, py: p.position.y,
        pvx: p.velocity.x, pvy: p.velocity.y, psize: p.currentSize, lastFeed: -1e9, nowMs: 0 };
}

function main() {
    const opt = parseArgs(process.argv);
    const valueNet = JSON.parse(fs.readFileSync(path.join(opt.js, 'value_net.json'), 'utf8'));
    const weightsJson = JSON.parse(fs.readFileSync(path.join(opt.js, 'predator_weights.json'), 'utf8'));
    const { sandbox } = buildSandbox(opt, valueNet, weightsJson);
    const controllers = opt.controllers.split(',');
    const sums = {}; const perSeed = {};
    controllers.forEach(function (c) { sums[c] = 0; perSeed[c] = []; });
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        const snap = snapForSeed(sandbox, seed, opt);
        for (const c of controllers) {
            // fresh copy: evalClosedLoop mutates its own buffers from s (reads only)
            const s = Object.assign({}, snap, { bx: snap.bx.slice(), by: snap.by.slice(), bvx: snap.bvx.slice(), bvy: snap.bvy.slice() });
            const r = sandbox.evalClosedLoop(s, opt.frames, opt.D, c);
            sums[c] += r.catches; perSeed[c].push(r.catches);
        }
        if ((i + 1) % 8 === 0) process.stderr.write('  ' + (i + 1) + '/' + opt.seeds + '\r');
    }
    const out = { seedStart: opt.seedStart, seeds: opt.seeds, frames: opt.frames, Hs: opt.Hs, twopass: true };
    controllers.forEach(function (c) { out['mean_' + c] = sums[c] / opt.seeds; });
    console.log(JSON.stringify(out));
}

main();
