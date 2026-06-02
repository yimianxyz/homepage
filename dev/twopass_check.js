// Boid-trajectory faithfulness check for the two-pass browser dynamics.
//
// Loads the REAL js/ boid+vector+rng into a VM sandbox, initializes boids from
// a seed exactly as the live page does, pins the predator at a FIXED position
// (so predator-avoidance is exercised but the predator never moves), then runs
// the boid update for F frames in one of two modes and dumps every boid's
// position. A matching sim_torch run (twopass_check.py) is compared against
// this; isolating the boid arithmetic with a fixed predator makes the match
// exact up to floating point.
//
//   single : per frame, sequential per-boid flock()+update()   (= render-only,
//            i.e. dev/oracle.js / eval_js_patrol single pass)
//   two    : per frame, pass1 = all boids flock() from frame-start positions
//            (= simulation.tick()), pass2 = sequential per-boid flock()+update()
//            (= simulation.render()/run()). This is the LIVE page's run().
//
//   node dev/twopass_check.js --js ./js --seed 200000 --frames 300 \
//        --mode two --px 840 --py 840 --out /tmp/js_two.json
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(argv) {
    const a = { js: './js', seed: 200000, frames: 300, mode: 'two',
                px: 840, py: 840, numBoids: 120, width: 1680, height: 1680,
                refreshIntervalMs: 12, out: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--js') a.js = argv[++i];
        else if (k === '--seed') a.seed = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--mode') a.mode = argv[++i];
        else if (k === '--px') a.px = +argv[++i];
        else if (k === '--py') a.py = +argv[++i];
        else if (k === '--numBoids') a.numBoids = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
    }
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

// boid.js + vector.js + rng.js are all we need for pure boid dynamics. We do
// NOT load predator.js / policy_features (no predator policy runs here).
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'simulation.js'];

function buildContext(opt) {
    const ctx = makeStubCtx();
    const win = { innerWidth: opt.width, innerHeight: opt.height,
        matchMedia: function () { return { matches: false, addEventListener: function () {} }; },
        addEventListener: function () {} };
    const sandbox = {
        navigator: { userAgent: 'NodeJSEval' }, window: win,
        document: { getElementById: function () {
            return { getContext: function () { return ctx; }, width: opt.width, height: opt.height }; },
            addEventListener: function () {} },
        fetch: function () { return new Promise(function () {}); },
        renderActivationViz: function () {}, Math: Math, Date: Date, console: console,
    };
    sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    for (const f of JS_FILES) {
        const code = fs.readFileSync(path.join(opt.js, f), 'utf8');
        vm.runInContext(code, context, { filename: 'js/' + f });
    }
    // sim.initialize() constructs a Predator; we pin our own fixed predator
    // afterward, so a position-only stub is all initialize needs.
    sandbox.Predator = function (x, y) { this.position = new sandbox.Vector(x, y); };
    return { sandbox, win };
}

function dumpPositions(sim) {
    const out = [];
    for (let i = 0; i < sim.boids.length; i++) {
        out.push([sim.boids[i].position.x, sim.boids[i].position.y]);
    }
    return out;
}

function main() {
    const opt = parseArgs(process.argv);
    const { sandbox } = buildContext(opt);
    sandbox.setSimSeed(opt.seed, opt.refreshIntervalMs);
    sandbox.NUM_BOIDS = opt.numBoids;
    const sim = new sandbox.Simulation('boids1');
    sim.canvasWidth = opt.width;
    sim.canvasHeight = opt.height;
    sim.initialize(false);
    sandbox.setFrameMs(opt.refreshIntervalMs);
    // Pin a fixed predator. boid.flock reads this.simulation.predator.position.
    sim.predator = { position: new sandbox.Vector(opt.px, opt.py) };
    const boids = sim.boids;

    const initPos = dumpPositions(sim);
    // count boids within PREDATOR_RANGE so we know avoidance is actually active
    const PR = sandbox.PREDATOR_RANGE;
    let inRange = 0;
    for (let i = 0; i < boids.length; i++) {
        const dx = boids[i].position.x - opt.px, dy = boids[i].position.y - opt.py;
        if (Math.sqrt(dx * dx + dy * dy) < PR) inRange++;
    }

    for (let f = 0; f < opt.frames; f++) {
        if (opt.mode === 'two') {
            // pass 1 = simulation.tick(): every boid flock() from frame-start, no move
            for (let i = 0; i < boids.length; i++) boids[i].flock(boids);
        }
        // pass 2 (or the only pass for 'single') = render()/run(): sequential
        for (let i = 0; i < boids.length; i++) { boids[i].flock(boids); boids[i].update(); }
    }
    const finalPos = dumpPositions(sim);
    const res = { mode: opt.mode, seed: opt.seed, frames: opt.frames,
        px: opt.px, py: opt.py, numBoids: boids.length, predator_range: PR,
        boids_in_range_at_start: inRange, init: initPos, final: finalPos };
    const s = JSON.stringify(res);
    if (opt.out) fs.writeFileSync(opt.out, s); else console.log(s);
}
main();
