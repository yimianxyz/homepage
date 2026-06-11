// exact_nn shared game-stepper — THE one faithful stepping module for the
// oracle logger (#5) and the lockstep differential harness (#6). Extracted
// from dev/fasteval.js buildHarness + the browser's Simulation.run() sequence
// so both consumers step bit-identical games; do not fork this.
//
// Faithfulness contract (mirrors index.html + boids.js):
//   * load order: rng, vector, boid, predator, simulation, cheap_planner,
//     predator_cheap — separate scripts (per-file hoisting, PREDATOR_RANGE
//     bakes to 80 exactly as in prod; see dev/fasteval.js for the full story).
//   * init: setSimSeed(seed, frameMs) -> new Simulation -> canvas dims ->
//     initialize(false) -> setFrameMs(frameMs) -> one pre-loop sim.tick().
//   * frame: simTick(); sim.tick(); sim.render()   (render() moves the
//     predator, applies catches, and is where __cheap.force runs).
//
// One game per createGame() call; policy closure state (target/frame/egBoid)
// is rebuilt fresh each call exactly like a browser page load. Games must be
// run sequentially within a process (globals are swapped per game).
//
// Seams for consumers (no js/ edits, no behavior change):
//   * opt.transform(filename, code) -> code'   — in-memory source hook
//     injection before vm.runInThisContext (the oracle uses anchored
//     one-line hooks; pristine runs pass no transform).
//   * the returned win lets a consumer wrap win.__cheap.force AFTER load
//     (the diff harness evaluates candidate policies on the same state).
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

// Same set + order index.html loads (DOM-only files skipped).
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js',
    'cheap_planner.js', 'predator_cheap.js'];

// opt: { policyDir, W, H, seed, frameMs (default 12 — see note below),
//        transform?: (filename, code) => code|null }
// Returns { win, sim, api, stepFrame, frame(), boidCount(), eaten() }.
//
// frameMs: the live page uses REFRESH_INTERVAL_IN_MS (18 if isMobileDevice()
// else 12) as the virtual ms-per-frame of rng.js's simNow() clock. The value
// is causally irrelevant to the policy output (lastFeed/nowMs are threaded
// through snapshots but never read back into any decision; decaySize() is
// never called) — verified empirically by dev/exact_nn/oracle_logger.js
// --selftest. We default to 12 everywhere, matching dev/fasteval.js.
async function createGame(opt) {
    if (!opt || !opt.policyDir || !opt.W || !opt.H || opt.seed == null) {
        throw new Error('createGame: policyDir, W, H, seed are required');
    }
    const frameMs = opt.frameMs == null ? 12 : opt.frameMs;
    const noop = function () {};
    const ctxMethods = ['beginPath', 'moveTo', 'lineTo', 'stroke', 'fill', 'arc',
        'clearRect', 'fillRect', 'strokeRect', 'closePath', 'save', 'restore',
        'translate', 'rotate', 'scale', 'fillText', 'setLineDash', 'ellipse',
        'quadraticCurveTo', 'bezierCurveTo'];
    const cctx = {};
    ctxMethods.forEach(m => cctx[m] = noop);
    cctx.createLinearGradient = () => ({ addColorStop: noop });
    const win = {
        innerWidth: opt.W, innerHeight: opt.H,
        matchMedia: () => ({ matches: false, addEventListener: noop }),
        addEventListener: noop,
    };
    global.window = win;
    global.self = global;
    global.navigator = { userAgent: 'Node' };
    global.document = {
        getElementById: () => ({ getContext: () => cctx, width: opt.W, height: opt.H }),
        addEventListener: noop,
    };
    global.renderActivationViz = noop;
    global.fetch = function (url) {
        const f = path.join(opt.policyDir, path.basename(url));
        return Promise.resolve({ ok: true, status: 200,
            json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, 'utf8'))) });
    };
    for (const f of JS_FILES) {
        let code = fs.readFileSync(path.join(opt.policyDir, f), 'utf8');
        if (opt.transform) {
            const out = opt.transform(f, code);
            if (out != null) code = out;
        }
        vm.runInThisContext(code, { filename: f });
    }
    // Production quirk, kept faithful: boid.js evaluates PREDATOR_RANGE before
    // simulation.js defines isMobileDevice, so it always bakes to 80. On a
    // re-load within one process the lingering isMobileDevice would flip it.
    global.PREDATOR_RANGE = 80;
    if (win.__predatorReady && typeof win.__predatorReady.then === 'function') {
        await win.__predatorReady;
    }
    global.setSimSeed(opt.seed, frameMs);
    const sim = new global.Simulation('boids1');
    sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
    sim.initialize(false);
    global.setFrameMs(frameMs);
    sim.tick();   // the one-time pre-loop tick() the browser's run() does
    let frame = 0;
    return {
        win, sim,
        api: {
            simTick: global.simTick, getSimFrame: global.getSimFrame,
            getNumBoids: () => global.NUM_BOIDS,
        },
        // One full browser frame. Returns the frame index just executed (1-based).
        stepFrame() { global.simTick(); sim.tick(); sim.render(); return ++frame; },
        frame: () => frame,
        boidCount: () => sim.boids.length,
        eaten: () => sim.boidsEaten,
    };
}

module.exports = { createGame, JS_FILES };
