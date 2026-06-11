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
//
// side-b extensions (#6), all OPT-IN — defaults reproduce the base behavior
// bit-for-bit (diff_harness --selftest proves each):
//   * opt.startBoids (int)  — override NUM_BOIDS before initialize (endgame games).
//   * opt.scatter (bool)    — re-place boids at simRandom()*W/H after initialize
//                             (dev/endgame_fasteval.js recipe, same draw order).
//   * opt.spawnScript       — [{frame,x,y}] deterministic sim.spawnBoid(x,y)
//                             immediately BEFORE that frame's simTick — the
//                             faithful stand-in for the live page's tap-to-spawn
//                             (the only prod path that re-crosses the N<=5 gate
//                             upward). Each spawn draws one simRandom().
//   * opt.uaMobile (bool)   — fake a mobile userAgent (iPad) so NUM_BOIDS=60 /
//                             REFRESH=18 derive as on real mobile hardware.
//                             PREDATOR_RANGE stays 80 on EVERY device: index.html
//                             loads boid.js (line 20) before simulation.js (line
//                             26), so at bake time isMobileDevice AND
//                             PREDATOR_DESKTOP_RANGE are both undefined -> the
//                             literal 80 fallback. UA only affects boid count.
//   * opt.fastRender (bool) — replace Boid/Predator.prototype.render with noops
//                             (they only write to the stubbed ctx; bitwise-
//                             identical state evolution, big speedup at N=120).
//   * setForce(fn)          — sugar for win.__cheap.force = fn.
//   * snapshotState()       — JSONable full sim state (exact f64 round-trip).
//   * loadPolicyAgain()     — re-eval predator_cheap.js (opt.transform applies)
//                             -> an INDEPENDENT prod closure for identity/debug
//                             candidates; restores win.__cheap & friends.
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
    global.navigator = { userAgent: opt.uaMobile
        ? 'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15'
        : 'Node' };
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
    if (opt.startBoids > 0) global.NUM_BOIDS = opt.startBoids;
    global.setSimSeed(opt.seed, frameMs);
    const sim = new global.Simulation('boids1');
    sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
    sim.initialize(false);
    if (opt.scatter) {
        for (let b = 0; b < sim.boids.length; b++) {
            sim.boids[b].position.x = global.simRandom() * opt.W;
            sim.boids[b].position.y = global.simRandom() * opt.H;
        }
    }
    global.setFrameMs(frameMs);
    if (opt.fastRender) {
        global.Boid.prototype.render = noop;
        global.Predator.prototype.render = noop;
    }
    let spawnAt = null;
    if (opt.spawnScript && opt.spawnScript.length) {
        spawnAt = new Map();
        for (const s of opt.spawnScript) {
            if (!spawnAt.has(s.frame)) spawnAt.set(s.frame, []);
            spawnAt.get(s.frame).push(s);
        }
    }
    sim.tick();   // the one-time pre-loop tick() the browser's run() does
    let frame = 0;
    return {
        win, sim,
        api: {
            simTick: global.simTick, getSimFrame: global.getSimFrame,
            getNumBoids: () => global.NUM_BOIDS,
        },
        // One full browser frame. Returns the frame index just executed (1-based).
        stepFrame() {
            if (spawnAt) {
                const sp = spawnAt.get(frame);
                if (sp) for (const s of sp) sim.spawnBoid(s.x, s.y);
            }
            global.simTick(); sim.tick(); sim.render(); return ++frame;
        },
        frame: () => frame,
        boidCount: () => sim.boids.length,
        eaten: () => sim.boidsEaten,
        setForce(fn) { win.__cheap.force = fn; },
        snapshotState() {
            const b = sim.boids, n = b.length, bs = new Array(n);
            for (let i = 0; i < n; i++) {
                bs[i] = { x: b[i].position.x, y: b[i].position.y,
                          vx: b[i].velocity.x, vy: b[i].velocity.y };
            }
            const p = sim.predator;
            return {
                simFrame: global.getSimFrame(), frame, n,
                W: opt.W, H: opt.H, seed: opt.seed,
                startBoids: opt.startBoids || 0, scatter: !!opt.scatter,
                uaMobile: !!opt.uaMobile,
                pred: { x: p.position.x, y: p.position.y,
                        vx: p.velocity.x, vy: p.velocity.y,
                        size: p.currentSize, lastFeed: p.lastFeedTime },
                boids: bs,
            };
        },
        async loadPolicyAgain() {
            const prevCheap = win.__cheap, prevReady = win.__predatorReady,
                  prevModel = win.__predatorModel, prevDebug = win.__cheapDebug;
            let code = fs.readFileSync(path.join(opt.policyDir, 'predator_cheap.js'), 'utf8');
            if (opt.transform) {
                const out = opt.transform('predator_cheap.js', code);
                if (out != null) code = out;
            }
            vm.runInThisContext(code, { filename: 'predator_cheap.js#extra' });
            const extra = { cheap: win.__cheap, ready: win.__predatorReady,
                            debug: win.__cheapDebug };
            win.__cheap = prevCheap; win.__predatorReady = prevReady;
            win.__predatorModel = prevModel; win.__cheapDebug = prevDebug;
            if (extra.ready && typeof extra.ready.then === 'function') await extra.ready;
            return extra;
        },
    };
}

module.exports = { createGame, JS_FILES };
