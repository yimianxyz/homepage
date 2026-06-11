// stepper.js — THE shared faithful game-stepper for the EXACT-NN program.
//
// One module used by BOTH issue #6 (side-b: diff_harness.js) and issue #5
// (side-a: oracle_logger.js), so the two pipelines step the byte-identical
// engine. It wraps dev/fasteval.js's proven buildHarness (same file set, same
// load order, same PREDATOR_RANGE=80 bake as the live page) and adds the hooks
// both consumers need:
//
//   const { createGame } = require('./stepper.js');
//   const game = await createGame({ policyDir, W, H, seed, startBoids, scatter,
//                                   frameMs, fastRender, spawnScript });
//   game.refForce              // detached reference force fn (frozen prod policy closure)
//   game.setForce(fn)          // swap window.__cheap.force — the per-frame hook point
//   await game.loadPolicyAgain() // SECOND independent prod closure (identity candidate / replay)
//   game.step()                // one frame: simTick(); sim.tick(); sim.render()
//   game.snapshotState()       // JSONable full sim state (exact f64 via JSON round-trip)
//   game.boidCount()           // live N
//
// Engine facts this module relies on (verified by reading main@6dce76f):
// - The policy is invoked from Predator.update INSIDE sim.render() — render is
//   load-bearing (boid run/move, predator force+integrate, catch+splice all
//   happen there). tick() only accumulates flock forces (first of two passes).
// - rng.js gives a virtual clock: simNow() = frame*frameMs. No wall-clock can
//   leak into the sim. frameMs is pinned (default 12, matching every existing
//   harness; REFRESH_INTERVAL_IN_MS only matters to the browser's setInterval).
// - Boid objects persist across frames (egBoid identity via indexOf is sound);
//   window.__cheap.force receives the LIVE boids array pre-catch-splice.
// - fastRender substitutes noop draw bodies for Boid/Predator.render. Those
//   methods only write to the (stubbed) canvas ctx — zero state mutation — so
//   the trace is bitwise identical; diff_harness --selftest re-proves this on
//   every run matrix anyway. renderSpawns is kept (it prunes sim.spawns).
// - spawnScript entries {frame,x,y} call sim.spawnBoid(x,y) immediately before
//   that frame's simTick — deterministic stand-in for the live page's tap-to-
//   spawn (the ONLY way prod can cross the N<=5 gate upward). Each spawn draws
//   one simRandom() (Boid ctor angle), so a scripted schedule stays replayable.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { buildHarness } = require('../fasteval.js');

const noop = function () {};

async function createGame(opt) {
    const o = Object.assign({
        policyDir: path.join(__dirname, '..', '..', 'js'),
        W: 390, H: 844, seed: 1,
        startBoids: 0,          // 0 = faithful device default (60 mobile / 120 desktop)
        scatter: false,         // endgame_fasteval-style random scatter after spawn
        frameMs: 12,
        fastRender: true,
        spawnScript: null,      // [{frame,x,y}, ...] sorted or not; applied pre-frame
        config: null,           // passed through to buildHarness (global.__POLICY)
    }, opt || {});

    // Fresh engine + policy per game: predator_cheap's closure state (target,
    // frame counter, egBoid, configured) must not leak across episodes.
    const built = buildHarness(o);
    const api = built.api, win = built.win;
    if (win.__predatorReady && typeof win.__predatorReady.then === 'function') {
        await win.__predatorReady;
    }
    const refCheap = win.__cheap;            // the reference policy instance
    const refForce = refCheap.force;         // closure-only, safe detached

    if (o.startBoids > 0) global.NUM_BOIDS = o.startBoids;

    api.setSimSeed(o.seed, 12);
    const sim = new api.Simulation('boids1');
    sim.canvasWidth = o.W; sim.canvasHeight = o.H;
    sim.initialize(false);
    if (o.scatter) {
        for (let b = 0; b < sim.boids.length; b++) {
            sim.boids[b].position.x = global.simRandom() * o.W;
            sim.boids[b].position.y = global.simRandom() * o.H;
        }
    }
    if (api.setFrameMs) api.setFrameMs(o.frameMs);

    if (o.fastRender) {
        global.Boid.prototype.render = noop;
        global.Predator.prototype.render = noop;
    }

    // spawn schedule -> frame-indexed map (multiple spawns per frame allowed)
    let spawnAt = null;
    if (o.spawnScript && o.spawnScript.length) {
        spawnAt = new Map();
        for (const s of o.spawnScript) {
            if (!spawnAt.has(s.frame)) spawnAt.set(s.frame, []);
            spawnAt.get(s.frame).push(s);
        }
    }

    sim.tick();   // the one-time pre-loop tick() the browser's run() does

    let frameIdx = 0;   // frames stepped so far (0-based; first step() => frame 0)
    function step() {
        if (spawnAt) {
            const sp = spawnAt.get(frameIdx);
            if (sp) for (const s of sp) sim.spawnBoid(s.x, s.y);
        }
        api.simTick(); sim.tick(); sim.render();
        frameIdx++;
    }

    function snapshotState() {
        const b = sim.boids, n = b.length;
        const bs = new Array(n);
        for (let i = 0; i < n; i++) {
            bs[i] = { x: b[i].position.x, y: b[i].position.y,
                      vx: b[i].velocity.x, vy: b[i].velocity.y };
        }
        const p = sim.predator;
        return {
            simFrame: global.getSimFrame(), frameIdx, n,
            W: o.W, H: o.H, seed: o.seed, startBoids: o.startBoids,
            scatter: o.scatter,
            pred: { x: p.position.x, y: p.position.y,
                    vx: p.velocity.x, vy: p.velocity.y,
                    size: p.currentSize, lastFeed: p.lastFeedTime },
            boids: bs,
        };
    }

    // Re-eval predator_cheap.js in this same context: an INDEPENDENT prod
    // closure (fresh target/frame/egBoid/cfg state, own NET parse). Restores
    // window.__cheap / __predatorReady / __predatorModel so the reference and
    // the running sim are untouched.
    async function loadPolicyAgain() {
        const prevCheap = win.__cheap, prevReady = win.__predatorReady,
              prevModel = win.__predatorModel;
        const src = fs.readFileSync(path.join(o.policyDir, 'predator_cheap.js'), 'utf8');
        vm.runInThisContext(src, { filename: 'predator_cheap.js#extra' });
        const extraCheap = win.__cheap, extraReady = win.__predatorReady;
        win.__cheap = prevCheap; win.__predatorReady = prevReady;
        win.__predatorModel = prevModel;
        if (extraReady && typeof extraReady.then === 'function') await extraReady;
        return extraCheap;
    }

    return {
        sim, api, win, opt: o,
        refCheap, refForce,
        setForce(fn) { win.__cheap.force = fn; },
        step,
        snapshotState,
        loadPolicyAgain,
        boidCount() { return sim.boids.length; },
        frame() { return frameIdx; },
    };
}

module.exports = { createGame };
