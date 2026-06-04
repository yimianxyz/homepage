// Benchmark the REAL JS flock rollout cost (predator_planner_worker.rolloutFlat),
// to learn the actual per-decision latency of a lookahead student in the browser
// regime. This is the load-bearing unknown: catches are good (0.93x) but can the
// rollout run fast enough (in a Web Worker, every D frames) to deploy?
//
// Times K rolloutFlat(Hs) calls at (a) frame-0 (boids spread) and (b) a clustered
// mid-episode state (worst case, where the grid degrades). Reuses the sandbox
// loader from eval_planner_worker.js.
//
//   node dev/bench_rollout.js --js ./js --K 16 --Hs 60 --seed 200000
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(a) {
    const o = { js: './js', K: 16, Hs: 60, seed: 200000, warm: 200, width: 1680, height: 1680,
                numBoids: 120, refreshIntervalMs: 12, POLICY_R: 80 };
    for (let i = 2; i < a.length; i++) {
        const k = a[i];
        if (k === '--js') o.js = a[++i]; else if (k === '--K') o.K = +a[++i];
        else if (k === '--Hs') o.Hs = +a[++i]; else if (k === '--seed') o.seed = +a[++i];
        else if (k === '--warm') o.warm = +a[++i];
    }
    return o;
}
function makeStubCtx() {
    const noop = function () {};
    const h = { get() { return ''; }, set() {} };
    return new Proxy({ createLinearGradient: () => ({ addColorStop: noop }) },
        { get: (t, p) => (p in t ? t[p] : (typeof p === 'string' && /^(get|set)/.test(p) ? undefined : noop)) });
}
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'policy_features.js',
                  'predator_nn.js', 'predator.js', 'simulation.js'];
function build(opt, weights) {
    const ctx = makeStubCtx();
    const win = { innerWidth: opt.width, innerHeight: opt.height,
        matchMedia: () => ({ matches: false, addEventListener() {} }), addEventListener() {} };
    const sandbox = { navigator: { userAgent: 'node' }, window: win,
        document: { getElementById: () => ({ getContext: () => ctx, width: opt.width, height: opt.height }), addEventListener() {} },
        fetch: () => new Promise(() => {}), renderActivationViz() {}, importScripts() {}, postMessage() {},
        Math, Date, console };
    sandbox.global = sandbox;
    const c = vm.createContext(sandbox);
    for (const f of JS_FILES) vm.runInContext(fs.readFileSync(path.join(opt.js, f), 'utf8'), c, { filename: f });
    win.__predatorModel = sandbox.PredatorNN.loadModel(weights);
    vm.runInContext(fs.readFileSync(path.join(opt.js, 'predator_planner_worker.js'), 'utf8'), c, { filename: 'predator_planner_worker.js' });
    sandbox.cfg.K = opt.K; sandbox.cfg.H = opt.Hs; sandbox.cfg.POLICY_R = opt.POLICY_R;
    sandbox.cfg.W = opt.width; sandbox.cfg.Hc = opt.height; sandbox.PREDATOR_RANGE = 80;
    return sandbox;
}
function snapFromSim(sandbox, opt, warmFrames) {
    sandbox.setSimSeed(opt.seed, opt.refreshIntervalMs);
    sandbox.NUM_BOIDS = opt.numBoids;
    const sim = new sandbox.Simulation('boids1');
    sim.canvasWidth = opt.width; sim.canvasHeight = opt.height;
    sim.initialize(false); sandbox.setFrameMs(opt.refreshIntervalMs);
    // warm the live sim forward warmFrames so boids cluster around the predator
    for (let f = 0; f < warmFrames; f++) { sandbox.simTick(); sim.render(); }
    const pred = sim.predator, boids = sim.boids.filter(b => b.alive !== false), n = boids.length;
    const bx = [], by = [], bvx = [], bvy = [];
    for (let i = 0; i < n; i++) { bx.push(boids[i].position.x); by.push(boids[i].position.y); bvx.push(boids[i].velocity.x); bvy.push(boids[i].velocity.y); }
    return { bx, by, bvx, bvy, px: pred.position.x, py: pred.position.y, pvx: pred.velocity.x, pvy: pred.velocity.y, psize: pred.currentSize, lastFeed: 0, nowMs: 0 };
}
function timeKRollouts(sandbox, snap, K, Hs, reps) {
    // candidates() gives K target points; time K rolloutFlat(Hs) = one decision
    const cand = sandbox.candidates(snap);
    let total = 0, t;
    for (let r = 0; r < reps; r++) {
        t = process.hrtime.bigint();
        for (let k = 0; k < cand.length; k++) sandbox.rolloutFlat(snap, cand[k].x, cand[k].y, Hs);
        total += Number(process.hrtime.bigint() - t) / 1e6;
    }
    return { perDecisionMs: total / reps, K: cand.length };
}
function main() {
    const opt = parseArgs(process.argv);
    const weights = JSON.parse(fs.readFileSync(path.join(opt.js, 'predator_weights.json'), 'utf8'));
    const sandbox = build(opt, weights);
    for (const warm of [0, opt.warm, 2000]) {
        const snap = snapFromSim(sandbox, opt, warm);
        const r = timeKRollouts(sandbox, snap, opt.K, opt.Hs, 5);
        console.log(JSON.stringify({ warmFrames: warm, K: r.K, Hs: opt.Hs, per_decision_ms: +r.perDecisionMs.toFixed(2),
            decisions_per_sec: +(1000 / r.perDecisionMs).toFixed(1) }));
    }
}
main();
