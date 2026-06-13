// THROUGHPUT GATE — node single-core baseline for prod's planCheap.
//
// Drives the REAL shipped policy (js/predator_cheap.js + cheap_planner.js +
// value_net.json) through the fasteval.js loader (same file set + load order as
// index.html; vm.runInThisContext for browser-class speed — see dev/fasteval.js
// header). Measures wall time of window.__cheap.force() calls on a frozen
// realistic state; the policy plans (planCheap: 16 cands, 80 net forwards,
// 4x90-step flock rollouts) on every 16th call (cfg.D=16) and only steers
// (O(N) trivial) on the rest, so plan calls are unambiguous in the per-call
// timing distribution (ms vs ~us). plans/sec = the gate's node-side number.
//
//   node dev/exact_nn/replica_gate/bench_node.js --N 120 --plans 60
//   node dev/exact_nn/replica_gate/bench_node.js --N 30  --plans 100
'use strict';
const path = require('path');
const { buildHarness } = require(path.join(__dirname, '..', '..', 'fasteval.js'));

function parseArgs(argv) {
    // W=1512,H=982 -> desktop device cell: NUM_BOIDS=120, PREDATOR_RANGE=80.
    const a = { N: 120, W: 1512, H: 982, seed: 200000, warmFrames: 150, plans: 60,
        policyDir: path.join(__dirname, '..', '..', '..', 'js') };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--N') a.N = +argv[++i];
        else if (k === '--plans') a.plans = +argv[++i];
        else if (k === '--seed') a.seed = +argv[++i];
        else if (k === '--warmFrames') a.warmFrames = +argv[++i];
        else if (k === '--policyDir') a.policyDir = argv[++i];
    }
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    const built = buildHarness({ W: opt.W, H: opt.H, policyDir: opt.policyDir });
    const win = built.win;
    if (win.__predatorReady && typeof win.__predatorReady.then === 'function') {
        await win.__predatorReady;     // value_net.json loaded (fetch stub)
    }
    built.api.setSimSeed(opt.seed, 12);
    const sim = new built.api.Simulation('boids1');
    sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
    sim.initialize(false);
    if (built.api.setFrameMs) built.api.setFrameMs(12);
    sim.tick();
    // Develop a realistic flock (clustered neighborhoods -> realistic grid
    // load). The predator catches a few boids during warmup, so keep the last
    // frame that still has >= N alive (flock structure develops in ~50 frames).
    const V = global.Vector;
    const snap = () => ({
        boids: sim.boids.map(b => ({
            position: new V(b.position.x, b.position.y),
            velocity: new V(b.velocity.x, b.velocity.y) })),
        pred: { x: sim.predator.position.x, y: sim.predator.position.y,
            vx: sim.predator.velocity.x, vy: sim.predator.velocity.y,
            currentSize: sim.predator.currentSize, lastFeedTime: sim.predator.lastFeedTime },
    });
    let frozen = snap(), warmed = 0;
    for (let f = 0; f < opt.warmFrames; f++) {
        built.api.simTick(); sim.tick(); sim.render();
        if (sim.boids.length >= opt.N) { frozen = snap(); warmed = f + 1; }
    }

    // Freeze the state into plain stubs (force() never mutates its inputs:
    // planCheap works on a snapshot copy, steer only reads).
    const all = frozen.boids;
    let boids = all;
    if (opt.N < all.length) {
        // Even subsample keeps the spatial spread of the developed flock
        // (a mid-game N=30 state; slightly sparser than a true late-game flock,
        // which under-counts node grid work -> conservative toward KILL).
        boids = [];
        const stride = all.length / opt.N;
        for (let i = 0; i < opt.N; i++) boids.push(all[Math.round(i * stride)]);
    } else if (opt.N > all.length) {
        throw new Error(`asked N=${opt.N} > ${all.length} boids in sim`);
    }
    const p = frozen.pred;
    const predStub = {
        position: new V(p.x, p.y),
        velocity: new V(p.vx, p.vy),
        currentSize: p.currentSize, lastFeedTime: p.lastFeedTime,
        simulation: { canvasWidth: opt.W, canvasHeight: opt.H },
    };
    const force = win.__cheap.force;

    // Timing loop: every call is timed; plan calls (1 per 16) are separated
    // from steer calls post-hoc (plan >= 50x the median call, which is a steer).
    const wantPlans = opt.plans + 5;            // first 5 plans dropped (JIT warmup)
    const durs = [];
    let t0all = process.hrtime.bigint();
    for (let c = 0; c < wantPlans * 16 + 1; c++) {
        const t0 = process.hrtime.bigint();
        force(predStub, boids);
        durs.push(Number(process.hrtime.bigint() - t0) / 1e6); // ms
    }
    const totalMs = Number(process.hrtime.bigint() - t0all) / 1e6;
    const sorted = durs.slice().sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)];          // a steer call
    const planDur = durs.filter(d => d > 50 * med && d > 0.2);
    if (Math.abs(planDur.length - Math.floor(durs.length / 16) - 1) > 2) {
        console.error(`WARN plan classification: got ${planDur.length} plans from ${durs.length} calls`);
    }
    const keep = planDur.slice(5);              // drop JIT warmup plans
    keep.sort((a, b) => a - b);
    const mean = keep.reduce((a, b) => a + b, 0) / keep.length;
    const median = keep[Math.floor(keep.length / 2)];
    const out = {
        N: boids.length, W: opt.W, H: opt.H, seed: opt.seed, node: process.version,
        nPlans: keep.length, plan_ms_mean: +mean.toFixed(3), plan_ms_median: +median.toFixed(3),
        plans_per_sec_mean: +(1000 / mean).toFixed(2),
        plans_per_sec_median: +(1000 / median).toFixed(2),
        steer_ms_median: +med.toFixed(4),
        total_ms: +totalMs.toFixed(1),
    };
    console.log(JSON.stringify(out));
}
main().catch(e => { console.error(e); process.exit(1); });
