// Clearance eval — the new north-star metric. Runs the real game until EVERY boid
// is caught (extinction) or a max-frame cap, and reports how long it takes. This
// unifies the two sub-problems the 1500-frame "catches" metric hid:
//   * FLOCK phase: clear the dense cluster (rate-limited, the deployed policy is good)
//   * ENDGAME phase: hunt the last few scattered singletons (the slow predator's
//     hard problem; where torus-edge interception matters)
// Some policies never clear the last boid, so non-clears are censored at maxFrames.
//
// Per seed it records the frame index of each catch, then decomposes:
//   t_flock   = frames to go from N down to ENDGAME_K boids
//   t_endgame = frames from ENDGAME_K down to 0 (maxFrames if never reached)
//   t_clear   = total frames to extinction (maxFrames if not cleared)
//
//   node dev/clear_eval.js --policyDir js --W 390 --H 844 --seeds 32 --maxFrames 12000
'use strict';
const path = require('path');
const { buildHarness } = require('./fasteval.js');

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', 'js'), W: 390, H: 844,
        seedStart: 200000, seeds: 32, maxFrames: 12000, endK: 5, config: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--endK') a.endK = +argv[++i];
        else if (k === '--config') a.config = argv[++i];
        else if (k === '--perseed') a.perseed = true;
    }
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    const tClear = [], tFlock = [], tEnd = [], cleared = [];
    // granular: frame at which boid count first reaches each checkpoint
    const CKPT = [45, 30, 20, 12, 8, 5, 3, 2, 1];
    const ckFrames = CKPT.map(() => []);
    let N0 = 0;
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        const built = buildHarness(opt);
        const api = built.api, win = built.win;
        if (win.__predatorReady && win.__predatorReady.then) await win.__predatorReady;
        api.setSimSeed(seed, 12);
        const sim = new api.Simulation('boids1');
        sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
        sim.initialize(false);
        if (api.setFrameMs) api.setFrameMs(12);
        sim.tick();
        N0 = sim.boids.length;
        let prevN = N0, flockFrame = -1, clearFrame = -1;
        const ckHit = CKPT.map(() => -1);
        for (let f = 0; f < opt.maxFrames; f++) {
            api.simTick(); sim.tick(); sim.render();
            const n = sim.boids.length;
            for (let c = 0; c < CKPT.length; c++) if (ckHit[c] < 0 && n <= CKPT[c]) ckHit[c] = f + 1;
            if (n <= opt.endK && flockFrame < 0) flockFrame = f + 1;
            if (n === 0) { clearFrame = f + 1; break; }
            prevN = n;
        }
        for (let c = 0; c < CKPT.length; c++) ckFrames[c].push(ckHit[c] >= 0 ? ckHit[c] : opt.maxFrames);
        const tc = clearFrame > 0 ? clearFrame : opt.maxFrames;
        const tf = flockFrame > 0 ? flockFrame : opt.maxFrames;
        tClear.push(tc); tFlock.push(tf); tEnd.push(tc - tf); cleared.push(clearFrame > 0 ? 1 : 0);
    }
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const se = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, a.length - 1)) / Math.sqrt(a.length); };
    const out = { policyDir: path.basename(opt.policyDir), W: opt.W, H: opt.H, N0,
        seeds: opt.seeds, maxFrames: opt.maxFrames, endK: opt.endK,
        clearRate: +mean(cleared).toFixed(3),
        tClear: +mean(tClear).toFixed(0), seClear: +se(tClear).toFixed(0),
        tFlock: +mean(tFlock).toFixed(0), tEnd: +mean(tEnd).toFixed(0) };
    // per-segment durations (frames to go between consecutive checkpoints)
    out.ckpt = {};
    let prev = 0;
    for (let c = 0; c < CKPT.length; c++) {
        const m = mean(ckFrames[c]);
        out.ckpt['>=' + CKPT[c]] = +(m - prev).toFixed(0);  // time spent at count in (CKPT[c-1], CKPT[c]]
        prev = m;
    }
    if (opt.perseed) { out.tClearArr = tClear; out.clearedArr = cleared; }
    console.log(JSON.stringify(out));
}
main().catch(e => { console.error(e); process.exit(1); });
