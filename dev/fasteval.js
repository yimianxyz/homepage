// Fast pure-node headless eval of a predator policy on the REAL game engine.
//
// Why this exists: eval_cheap_production.js runs the same files via
// vm.runInContext, where every hot-loop access to a sandbox global goes through
// a V8 interceptor (~300 ms/frame). Loading the SAME source concatenated into
// one CommonJS module scope makes those accesses ordinary fast var lookups
// (~10-15 ms/frame, browser-class), so a 1500-frame episode is ~20s -> <2s.
// That turns policy search from "minutes per variant" into "hundreds per minute"
// across cores/VMs, with NO GPU-inflation or sim_torch parity burden — this is
// the actual shipped engine.
//
// Faithfulness: replicates index.html's load order + boids.js's two-pass frame
// loop (simTick; sim.tick; sim.render). NUM_BOIDS/PREDATOR_RANGE are set from the
// requested viewport exactly as the page derives them (isMobileDevice via a faked
// window.innerWidth; PREDATOR_RANGE bakes to 80 because boid.js evaluates it
// before simulation.js defines isMobileDevice — matches production).
//
//   node dev/fasteval.js --policyDir js --W 390 --H 844 --seedStart 200000 --seeds 16 --frames 1500
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', 'js'), W: 390, H: 844,
        seedStart: 200000, seeds: 16, frames: 1500, json: true };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--perseed') a.perseed = true;
        else if (k === '--config') a.config = argv[++i];
    }
    return a;
}

// Same set + order index.html loads (the policy/sim files; DOM-only files skipped).
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js',
    'cheap_planner.js', 'predator_cheap.js'];

function buildHarness(opt) {
    // Browser-API stubs as REAL globals; each policy file then runs via
    // vm.runInThisContext (real V8 context → fast global access, AND a separate
    // script per file → per-file function hoisting that matches the browser's
    // separate <script> tags — so PREDATOR_RANGE bakes to 80 exactly as in prod).
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
    if (opt.config) {
        try { global.__POLICY = JSON.parse(opt.config); } catch (e) { /* ignore */ }
    }
    global.window = win;
    global.self = global;
    global.navigator = { userAgent: 'Node' };
    global.document = { getElementById: () => ({ getContext: () => cctx, width: opt.W, height: opt.H }),
        addEventListener: noop };
    global.renderActivationViz = noop;
    global.fetch = function (url) {
        const f = path.join(opt.policyDir, path.basename(url));
        return Promise.resolve({ ok: true, status: 200,
            json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, 'utf8'))) });
    };
    for (const f of JS_FILES) {
        const code = fs.readFileSync(path.join(opt.policyDir, f), 'utf8');
        vm.runInThisContext(code, { filename: f });
    }
    // Production loads boid.js BEFORE simulation.js defines isMobileDevice, so
    // `var PREDATOR_RANGE = getBoidPredatorRange()` always bakes to 80 (desktop
    // fallback). On our per-episode rebuild, isMobileDevice lingers in the global
    // from the prior episode and would flip it to 60 — so lock it to the faithful
    // value (config override allowed for range experiments).
    global.PREDATOR_RANGE = (global.__POLICY && global.__POLICY.predRange != null)
        ? global.__POLICY.predRange : 80;
    const api = {
        setSimSeed: global.setSimSeed, setFrameMs: global.setFrameMs, simTick: global.simTick,
        Simulation: global.Simulation,
        getNumBoids: () => global.NUM_BOIDS,
        getPredRange: () => (typeof global.PREDATOR_RANGE !== 'undefined' ? global.PREDATOR_RANGE : null),
    };
    return { api, win };
}

async function main() {
    const opt = parseArgs(process.argv);
    const per = [];
    let api = null;
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        // Rebuild the policy fresh PER EPISODE: the cheap policy holds module
        // state (committed target, frame counter, configured flag, grid scratch)
        // in its IIFE closure that would otherwise leak from seed k into seed k+1,
        // making episodes non-independent (seed 200002 run alone != as the 3rd of
        // a batch). Production loads a fresh policy per page, so independent
        // episodes are the faithful measurement. Re-eval cost is ~ms vs the episode.
        const built = buildHarness(opt);
        api = built.api;
        const win = built.win;
        if (win.__predatorReady && typeof win.__predatorReady.then === 'function') {
            await win.__predatorReady;
        }
        api.setSimSeed(seed, 12);
        const sim = new api.Simulation('boids1');
        sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
        sim.initialize(false);
        if (api.setFrameMs) api.setFrameMs(12);
        sim.tick();   // the one-time pre-loop tick() the browser's run() does before its interval
        for (let f = 0; f < opt.frames; f++) { api.simTick(); sim.tick(); sim.render(); }
        per.push(sim.boidsEaten);
    }
    const sum = per.reduce((a, b) => a + b, 0);
    const mean = sum / per.length;
    const sd = Math.sqrt(per.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, per.length - 1));
    const se = sd / Math.sqrt(per.length);
    const out = { policyDir: path.basename(opt.policyDir), W: opt.W, H: opt.H,
        numBoids: api.getNumBoids(), predRange: api.getPredRange(),
        seedStart: opt.seedStart, seeds: opt.seeds, frames: opt.frames, mean, se };
    if (opt.perseed) out.per = per;
    console.log(JSON.stringify(out));
}
if (require.main === module) {
    main().catch(e => { console.error(e); process.exit(1); });
}
module.exports = { buildHarness, JS_FILES };
