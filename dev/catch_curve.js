// Per-catch timing curve: average frame at which the k-th catch happens, and the
// inter-catch gap as a function of boids-remaining. Reveals exactly where the
// clearance time is spent (the slow tail). Runs to extinction or maxFrames.
//   node dev/catch_curve.js --policyDir js --W 390 --H 844 --seeds 24 --maxFrames 30000
'use strict';
const path = require('path');
const fs = require('fs');
const vm = require('vm');
const { JS_FILES } = require('./fasteval.js');

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', 'js'), W: 390, H: 844,
        seedStart: 200000, seeds: 24, maxFrames: 30000, config: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--config') a.config = argv[++i];
    }
    return a;
}

function build(opt) {
    const noop = function () {};
    const cctx = {};
    ['beginPath','moveTo','lineTo','stroke','fill','arc','clearRect','fillRect','strokeRect','closePath','save','restore','translate','rotate','scale','fillText','setLineDash','ellipse','quadraticCurveTo','bezierCurveTo'].forEach(m => cctx[m] = noop);
    cctx.createLinearGradient = () => ({ addColorStop: noop });
    const win = { innerWidth: opt.W, innerHeight: opt.H, matchMedia: () => ({ matches: false, addEventListener: noop }), addEventListener: noop };
    global.__POLICY = opt.config ? JSON.parse(opt.config) : {};
    global.window = win; global.self = global; global.navigator = { userAgent: 'Node' };
    global.document = { getElementById: () => ({ getContext: () => cctx, width: opt.W, height: opt.H }), addEventListener: noop };
    global.renderActivationViz = noop;
    global.fetch = function (url) { const f = path.join(opt.policyDir, path.basename(url)); return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, 'utf8'))) }); };
    for (const f of JS_FILES) vm.runInThisContext(fs.readFileSync(path.join(opt.policyDir, f), 'utf8'), { filename: f });
    global.PREDATOR_RANGE = (global.__POLICY && global.__POLICY.predRange != null) ? global.__POLICY.predRange : 80;
    return { win, api: { setSimSeed: global.setSimSeed, setFrameMs: global.setFrameMs, simTick: global.simTick, Simulation: global.Simulation, getNumBoids: () => global.NUM_BOIDS } };
}

async function main() {
    const opt = parseArgs(process.argv);
    let N0 = 0;
    const catchFrameByK = [];   // catchFrameByK[k] = array over seeds of frame of (k+1)-th catch (or maxFrames if never)
    for (let i = 0; i < opt.seeds; i++) {
        const built = build(opt); const api = built.api, win = built.win;
        if (win.__predatorReady && win.__predatorReady.then) await win.__predatorReady;
        api.setSimSeed(opt.seedStart + i, 12);
        const sim = new api.Simulation('boids1');
        sim.canvasWidth = opt.W; sim.canvasHeight = opt.H; sim.initialize(false);
        if (api.setFrameMs) api.setFrameMs(12);
        sim.tick();
        N0 = sim.boids.length;
        let prevEaten = 0;
        const cf = new Array(N0).fill(opt.maxFrames);
        for (let f = 0; f < opt.maxFrames; f++) {
            api.simTick(); sim.tick(); sim.render();
            while (sim.boidsEaten > prevEaten) { cf[prevEaten] = f + 1; prevEaten++; }
            if (sim.boids.length === 0) break;
        }
        for (let k = 0; k < N0; k++) { if (!catchFrameByK[k]) catchFrameByK[k] = []; catchFrameByK[k].push(cf[k]); }
    }
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    // report cumulative catch frame and per-catch gap at selected remaining-counts
    const rows = [];
    let prevMean = 0;
    for (let k = 0; k < N0; k++) {
        const m = mean(catchFrameByK[k]);
        const remainingAfter = N0 - (k + 1);
        const gap = m - prevMean;
        rows.push({ kth: k + 1, remainingBefore: N0 - k, cumFrame: Math.round(m), gapFromPrev: Math.round(gap) });
        prevMean = m;
    }
    // print compact: show every catch for last 15, sampled for the rest
    console.log('# policy=' + path.basename(opt.policyDir) + ' config=' + (opt.config || '{}') + ' W=' + opt.W + ' H=' + opt.H + ' N0=' + N0 + ' seeds=' + opt.seeds + ' maxFrames=' + opt.maxFrames);
    console.log('kth\tremain\tcumFrame\tgap');
    for (const r of rows) {
        if (r.remainingBefore <= 12 || r.kth % 10 === 0 || r.kth === 1) {
            console.log(r.kth + '\t' + r.remainingBefore + '\t' + r.cumFrame + '\t' + r.gapFromPrev);
        }
    }
    // total clearance (mean of last catch frame)
    console.log('# tClear(mean last catch) = ' + Math.round(mean(catchFrameByK[N0 - 1])));
}
main().catch(e => { console.error(e); process.exit(1); });
