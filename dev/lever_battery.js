// Lever battery — A/B every candidate lever from the skeptic's brief on the REAL
// engine via the exp/js config harness. Reports mean catches/1500f + SE, and the
// paired delta vs the no-config baseline on the SAME seed set (so the comparison
// is within-seed paired -> low-variance). Run per device size.
//
//   node dev/lever_battery.js --W 390 --H 844 --seeds 32 --frames 1500
'use strict';
const path = require('path');
const fs = require('fs');
const vm = require('vm');
const { JS_FILES } = require('./fasteval.js');

function parseArgs(argv) {
    const a = { W: 390, H: 844, seedStart: 200000, seeds: 32, frames: 1500,
        policyDir: path.join(__dirname, 'exp', 'js') };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--policyDir') a.policyDir = argv[++i];
    }
    return a;
}

function build(opt, config) {
    const noop = function () {};
    const cctx = {};
    ['beginPath','moveTo','lineTo','stroke','fill','arc','clearRect','fillRect','strokeRect','closePath','save','restore','translate','rotate','scale','fillText','setLineDash','ellipse','quadraticCurveTo','bezierCurveTo'].forEach(m => cctx[m] = noop);
    cctx.createLinearGradient = () => ({ addColorStop: noop });
    const win = { innerWidth: opt.W, innerHeight: opt.H, matchMedia: () => ({ matches: false, addEventListener: noop }), addEventListener: noop };
    global.__POLICY = config || {};
    global.window = win; global.self = global; global.navigator = { userAgent: 'Node' };
    global.document = { getElementById: () => ({ getContext: () => cctx, width: opt.W, height: opt.H }), addEventListener: noop };
    global.renderActivationViz = noop;
    global.fetch = function (url) { const f = path.join(opt.policyDir, path.basename(url)); return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, 'utf8'))) }); };
    for (const f of JS_FILES) vm.runInThisContext(fs.readFileSync(path.join(opt.policyDir, f), 'utf8'), { filename: f });
    global.PREDATOR_RANGE = (config && config.predRange != null) ? config.predRange : 80;
    return { win, api: { setSimSeed: global.setSimSeed, setFrameMs: global.setFrameMs, simTick: global.simTick, Simulation: global.Simulation } };
}

async function runConfig(opt, config) {
    const per = [];
    for (let i = 0; i < opt.seeds; i++) {
        const built = build(opt, config);
        const api = built.api, win = built.win;
        if (win.__predatorReady && win.__predatorReady.then) await win.__predatorReady;
        api.setSimSeed(opt.seedStart + i, 12);
        const sim = new api.Simulation('boids1');
        sim.canvasWidth = opt.W; sim.canvasHeight = opt.H; sim.initialize(false);
        if (api.setFrameMs) api.setFrameMs(12);
        sim.tick();
        for (let f = 0; f < opt.frames; f++) { api.simTick(); sim.tick(); sim.render(); }
        per.push(sim.boidsEaten);
    }
    return per;
}

const LEVERS = [
    { name: 'baseline (prod)', cfg: {} },
    // selector / search-depth levers
    { name: 'K_roll 8 (deeper prune)', cfg: { K_roll: 8 } },
    { name: 'K_roll 16 (roll all)', cfg: { K_roll: 16 } },
    { name: 'prune=net', cfg: { prune: 'net' } },
    { name: 'prune=sum', cfg: { prune: 'sum' } },
    { name: 'Hs 140 (deeper roll)', cfg: { Hs: 140 } },
    { name: 'D 8 (replan often)', cfg: { D: 8 } },
    { name: 'D 4 (replan v.often)', cfg: { D: 4 } },
    // toroidal distance
    { name: 'wrap on', cfg: { wrap: true } },
    // POLICY_R (when to chase vs patrol)
    { name: 'POLICY_R 120', cfg: { POLICY_R: 120 } },
    { name: 'POLICY_R 200', cfg: { POLICY_R: 200 } },
    // ambush / compression / herding
    { name: 'ambush', cfg: { ambush: true } },
    { name: 'compress 0.3', cfg: { compress: 0.3 } },
    { name: 'compress 1.0', cfg: { compress: 1.0 } },
    // endgame interceptor
    { name: 'endgame K5', cfg: { endgame: true, endgameK: 5 } },
    { name: 'endgame K8 + isolate', cfg: { endgame: true, endgameK: 8, egIsolate: true, egIsoMax: 30 } },
    { name: 'endgame K12 + isolate40', cfg: { endgame: true, endgameK: 12, egIsolate: true, egIsoMax: 40 } },
    // combos
    { name: 'wrap + K_roll16 + endgameIso', cfg: { wrap: true, K_roll: 16, endgame: true, endgameK: 8, egIsolate: true, egIsoMax: 30 } },
];

async function main() {
    const opt = parseArgs(process.argv);
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const results = [];
    let base = null;
    for (const lev of LEVERS) {
        const per = await runConfig(opt, lev.cfg);
        if (lev.name.startsWith('baseline')) base = per;
        const m = mean(per);
        const sd = Math.sqrt(per.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, per.length - 1));
        const se = sd / Math.sqrt(per.length);
        // paired delta vs baseline
        let pd = null, pdse = null;
        if (base) {
            const d = per.map((v, i) => v - base[i]);
            const dm = mean(d);
            const dsd = Math.sqrt(d.reduce((x, y) => x + (y - dm) ** 2, 0) / Math.max(1, d.length - 1));
            pd = dm; pdse = dsd / Math.sqrt(d.length);
        }
        results.push({ name: lev.name, mean: m, se, pd, pdse });
        const pct = base ? (100 * (m - mean(base)) / mean(base)) : 0;
        console.log(lev.name.padEnd(30), 'mean=' + m.toFixed(2).padStart(6), 'se=' + se.toFixed(2),
            pd != null ? ('  Δ=' + (pd >= 0 ? '+' : '') + pd.toFixed(2) + '±' + pdse.toFixed(2) + ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%)') : '');
    }
}
main().catch(e => { console.error(e); process.exit(1); });
