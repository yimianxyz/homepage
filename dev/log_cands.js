// Distillation data-gen: run the cheap policy with PC.logCands on the device mix,
// rolling EVERY candidate each decision to record (features, ctx, per-candidate
// rollout-gains). Output = JSON lines, one per decision: {f:[[19]xK], c:[4], g:[K]}.
// A net trained on this to RANK candidates is a better prune than the ballistic
// score (roll-all-16 diagnostic: +9.6% available at deployed compute).
//
//   node dev/log_cands.js --policyDir dev/exp/js --W 390 --H 844 --seeds 40 --frames 1500 --out /tmp/cands_phone.jsonl
'use strict';
const fs = require('fs');
const path = require('path');
const { buildHarness } = require('./fasteval.js');

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, 'exp', 'js'), W: 390, H: 844,
        seedStart: 200000, seeds: 40, frames: 1500, out: '/tmp/cands.jsonl' };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--config') a.baseConfig = argv[++i];
    }
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    // Log on the ship-candidate's (ES-params) visited states — DAgger principle:
    // train the prune net on the distribution it will actually be deployed under.
    const base = opt.baseConfig ? JSON.parse(opt.baseConfig) : {};
    base.logCands = true;
    opt.config = JSON.stringify(base);
    const ws = fs.createWriteStream(opt.out);
    let rows = 0;
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        global.__CANDLOG = [];
        const built = buildHarness(opt);
        const api = built.api, win = built.win;
        if (win.__predatorReady && win.__predatorReady.then) await win.__predatorReady;
        api.setSimSeed(seed, 12);
        const sim = new api.Simulation('boids1');
        sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
        sim.initialize(false);
        if (api.setFrameMs) api.setFrameMs(12);
        sim.tick();
        for (let f = 0; f < opt.frames; f++) { api.simTick(); sim.tick(); sim.render(); }
        for (const rec of global.__CANDLOG) {
            // round to 4 dp to keep the file small
            const f4 = rec.feat.map(v => v.map(x => +x.toFixed(4)));
            const c4 = rec.ctx.map(x => +x.toFixed(4));
            const g4 = rec.gains.map(x => +x.toFixed(4));
            ws.write(JSON.stringify({ f: f4, c: c4, g: g4 }) + '\n');
            rows++;
        }
        process.stderr.write('  seed ' + (i + 1) + '/' + opt.seeds + ' rows=' + rows + '\r');
    }
    ws.end();
    console.log(JSON.stringify({ out: opt.out, rows, W: opt.W, H: opt.H, seeds: opt.seeds }));
}
main().catch(e => { console.error(e); process.exit(1); });
