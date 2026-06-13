// margin_cdf.js — SPEC §6.3 "deliverable zero": prod's plan-decision near-tie
// density. The deduped top1−top2 score margin per plan, as a CDF stratified by
// N-bucket and device. This number bounds every L1 NN-share BEFORE any net is
// trained: a plan whose margin < τ must fall back to prod scoring (L1h), so the
// CDF's left tail is the achievable trusted fraction's ceiling.
//
// Uses the certified stepper with an anchored in-memory transform that captures
// planCheap's closure-local `score`/`cands`/`bi` (a logging line; no behavior
// change — the deterministic digest is unaffected, asserted by --selftest in
// diff_harness). Run on the PUBLISHED calibration range only (never sealed).
//
//   node margin_cdf.js --seeds 200 --seedStart 270000 --maxFrames 20000 \
//        --cells 390x844,1512x982,2560x1440 --out margin_cdf.json
//
// "Deduped margin" (SPEC §3): candidates() pads slots k≥N with the E3D point,
// so duplicate-coordinate candidates share bitwise-equal scores. Collapse
// candidates to unique (x,y) classes, then margin = (best class score) −
// (2nd-best DISTINCT class score). A raw index margin reads 0 on every padded
// plan and is meaningless.
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

// capture planCheap's score+cands+bi right before its return (unique anchor)
const ANCHOR = '        return { x: cands[bi].x, y: cands[bi].y };';
function planCaptureTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(ANCHOR) < 0) throw new Error('margin_cdf: planCheap anchor not found');
    const inject =
        '        if (window.__planLog) window.__planLog.push({ '
        + 'score: score.slice(), cx: cands.map(function(c){return c.x;}), '
        + 'cy: cands.map(function(c){return c.y;}), bi: bi, n: s.bx.length });\n';
    return code.replace(ANCHOR, inject + ANCHOR);
}

// bitwise-equal coordinate key (f64 bits, so −0 vs +0 distinguished)
const _dv = new DataView(new ArrayBuffer(16));
function coordKey(x, y) { _dv.setFloat64(0, x); _dv.setFloat64(8, y); return _dv.getBigUint64(0).toString(16) + ':' + _dv.getBigUint64(8).toString(16); }

function dedupMargin(rec) {
    // best score per distinct coordinate class
    const byCoord = new Map();
    for (let k = 0; k < rec.score.length; k++) {
        const key = coordKey(rec.cx[k], rec.cy[k]);
        const sc = rec.score[k];
        if (!byCoord.has(key) || sc > byCoord.get(key)) byCoord.set(key, sc);
    }
    const scores = Array.from(byCoord.values()).sort((a, b) => b - a);
    const nClasses = scores.length;
    const margin = nClasses >= 2 ? scores[0] - scores[1] : Infinity; // 1 distinct class = no contest
    return { margin, nClasses };
}

function parseArgs(argv) {
    const a = { seeds: 200, seedStart: 270000, maxFrames: 20000,
        cells: '390x844,820x1180,1024x768,1512x982,1680x1050,2560x1440',
        out: null, raw: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--raw') a.raw = argv[++i];   // dump raw {cell:[margins]} + N-bucket tags for sharded merge
        else throw new Error('unknown arg ' + k);
    }
    if (a.seedStart >= 290000) throw new Error('refusing to run margin CDF on sealed range (>=290000)');
    return a;
}

// CDF percentiles + fraction below a set of τ candidates
function summarize(margins) {
    const s = margins.filter(m => Number.isFinite(m)).sort((a, b) => a - b);
    const n = s.length;
    const pct = p => n ? s[Math.min(n - 1, Math.floor(p / 100 * n))] : null;
    const TAUS = [0, 1e-9, 1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0];
    const fracBelow = {};
    for (const t of TAUS) fracBelow[t] = n ? s.filter(m => m < t).length / n : null;
    return {
        nPlansFinite: n, nPlansTotal: margins.length,
        exactTies_margin0: margins.filter(m => m === 0).length,
        oneClassPlans: margins.filter(m => !Number.isFinite(m)).length,
        pctiles: { p1: pct(1), p5: pct(5), p10: pct(10), p25: pct(25), p50: pct(50),
                   p75: pct(75), p90: pct(90), p99: pct(99) },
        fracBelowTau: fracBelow,
    };
}

async function main() {
    const opt = parseArgs(process.argv);
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const cells = opt.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });
    // strat buckets: N in [6,14] (padded plans) vs N>=15 (dense); device
    const all = [], byBucket = { 'N6-14': [], 'N15+': [] }, byCell = {};
    const rawRecs = [];   // {m,n,cell} for sharded merge (only kept if --raw)
    let totalPlans = 0;

    for (const c of cells) {
        const key = c.W + 'x' + c.H; byCell[key] = [];
        for (let i = 0; i < opt.seeds; i++) {
            const seed = opt.seedStart + i;
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed,
                fastRender: true, transform: planCaptureTransform });
            game.win.__planLog = [];
            while (game.boidCount() > 0 && game.frame() < opt.maxFrames) game.stepFrame();
            for (const rec of game.win.__planLog) {
                const { margin, nClasses } = dedupMargin(rec);
                all.push(margin); byCell[key].push(margin);
                (rec.n <= 14 ? byBucket['N6-14'] : byBucket['N15+']).push(margin);
                if (opt.raw) rawRecs.push({ m: Number.isFinite(margin) ? margin : null, n: rec.n, cell: key });
                totalPlans++;
            }
            game.win.__planLog = null;
        }
        process.stderr.write(`[cell ${key}] plans=${byCell[key].length}\n`);
    }
    if (opt.raw) { fs.writeFileSync(opt.raw, JSON.stringify(rawRecs)); process.stderr.write(`raw -> ${opt.raw} (${rawRecs.length} plans)\n`); }

    const report = {
        spec: 'SPEC §6.3 margin CDF (deduped top1-top2 plan-score margin)',
        seedSet: 'calibration[' + opt.seedStart + ',+' + opt.seeds + ')',
        cells: cells.map(c => c.W + 'x' + c.H), maxFrames: opt.maxFrames,
        totalPlans,
        pooled: summarize(all),
        byBucket: { 'N6-14': summarize(byBucket['N6-14']), 'N15+': summarize(byBucket['N15+']) },
        byCell: Object.fromEntries(Object.entries(byCell).map(([k, v]) => [k, summarize(v)])),
        interpretation: 'fracBelowTau[τ] ≈ the fraction of plans an L1h student must '
            + 'route to the rollout fallback at threshold τ; (1 − that) bounds NN-alone share. '
            + 'exactTies_margin0 + the small-τ mass are the irreducible near-tie floor.',
    };
    console.log(JSON.stringify(report, null, 1));
    if (opt.out) fs.writeFileSync(opt.out, JSON.stringify(report, null, 1));
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { dedupMargin, planCaptureTransform };
