// eg_logger.js — ENDGAME egBoid-COMMIT oracle logger for the L1e student (task #18).
//
// The N<=5 decision is intercept()'s egBoid commit = argmin over the present boids
// of scan(boid).t (earliest frame the predator can stand where the boid will be,
// torus min-image; nearest-distance fallback if none reachable within TMAX=1400).
// scan().t depends ONLY on the boid's (relpos, vel) + torus dims + sM=2.5 — it is
// per-boid SEPARABLE (no inter-boid interaction), unlike the D1 90-frame rollout.
//
// This logs, at EACH commit (egBoid (re)assigned: first N<=5 entry + after each
// catch), the full state + every present boid's exact scan().t (the LABEL) + prod's
// committed egBoid index. We recompute scan() via prod's OWN closure (digest-inert
// anchored transform), so the logged t IS prod's scan-t bitwise.
//
//   node eg_logger.js --seeds 400 --seedStart 100000 --startBoids 5 \
//     --cells 390x844,820x1180,1024x768,1512x982,1680x1050,2560x1440 \
//     --maxFrames 8000 --outDir data_eg
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { createGame } = require('../stepper.js');

// --- digest-inert transform: log per-boid scan-t at each commit -----------------
const ANCHOR = '        // aim at the earliest-reachable point (perpendicular cut-off onto its line if none)';
const COMMIT = '        if (!egBoid) {';
function egTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(ANCHOR) < 0 || code.indexOf(COMMIT) < 0) throw new Error('eg_logger: intercept anchors not found');
    const inj =
        '        if (window.__egLog && window.__egJustCommitted) {\n'
        + '            var __recs = [];\n'
        + '            for (var __z = 0; __z < boids.length; __z++) {\n'
        + '                var __c = scan(boids[__z]);\n'
        + '                __recs.push({ x: boids[__z].position.x, y: boids[__z].position.y,\n'
        + '                    vx: boids[__z].velocity.x, vy: boids[__z].velocity.y, t: __c ? __c.t : null });\n'
        + '            }\n'
        + '            window.__egLog.push({ W: cfg.W, Hc: cfg.Hc, px: px, py: py,\n'
        + '                pvx: pred.velocity.x, pvy: pred.velocity.y, psize: pred.currentSize,\n'
        + '                egIdx: boids.indexOf(egBoid), boids: __recs });\n'
        + '            window.__egJustCommitted = false;\n'
        + '        }\n';
    let out = code.replace(ANCHOR, inj + ANCHOR);
    out = out.replace(COMMIT, '        if (!egBoid) { if (window.__egLog) window.__egJustCommitted = true;');
    return out;
}

function parseArgs() {
    const a = { seeds: 400, seedStart: 100000, startBoids: 5, maxFrames: 8000,
        cells: '390x844,820x1180,1024x768,1512x982,1680x1050,2560x1440',
        outDir: path.join(__dirname, 'data_eg'), scatter: true };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i];
        else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--startBoids') a.startBoids = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i];
        else if (k === '--cells') a.cells = process.argv[++i];
        else if (k === '--outDir') a.outDir = process.argv[++i];
        else if (k === '--noScatter') a.scatter = false;
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

async function main() {
    const a = parseArgs();
    if (a.seedStart >= 290000) throw new Error('refusing sealed range (>=290000)');
    if (a.seedStart < 270000 && a.seedStart + a.seeds > 270000) throw new Error('train run must stay <270000');
    fs.mkdirSync(a.outDir, { recursive: true });
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const cells = a.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });
    const cellTag = c => `${c.W}x${c.H}`;
    let totalCommits = 0;
    for (const c of cells) {
        const recs = [];
        for (let i = 0; i < a.seeds; i++) {
            const seed = a.seedStart + i;
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed,
                startBoids: a.startBoids, scatter: a.scatter, fastRender: true, transform: egTransform });
            game.win.__egLog = []; game.win.__egJustCommitted = false;
            while (game.boidCount() > 0 && game.frame() < a.maxFrames) game.stepFrame();
            for (const r of game.win.__egLog) { r.seed = seed; r.cell = cellTag(c); recs.push(r); }
            game.win.__egLog = null;
        }
        const fn = path.join(a.outDir, `${cellTag(c)}_${a.seedStart}.commits.jsonl.gz`);
        const buf = recs.map(r => JSON.stringify(r)).join('\n') + '\n';
        fs.writeFileSync(fn, zlib.gzipSync(buf));
        totalCommits += recs.length;
        process.stderr.write(`[${cellTag(c)}] ${recs.length} commits -> ${path.basename(fn)}\n`);
    }
    const meta = { spec: 'L1e endgame egBoid-commit oracle (task #18)', seedStart: a.seedStart,
        seeds: a.seeds, startBoids: a.startBoids, scatter: a.scatter, cells: cells.map(cellTag),
        maxFrames: a.maxFrames, totalCommits, sM: 2.5, BORDER_OFFSET: 10, TMAX: 1400 };
    fs.writeFileSync(path.join(a.outDir, `manifest_${a.seedStart}.json`), JSON.stringify(meta, null, 1));
    process.stderr.write(`TOTAL ${totalCommits} commits across ${cells.length} cells\n`);
}

main().catch(e => { console.error(e); process.exit(1); });
