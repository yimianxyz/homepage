// throughput_farm.js — the authoritative JS PAIRED-SEED farm for the optimal split.
// Metric = THROUGHPUT (total catches / total frames to extinction). Stuck-rate is moot
// (the deployed endgame never freezes), kept as a ≈0 sanity. PAIRED seeds across split
// configs (same seed → same game until the split diverges) → paired Wilcoxon (huge
// variance reduction). Search-set vs sealed-confirm-set split (anti-overfit).
//
// Runs the EXACT deployed policy (candidates/split.js) per (config, screen, seed),
// recording throughput, time-to-clear, cleared?, neverCatch300 (stuck sanity).
//
//   node throughput_farm.js --configs "count:T=3,count:T=5,count:T=8,density:Tref=5,horizon:H=90" \
//        --cells 1024x768,2560x1440 --seeds 200 --seedBase 270000 --out farm.json
//   (sealed: --sealed  → draws from seal_seeds p2 salt instead of seedBase+i)
'use strict';
const path = require('path');
const fs = require('fs');
const { runGame } = require('../diff_harness.js');

const DEVICE_MATRIX = ['390x844', '820x1180', '1024x768', '1512x982', '1680x1050', '2560x1440'];

function parseArgs(argv) {
    const a = { configs: 'count:T=5', cells: null, seeds: 200, seedBase: 270000, maxFrames: 50000,
        natural: true, startBoids: 28, sealed: false, sealOffset: 0, out: null };
    for (let i = 2; i < argv.length; i++) { const k = argv[i];
        if (k === '--configs') a.configs = argv[++i]; else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i]; else if (k === '--seedBase') a.seedBase = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i]; else if (k === '--scatter') a.natural = false;
        else if (k === '--startBoids') a.startBoids = +argv[++i];
        else if (k === '--sealed') a.sealed = true; else if (k === '--sealOffset') a.sealOffset = +argv[++i];
        else if (k === '--out') a.out = argv[++i]; else throw new Error('unknown arg ' + k); }
    return a;
}
// "count:T=8" -> {rule:'count', env:{EXACTNN_SPLIT_RULE:'count', EXACTNN_SPLIT_T:'8'}, label:'count:T=8'}
function parseConfig(spec) {
    const [rule, kv] = spec.split(':');
    const env = { EXACTNN_SPLIT_RULE: rule };
    if (kv) for (const p of kv.split(',')) { const [k, v] = p.split('='); env['EXACTNN_SPLIT_' + k.toUpperCase()] = v; }
    return { rule, env, label: spec };
}
function cells(opt) { return opt.cells ? opt.cells.split(',') : DEVICE_MATRIX; }
function seedList(opt) {
    if (opt.sealed) { const seal = require('./seal_seeds.js');
        const all = seal.sealedSeeds(fs.readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
        return all.slice(opt.sealOffset, opt.sealOffset + opt.seeds); }
    const o = []; for (let i = 0; i < opt.seeds; i++) o.push(opt.seedBase + i); return o;   // held-out search block
}

async function main() {
    const opt = parseArgs(process.argv);
    const configs = opt.configs.split(';').map(parseConfig);
    const cs = cells(opt);
    const seeds = seedList(opt);
    const candidate = path.join(__dirname, '..', 'candidates', 'split.js');
    const base = { policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        maxFrames: opt.maxFrames, postExtinct: 0, decisions: false, fastRender: true, mismatchLimit: 0,
        mode: 'fork', resync: false };
    // result[cell][configLabel] = array of per-seed {seed, thru, frames, cleared, neverCatch}
    const R = {};
    for (const cell of cs) {
        const [W, H] = cell.split('x').map(Number);
        R[cell] = {};
        for (const cfg of configs) R[cell][cfg.label] = [];
        for (const seed of seeds) {                       // PAIRED: same seed across all configs
            for (const cfg of configs) {
                for (const k in cfg.env) process.env[k] = cfg.env[k];
                const startBoids = opt.natural ? 0 : opt.startBoids;   // late-game scatter (config-dependent regime)
                const r = await runGame(Object.assign({}, base, { W, H, seed, startBoids, scatter: !opt.natural }), seed, candidate);
                // throughput = boids caught / frames elapsed (caught = initial N - survivors;
                // since runGame steps to extinction-or-cap, caught = eaten count via boidCount delta)
                const caught = r.eaten != null ? r.eaten : null;
                const thru = caught != null ? caught / r.frames : null;
                R[cell][cfg.label].push({ seed, thru, frames: r.frames, cleared: r.cleared, caught });
                for (const k in cfg.env) delete process.env[k];
            }
        }
        // per-config summary for this cell
        const summ = {};
        for (const cfg of configs) {
            const rows = R[cell][cfg.label];
            const thrus = rows.map(x => x.thru).filter(x => x != null);
            const frames = rows.map(x => x.frames);
            summ[cfg.label] = { n: rows.length, clearRate: mean(rows.map(x => x.cleared ? 1 : 0)),
                meanThru: mean(thrus), medFrames: median(frames), p90Frames: pct90(frames) };
        }
        console.error(`[${cell}] ` + configs.map(c => `${c.label}=${(summ[c.label].meanThru * 1e4).toFixed(3)}e-4/clr${(summ[c.label].clearRate * 100).toFixed(0)}%`).join(' '));
        R[cell].__summary = summ;
    }
    const report = { metric: 'throughput=caught/frames (paired seeds)', configs: configs.map(c => c.label),
        cells: cs, seeds: seeds.length, seedSet: opt.sealed ? ('SEALED@off' + opt.sealOffset) : ('held-out@' + opt.seedBase),
        maxFrames: opt.maxFrames, R };
    if (opt.out) fs.writeFileSync(opt.out, JSON.stringify(report));
    console.log('DONE ' + (opt.out || ''));
}
function mean(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; }
function median(a) { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; }
function pct90(a) { if (!a.length) return null; const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length * 0.9)]; }
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
