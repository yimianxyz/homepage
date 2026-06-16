// clearrate_verdict.js — the PANEL metric: OUTCOME (clear-rate + time-to-catch),
// not decision-agreement. Does a genuine 88%-decision endgame NN still CLEAR ≥95%
// of full games (its egBoid disagreements being outcome-equivalent near-ties), even
// though it matches only ~88% of individual picks?
//
// Drives a full policy (prod planner + an endgame decider) FREE-RUNNING (diff_harness
// fork mode) over sealed full games to extinction, per cell. Reports clear-rate,
// time-to-catch distribution, and STUCK games (never clear = real failures vs harmless
// swaps). Policies (--policy):
//   prod      — identity (prod intercept, the 100% baseline)
//   oracle    — egnn oracle (prod intercept via injection = harness control, ~100%)
//   rawnn     — egnn nn + egboidPickRaw (the GENUINE 88% raw-kinematics NN)
//   analytic  — egnn raw_geom (argmin wrap-aware analytic wa0, the 98.4% formula)
//
//   EXACTNN_SALT_PATH=~/.exactnn_seal_salt_p2 node clearrate_verdict.js \
//     --policy rawnn --seeds 24 --sealOffset 0 --cells 2560x1440 --out out.json
'use strict';
const path = require('path');
const fs = require('fs');
const { runGame } = require('../diff_harness.js');
const seal = require('./seal_seeds.js');

const DEVICE_MATRIX = ['390x844', '820x1180', '1024x768', '1512x982', '1680x1050', '2560x1440'];
const RAW = '/workspace/.team/exact-nn-endgame-student';

function parseArgs(argv) {
    const a = { policy: 'rawnn', seeds: 24, sealOffset: 0, calibration: false, cells: null,
        maxFrames: 30000, natural: true, out: null };
    for (let i = 2; i < argv.length; i++) { const k = argv[i];
        if (k === '--policy') a.policy = argv[++i]; else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--sealOffset') a.sealOffset = +argv[++i]; else if (k === '--calibration') a.calibration = true;
        else if (k === '--cells') a.cells = argv[++i]; else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--scatter') a.natural = false; else if (k === '--out') a.out = argv[++i];
        else throw new Error('unknown arg ' + k); }
    return a;
}
function setPolicyEnv(policy) {
    delete process.env.EXACTNN_EGNN_MODE; delete process.env.EXACTNN_EGNN_STUDENT; delete process.env.EXACTNN_EGNN_WEIGHTS;
    if (policy === 'prod') return 'identity';
    if (policy === 'oracle') { process.env.EXACTNN_EGNN_MODE = 'oracle'; return path.join(__dirname, '..', 'candidates', 'egnn.js'); }
    if (policy === 'analytic') { process.env.EXACTNN_EGNN_MODE = 'raw_geom'; return path.join(__dirname, '..', 'candidates', 'egnn.js'); }
    if (policy === 'rawnn') {
        process.env.EXACTNN_EGNN_MODE = 'nn';
        process.env.EXACTNN_EGNN_STUDENT = RAW + '/egboidPickRaw.js';
        process.env.EXACTNN_EGNN_WEIGHTS = RAW + '/eg_weights_raw.json';
        return path.join(__dirname, '..', 'candidates', 'egnn.js');
    }
    throw new Error('unknown policy ' + policy);
}
function seedList(opt) {
    if (opt.calibration) { const o = []; for (let i = 0; i < opt.seeds; i++) o.push(270000 + i); return { seeds: o, label: 'calib' }; }
    const all = seal.sealedSeeds(fs.readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
    return { seeds: all.slice(opt.sealOffset, opt.sealOffset + opt.seeds), label: 'SEALED@off' + opt.sealOffset };
}

async function main() {
    const opt = parseArgs(process.argv);
    const candidate = setPolicyEnv(opt.policy);
    const { seeds, label } = seedList(opt);
    const cells = opt.cells ? opt.cells.split(',') : DEVICE_MATRIX;
    const base = { policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        maxFrames: opt.maxFrames, postExtinct: 0, decisions: false, fastRender: true, mismatchLimit: 0,
        mode: 'fork', resync: false };
    const perCell = [];
    let totGames = 0, totClear = 0, allTimes = [];
    for (const cell of cells) {
        const [W, H] = cell.split('x').map(Number);
        let cleared = 0, games = 0; const times = [], stuck = [];
        for (const seed of seeds) {
            const r = await runGame(Object.assign({}, base, { W, H, seed, startBoids: opt.natural ? 0 : 5, scatter: !opt.natural }), seed, candidate);
            games++;
            if (r.cleared) { cleared++; times.push(r.clearedAt); allTimes.push(r.clearedAt); }
            else stuck.push(seed);     // never cleared within maxFrames = a real failure
        }
        const med = times.length ? times.slice().sort((a, b) => a - b)[times.length >> 1] : null;
        perCell.push({ cell, games, cleared, clearRate: +(cleared / games).toFixed(4),
            stuck: stuck.length, stuckSeeds: stuck, medTimeToCatch: med });
        totGames += games; totClear += cleared;
        console.error(`[${opt.policy} ${cell}] clear ${cleared}/${games} (${(100 * cleared / games).toFixed(1)}%) stuck=${stuck.length} medCatch=${med}`);
    }
    const report = { policy: opt.policy, candidate: path.basename(candidate), seedSet: label,
        seedCount: seeds.length, sealOffset: opt.calibration ? null : opt.sealOffset,
        distribution: opt.natural ? 'natural(full-game)' : 'scatter', maxFrames: opt.maxFrames,
        commitment_sha256_salt: safeCommitSha(), games: totGames, cleared: totClear,
        clearRate: +(totClear / totGames).toFixed(4),
        medTimeToCatch: allTimes.length ? allTimes.slice().sort((a, b) => a - b)[allTimes.length >> 1] : null,
        p90TimeToCatch: allTimes.length ? allTimes.slice().sort((a, b) => a - b)[Math.floor(allTimes.length * 0.9)] : null,
        stuck_total: perCell.reduce((s, c) => s + c.stuck, 0), perCell };
    console.log(JSON.stringify(report, null, 1));
    if (opt.out) fs.writeFileSync(opt.out, JSON.stringify(report, null, 1));
    console.error(`\n=== ${opt.policy}: clear-rate ${(report.clearRate * 100).toFixed(1)}% (stuck ${report.stuck_total}) — gate(≥95% clear) ${report.clearRate >= 0.95 ? 'PASS' : 'FAIL'} ===`);
}
function safeCommitSha() { try { return JSON.parse(fs.readFileSync(seal.COMMIT_PATH, 'utf8')).sha256_salt; } catch (e) { return null; } }
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
