// verdict.js — independent-verifier sealed-set verdict (SPEC §4b/4c).
//
// Runs a candidate policy against frozen prod on the SEALED seed set across the
// §5 device matrix and reports the goal-gate similarity numbers + the
// rule-of-three residual-risk bound. Wraps dev/exact_nn/diff_harness.js's
// runGame (the certified lockstep instrument) — no metric is recomputed here,
// only aggregated into the §4b form.
//
//   node verdict.js --candidate ../candidates/l0.js            # default: full sealed sweep
//   node verdict.js --candidate <mod> --seeds 256 --maxFrames 30000
//   node verdict.js --candidate <mod> --cells 390x844,1512x982 # subset of the matrix
//   node verdict.js --candidate <mod> --calibration            # run the [270000,280000) set instead of sealed
//
// Metrics (SPEC §4b), per cell and pooled:
//   S_dec   = 1 − planDisagree/plans         (committed target coords, deduped)
//   S_eg    = 1 − egDisagree/egCommits       (endgame egBoid identity)
//   S_frame = 1 − forceMismatches/forceFrames(bitwise fx,fy over every frame, every regime)
//   S_traj  = fork-mode: fraction of games bit-identical to extinction + median
//             first-divergence frame
// Residual risk (rule of three): with 0 mismatches in n trusted decisions,
// per-decision mismatch prob ≤ 3/n at 95%; per-game ≤ 1−(1−3/n)^(plans/game).
//
// SEALED DISCIPLINE: this reads sealed seeds from seal_seeds.js (secret salt).
// It NEVER prints a seed — only seed *counts* and aggregate metrics — so the
// console/log output is safe to paste into an issue. Use seal_seeds --reveal
// for the audit trail at verdict time.
'use strict';
const path = require('path');
const crypto = require('crypto');
const { runGame } = require('../diff_harness.js');
const seal = require('./seal_seeds.js');

const DEVICE_MATRIX = [
    { W: 390, H: 844 }, { W: 820, H: 1180 }, { W: 1024, H: 768 },
    { W: 1512, H: 982 }, { W: 1680, H: 1050 }, { W: 2560, H: 1440 },
];

function parseArgs(argv) {
    const a = { candidate: path.join(__dirname, '..', 'candidates', 'l0.js'),
        seeds: 256, maxFrames: 30000, cells: null, calibration: false,
        postExtinct: 60, out: null, sealOffset: 0 };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--candidate') a.candidate = argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--calibration') a.calibration = true;
        else if (k === '--sealOffset') a.sealOffset = +argv[++i];   // slice a fresh sealed range (one-shot discipline)
        else if (k === '--postExtinct') a.postExtinct = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}

function cells(opt) {
    if (!opt.cells) return DEVICE_MATRIX;
    return opt.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });
}

// calibration seeds are the PUBLISHED [270000,280000) range; sealed seeds come
// from the secret salt. Either way we never echo individual seeds.
function seedList(opt) {
    if (opt.calibration) {
        const out = [];
        for (let i = 0; i < opt.seeds; i++) out.push(270000 + i);
        return { seeds: out, label: 'calibration[270000,280000)' };
    }
    const all = seal.sealedSeeds(require('fs').readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
    return { seeds: all.slice(opt.sealOffset, opt.sealOffset + opt.seeds), label: 'SEALED(hidden)@off' + opt.sealOffset };
}

async function main() {
    const opt = parseArgs(process.argv);
    const { seeds, label } = seedList(opt);
    const cs = cells(opt);
    const base = { policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        maxFrames: opt.maxFrames, postExtinct: opt.postExtinct,
        decisions: true, resync: true, fastRender: true, mismatchLimit: 5,
        startBoids: 0, scatter: false, uaMobile: false };

    const pooled = { plans: 0, planDisagree: 0, egCommits: 0, egDisagree: 0,
        forceFrames: 0, forceMismatch: 0, games: 0, cleared: 0,
        trajIdentical: 0, firstDiv: [] };
    const perCell = [];

    for (const c of cs) {
        const cell = { W: c.W, H: c.H, plans: 0, planDisagree: 0, egCommits: 0,
            egDisagree: 0, forceFrames: 0, forceMismatch: 0, games: 0, cleared: 0,
            trajIdentical: 0, firstDiv: [] };
        for (const seed of seeds) {
            // lockstep (S_dec, S_eg, S_frame)
            const L = await runGame(Object.assign({}, base, { W: c.W, H: c.H, mode: 'lockstep' }), seed, opt.candidate);
            cell.games++; if (L.cleared) cell.cleared++;
            cell.forceFrames += L.framesByRegime.planner + L.framesByRegime.intercept + L.framesByRegime.zero;
            cell.forceMismatch += L.mismatchCount;
            if (L.decisions) { cell.plans += L.decisions.plans; cell.planDisagree += L.decisions.planDisagree;
                cell.egCommits += L.decisions.egCommits; cell.egDisagree += L.decisions.egDisagree; }
            // fork (S_traj) — no resync so divergence isn't masked
            const F = await runGame(Object.assign({}, base, { W: c.W, H: c.H, mode: 'fork', resync: false }), seed, opt.candidate);
            if (F.firstMismatchFrame < 0) cell.trajIdentical++;
            else cell.firstDiv.push(F.firstMismatchFrame);
        }
        for (const k of ['plans', 'planDisagree', 'egCommits', 'egDisagree', 'forceFrames', 'forceMismatch', 'games', 'cleared', 'trajIdentical']) pooled[k] += cell[k];
        pooled.firstDiv.push(...cell.firstDiv);
        cell.S_dec = cell.plans ? 1 - cell.planDisagree / cell.plans : null;
        cell.S_eg = cell.egCommits ? 1 - cell.egDisagree / cell.egCommits : null;
        cell.S_frame = cell.forceFrames ? 1 - cell.forceMismatch / cell.forceFrames : null;
        cell.S_traj_identical = cell.games ? cell.trajIdentical / cell.games : null;
        perCell.push(cell);
        console.error(`[cell ${c.W}x${c.H}] S_dec=${fmt(cell.S_dec)} S_frame=${fmt(cell.S_frame)} `
            + `traj_identical=${cell.trajIdentical}/${cell.games} plans=${cell.plans} mm=${cell.forceMismatch}`);
    }

    const median = arr => { if (!arr.length) return null; const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
    const n = pooled.plans;
    const ruleOf3 = n ? 3 / n : null;
    const plansPerGame = pooled.games ? pooled.plans / pooled.games : 0;
    const report = {
        candidate: path.basename(opt.candidate), seedSet: label, seedCount: seeds.length,
        cells: cs.length, devicematrix: cs.map(c => c.W + 'x' + c.H),
        commitment_sha256_salt: opt.calibration ? null : safeCommitSha(),
        games: pooled.games, cleared: pooled.cleared,
        S_dec: rnd(pooled.plans ? 1 - pooled.planDisagree / pooled.plans : null),
        S_eg: rnd(pooled.egCommits ? 1 - pooled.egDisagree / pooled.egCommits : null),
        S_frame: rnd(pooled.forceFrames ? 1 - pooled.forceMismatch / pooled.forceFrames : null),
        S_traj_identical_frac: rnd(pooled.games ? pooled.trajIdentical / pooled.games : null),
        S_traj_median_first_div: median(pooled.firstDiv),
        plans_total: pooled.plans, planDisagree: pooled.planDisagree,
        egCommits_total: pooled.egCommits, egDisagree: pooled.egDisagree,
        forceFrames_total: pooled.forceFrames, forceMismatch_total: pooled.forceMismatch,
        residual_risk: {
            trusted_decisions: n,
            per_decision_upper95: rnd(ruleOf3),
            per_game_upper95: ruleOf3 != null ? rnd(1 - Math.pow(1 - ruleOf3, plansPerGame)) : null,
            plans_per_game: rnd(plansPerGame),
            note: '0 mismatches assumed for the rule-of-three columns; if mismatches>0, '
                + 'this is NOT a clean upper bound — see planDisagree/forceMismatch.',
        },
        goal_gate_SPEC_4b: {
            requires: 'S_dec>0.95 AND S_frame>0.95 for an NN-alone decision (L0 passes trivially as the floor)',
            S_dec_pass: pooled.plans ? (1 - pooled.planDisagree / pooled.plans) > 0.95 : null,
            S_frame_pass: pooled.forceFrames ? (1 - pooled.forceMismatch / pooled.forceFrames) > 0.95 : null,
        },
        perCell,
    };
    console.log(JSON.stringify(report, null, 1));
    if (opt.out) require('fs').writeFileSync(opt.out, JSON.stringify(report, null, 1));
    // non-zero exit if any mismatch (a real verdict failure for an exact claim)
    process.exit((pooled.forceMismatch || pooled.planDisagree || pooled.egDisagree) ? 2 : 0);
}

function fmt(x) { return x == null ? 'n/a' : (x * 100).toFixed(4) + '%'; }
function rnd(x) { return x == null ? null : +x.toFixed(9); }
function safeCommitSha() {
    try { return JSON.parse(require('fs').readFileSync(seal.COMMIT_PATH, 'utf8')).sha256_salt; }
    catch (e) { return null; }
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { DEVICE_MATRIX };
