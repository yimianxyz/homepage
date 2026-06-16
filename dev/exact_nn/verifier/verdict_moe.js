// verdict_moe.js — the PHASE-2 independent-verifier similarity verdict for the
// PURE single MoE-NN policy (no fallback). Wraps diff_harness.runGame (the
// certified lockstep instrument, +the additive opt.forceSim accumulator) — no
// metric is recomputed by hand. Drives candidates/moe.js with NO fallback (the
// NN's argmax IS the committed decision in every boid case).
//
// METRICS (SPEC_PHASE2_MOE.md §2):
//   * S_dec (PRIMARY, the GATE) — committed target-coord (planner, coord-deduped)
//     / egBoid-identity (endgame) agreement with prod, reported:
//        - POOLED across all N      = 1 − (planDisagree+egDisagree)/(plans+egCommits)
//        - per-REGIME planner       = 1 − planDisagree/plans
//        - per-REGIME endgame       = 1 − egDisagree/egCommits
//        - per device CELL (so neither regime/cell hides behind another)
//     GATE = S_dec ≥ 95% POOLED **and** per-regime (planner AND endgame).
//   * S_traj  — fork-mode (no resync, candidate force applied): median first-
//     divergence frame + fraction of games the pure-NN still clears the board
//     (catch agreement). Honesty, not the gate.
//   * S_force — per-frame force cosine + relative-magnitude (min/max), per regime,
//     in BOTH lockstep (texture on prod's trajectory) and fork (free-run texture).
//     A pure-NN texture check; not the gate.
//
// SEALED + ONE-SHOT DISCIPLINE (carried from Phase-1): sealed seeds via
// seal_seeds.js (secret salt, never printed); Phase-2 uses a FRESH disjoint slice
// --sealOffset 60 (Phase-1 used 0/20/40). Held-out [270000,280000) (--calibration)
// is the only set touched before the verdict; the sealed slice is run ONCE.
//
//   MODE (env or --mode): nn (default, side-a's model) | oracle | raw_prior | perturb
//   node verdict_moe.js --mode nn --natural --seeds 64 --sealOffset 60 \
//        --student ../student/moePolicy.js --weights ../student/moe_weights.json
//   node verdict_moe.js --selftest      # proves the verdict on oracle/perturb stubs
'use strict';
const path = require('path');
const fs = require('fs');
const { runGame } = require('../diff_harness.js');
const seal = require('./seal_seeds.js');

const DEVICE_MATRIX = [
    { W: 390, H: 844 }, { W: 820, H: 1180 }, { W: 1024, H: 768 },
    { W: 1512, H: 982 }, { W: 1680, H: 1050 }, { W: 2560, H: 1440 },
];
const GATE = 0.95;

function parseArgs(argv) {
    const a = { candidate: path.join(__dirname, '..', 'candidates', 'moe.js'),
        mode: process.env.EXACTNN_MOE_MODE || 'nn', perturb: null,
        seeds: 64, natural: true, cells: null, calibration: false,
        sealOffset: 60, maxFrames: null, out: null, forceSim: true,
        student: process.env.EXACTNN_MOE_STUDENT || null, weights: process.env.EXACTNN_MOE_WEIGHTS || null,
        selftest: false };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--candidate') a.candidate = argv[++i];
        else if (k === '--mode') a.mode = argv[++i];
        else if (k === '--perturb') a.perturb = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--natural') a.natural = true;
        else if (k === '--scatter') a.natural = false;
        else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--calibration') a.calibration = true;
        else if (k === '--sealOffset') a.sealOffset = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--student') a.student = argv[++i];
        else if (k === '--weights') a.weights = argv[++i];
        else if (k === '--noForceSim') a.forceSim = false;
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--selftest') a.selftest = true;
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}
function cells(opt) {
    if (!opt.cells) return DEVICE_MATRIX;
    return opt.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });
}
function seedList(opt) {
    if (opt.calibration) {
        const out = []; for (let i = 0; i < opt.seeds; i++) out.push(270000 + i);
        return { seeds: out, label: 'calibration[270000,280000)' };
    }
    const all = seal.sealedSeeds(fs.readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
    return { seeds: all.slice(opt.sealOffset, opt.sealOffset + opt.seeds),
        label: 'SEALED(hidden)@off' + opt.sealOffset };
}

function newAcc(extra) {
    return Object.assign({ plans: 0, planDisagree: 0, egCommits: 0, egDisagree: 0,
        games: 0, lockstepCleared: 0, forkCleared: 0, trajIdentical: 0, firstDiv: [],
        gateFlips: 0, gateMalformed: 0, soleN1: 0,   // candidate self-reported flips + n==1 sole-boid commits
        fsLockP: { cos: 0, rel: 0, n: 0 }, fsLockE: { cos: 0, rel: 0, n: 0 },
        fsForkP: { cos: 0, rel: 0, n: 0 }, fsForkE: { cos: 0, rel: 0, n: 0 } }, extra || {});
}
function addForceSim(dst, fs) { if (!fs) return; dst.cos += fs.cos; dst.rel += fs.rel; dst.n += fs.n; }
function sdec(dis, tot) { return tot ? 1 - dis / tot : null; }
function mean(acc) { return acc.n ? acc.cos / acc.n : null; }
function meanRel(acc) { return acc.n ? acc.rel / acc.n : null; }
function median(arr) { if (!arr.length) return null; const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }
function rnd(x) { return x == null ? null : +x.toFixed(9); }
function pct(x) { return x == null ? 'n/a' : (x * 100).toFixed(3) + '%'; }

async function runCell(opt, c, seeds, base) {
    const C = newAcc({ W: c.W, H: c.H });
    for (let si = 0; si < seeds.length; si++) {
        const seed = seeds[si];
        const startBoids = opt.natural ? 0 : (2 + (si % 4));   // 2..5
        const cfg = Object.assign({}, base, { W: c.W, H: c.H, startBoids, scatter: !opt.natural });
        // lockstep (resync) — S_dec + S_force texture on prod's trajectory
        const L = await runGame(Object.assign({}, cfg, { mode: 'lockstep', resync: true, forceSim: opt.forceSim }), seed, opt.candidate);
        // the candidate's own counters for THIS (lockstep) game (moe.js OR egnn.js —
        // audit caught that egnn sets __egnnStatsLast, so read both prefixes).
        const gs = global.__egnnStatsLast || global.__moeStatsLast;
        C.games++; if (L.cleared) C.lockstepCleared++;
        if (L.decisions) { C.plans += L.decisions.plans; C.planDisagree += L.decisions.planDisagree;
            C.egCommits += L.decisions.egCommits; C.egDisagree += L.decisions.egDisagree; }
        if (gs) { C.gateFlips += gs.flips || 0; C.gateMalformed += gs.malformed || 0; C.soleN1 += gs.soleN1 || 0; }
        if (L.forceSim) { addForceSim(C.fsLockP, L.forceSim.planner); addForceSim(C.fsLockE, L.forceSim.endgame); }
        // fork (no resync, candidate force applied) — S_traj + free-run S_force
        const F = await runGame(Object.assign({}, cfg, { mode: 'fork', resync: false, forceSim: opt.forceSim }), seed, opt.candidate);
        if (F.cleared) C.forkCleared++;
        if (F.firstMismatchFrame < 0) C.trajIdentical++; else C.firstDiv.push(F.firstMismatchFrame);
        if (F.forceSim) { addForceSim(C.fsForkP, F.forceSim.planner); addForceSim(C.fsForkE, F.forceSim.endgame); }
    }
    return C;
}

function cellReport(C) {
    return {
        cell: C.W + 'x' + C.H, games: C.games,
        S_dec_pooled: rnd(sdec(C.planDisagree + C.egDisagree, C.plans + C.egCommits)),
        S_dec_planner: rnd(sdec(C.planDisagree, C.plans)),
        S_dec_endgame: rnd(sdec(C.egDisagree, C.egCommits)),
        plans: C.plans, planDisagree: C.planDisagree, egCommits: C.egCommits, egDisagree: C.egDisagree,
        S_force_lockstep: { planner_cos: rnd(mean(C.fsLockP)), planner_rel: rnd(meanRel(C.fsLockP)),
            endgame_cos: rnd(mean(C.fsLockE)), endgame_rel: rnd(meanRel(C.fsLockE)) },
        S_force_fork: { planner_cos: rnd(mean(C.fsForkP)), planner_rel: rnd(meanRel(C.fsForkP)),
            endgame_cos: rnd(mean(C.fsForkE)), endgame_rel: rnd(meanRel(C.fsForkE)) },
        S_traj_identical_frac: rnd(C.games ? C.trajIdentical / C.games : null),
        S_traj_median_first_div: median(C.firstDiv),
        forkClearedFrac: rnd(C.games ? C.forkCleared / C.games : null),
        gateFlips: C.gateFlips, gateMalformed: C.gateMalformed, soleN1: C.soleN1,
        // n==1 sole-boid commits trivially agree (single boid = the only pick) → report the
        // non-trivial (contested, n≥2) endgame S_dec too (audit F2: n=1 inflates the denom).
        S_dec_endgame_nonTrivial: rnd(sdec(C.egDisagree, C.egCommits - C.soleN1)),
        // raw summable counters so per-cell shard reports merge EXACTLY (merge_moe_reports.js)
        _raw: { plans: C.plans, planDisagree: C.planDisagree, egCommits: C.egCommits, egDisagree: C.egDisagree,
            games: C.games, lockstepCleared: C.lockstepCleared, forkCleared: C.forkCleared,
            trajIdentical: C.trajIdentical, firstDiv: C.firstDiv, gateFlips: C.gateFlips, gateMalformed: C.gateMalformed,
            soleN1: C.soleN1, fsLockP: C.fsLockP, fsLockE: C.fsLockE, fsForkP: C.fsForkP, fsForkE: C.fsForkE },
    };
}

async function runVerdict(opt) {
    if (opt.mode) process.env.EXACTNN_MOE_MODE = opt.mode;
    if (opt.perturb != null) process.env.EXACTNN_MOE_PERTURB = String(opt.perturb);
    if (opt.student) process.env.EXACTNN_MOE_STUDENT = path.resolve(opt.student);
    if (opt.weights) process.env.EXACTNN_MOE_WEIGHTS = path.resolve(opt.weights);
    const { seeds, label } = seedList(opt);
    const cs = cells(opt);
    const maxFrames = opt.maxFrames || (opt.natural ? 40000 : 4000);
    const base = { policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        maxFrames, postExtinct: opt.natural ? 60 : 60, decisions: true, fastRender: true, mismatchLimit: 3 };

    const P = newAcc();
    const perCell = [];
    for (const c of cs) {
        const C = await runCell(opt, c, seeds, base);
        for (const k of ['plans', 'planDisagree', 'egCommits', 'egDisagree', 'games', 'lockstepCleared', 'forkCleared', 'trajIdentical', 'gateFlips', 'gateMalformed', 'soleN1']) P[k] += C[k];
        P.firstDiv.push(...C.firstDiv);
        for (const k of ['fsLockP', 'fsLockE', 'fsForkP', 'fsForkE']) { P[k].cos += C[k].cos; P[k].rel += C[k].rel; P[k].n += C[k].n; }
        const cr = cellReport(C);
        perCell.push(cr);
        console.error(`[cell ${cr.cell}] S_dec pooled=${pct(cr.S_dec_pooled)} planner=${pct(cr.S_dec_planner)} `
            + `endgame=${pct(cr.S_dec_endgame)} | plans=${C.plans} egC=${C.egCommits} `
            + `Sforce(fork)pl=${pct(cr.S_force_fork.planner_cos)} eg=${pct(cr.S_force_fork.endgame_cos)}`);
    }

    const S_dec_pooled = sdec(P.planDisagree + P.egDisagree, P.plans + P.egCommits);
    const S_dec_planner = sdec(P.planDisagree, P.plans);
    const S_dec_endgame = sdec(P.egDisagree, P.egCommits);
    const gate = {
        threshold: GATE,
        pooled_pass: S_dec_pooled != null && S_dec_pooled >= GATE,
        planner_pass: S_dec_planner == null || S_dec_planner >= GATE,
        endgame_pass: S_dec_endgame == null || S_dec_endgame >= GATE,
    };
    gate.PASS = gate.pooled_pass && gate.planner_pass && gate.endgame_pass;

    const report = {
        phase: 2, candidate: path.basename(opt.candidate), mode: opt.mode,
        weights: opt.weights ? path.basename(opt.weights) : null,
        weights_sha256: opt.weights && fs.existsSync(opt.weights) ? sha256(opt.weights) : null,
        seedSet: label, seedCount: seeds.length, sealOffset: opt.calibration ? null : opt.sealOffset,
        commitment_sha256_salt: opt.calibration ? null : safeCommitSha(),
        distribution: opt.natural ? 'natural(full-game, all-N)' : 'scatter(endgame, startBoids 2..5)',
        cells: cs.length, devicematrix: cs.map(c => c.W + 'x' + c.H),
        games: P.games, maxFrames,
        S_dec_pooled: rnd(S_dec_pooled), S_dec_planner: rnd(S_dec_planner), S_dec_endgame: rnd(S_dec_endgame),
        plans_total: P.plans, planDisagree_total: P.planDisagree,
        egCommits_total: P.egCommits, egDisagree_total: P.egDisagree,
        decisions_total: P.plans + P.egCommits, disagree_total: P.planDisagree + P.egDisagree,
        // gateMalformed > 0 ⇒ the NN returned out-of-range slots (a model defect,
        // penalized as disagreements — NOT silently corrected to prod). Honesty flag.
        gateMalformed_total: P.gateMalformed, gateFlips_total: P.gateFlips,
        soleN1_total: P.soleN1, soleN1_frac: rnd(P.egCommits ? P.soleN1 / P.egCommits : null),
        nonTrivial_egCommits: P.egCommits - P.soleN1,
        S_dec_endgame_nonTrivial: rnd(sdec(P.egDisagree, P.egCommits - P.soleN1)),
        S_force_lockstep: { planner_cos: rnd(mean(P.fsLockP)), planner_rel: rnd(meanRel(P.fsLockP)),
            endgame_cos: rnd(mean(P.fsLockE)), endgame_rel: rnd(meanRel(P.fsLockE)) },
        S_force_fork: { planner_cos: rnd(mean(P.fsForkP)), planner_rel: rnd(meanRel(P.fsForkP)),
            endgame_cos: rnd(mean(P.fsForkE)), endgame_rel: rnd(meanRel(P.fsForkE)) },
        S_traj_identical_frac: rnd(P.games ? P.trajIdentical / P.games : null),
        S_traj_median_first_div: median(P.firstDiv),
        forkClearedFrac: rnd(P.games ? P.forkCleared / P.games : null),
        lockstepClearedFrac: rnd(P.games ? P.lockstepCleared / P.games : null),
        gate,
        perCell,
    };
    return report;
}

function printAndExit(report, opt) {
    console.log(JSON.stringify(report, null, 1));
    if (opt && opt.out) fs.writeFileSync(opt.out, JSON.stringify(report, null, 1));
    console.error('\n=== S_dec VERDICT (' + report.mode + ', ' + report.distribution + ', ' + report.seedSet + ') ===');
    console.error('  POOLED   S_dec = ' + pct(report.S_dec_pooled) + '  (gate ' + (report.gate.pooled_pass ? 'PASS' : 'FAIL') + ')');
    console.error('  PLANNER  S_dec = ' + pct(report.S_dec_planner) + '  (gate ' + (report.gate.planner_pass ? 'PASS' : 'FAIL') + ')');
    console.error('  ENDGAME  S_dec = ' + pct(report.S_dec_endgame) + '  (gate ' + (report.gate.endgame_pass ? 'PASS' : 'FAIL') + ')');
    console.error('  GATE (≥95% pooled+per-regime): ' + (report.gate.PASS ? '✅ PASS' : '❌ FAIL'));
    process.exit(report.gate.PASS ? 0 : 2);
}

function sha256(fp) { return require('crypto').createHash('sha256').update(fs.readFileSync(fp)).digest('hex'); }
function safeCommitSha() { try { return JSON.parse(fs.readFileSync(seal.COMMIT_PATH, 'utf8')).sha256_salt; } catch (e) { return null; } }

// ---- self-test: the verdict's own proof on oracle/perturb stubs -------------
// Uses BUFFER seeds [280000,290000) (no tuning, no verdict) so the sealed pool is
// untouched. oracle ⇒ S_dec==100% (no-fallback override commits prod's exact
// decision, harness measures agreement, decision-match ⇒ force-exact). perturb p
// ⇒ S_dec ≈ 1−p (the metric counts disagreements proportionally) in BOTH regimes.
async function selftest() {
    let fail = 0;
    const check = (name, ok, extra) => { console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + (extra ? '  ' + extra : '')); if (!ok) fail++; };
    const bufCells = '1024x768';
    // oracle on full games (both regimes) — must be exactly 100% + 0 traj divergence
    {
        const opt = parseArgs(['', '', '--mode', 'oracle', '--natural', '--seeds', '3', '--cells', bufCells, '--calibration']);
        // override calibration seeds to BUFFER range (no calib/sealed touch)
        opt._seedOverride = [280100, 280101, 280102];
        const rep = await runVerdictWithSeeds(opt, opt._seedOverride, 'buffer[280100..]');
        check('oracle full-game: S_dec pooled == 100%', rep.S_dec_pooled === 1, '(' + pct(rep.S_dec_pooled) + ', ' + rep.decisions_total + ' decisions)');
        check('oracle full-game: planner == 100%', rep.S_dec_planner === 1, '(' + pct(rep.S_dec_planner) + ', ' + rep.plans_total + ' plans)');
        check('oracle full-game: endgame == 100%', rep.S_dec_endgame === 1, '(' + pct(rep.S_dec_endgame) + ', ' + rep.egCommits_total + ' egCommits)');
        check('oracle: S_traj fully identical (no divergence, no fallback)', rep.S_traj_identical_frac === 1, '(frac ' + rep.S_traj_identical_frac + ')');
        check('oracle: S_force fork cos≈1 both regimes',
            rep.S_force_fork.planner_cos > 1 - 1e-9 && rep.S_force_fork.endgame_cos > 1 - 1e-9,
            '(pl ' + rep.S_force_fork.planner_cos + ' eg ' + rep.S_force_fork.endgame_cos + ')');
        check('oracle: gate PASS', rep.gate.PASS === true);
    }
    // perturb p=0.2 endgame scatter — the EXACT calibration property: the harness's
    // disagreement count == the flips the gate actually made (assumption-free; a
    // fuzzy "≈80%" band is wrong because n=1 sole-boid commits can't flip but sit in
    // the denominator). Also: flips actually happened, and the gate correctly FAILS.
    {
        const opt = parseArgs(['', '', '--mode', 'perturb', '--perturb', '0.2', '--scatter', '--seeds', '40', '--cells', bufCells]);
        const rep = await runVerdictWithSeeds(opt, range(280200, 40), 'buffer[280200..]');
        check('perturb endgame: harness egDisagree == gate flips (exact calibration)',
            rep.egDisagree_total === rep.gateFlips_total && rep.gateFlips_total > 0,
            '(egDisagree=' + rep.egDisagree_total + ' flips=' + rep.gateFlips_total + ' over ' + rep.egCommits_total + ' egCommits)');
        check('perturb: gate correctly FAILS (<95%)', rep.gate.PASS === false);
        check('perturb: 0 malformed (clean flip path)', rep.gateMalformed_total === 0);
    }
    // perturb p=0.1 full game — same exact check across BOTH regimes pooled.
    {
        const opt = parseArgs(['', '', '--mode', 'perturb', '--perturb', '0.1', '--natural', '--seeds', '3', '--cells', bufCells]);
        const rep = await runVerdictWithSeeds(opt, [280300, 280301, 280302], 'buffer[280300..]');
        check('perturb full-game: harness disagree == gate flips, both regimes (exact)',
            rep.disagree_total === rep.gateFlips_total && rep.gateFlips_total > 0,
            '(disagree=' + rep.disagree_total + ' flips=' + rep.gateFlips_total + ' over ' + rep.decisions_total + ' decisions, S_dec_pl=' + pct(rep.S_dec_planner) + ')');
    }
    console.log(fail === 0 ? 'MOE-VERDICT SELFTEST: ALL PASS' : 'MOE-VERDICT SELFTEST: ' + fail + ' FAILURES');
    return fail === 0 ? 0 : 1;
}
function range(a, n) { const o = []; for (let i = 0; i < n; i++) o.push(a + i); return o; }
async function runVerdictWithSeeds(opt, seeds, label) {
    // thin wrapper: inject an explicit seed list (selftest only) into runVerdict's flow
    if (opt.mode) process.env.EXACTNN_MOE_MODE = opt.mode;
    if (opt.perturb != null) process.env.EXACTNN_MOE_PERTURB = String(opt.perturb);
    const cs = cells(opt);
    const maxFrames = opt.maxFrames || (opt.natural ? 40000 : 4000);
    const base = { policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        maxFrames, postExtinct: 60, decisions: true, fastRender: true, mismatchLimit: 3 };
    const P = newAcc();
    const perCell = [];
    for (const c of cs) {
        const C = await runCell(opt, c, seeds, base);
        for (const k of ['plans', 'planDisagree', 'egCommits', 'egDisagree', 'games', 'lockstepCleared', 'forkCleared', 'trajIdentical', 'gateFlips', 'gateMalformed', 'soleN1']) P[k] += C[k];
        P.firstDiv.push(...C.firstDiv);
        for (const k of ['fsLockP', 'fsLockE', 'fsForkP', 'fsForkE']) { P[k].cos += C[k].cos; P[k].rel += C[k].rel; P[k].n += C[k].n; }
        perCell.push(cellReport(C));
    }
    const S_dec_pooled = sdec(P.planDisagree + P.egDisagree, P.plans + P.egCommits);
    const S_dec_planner = sdec(P.planDisagree, P.plans), S_dec_endgame = sdec(P.egDisagree, P.egCommits);
    const gate = { threshold: GATE,
        pooled_pass: S_dec_pooled != null && S_dec_pooled >= GATE,
        planner_pass: S_dec_planner == null || S_dec_planner >= GATE,
        endgame_pass: S_dec_endgame == null || S_dec_endgame >= GATE };
    gate.PASS = gate.pooled_pass && gate.planner_pass && gate.endgame_pass;
    return { mode: opt.mode, seedSet: label, S_dec_pooled: rnd(S_dec_pooled), S_dec_planner: rnd(S_dec_planner),
        S_dec_endgame: rnd(S_dec_endgame), plans_total: P.plans, egCommits_total: P.egCommits,
        planDisagree_total: P.planDisagree, egDisagree_total: P.egDisagree,
        disagree_total: P.planDisagree + P.egDisagree, gateFlips_total: P.gateFlips, gateMalformed_total: P.gateMalformed,
        decisions_total: P.plans + P.egCommits,
        S_traj_identical_frac: rnd(P.games ? P.trajIdentical / P.games : null),
        S_force_fork: { planner_cos: rnd(mean(P.fsForkP)), endgame_cos: rnd(mean(P.fsForkE)) }, gate };
}

async function main() {
    const opt = parseArgs(process.argv);
    if (opt.selftest) process.exit(await selftest());
    const report = await runVerdict(opt);
    printAndExit(report, opt);
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { DEVICE_MATRIX, runVerdict, GATE };
