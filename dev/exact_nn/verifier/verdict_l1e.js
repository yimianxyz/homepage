// verdict_l1e.js — the independent-verifier SEALED verdict for the L1e endgame
// (N≤5) policy. The D4 twin of verdict.js. Wraps diff_harness.runGame (the
// certified lockstep instrument) — no metric is recomputed; the gate's NN-share
// counters come from candidates/l1e.js via global.__l1eStatsLast.
//
// What it proves, per the L1e exactness fact (egBoid identity agreement ⇒ force
// bitwise-exact, since intercept()'s downstream scan/aim/steer is verbatim prod):
//   * S_eg     = 1 − egDisagree/egCommits   (endgame egBoid identity vs prod)
//   * S_frame  = 1 − forceMismatch/forceFrames (bitwise fx,fy, every frame)
//   * S_traj   = fork-mode trajectory identity to extinction (no resync)
//   * NN-share = (cert + trusted)/commits   — the headline endgame NN-share
//        cert-share  : provably-exact certificate commits (zero-risk, no τ)
//        trusted-share: NN margin ≥ τ commits (the τ-residual carries here)
//   * residual = rule-of-three on the TRUSTED commits ONLY (cert = zero-risk)
//
// SEALED DISCIPLINE (verdict.js's): reads sealed seeds from seal_seeds.js (secret
// salt), never prints a seed. Use --sealOffset 40 for the L1e fresh slice (v1=0,
// v2a=20). τ comes from the FROZEN verifier/frozen_tau_eg.json (one-shot). A
// sealed run with egDisagree>0 means the frozen τ did NOT generalize → FAIL
// (exit 2), exactly the anti-Goodhart catch (cf. L1h v2a's 2 disagreements).
//
//   node verdict_l1e.js --seeds 800 --sealOffset 40 [--scatter|--natural]
//   node verdict_l1e.js --seeds 800 --calibration         # τ-freeze validation
//   EXACTNN_EG_WEIGHTS=<final> EXACTNN_EG_STUDENT=<egboidPick> node verdict_l1e.js ...
'use strict';
const path = require('path');
const { runGame } = require('../diff_harness.js');
const seal = require('./seal_seeds.js');

const DEVICE_MATRIX = [
    { W: 390, H: 844 }, { W: 820, H: 1180 }, { W: 1024, H: 768 },
    { W: 1512, H: 982 }, { W: 1680, H: 1050 }, { W: 2560, H: 1440 },
];

function parseArgs(argv) {
    const a = { candidate: path.join(__dirname, '..', 'candidates', 'l1e.js'),
        seeds: 256, natural: false, cells: null, calibration: false,
        sealOffset: 40, maxFrames: null, out: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--candidate') a.candidate = argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--natural') a.natural = true;
        else if (k === '--scatter') a.natural = false;
        else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--calibration') a.calibration = true;
        else if (k === '--sealOffset') a.sealOffset = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
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
    const all = seal.sealedSeeds(require('fs').readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
    return { seeds: all.slice(opt.sealOffset, opt.sealOffset + opt.seeds),
        label: 'SEALED(hidden)@off' + opt.sealOffset };
}

async function main() {
    const opt = parseArgs(process.argv);
    const { seeds, label } = seedList(opt);
    const cs = cells(opt);
    const maxFrames = opt.maxFrames || (opt.natural ? 30000 : 4000);
    const base = { policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        maxFrames, postExtinct: 0, decisions: true, fastRender: true,
        mismatchLimit: 5, uaMobile: false };

    const P = { egCommits: 0, egDisagree: 0, forceFrames: 0, forceMismatch: 0,
        games: 0, cleared: 0, trajIdentical: 0, firstDiv: [],
        commits: 0, cert: 0, trusted: 0, fallback: 0, interceptFrames: 0 };
    const perCell = [];

    for (const c of cs) {
        const C = { W: c.W, H: c.H, egCommits: 0, egDisagree: 0, forceFrames: 0,
            forceMismatch: 0, games: 0, cleared: 0, trajIdentical: 0, firstDiv: [],
            commits: 0, cert: 0, trusted: 0, fallback: 0, interceptFrames: 0 };
        for (let si = 0; si < seeds.length; si++) {
            const seed = seeds[si];
            const startBoids = opt.natural ? 0 : (2 + (si % 4));   // 2..5
            const cfg = Object.assign({}, base, { W: c.W, H: c.H,
                startBoids, scatter: !opt.natural });
            // lockstep: S_eg, S_frame, gate NN-share
            global.__l1eStatsLast = null;
            const L = await runGame(Object.assign({}, cfg, { mode: 'lockstep', resync: true }), seed, opt.candidate);
            const gs = global.__l1eStatsLast || { commits: 0, cert: 0, trusted: 0, fallback: 0 };
            C.games++; if (L.cleared) C.cleared++;
            C.interceptFrames += L.framesByRegime.intercept;
            C.forceFrames += L.framesByRegime.planner + L.framesByRegime.intercept + L.framesByRegime.zero;
            C.forceMismatch += L.mismatchCount;
            if (L.decisions) { C.egCommits += L.decisions.egCommits; C.egDisagree += L.decisions.egDisagree; }
            C.commits += gs.commits; C.cert += gs.cert; C.trusted += gs.trusted; C.fallback += gs.fallback;
            // fork: S_traj (no resync → a wrong commit cascades through the endgame)
            const F = await runGame(Object.assign({}, cfg, { mode: 'fork', resync: false }), seed, opt.candidate);
            if (F.firstMismatchFrame < 0) C.trajIdentical++; else C.firstDiv.push(F.firstMismatchFrame);
        }
        for (const k of ['egCommits', 'egDisagree', 'forceFrames', 'forceMismatch', 'games', 'cleared',
            'trajIdentical', 'commits', 'cert', 'trusted', 'fallback', 'interceptFrames']) P[k] += C[k];
        P.firstDiv.push(...C.firstDiv);
        C.S_eg = C.egCommits ? 1 - C.egDisagree / C.egCommits : null;
        C.S_frame = C.forceFrames ? 1 - C.forceMismatch / C.forceFrames : null;
        C.NNshare = C.commits ? (C.cert + C.trusted) / C.commits : null;
        perCell.push(C);
        console.error(`[cell ${c.W}x${c.H}] S_eg=${fmt(C.S_eg)} S_frame=${fmt(C.S_frame)} `
            + `NN-share=${fmt(C.NNshare)} (cert=${fmt(C.commits ? C.cert / C.commits : null)}) `
            + `egCommits=${C.egCommits} egDis=${C.egDisagree} mm=${C.forceMismatch}`);
    }

    const median = arr => { if (!arr.length) return null; const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
    const nTrusted = P.trusted;     // rule-of-three applies to TRUSTED commits (cert = zero-risk)
    const ruleOf3 = nTrusted ? 3 / nTrusted : null;
    const commitsPerGame = P.games ? P.commits / P.games : 0;
    const report = {
        candidate: path.basename(opt.candidate), seedSet: label, seedCount: seeds.length,
        distribution: opt.natural ? 'natural(full-game)' : 'scatter(startBoids 2..5)',
        cells: cs.length, devicematrix: cs.map(c => c.W + 'x' + c.H),
        commitment_sha256_salt: opt.calibration ? null : safeCommitSha(),
        tauFrozen: frozenTau(),
        games: P.games, cleared: P.cleared, interceptFrames: P.interceptFrames,
        S_eg: rnd(P.egCommits ? 1 - P.egDisagree / P.egCommits : null),
        S_frame: rnd(P.forceFrames ? 1 - P.forceMismatch / P.forceFrames : null),
        S_traj_identical_frac: rnd(P.games ? P.trajIdentical / P.games : null),
        S_traj_median_first_div: median(P.firstDiv),
        egCommits_total: P.egCommits, egDisagree_total: P.egDisagree,
        forceFrames_total: P.forceFrames, forceMismatch_total: P.forceMismatch,
        NNshare: rnd(P.commits ? (P.cert + P.trusted) / P.commits : null),
        certShare: rnd(P.commits ? P.cert / P.commits : null),
        trustedShare: rnd(P.commits ? P.trusted / P.commits : null),
        fallbackShare: rnd(P.commits ? P.fallback / P.commits : null),
        commits_total: P.commits, cert_total: P.cert, trusted_total: P.trusted, fallback_total: P.fallback,
        residual_risk: {
            basis: 'rule-of-three on TRUSTED (NN-margin) commits only; cert commits are zero-risk (sound).',
            trusted_commits: nTrusted, per_trusted_commit_upper95: rnd(ruleOf3),
            per_game_upper95: ruleOf3 != null ? rnd(1 - Math.pow(1 - ruleOf3, commitsPerGame)) : null,
            commits_per_game: rnd(commitsPerGame),
            note: '0 egDisagree assumed for these columns; if egDisagree>0 the frozen τ did '
                + 'NOT generalize → NOT bitwise-exact (see egDisagree_total).',
        },
        exactness_gate: {
            requires: 'sealed S_eg==100% AND S_frame==100% (0 egDisagree, 0 forceMismatch) on cert+trusted commits',
            S_eg_exact: P.egCommits ? P.egDisagree === 0 : null,
            S_frame_exact: P.forceFrames ? P.forceMismatch === 0 : null,
        },
        perCell,
    };
    console.log(JSON.stringify(report, null, 1));
    if (opt.out) require('fs').writeFileSync(opt.out, JSON.stringify(report, null, 1));
    process.exit((P.forceMismatch || P.egDisagree) ? 2 : 0);
}

function fmt(x) { return x == null ? 'n/a' : (x * 100).toFixed(4) + '%'; }
function rnd(x) { return x == null ? null : +x.toFixed(9); }
function frozenTau() {
    try { return JSON.parse(require('fs').readFileSync(path.join(__dirname, 'frozen_tau_eg.json'), 'utf8')).chosenTau; }
    catch (e) { return null; }
}
function safeCommitSha() {
    try { return JSON.parse(require('fs').readFileSync(seal.COMMIT_PATH, 'utf8')).sha256_salt; }
    catch (e) { return null; }
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { DEVICE_MATRIX };
