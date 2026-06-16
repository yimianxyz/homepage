// ungated_sdec.js — the ELEGANT-UNIFICATION decisive measurement (#6): does prod's
// ROLLOUT-PLANNER (planCheap) reproduce prod's N≤5 endgame egBoid decisions?
//
// Final policy idea = un-gate planCheap to run for ALL N≥1 (prod's own value-net +
// rollout, NN genuinely in-loop everywhere — no separate endgame NN, no formula). The
// decisive question: at an N≤5 endgame state, does planCheap commit the SAME boid as
// prod's intercept() scan-argmin egBoid?
//
// S_dec(N≤5) = fraction of prod egBoid-commit states where planCheap's committed
// candidate maps to prod's egBoid. Candidate→boid map (from candidates()): cand0 =
// E3D patrol → NOT a boid → DISAGREEMENT; cand[bi] for 1≤bi≤n → boid order[bi-1]
// (bi-th nearest by dist²); padded cand (bi>n) = E3D → DISAGREEMENT. We measure at
// PROD's egBoid (re)commit frames (the fresh decision points), running planCheap fresh.
//
// Also reports the value-net/rollout DECISION SHARE (honesty): per endgame commit,
// did the committed candidate's winning score come from a ROLLED slot (rollout
// catch-count + value-net bootstrap) or an UNROLLED slot (pure value-net vprior)?
//
//   node ungated_sdec.js --seeds 512 --scatter --sealOffset 0   [--natural] [--calibration]
//   EXACTNN_SALT_PATH=~/.exactnn_seal_salt_p2 node ungated_sdec.js ...   (fresh salt)
'use strict';
const path = require('path');
const fs = require('fs');
const { createGame } = require('../stepper.js');
const seal = require('./seal_seeds.js');
const egScan = require('../endgame/eg_scan.js');

const DEVICE_MATRIX = [
    { W: 390, H: 844 }, { W: 820, H: 1180 }, { W: 1024, H: 768 },
    { W: 1512, H: 982 }, { W: 1680, H: 1050 }, { W: 2560, H: 1440 },
];

// ---- instrument predator_cheap: expose planCheap as a pure probe + capture the
// decision provenance (committed index + rollout/value-net source). Pure inserts;
// no arithmetic changed (a pristine vs transformed digest check guards this).
const DEBUG_ANCHOR = '    window.__cheap = {';
const SCORE_INIT = '        var score = vprior.slice();';
const ROLL_CAP = '            score[ci] = rr.catches + boot;';
const VIZ_LINE = '        if (vizModel) cp_value_viz(NET, fr.feat[bi], fr.ctx, vizModel);';

function instrument(code) {
    code = code.replace(SCORE_INIT, SCORE_INIT + '\n        window.__rolledTmp = [];');
    code = code.replace(ROLL_CAP, ROLL_CAP + '\n            if (window.__rolledTmp) window.__rolledTmp.push({ ci: ci, catches: rr.catches, boot: boot });');
    code = code.replace(VIZ_LINE,
        '        window.__lastPlan = { bi: bi, isRolled: false, catches: null, boot: null, vprior_bi: vprior[bi], score_bi: score[bi] };\n'
        + '        if (window.__rolledTmp) { for (var __ri = 0; __ri < window.__rolledTmp.length; __ri++) { if (window.__rolledTmp[__ri].ci === bi) { window.__lastPlan.isRolled = true; window.__lastPlan.catches = window.__rolledTmp[__ri].catches; window.__lastPlan.boot = window.__rolledTmp[__ri].boot; } } }\n'
        + VIZ_LINE);
    // expose planCheap (returns committed target + provenance) and egBoid debug
    code = code.replace(DEBUG_ANCHOR,
        '    window.__cheapProbe = { plan: function (s) { var t = planCheap(s); return { target: t, prov: window.__lastPlan }; } };\n'
        + '    window.__cheapDebug = { get: function () { return { egBoid: egBoid, frame: frame }; } };\n'
        + DEBUG_ANCHOR);
    return code;
}

// candidates() order: boids sorted by dist² to predator (V8 stable sort). cand[bi]
// for bi≥1 = boid order[bi-1] while bi-1 < n, else E3D (padded). bi==0 = E3D patrol.
function committedBoidIdx(px, py, bx, by, bi) {
    const n = bx.length;
    if (bi === 0) return -1;          // E3D patrol — not a boid
    const k = bi - 1;
    if (k >= n) return -1;            // padded E3D — not a boid
    const order = [];
    for (let i = 0; i < n; i++) { const dx = bx[i] - px, dy = by[i] - py; order.push([dx * dx + dy * dy, i]); }
    order.sort((a, b) => a[0] - b[0]);
    return order[k][1];               // the boid index the planner committed to
}

function snapshotFor(sim) {
    const b = sim.boids, n = b.length;
    const bx = new Array(n), by = new Array(n), bvx = new Array(n), bvy = new Array(n);
    for (let i = 0; i < n; i++) { bx[i] = b[i].position.x; by[i] = b[i].position.y; bvx[i] = b[i].velocity.x; bvy[i] = b[i].velocity.y; }
    const p = sim.predator;
    return { bx, by, bvx, bvy, px: p.position.x, py: p.position.y, pvx: p.velocity.x, pvy: p.velocity.y,
        psize: p.currentSize, lastFeed: p.lastFeedTime, nowMs: 0 };
}

async function runGame(opt, seed) {
    const game = await createGame({ policyDir: opt.policyDir, W: opt.W, H: opt.H, seed,
        startBoids: opt.startBoids, scatter: opt.scatter, fastRender: true, transform: (f, c) => f === 'predator_cheap.js' ? instrument(c) : null });
    const refForce = game.win.__cheap.force;       // prod force (reference trajectory)
    const probe = game.win.__cheapProbe;
    const dbg = game.win.__cheapDebug;
    const C = { commits: 0, agree: 0, n1: 0, n1agree: 0, e3dPick: 0, rolled: 0, vnetOnly: 0,
        rolledCatchPos: 0, byN: {} };
    let prevEg = dbg.get().egBoid;
    game.setForce(function (pred, boids) {
        const f = refForce(pred, boids);            // prod commits/holds egBoid
        const eg = dbg.get().egBoid;
        if (boids.length >= 1 && boids.length <= 5 && eg && eg !== prevEg) {
            // a FRESH prod egBoid commit at N≤5 → ask planCheap on this exact state
            const s = snapshotFor(game.sim);
            const r = probe.plan(s);
            const bi = r.prov ? r.prov.bi : -1;
            const boidIdx = committedBoidIdx(s.px, s.py, s.bx, s.by, bi);
            const agree = boidIdx >= 0 && boids[boidIdx] === eg;
            C.commits++;
            const N = boids.length; C.byN[N] = C.byN[N] || { c: 0, a: 0 }; C.byN[N].c++;
            if (agree) { C.agree++; C.byN[N].a++; }
            if (N === 1) { C.n1++; if (agree) C.n1agree++; }
            if (boidIdx < 0) C.e3dPick++;            // planner picked the E3D patrol, not a boid
            if (r.prov && r.prov.isRolled) { C.rolled++; if (r.prov.catches > 0) C.rolledCatchPos++; }
            else C.vnetOnly++;                       // committed an unrolled (pure value-net) candidate
        }
        prevEg = eg;
        return f;
    });
    while (game.frame() < opt.maxFrames) {
        if (game.boidCount() === 0) break;
        game.stepFrame();
    }
    return C;
}

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', '..', '..', 'js'), seeds: 256, natural: false,
        cells: null, sealOffset: 0, calibration: false, maxFrames: null, out: null };
    for (let i = 2; i < argv.length; i++) { const k = argv[i];
        if (k === '--seeds') a.seeds = +argv[++i]; else if (k === '--natural') a.natural = true;
        else if (k === '--scatter') a.natural = false; else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--sealOffset') a.sealOffset = +argv[++i]; else if (k === '--calibration') a.calibration = true;
        else if (k === '--maxFrames') a.maxFrames = +argv[++i]; else if (k === '--out') a.out = argv[++i];
        else throw new Error('unknown arg ' + k); }
    return a;
}
function cells(opt) { return opt.cells ? opt.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; }) : DEVICE_MATRIX; }
function seedList(opt) {
    if (opt.calibration) { const o = []; for (let i = 0; i < opt.seeds; i++) o.push(270000 + i); return { seeds: o, label: 'calibration[270000,280000)' }; }
    const all = seal.sealedSeeds(fs.readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
    return { seeds: all.slice(opt.sealOffset, opt.sealOffset + opt.seeds), label: 'SEALED@off' + opt.sealOffset };
}

async function main() {
    const opt = parseArgs(process.argv);
    const { seeds, label } = seedList(opt);
    const cs = cells(opt);
    const maxFrames = opt.maxFrames || (opt.natural ? 40000 : 4000);
    const P = { commits: 0, agree: 0, n1: 0, n1agree: 0, e3dPick: 0, rolled: 0, vnetOnly: 0, rolledCatchPos: 0 };
    const perCell = [];
    for (const c of cs) {
        const CC = { commits: 0, agree: 0, n1: 0, n1agree: 0, e3dPick: 0, rolled: 0, vnetOnly: 0, rolledCatchPos: 0 };
        for (let si = 0; si < seeds.length; si++) {
            const startBoids = opt.natural ? 0 : (2 + (si % 4));
            const C = await runGame(Object.assign({}, opt, { W: c.W, H: c.H, startBoids, scatter: !opt.natural, maxFrames }), seeds[si]);
            for (const k of ['commits', 'agree', 'n1', 'n1agree', 'e3dPick', 'rolled', 'vnetOnly', 'rolledCatchPos']) CC[k] += C[k];
        }
        for (const k of Object.keys(P)) P[k] += CC[k];
        const sdec = CC.commits ? CC.agree / CC.commits : null;
        const nt = CC.commits - CC.n1;
        const sdecNT = nt ? (CC.agree - CC.n1agree) / nt : null;
        perCell.push({ cell: c.W + 'x' + c.H, commits: CC.commits, S_dec: rnd(sdec), S_dec_nonTrivial: rnd(sdecNT),
            e3dPickFrac: rnd(CC.commits ? CC.e3dPick / CC.commits : null), vnetOnlyFrac: rnd(CC.commits ? CC.vnetOnly / CC.commits : null) });
        console.error(`[cell ${c.W}x${c.H}] S_dec(N≤5)=${pct(sdec)} nonTrivial=${pct(sdecNT)} commits=${CC.commits} e3dPick=${CC.e3dPick} vnetOnly=${CC.vnetOnly} rolled=${CC.rolled}`);
    }
    const S_dec = P.commits ? P.agree / P.commits : null;
    const nt = P.commits - P.n1;
    const report = {
        metric: 'S_dec(N≤5) = un-gated planCheap committed-boid == prod intercept egBoid',
        seedSet: label, seedCount: seeds.length, sealOffset: opt.calibration ? null : opt.sealOffset,
        commitment_sha256_salt: opt.calibration ? null : safeCommitSha(),
        distribution: opt.natural ? 'natural(full-game endgame frames)' : 'scatter(startBoids 2..5)',
        cells: cs.length, commits_total: P.commits,
        S_dec: rnd(S_dec), S_dec_nonTrivial: rnd(nt ? (P.agree - P.n1agree) / nt : null),
        agree_total: P.agree, disagree_total: P.commits - P.agree,
        soleN1_total: P.n1, soleN1_frac: rnd(P.commits ? P.n1 / P.commits : null),
        gate_pass: S_dec != null && S_dec >= 0.95,
        // honesty — value-net / rollout decision share in the endgame
        e3dPatrolPick_total: P.e3dPick, e3dPatrolPick_frac: rnd(P.commits ? P.e3dPick / P.commits : null),
        rolled_share: rnd(P.commits ? P.rolled / P.commits : null),
        valueNetOnly_share: rnd(P.commits ? P.vnetOnly / P.commits : null),
        rolledWithPositiveCatch_share: rnd(P.commits ? P.rolledCatchPos / P.commits : null),
        perCell,
    };
    console.log(JSON.stringify(report, null, 1));
    if (opt.out) fs.writeFileSync(opt.out, JSON.stringify(report, null, 1));
    console.error(`\n=== S_dec(N≤5) = ${pct(S_dec)} (non-trivial ${pct(nt ? (P.agree - P.n1agree) / nt : null)}) — gate ${report.gate_pass ? '✅ PASS' : '❌ FAIL'} ===`);
    process.exit(report.gate_pass ? 0 : 2);
}
function rnd(x) { return x == null ? null : +x.toFixed(6); }
function pct(x) { return x == null ? 'n/a' : (x * 100).toFixed(3) + '%'; }
function safeCommitSha() { try { return JSON.parse(fs.readFileSync(seal.COMMIT_PATH, 'utf8')).sha256_salt; } catch (e) { return null; } }
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { committedBoidIdx, instrument, DEVICE_MATRIX };
