// cheap_gate_probe.js — ESCAPE HUNT for the L1h NN-share barrier.
//
// CLAIM: under the exact gate (trust student deduped-argmax iff deduped top-2
// margin >= tau where tau holds 0 trusted disagreements), NN-share floors ~0
// because the committed target is decided by 'boot' (rollout-bound).
//
// ANGLE TESTED HERE: a CHEAP DETERMINISTIC gate that needs NO 90-step rollout.
// The prod ranking that picks the 4 rolled candidates uses the ballistic pscore
// = cp_features feat[18]-feat[16] = (caught - tCatchNorm), available with no
// rollout. cp_top1(feat) is its argmax. Hypothesis: when the student's
// deduped-argmax COINCIDES with the cheap ballistic-pscore argmax (and/or with
// the dedup of vprior, or with simple structural signals), prod's committed
// winner is reliably the same target — carving a 0-disagreement band at
// non-trivial coverage that the margin-only gate misses.
//
// We log, per prod plan (anchored, digest-inert):
//   snapshot, candidates, prod's full 16 score vector, prod committed bi,
//   the rolled indices pidx[0:4].
// Offline we compute:
//   - student deduped argmax + margin + agree(vs prod, deduped coord)
//   - cheap ballistic pscore argmax (raw cp_top1) + a coord-deduped version
//   - vprior deduped argmax
//   - prod-winner-is-rolled?
// and several CHEAP gates, reporting for each: coverage (trusted frac) and
// trusted-disagreement count/rate. An ESCAPE = a cheap gate with 0 trusted
// disagreements at coverage >> 0.0003.
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

const JS_DIR = process.env.EXACTNN_JS_DIR || path.join(__dirname, '..', '..', '..', 'js');
const planner = require(path.join(JS_DIR, 'cheap_planner.js'));
const NET = require(path.join(JS_DIR, 'value_net.json'));
const PMAX_S = 2.5, PMAX_F = 0.05;

// anchored transform: log prod's full score vector + bi + rolled pidx + state
const A_RET = '        return { x: cands[bi].x, y: cands[bi].y };';
const A_RESET = '        var score = vprior.slice();';
function tf(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(A_RET) < 0 || code.indexOf(A_RESET) < 0) throw new Error('anchors not found');
    return code.replace(A_RET,
        '        if (window.__cg) window.__cg.push({ px:s.px,py:s.py,pvx:s.pvx,pvy:s.pvy,psize:s.psize,'
        + 'bx:s.bx.slice(),by:s.by.slice(),bvx:s.bvx.slice(),bvy:s.bvy.slice(),'
        + 'cx:cands.map(function(c){return c.x;}),cy:cands.map(function(c){return c.y;}),'
        + 'score:score.slice(), bi:bi, pidx4:pidx.slice(0,4) });\n' + A_RET);
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
function dedupTop(score, cx, cy) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) { const key = ckey(cx[k], cy[k]); const c = best.get(key);
        if (!c || score[k] > c.s || (score[k] === c.s && k < c.idx)) best.set(key, { key, s: score[k], idx: k }); }
    return Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
}
// coord-deduped argmax of an arbitrary per-candidate scalar (ties -> lowest idx)
function dedupArg(vals, cx, cy) {
    const t = dedupTop(vals, cx, cy);
    return t[0].idx;
}

function main() {
    const a = { seeds: 60, seedStart: 270000, maxFrames: 20000,
        cells: '390x844,820x1180,1024x768,1512x982,1680x1050,2560x1440',
        student: path.join(__dirname, '..', 'student_v2a', 'studentScores.js'),
        weights: path.join(__dirname, '..', 'student_v2a', 'student_weights.json'),
        out: null };
    for (let i = 2; i < process.argv.length; i++) { const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i]; else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i]; else if (k === '--cells') a.cells = process.argv[++i];
        else if (k === '--student') a.student = process.argv[++i]; else if (k === '--weights') a.weights = process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i]; }
    return run(a);
}

async function run(a) {
    const policyDir = JS_DIR;
    const { loadStudent } = require(path.resolve(a.student));
    const student = loadStudent(path.resolve(a.weights));
    const cells = a.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    const recs = [];   // per plan
    for (const c of cells) {
        const cfg = { W: c.W, Hc: c.H, PREDATOR_RANGE: 80, NUM_BOIDS: c.W <= 768 ? 60 : 120 };
        for (let i = 0; i < a.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: a.seedStart + i, fastRender: true, transform: tf });
            game.win.__cg = [];
            while (game.boidCount() > 0 && game.frame() < a.maxFrames) game.stepFrame();
            for (const r of game.win.__cg) {
                const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                    bx: r.bx, by: r.by, bvx: r.bvx, bvy: r.bvy, nAlive: r.bx.length };
                const cands = r.cx.map((x, k) => ({ x, y: r.cy[k] }));
                // cheap features (no rollout) — EXACT prod cp_features/cp_value
                const fr = planner.cp_features(snap, cands, PMAX_S, PMAX_F);
                const vprior = planner.cp_value(NET, fr.feat, fr.ctx);
                // cheap signals:
                const psArg = planner.cp_top1(fr.feat);                       // ballistic pscore raw argmax
                const psVals = fr.feat.map(rr => rr[18] - rr[16]);
                const psArgD = dedupArg(psVals, r.cx, r.cy);                  // deduped ballistic argmax
                const vpriorArgD = dedupArg(vprior, r.cx, r.cy);              // deduped vprior argmax (no rollout)
                // ballistic-pscore deduped top-2 margin (a cheap confidence signal)
                const psTop = dedupTop(psVals, r.cx, r.cy);
                const psMargin = psTop.length >= 2 ? psTop[0].s - psTop[1].s : Infinity;
                // vprior deduped top-2 margin (a cheap confidence signal, no rollout)
                const vpTop = dedupTop(vprior, r.cx, r.cy);
                const vpMargin = vpTop.length >= 2 ? vpTop[0].s - vpTop[1].s : Infinity;
                // student
                const sc = student(snap, cands, cfg);
                const st = dedupTop(sc, r.cx, r.cy);
                const sArg = st[0].idx, margin = st.length >= 2 ? st[0].s - st[1].s : Infinity;
                // prod committed (deduped) + agreement
                const prodKey = ckey(r.cx[r.bi], r.cy[r.bi]);
                const agree = ckey(r.cx[sArg], r.cy[sArg]) === prodKey;
                // prod-winner is a rolled candidate?
                const rolled = new Set(r.pidx4);
                const pt = dedupTop(r.score, r.cx, r.cy);
                const prodWinnerRolled = rolled.has(pt[0].idx);
                recs.push({
                    margin: Number.isFinite(margin) ? margin : 1e9,
                    psMargin: Number.isFinite(psMargin) ? psMargin : 1e9,
                    vpMargin: Number.isFinite(vpMargin) ? vpMargin : 1e9,
                    agree,
                    // cheap-gate booleans:
                    studEqBallistic: ckey(r.cx[sArg], r.cy[sArg]) === ckey(r.cx[psArgD], r.cy[psArgD]),
                    studEqBallisticRaw: sArg === psArg,
                    studEqVprior: ckey(r.cx[sArg], r.cy[sArg]) === ckey(r.cx[vpriorArgD], r.cy[vpriorArgD]),
                    ballisticEqVprior: ckey(r.cx[psArgD], r.cy[psArgD]) === ckey(r.cx[vpriorArgD], r.cy[vpriorArgD]),
                    studIsE3d: sArg === 0,                                    // student picked cand0 (E3D patrol)
                    cell: c.W + 'x' + c.H, n: snap.nAlive,
                });
            }
            game.win.__cg = null;
        }
        process.stderr.write(`[${c.W}x${c.H}] plans=${recs.length}\n`);
    }

    // ---- gate evaluation helpers ----
    const N = recs.length;
    const totalDisagree = recs.filter(r => !r.agree).length;
    // For a boolean subset + a margin threshold tau, report coverage + trusted disagreements.
    function gate(name, pred, marginKey) {
        const sub = recs.filter(pred);
        // smallest tau (on marginKey) with 0 disagreements among {sub & margin>=tau}
        const sorted = sub.slice().sort((x, y) => y[marginKey] - x[marginKey]);
        let dis = 0, bestIdx = -1;
        for (let i = 0; i < sorted.length; i++) {
            if (!sorted[i].agree) dis++;
            if (dis === 0) bestIdx = i;            // contiguous-from-top 0-disagree prefix
        }
        const cov0 = bestIdx + 1;                  // count trusted at 0 disagree
        const tau0 = bestIdx >= 0 ? sorted[bestIdx][marginKey] : null;
        // also: the WHOLE subset (no margin) — its disagreement count
        const subDis = sub.filter(r => !r.agree).length;
        // rule-of-three: with 0 disagreements in cov0 trusted decisions,
        // per-decision disagree prob <= 3/cov0 at 95%.
        const ruleOf3 = cov0 > 0 ? +(3 / cov0).toExponential(2) : null;
        return { name, marginKey, subsetN: sub.length, subsetCoverageOfAll: +(sub.length / N).toFixed(6),
            subsetDisagree: subDis, subsetDisagreeRate: sub.length ? +(subDis / sub.length).toFixed(4) : null,
            zeroDisagree_count: cov0, zeroDisagree_shareOfAll: +(cov0 / N).toFixed(6),
            zeroDisagree_tau: tau0, zeroDisagree_ruleOf3_perDecision: ruleOf3 };
    }

    const report = {
        spec: 'CHEAP-GATE escape hunt for L1h NN-share barrier (no 90-step rollout in any gate)',
        plans: N, overallAgree: +((N - totalDisagree) / N).toFixed(6),
        // BASELINE: margin-only gate (the committed L1h gate) for reference
        gates: [
            gate('margin-only (baseline)', () => true, 'margin'),
            // CHEAP-SIGNAL AGREEMENT GATES:
            gate('student==ballistic(dedup)', r => r.studEqBallistic, 'margin'),
            gate('student==ballistic(dedup) [gate on ballistic margin]', r => r.studEqBallistic, 'psMargin'),
            gate('student==ballistic(raw)', r => r.studEqBallisticRaw, 'margin'),
            gate('student==vprior(dedup)', r => r.studEqVprior, 'margin'),
            gate('student==vprior(dedup) [gate on vprior margin]', r => r.studEqVprior, 'vpMargin'),
            gate('student==vprior(dedup) [gate on ballistic margin]', r => r.studEqVprior, 'psMargin'),
            gate('student==ballistic AND student==vprior', r => r.studEqBallistic && r.studEqVprior, 'margin'),
            gate('student==ballistic==vprior (triple)', r => r.studEqBallistic && r.studEqVprior && r.ballisticEqVprior, 'margin'),
            gate('student picked E3D (cand0)', r => r.studIsE3d, 'margin'),
            gate('student==ballistic AND studIsE3d', r => r.studEqBallistic && r.studIsE3d, 'margin'),
        ],
        // unconditional cheap-signal agreement rates (diagnostic)
        diag: {
            studEqBallistic_frac: +(recs.filter(r => r.studEqBallistic).length / N).toFixed(4),
            studEqVprior_frac: +(recs.filter(r => r.studEqVprior).length / N).toFixed(4),
            studIsE3d_frac: +(recs.filter(r => r.studIsE3d).length / N).toFixed(4),
            // when student==ballistic, agreement with prod:
            agree_given_studEqBallistic: (() => { const s = recs.filter(r => r.studEqBallistic); return s.length ? +(s.filter(r => r.agree).length / s.length).toFixed(4) : null; })(),
            agree_given_studEqVprior: (() => { const s = recs.filter(r => r.studEqVprior); return s.length ? +(s.filter(r => r.agree).length / s.length).toFixed(4) : null; })(),
            agree_given_studEqBallistic_AND_vprior: (() => { const s = recs.filter(r => r.studEqBallistic && r.studEqVprior); return s.length ? +(s.filter(r => r.agree).length / s.length).toFixed(4) : null; })(),
        },
    };
    console.log(JSON.stringify(report, null, 1));
    if (a.out) fs.writeFileSync(a.out, JSON.stringify(report, null, 1));
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
