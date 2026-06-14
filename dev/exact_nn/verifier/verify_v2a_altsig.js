// verify_v2a_altsig.js — EXTENDED v2a verifier probing ALTERNATIVE confidence
// signals beyond the deduped score-margin (the ANGLE: "a better confidence signal
// than the score-margin").
//
// The L1h gate trusts the student's deduped-argmax committed target iff its deduped
// top-2 score-margin >= tau. The margin-only floor is ~18% high-conf disagreement
// (margin>1 band) -> NN-share ~0.0003. This harness recomputes, per plan, the RAW
// deep-set head outputs for the rolled candidates and derives alternative
// confidence signals on the STUDENT'S WINNING candidate:
//
//   (a) winSoftmaxProb  : max over 24 catch-classes of softmax(logits) of the winner
//       winEntropy      : -sum p log p  of that softmax
//   (b) ecArgmaxGap     : | E[catch] - argmax_c |  (smooth-vs-discrete catch gap)
//   (c) bootMargin      : winner.boot - runnerUp.boot  (the boot scalar gap, deduped)
//       scoreVsBoot     : winner finalScore - winner.boot
//   plus margin (the baseline) and agree (vs side-b's own frozen prod).
//
// Then it does an EXACTNESS sweep: for the baseline margin gate and for each
// combined gate (margin>=m AND altsig in trusted region), find the gate with the
// MAX trusted fraction (= NN-share) holding 0 trusted disagreements. The question:
// does any alt-signal carve a 0-disagreement band at non-trivial coverage that the
// margin-only gate misses?
//
//   EXACTNN_JS_DIR=/workspace/.team/wt-exact-nn/js \
//   node verify_v2a_altsig.js --seeds 12 --seedStart 270000 --cells 1024x768,2560x1440 \
//     --student ../student_v2a/studentScores.js --weights ../student_v2a/student_weights.json \
//     --out altsig_report.json --dump altsig_recs.json
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

const A_RET = '        return { x: cands[bi].x, y: cands[bi].y };';
function tf(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(A_RET) < 0) throw new Error('anchor not found');
    return code.replace(A_RET,
        '        if (window.__vl) window.__vl.push({ px:s.px,py:s.py,pvx:s.pvx,pvy:s.pvy,psize:s.psize,'
        + 'bx:s.bx.slice(),by:s.by.slice(),bvx:s.bvx.slice(),bvy:s.bvy.slice(),'
        + 'cx:cands.map(function(c){return c.x;}),cy:cands.map(function(c){return c.y;}),'
        + 'score:score.slice(), bi:bi, pidx4:pidx.slice(0,4) });\n' + A_RET);
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }

// deduped top: returns sorted array of {key, s, idx} (best first), prod-identical rule.
function dedupTop(score, cx, cy) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) {
        const key = ckey(cx[k], cy[k]); const c = best.get(key);
        if (!c || score[k] > c.s || (score[k] === c.s && k < c.idx)) best.set(key, { key, s: score[k], idx: k });
    }
    return Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
}

const CATCH_CLASSES = 24;
// derive alt-signals from a raw 25-vector head output row (logits[0..23] + boot[24])
function rowSignals(row) {
    let mx = -Infinity, amax = 0;
    for (let c = 0; c < CATCH_CLASSES; c++) if (row[c] > mx) { mx = row[c]; amax = c; }
    let Z = 0; const e = new Array(CATCH_CLASSES);
    for (let c = 0; c < CATCH_CLASSES; c++) { e[c] = Math.exp(row[c] - mx); Z += e[c]; }
    let ec = 0, ent = 0, maxp = 0;
    for (let c = 0; c < CATCH_CLASSES; c++) {
        const p = e[c] / Z; ec += c * p;
        if (p > maxp) maxp = p;
        if (p > 0) ent += -p * Math.log(p);
    }
    return { boot: row[CATCH_CLASSES], argmaxCatch: amax, eCatch: ec,
        maxSoftmaxProb: maxp, entropy: ent, ecArgmaxGap: Math.abs(ec - amax) };
}

async function main() {
    const a = { seeds: 12, seedStart: 270000, maxFrames: 12000, cells: '1024x768,2560x1440',
        student: path.join(__dirname, '..', 'student_v2a', 'studentScores.js'),
        weights: path.join(__dirname, '..', 'student_v2a', 'student_weights.json'),
        out: null, dump: null };
    for (let i = 2; i < process.argv.length; i++) { const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i]; else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i]; else if (k === '--cells') a.cells = process.argv[++i];
        else if (k === '--student') a.student = process.argv[++i]; else if (k === '--weights') a.weights = process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i]; else if (k === '--dump') a.dump = process.argv[++i]; }

    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const { loadStudent } = require(path.resolve(a.student));
    const studentScores = loadStudent(path.resolve(a.weights));
    const cells = a.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    // We need the RAW per-candidate head outputs. studentScores recomputes them
    // internally; replicate that exact path here using the exposed featurize+deepset.
    const planner = require(path.join(process.env.EXACTNN_JS_DIR || path.join(policyDir), 'cheap_planner.js'));
    const NET = require(path.join(process.env.EXACTNN_JS_DIR || path.join(policyDir), 'value_net.json'));
    const { PMAX_S, PMAX_F } = require(path.resolve(a.student));

    const recs = [];   // per-plan: {margin, agree, winSoftmaxProb, winEntropy, ecArgmaxGap, bootMargin, scoreVsBoot, winnerRolled, n, cell}
    for (const c of cells) {
        const cellName = c.W + 'x' + c.H;
        const cfg = { W: c.W, Hc: c.H, PREDATOR_RANGE: 80, NUM_BOIDS: c.W <= 768 ? 60 : 120 };
        for (let i = 0; i < a.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: a.seedStart + i, fastRender: true, transform: tf });
            game.win.__vl = [];
            while (game.boidCount() > 0 && game.frame() < a.maxFrames) game.stepFrame();
            for (const r of game.win.__vl) {
                const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                    bx: r.bx, by: r.by, bvx: r.bvx, bvy: r.bvy, nAlive: r.bx.length };
                const cands = r.cx.map((x, k) => ({ x, y: r.cy[k] }));

                // 1) final student scores (== prod-side score logging path) -> margin/agree
                const sc = studentScores(snap, cands, cfg);
                const st = dedupTop(sc, r.cx, r.cy);
                const sArg = st[0].idx;
                const margin = st.length >= 2 ? st[0].s - st[1].s : Infinity;
                const ok = ckey(r.cx[sArg], r.cy[sArg]) === ckey(r.cx[r.bi], r.cy[r.bi]);

                // 2) RAW head outputs for ALL candidates (recompute the deepset forward)
                const fr = planner.cp_features(snap, cands, PMAX_S, PMAX_F);
                const vprior = planner.cp_value(NET, fr.feat, fr.ctx);
                const inp = studentScores.featurize(snap, cands, cfg, fr.feat, vprior);
                const out = studentScores.deepset(inp.boidTok, inp.glob, inp.candTok);   // [16][25]

                // signals on the STUDENT'S WINNING deduped candidate (the committed target)
                const winSig = rowSignals(out[sArg]);
                const runnerIdx = st.length >= 2 ? st[1].idx : sArg;
                const runSig = rowSignals(out[runnerIdx]);
                const rolledSet = new Set(r.pidx4);
                const winnerRolled = rolledSet.has(sArg);

                recs.push({
                    margin: Number.isFinite(margin) ? margin : null,
                    agree: ok,
                    winSoftmaxProb: winSig.maxSoftmaxProb,
                    winEntropy: winSig.entropy,
                    ecArgmaxGap: winSig.ecArgmaxGap,
                    bootMargin: winSig.boot - runSig.boot,
                    scoreVsBoot: st[0].s - winSig.boot,
                    winnerRolled,
                    n: snap.nAlive, cell: cellName,
                });
            }
            game.win.__vl = null;
            // incremental dump after EACH seed so a timeout never loses progress
            if (a.dump) fs.writeFileSync(a.dump, JSON.stringify(recs));
        }
        process.stderr.write(`[${cellName}] plans=${recs.length}\n`);
        if (a.dump) fs.writeFileSync(a.dump, JSON.stringify(recs));
    }

    if (a.dump) fs.writeFileSync(a.dump, JSON.stringify(recs));

    // ---------- EXACTNESS sweep ----------
    // Baseline: best margin-only gate with 0 trusted disagreements.
    const N = recs.length;
    const disagrees = recs.filter(r => !r.agree).length;

    // For an arbitrary gate predicate, compute (trustedFrac, trustedDisagree).
    function gateStats(pred) {
        let t = 0, td = 0;
        for (const r of recs) if (pred(r)) { t++; if (!r.agree) td++; }
        return { trustedFrac: t / N, trustedN: t, trustedDisagree: td };
    }

    // margin-only: smallest margin tau (over observed margins) holding 0 trusted disagrees.
    // trustedFrac = fraction with margin>=tau. Sweep tau over the sorted UNIQUE margins.
    function bestMarginOnly() {
        // collect disagreeing margins; the gate must exclude ALL of them.
        // any margin > max(disagree margin) is safe. tau = nextafter(maxDisagreeMargin).
        let maxBad = -Infinity, anyInfBad = false;
        for (const r of recs) if (!r.agree) {
            if (r.margin === null) anyInfBad = true;
            else if (r.margin > maxBad) maxBad = r.margin;
        }
        // gate: margin > maxBad (strict). If a disagree had margin=null(Infinity), unbeatable->0.
        if (anyInfBad) return { tau: Infinity, ...gateStats(() => false), note: 'a disagree had infinite margin' };
        const pred = (r) => r.margin !== null && r.margin > maxBad;
        return { tau: maxBad, ...gateStats(pred) };
    }

    // 1-D alt-signal gate alone (direction: higher-is-more-confident for prob;
    // lower for entropy/gap). For "higher-confident" signal sig with field f:
    // gate = f >= thr, thr = nextafter(max f over disagrees). For "lower-confident": f <= thr.
    function bestHigher(field) {
        let maxBad = -Infinity, anyNull = false;
        for (const r of recs) if (!r.agree) { const v = r[field]; if (v == null || !Number.isFinite(v)) anyNull = true; else if (v > maxBad) maxBad = v; }
        if (anyNull) return { thr: Infinity, ...gateStats(() => false) };
        const pred = (r) => Number.isFinite(r[field]) && r[field] > maxBad;
        return { thr: maxBad, dir: 'higher', ...gateStats(pred) };
    }
    function bestLower(field) {
        let minBad = Infinity, anyNull = false;
        for (const r of recs) if (!r.agree) { const v = r[field]; if (v == null || !Number.isFinite(v)) anyNull = true; else if (v < minBad) minBad = v; }
        if (anyNull) return { thr: -Infinity, ...gateStats(() => false) };
        const pred = (r) => Number.isFinite(r[field]) && r[field] < minBad;
        return { thr: minBad, dir: 'lower', ...gateStats(pred) };
    }

    // 2-D gate: margin>=m AND altsig in trusted region. We sweep a grid of margin
    // thresholds; for each, among the records passing the margin floor, find the
    // best 1-D altsig sub-gate with 0 trusted disagrees in the SUBSET. Report the
    // overall (margin AND altsig) trustedFrac. This searches the joint region.
    function best2D(field, dir) {
        // candidate margin thresholds: a grid of quantiles of observed margins.
        const ms = recs.map(r => (r.margin === null ? Infinity : r.margin)).filter(Number.isFinite).sort((x, y) => x - y);
        const cand = new Set([0]);
        for (let q = 0; q <= 1.0001; q += 0.02) cand.add(ms[Math.min(ms.length - 1, Math.floor(q * (ms.length - 1)))]);
        let best = { trustedFrac: 0 };
        for (const m of cand) {
            const subset = recs.filter(r => r.margin !== null && r.margin >= m);
            if (!subset.length) continue;
            // best altsig sub-gate over subset with 0 trusted disagree
            let bad;
            if (dir === 'higher') {
                bad = -Infinity; let nl = false;
                for (const r of subset) if (!r.agree) { const v = r[field]; if (!Number.isFinite(v)) nl = true; else if (v > bad) bad = v; }
                if (nl) continue;
                const t = subset.filter(r => Number.isFinite(r[field]) && r[field] > bad);
                const tf2 = t.length / N;
                if (tf2 > best.trustedFrac) best = { trustedFrac: tf2, trustedN: t.length, marginFloor: m, sigThr: bad, dir, trustedDisagree: t.filter(r => !r.agree).length };
            } else {
                bad = Infinity; let nl = false;
                for (const r of subset) if (!r.agree) { const v = r[field]; if (!Number.isFinite(v)) nl = true; else if (v < bad) bad = v; }
                if (nl) continue;
                const t = subset.filter(r => Number.isFinite(r[field]) && r[field] < bad);
                const tf2 = t.length / N;
                if (tf2 > best.trustedFrac) best = { trustedFrac: tf2, trustedN: t.length, marginFloor: m, sigThr: bad, dir, trustedDisagree: t.filter(r => !r.agree).length };
            }
        }
        return best;
    }

    // High-confidence band breakdown: among margin>=1 records, what is the disagree
    // rate, and does conditioning on each alt-signal carve a 0-disagree pocket?
    const hi = recs.filter(r => r.margin !== null && r.margin >= 1);
    function bandFloor(field, dir) {
        // within the margin>=1 band, best 1-D altsig gate with 0 trusted disagree (frac of WHOLE set)
        if (!hi.length) return null;
        let bad = dir === 'higher' ? -Infinity : Infinity, nl = false;
        for (const r of hi) if (!r.agree) { const v = r[field]; if (!Number.isFinite(v)) nl = true; else if (dir === 'higher' ? v > bad : v < bad) bad = v; }
        if (nl) return { trustedFrac: 0, note: 'non-finite disagree in band' };
        const t = hi.filter(r => Number.isFinite(r[field]) && (dir === 'higher' ? r[field] > bad : r[field] < bad));
        return { thr: bad, dir, trustedN_inBand: t.length, trustedFrac_ofAll: t.length / N, disagreeInGate: t.filter(r => !r.agree).length };
    }

    const SIGS = [
        ['winSoftmaxProb', 'higher'],
        ['winEntropy', 'lower'],
        ['ecArgmaxGap', 'lower'],
        ['bootMargin', 'higher'],
        ['scoreVsBoot', 'higher'],
    ];

    const report = {
        spec: 'side-b ALT-SIGNAL v2a verification — exactness sweep over confidence signals beyond score-margin',
        plans: N, disagrees, overallAgree: +((N - disagrees) / N).toFixed(6),
        cells: a.cells, seeds: a.seeds, seedStart: a.seedStart,
        hiBand_margin_ge1: { n: hi.length, disagreeRate: hi.length ? +(hi.filter(r => !r.agree).length / hi.length).toFixed(4) : null },
        baseline_marginOnly: bestMarginOnly(),
        altsig_1D_alone: Object.fromEntries(SIGS.map(([f, d]) => [f, (d === 'higher' ? bestHigher(f) : bestLower(f))])),
        altsig_2D_marginAND: Object.fromEntries(SIGS.map(([f, d]) => [f, best2D(f, d)])),
        altsig_floorInHiBand: Object.fromEntries(SIGS.map(([f, d]) => [f + '_' + d, bandFloor(f, d)])),
    };
    // best exact NN-share over everything
    const cands2 = [report.baseline_marginOnly.trustedFrac];
    for (const k in report.altsig_1D_alone) cands2.push(report.altsig_1D_alone[k].trustedFrac);
    for (const k in report.altsig_2D_marginAND) cands2.push(report.altsig_2D_marginAND[k].trustedFrac);
    report.bestExactNNshare = Math.max(...cands2.filter(x => Number.isFinite(x)));

    console.log(JSON.stringify(report, null, 1));
    if (a.out) fs.writeFileSync(a.out, JSON.stringify(report, null, 1));
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
