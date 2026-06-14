// verify_v2a.js — INDEPENDENT verifier analysis of side-a's v2a "NN-share ~0 is
// a fundamental barrier" claim (#5 EVIDENCE_nnshare.md). side-a stratified the
// ~20% high-confidence error floor by cell + N-bucket (uniform). This adds the
// stratification they did NOT run: by DECISION-TYPE and by whether the STUDENT's
// committed winner is a rolled candidate or a non-rolled vprior candidate.
//
// The hypothesis the verifier must test: on plans where the student commits a
// NON-ROLLED vprior candidate, it holds prod's EXACT vprior for that pick — is it
// then reliably exact at high margin (a gateable NN-share the margin-only gate
// misses)? Or is the ~20% floor uniform across decision-types (barrier real)?
//
// Per prod plan, an anchored transform logs: snapshot, cands, prod's 16 scores,
// committed bi, and the rolled indices pidx[0:4]. Offline: run the v2a student,
// get its deduped argmax (rolled? coord) + margin + agree-vs-prod, and the prod
// decision-type. Output: overall calib record + reliability stratified by
// {student-winner-rolled?} and {prod decision-type}.
//
//   node verify_v2a.js --seeds 12 --seedStart 270000 --cells 1024x768,2560x1440 \
//     --student ../student_v2a/studentScores.js --weights ../student_v2a/student_weights.json
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

const A_RET = '        return { x: cands[bi].x, y: cands[bi].y };';
const A_RESET = '        var score = vprior.slice();';
function tf(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(A_RET) < 0 || code.indexOf(A_RESET) < 0) throw new Error('anchors not found');
    let out = code.replace(A_RET,
        '        if (window.__vl) window.__vl.push({ px:s.px,py:s.py,pvx:s.pvx,pvy:s.pvy,psize:s.psize,'
        + 'bx:s.bx.slice(),by:s.by.slice(),bvx:s.bvx.slice(),bvy:s.bvy.slice(),'
        + 'cx:cands.map(function(c){return c.x;}),cy:cands.map(function(c){return c.y;}),'
        + 'score:score.slice(), bi:bi, pidx4:pidx.slice(0,4) });\n' + A_RET);
    return out;
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
function dedupTop(score, cx, cy) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) { const key = ckey(cx[k], cy[k]); const c = best.get(key);
        if (!c || score[k] > c.s || (score[k] === c.s && k < c.idx)) best.set(key, { key, s: score[k], idx: k }); }
    return Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
}

function relCurve(recs) {   // recs: [{margin, ok}]
    const BINS = [0, 1e-3, 0.01, 0.03, 0.1, 0.3, 1, Infinity];
    const out = [];
    for (let i = 0; i < BINS.length - 1; i++) {
        const lo = BINS[i], hi = BINS[i + 1];
        const inb = recs.filter(r => Number.isFinite(r.margin) && r.margin >= lo && r.margin < hi);
        if (inb.length) out.push({ range: [lo, hi === Infinity ? null : hi], n: inb.length, disagree: +((inb.filter(r => !r.ok).length / inb.length)).toFixed(4) });
    }
    return out;
}

async function main() {
    const a = { seeds: 12, seedStart: 270000, maxFrames: 12000, cells: '1024x768,2560x1440',
        student: path.join(__dirname, '..', 'student_v2a', 'studentScores.js'),
        weights: path.join(__dirname, '..', 'student_v2a', 'student_weights.json'), out: null };
    for (let i = 2; i < process.argv.length; i++) { const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i]; else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i]; else if (k === '--cells') a.cells = process.argv[++i];
        else if (k === '--student') a.student = process.argv[++i]; else if (k === '--weights') a.weights = process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i]; }
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const { loadStudent } = require(path.resolve(a.student));
    const student = loadStudent(path.resolve(a.weights));
    const cells = a.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    const all = [];                       // {margin, ok, studentWinnerRolled, prodType}
    for (const c of cells) {
        const cfg = { W: c.W, Hc: c.H, PREDATOR_RANGE: 80, NUM_BOIDS: c.W <= 768 ? 60 : 120 };
        for (let i = 0; i < a.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: a.seedStart + i, fastRender: true, transform: tf });
            game.win.__vl = [];
            while (game.boidCount() > 0 && game.frame() < a.maxFrames) game.stepFrame();
            for (const r of game.win.__vl) {
                const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                    bx: r.bx, by: r.by, bvx: r.bvx, bvy: r.bvy, nAlive: r.bx.length };
                const cands = r.cx.map((x, k) => ({ x, y: r.cy[k] }));
                const sc = student(snap, cands, cfg);
                const st = dedupTop(sc, r.cx, r.cy);
                const sArg = st[0].idx, margin = st.length >= 2 ? st[0].s - st[1].s : Infinity;
                const rolled = new Set(r.pidx4);
                const ok = ckey(r.cx[sArg], r.cy[sArg]) === ckey(r.cx[r.bi], r.cy[r.bi]);
                // prod decision type (deduped top-2 of PROD's scores)
                const pt = dedupTop(r.score, r.cx, r.cy);
                let prodType;
                const p1r = rolled.has(pt[0].idx), p2r = pt.length >= 2 && rolled.has(pt[1].idx);
                if (!p1r) prodType = 'vprior_top'; else if (p1r && p2r) prodType = 'rolled_both'; else prodType = 'rolled_vs_vprior';
                all.push({ margin, ok, studentWinnerRolled: rolled.has(sArg), prodType });
            }
            game.win.__vl = null;
        }
        process.stderr.write(`[${c.W}x${c.H}] plans=${all.length}\n`);
    }

    const sub = (pred) => all.filter(pred);
    const summ = (recs) => ({ n: recs.length, agree: +(recs.filter(r => r.ok).length / Math.max(1, recs.length)).toFixed(4),
        highMarginDisagree_ge1: (() => { const h = recs.filter(r => Number.isFinite(r.margin) && r.margin >= 1); return h.length ? +(h.filter(r => !r.ok).length / h.length).toFixed(4) : null; })(),
        reliability: relCurve(recs) });
    const report = {
        spec: 'side-b INDEPENDENT v2a verification — disagreement stratified by student-winner-type & prod decision-type',
        plans: all.length, overallAgree: +(all.filter(r => r.ok).length / all.length).toFixed(4),
        overall: summ(all),
        byStudentWinner: { nonRolled_vprior: summ(sub(r => !r.studentWinnerRolled)), rolled: summ(sub(r => r.studentWinnerRolled)) },
        byProdDecisionType: { vprior_top: summ(sub(r => r.prodType === 'vprior_top')),
            rolled_both: summ(sub(r => r.prodType === 'rolled_both')), rolled_vs_vprior: summ(sub(r => r.prodType === 'rolled_vs_vprior')) },
        // the crux: student commits a non-rolled vprior winner AT HIGH MARGIN — reliably exact?
        crux_studentNonRolled_highMargin: summ(sub(r => !r.studentWinnerRolled && Number.isFinite(r.margin) && r.margin >= 0.3)),
    };
    console.log(JSON.stringify(report, null, 1));
    if (a.out) fs.writeFileSync(a.out, JSON.stringify(report, null, 1));
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
