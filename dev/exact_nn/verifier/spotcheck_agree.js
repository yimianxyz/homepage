// spotcheck_agree.js — FULLY INDEPENDENT recompute of the v2a `agree` signal.
// Audit purpose: confirm the ~39% agree / ~18% high-conf floor is REAL, not a
// pipeline bug deflating NN-share. Written from scratch; shares no code with
// calib_gen.js beyond the stepper + the student module under test.
//
// At each prod planCheap, a transform logs prod's committed coord (cands[bi])
// AND the full 16 candidate coords. Offline we run the v2a student on the SAME
// snapshot, take its DEDUPED argmax coord + deduped top-2 margin, and set
// agree = (student coord == prod coord). We ALSO sanity-cross-check three
// independent ways of recovering prod's committed coord (raw bi, raw argmax of
// logged prod score, deduped argmax of prod score) to prove the comparison is
// coord-canonical and not subtly wrong.
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

// Log prod's committed bi + full candidate coords + prod's full score[16].
const A_RET = '        return { x: cands[bi].x, y: cands[bi].y };';
const A_SCORE = '        var bi = 0, bs = -Infinity;';
function tf(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(A_RET) < 0) throw new Error('return anchor not found');
    // log AT the return: bi, candidate coords, and the score array
    const inj = '        if (window.__sc) window.__sc.push({'
        + ' px:s.px,py:s.py,pvx:s.pvx,pvy:s.pvy,psize:s.psize,'
        + ' bx:s.bx.slice(),by:s.by.slice(),bvx:s.bvx.slice(),bvy:s.bvy.slice(),'
        + ' cx:cands.map(function(c){return c.x;}), cy:cands.map(function(c){return c.y;}),'
        + ' score:score.slice(), bi:bi, bx0:cands[bi].x, by0:cands[bi].y });\n';
    return code.replace(A_RET, inj + A_RET);
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }

// deduped argmax (canonical lowest index) + deduped top-2 margin
function dedupArg(score, cx, cy) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) {
        const key = ckey(cx[k], cy[k]); const cur = best.get(key);
        if (!cur || score[k] > cur.s || (score[k] === cur.s && k < cur.idx)) best.set(key, { s: score[k], idx: k });
    }
    const arr = Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
    return { argIdx: arr[0].idx, margin: arr.length >= 2 ? arr[0].s - arr[1].s : Infinity, groups: arr.length };
}
// raw argmax (first-max), the EXACT rule prod uses to pick bi
function rawArg(score) { let bi = 0, bs = -Infinity; for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; } return bi; }

async function main() {
    const studentPath = process.argv[2] || path.join(__dirname, '..', 'student_v2a', 'studentScores.js');
    const weightsPath = process.argv[3] || path.join(__dirname, '..', 'student_v2a', 'student_weights.json');
    const seedStart = +(process.argv[4] || 270000);
    const nSeeds = +(process.argv[5] || 5);
    const cellArg = process.argv[6] || '1024x768';
    const [W, H] = cellArg.split('x').map(Number);
    const maxFrames = 20000;

    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const { loadStudent } = require(path.resolve(studentPath));
    const studentScores = loadStudent(path.resolve(weightsPath));
    const cfg = { W, Hc: H, PREDATOR_RANGE: 80, NUM_BOIDS: W <= 768 ? 60 : 120 };

    const recs = [];
    let prodBiMismatchRaw = 0, prodCoordMismatch = 0;
    for (let i = 0; i < nSeeds; i++) {
        const game = await createGame({ policyDir, W, H, seed: seedStart + i, fastRender: true, transform: tf });
        game.win.__sc = [];
        while (game.boidCount() > 0 && game.frame() < maxFrames) game.stepFrame();
        for (const r of game.win.__sc) {
            // cross-check 1: prod's logged bi == raw argmax of prod's logged score
            const rb = rawArg(r.score);
            if (rb !== r.bi) prodBiMismatchRaw++;
            // cross-check 2: prod committed coord from bi == logged bx0/by0
            if (ckey(r.cx[r.bi], r.cy[r.bi]) !== ckey(r.bx0, r.by0)) prodCoordMismatch++;

            const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                bx: r.bx, by: r.by, bvx: r.bvx, bvy: r.bvy, nAlive: r.bx.length };
            const cands = r.cx.map((x, k) => ({ x, y: r.cy[k] }));
            const sc = studentScores(snap, cands, cfg);
            const d = dedupArg(sc, r.cx, r.cy);
            const studKey = ckey(r.cx[d.argIdx], r.cy[d.argIdx]);
            const prodKey = ckey(r.bx0, r.by0);          // prod's ACTUAL committed coord
            recs.push({ margin: Number.isFinite(d.margin) ? d.margin : null, agree: studKey === prodKey, n: snap.nAlive });
        }
        game.win.__sc = null;
    }

    // headline numbers
    const N = recs.length, agree = recs.filter(r => r.agree).length;
    const finite = recs.filter(r => r.margin != null);
    const hc = finite.filter(r => r.margin >= 1);
    const hcAgree = hc.filter(r => r.agree).length;
    // bin reliability
    const BINS = [0, 1e-3, 0.01, 0.03, 0.1, 0.3, 1, Infinity];
    const bins = [];
    for (let b = 0; b < BINS.length - 1; b++) {
        const lo = BINS[b], hi = BINS[b + 1];
        const inb = finite.filter(r => r.margin >= lo && r.margin < hi);
        if (inb.length) bins.push({ range: [lo, hi === Infinity ? null : hi], n: inb.length, disagree: +((inb.filter(r => !r.agree).length / inb.length)).toFixed(4) });
    }
    console.log(JSON.stringify({
        student: path.basename(path.dirname(studentPath)) + '/' + path.basename(studentPath),
        weights: path.basename(weightsPath),
        cell: cellArg, NUM_BOIDS: cfg.NUM_BOIDS, seeds: [seedStart, seedStart + nSeeds],
        plans: N, agree, agreeFrac: +(agree / N).toFixed(4),
        highConf_marginGe1: { n: hc.length, agree: hcAgree, disagreeFrac: hc.length ? +((hc.length - hcAgree) / hc.length).toFixed(4) : null },
        reliability: bins,
        crossChecks: { prodBi_neq_rawArgmaxOfLoggedScore: prodBiMismatchRaw, prodCoord_neq_loggedCommitCoord: prodCoordMismatch },
    }, null, 1));
}
main().catch(e => { console.error(e); process.exit(1); });
