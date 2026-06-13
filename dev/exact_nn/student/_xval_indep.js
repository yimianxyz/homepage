// _xval_indep.js — INDEPENDENT cross-validation of the L1h student.
// Written by the verifier (#6 cross-check). Does NOT reuse calib_gen.js or
// l1h.js. It:
//   * runs prod games via the shared stepper at a handful of seeds/cells,
//   * captures at EACH planCheap call: the snapshot, the 16 candidates, AND
//     prod's actual committed (x,y) — via its OWN one-line injection hook
//     (distinct from calib_gen's anchor; logs cands[bi] directly),
//   * loads the delivered student (loadStudent) and scores the SAME snapshot,
//   * (a) reuse check: student's 12 non-rolled scores == prod cp_value bitwise,
//     and feat/vprior recomputed == student's internal pieces,
//   * (b) agreement: my OWN dedup-argmax committed coord == prod committed coord.
//
// Goal: confirm S_dec ~ 0.37 (NOT artificially 0), and reuse_exact, via a path
// that shares NONE of side-b's calib/l1h plumbing.
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');
const { loadStudent } = require('./studentScores.js');

const STUDENT_DIR = __dirname;
const JS_DIR = path.join(__dirname, '..', '..', '..', 'js');
const planner = require(path.join(JS_DIR, 'cheap_planner.js'));
const NET = require(path.join(JS_DIR, 'value_net.json'));

// --- my own committed-coord logger: inject right at planCheap's return ---
// prod ends planCheap with:  return { x: cands[bi].x, y: cands[bi].y };
// I capture the snapshot `s`, the candidates, and the committed (x,y) BY VALUE.
const PROD_RETURN = '        return { x: cands[bi].x, y: cands[bi].y };';
function myTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(PROD_RETURN) < 0) throw new Error('xval: prod return anchor not found');
    const log =
        '        if (window.__xvalLog) window.__xvalLog.push({'
        + ' px:s.px,py:s.py,pvx:s.pvx,pvy:s.pvy,psize:s.psize,'
        + ' bx:s.bx.slice(),by:s.by.slice(),bvx:s.bvx.slice(),bvy:s.bvy.slice(),'
        + ' cx:cands.map(function(c){return c.x;}), cy:cands.map(function(c){return c.y;}),'
        + ' wx:cands[bi].x, wy:cands[bi].y });\n';
    return code.replace(PROD_RETURN, log + PROD_RETURN);
}

// my OWN dedup-argmax committed coord + top-2 margin (independent impl)
const _dv = new DataView(new ArrayBuffer(8));
function fhex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16); }
function ckey(x, y) { return fhex(x) + ',' + fhex(y); }
function dedupArgmax(score, cx, cy) {
    // group by exact (x,y); per group keep the max score (lowest idx on tie);
    // overall winner = highest group max (lowest idx on tie).
    const grp = new Map();
    for (let k = 0; k < score.length; k++) {
        const key = ckey(cx[k], cy[k]);
        const g = grp.get(key);
        if (!g) grp.set(key, { s: score[k], idx: k });
        else if (score[k] > g.s || (score[k] === g.s && k < g.idx)) { g.s = score[k]; g.idx = k; }
    }
    const arr = [...grp.values()].sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
    const win = arr[0];
    return { wx: cx[win.idx], wy: cy[win.idx], margin: arr.length >= 2 ? win.s - arr[1].s : Infinity };
}

async function main() {
    const studentScores = loadStudent(path.join(STUDENT_DIR, 'student_weights.json'));
    const cellArg = process.argv[2] || '2560x1440';
    const [W, H] = cellArg.split('x').map(Number);
    const seedStart = +(process.argv[3] || 270000);
    const nSeeds = +(process.argv[4] || 6);
    const maxFrames = +(process.argv[5] || 20000);
    const cfg = { W, Hc: H, PREDATOR_RANGE: 80, NUM_BOIDS: W <= 768 ? 60 : 120 };

    let plans = 0, agree = 0, vpriorMis = 0, featMis = 0, nonrollMis = 0;
    const byBucket = {};   // margin bucket -> {n, dis}
    const buckets = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, Infinity];
    function bucketOf(m) {
        if (m === Infinity || m === null) return 'inf';
        for (let i = 0; i < buckets.length - 1; i++) if (m >= buckets[i] && m < buckets[i + 1]) return buckets[i] + '-' + buckets[i + 1];
        return 'inf';
    }

    for (let si = 0; si < nSeeds; si++) {
        const game = await createGame({ policyDir: JS_DIR, W, H, seed: seedStart + si,
            fastRender: true, transform: myTransform });
        game.win.__xvalLog = [];
        while (game.boidCount() > 0 && game.frame() < maxFrames) game.stepFrame();
        for (const r of game.win.__xvalLog) {
            plans++;
            const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                bx: r.bx, by: r.by, bvx: r.bvx, bvy: r.bvy, nAlive: r.bx.length };
            const cands = r.cx.map((x, k) => ({ x, y: r.cy[k] }));
            // student scores (the deployed artifact)
            const sc = studentScores(snap, cands, cfg);

            // (a) reuse: recompute prod feat/vprior INDEPENDENTLY and compare.
            const st = { px: snap.px, py: snap.py, pvx: snap.pvx, pvy: snap.pvy, psize: snap.psize,
                bx: snap.bx, by: snap.by, bvx: snap.bvx, bvy: snap.bvy, nAlive: snap.nAlive };
            const fr = planner.cp_features(st, cands, planner.CP ? 2.5 : 2.5, 0.05);
            const vp = planner.cp_value(NET, fr.feat, fr.ctx);
            // which 4 indices are rolled (NN-replaced) by the ballistic pidx?
            const ps = fr.feat.map(row => row[18] - row[16]);
            const pidx = fr.feat.map((_, k) => k).sort((a, b) => (ps[b] - ps[a]) || (a - b));
            const rolled = new Set(pidx.slice(0, 4));
            // The 12 NON-rolled student scores must equal prod vprior bitwise.
            for (let k = 0; k < 16; k++) {
                if (rolled.has(k)) continue;
                if (fhex(sc[k]) !== fhex(vp[k])) nonrollMis++;
            }

            // (b) agreement: my dedup-argmax committed coord vs prod's logged commit.
            const d = dedupArgmax(sc, r.cx, r.cy);
            const ok = (fhex(d.wx) === fhex(r.wx) && fhex(d.wy) === fhex(r.wy));
            if (ok) agree++;
            const bk = bucketOf(d.margin);
            if (!byBucket[bk]) byBucket[bk] = { n: 0, dis: 0 };
            byBucket[bk].n++; if (!ok) byBucket[bk].dis++;
        }
        game.win.__xvalLog = null;
    }
    const out = { cell: cellArg, seeds: [seedStart, seedStart + nSeeds], plans,
        nonrolled_score_mismatches: nonrollMis,
        reuse_exact_12scores: nonrollMis === 0,
        agree, disagree: plans - agree,
        S_dec: +(agree / plans).toFixed(4),
        disagree_rate: +((plans - agree) / plans).toFixed(4),
        margin_reliability: {} };
    for (const [k, v] of Object.entries(byBucket))
        out.margin_reliability[k] = { n: v.n, disagreeRate: +(v.dis / v.n).toFixed(4) };
    console.log(JSON.stringify(out, null, 1));
}
main().catch(e => { console.error(e); process.exit(1); });
