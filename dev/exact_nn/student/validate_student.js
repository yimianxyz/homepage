// validate_student.js — proves the JS studentScores is a faithful deploy artifact.
//   (1) cp_features/cp_value/pidx reuse: recomputed from the snapshot == the
//       oracle-logged feat/vprior/pidx BITWISE (the 12 non-rolled scores are
//       prod's exact value-net prior).
//   (2) end-to-end: student deduped-argmax agreement (S_dec) on logged plans —
//       compare to the torch checkpoint's val agree_dedup to confirm the JS
//       float64 forward reproduces the trained student.
//
//   node validate_student.js --weights student_weights.json --data <dir> [--glob desk_2560] [--max N]
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { loadStudent } = require('./studentScores.js');

function parseArgs() {
    const a = { weights: path.join(__dirname, 'student_weights.json'),
        data: path.join(__dirname, '..', 'data'), glob: null, max: Infinity };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--weights') a.weights = process.argv[++i];
        else if (k === '--data') a.data = process.argv[++i];
        else if (k === '--glob') a.glob = process.argv[++i];
        else if (k === '--max') a.max = +process.argv[++i];
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

const _dv = new DataView(new ArrayBuffer(8));
function hex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); }
function coordKey(x, y) { return hex(x) + '_' + hex(y); }

// deduped argmax committed coord + top-2 margin over coordinate groups
function dedupDecision(cands, score) {
    const groups = new Map();
    for (let k = 0; k < cands.length; k++) {
        const key = coordKey(cands[k][0], cands[k][1]);
        const g = groups.get(key);
        if (!g) groups.set(key, { max: score[k], minIdx: k });
        else { if (score[k] > g.max) g.max = score[k]; if (k < g.minIdx) g.minIdx = k; }
    }
    let bi = 0, bs = -Infinity;
    for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; }
    const winKey = coordKey(cands[bi][0], cands[bi][1]);
    let ru = -Infinity;
    for (const [key, g] of groups) if (key !== winKey && g.max > ru) ru = g.max;
    return { winKey, margin: ru === -Infinity ? Infinity : groups.get(winKey).max - ru };
}

function* iter(dir, globSub) {
    for (const f of fs.readdirSync(dir).sort()) {
        if (!f.endsWith('.decisions.jsonl.gz')) continue;
        if (globSub && !f.includes(globSub)) continue;
        for (const line of zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString().split('\n'))
            if (line) yield JSON.parse(line);
    }
}

function main() {
    const opt = parseArgs();
    const scorer = loadStudent(opt.weights);
    let n = 0, featMis = 0, vpriorMis = 0, pidxMis = 0, agree = 0;
    const marginRec = [];
    for (const r of iter(opt.data, opt.glob)) {
        if (n >= opt.max) break;
        n++;
        const snap = { px: r.s.px, py: r.s.py, pvx: r.s.pvx, pvy: r.s.pvy, psize: r.s.psize,
            bx: r.s.bx, by: r.s.by, bvx: r.s.bvx, bvy: r.s.bvy, nAlive: r.s.bx.length };
        const cands = r.cands.map(c => ({ x: c[0], y: c[1] }));
        const score = scorer(snap, cands, r.cfg);
        // (1) reuse check — recompute prod pieces the same way studentScores does
        const planner = require('../../../js/cheap_planner.js');
        const NET = require('../../../js/value_net.json');
        const st = { px: snap.px, py: snap.py, pvx: snap.pvx, pvy: snap.pvy, psize: snap.psize,
            bx: snap.bx, by: snap.by, bvx: snap.bvx, bvy: snap.bvy, nAlive: snap.nAlive };
        const fr = planner.cp_features(st, cands, 2.5, 0.05);
        const vp = planner.cp_value(NET, fr.feat, fr.ctx);
        for (let k = 0; k < 16; k++) {
            // numeric equality: -0 === +0 (JSON dropped logged -0; causally dead
            // in cp_value and the NN's linear+GELU path — vprior bitwise match proves it)
            for (let i = 0; i < 19; i++) if (fr.feat[k][i] !== r.feat[k][i]) { featMis++; break; }
            if (hex(vp[k]) !== hex(r.vprior[k])) vpriorMis++;
        }
        const ps = fr.feat.map(row => row[18] - row[16]);
        const pidx = fr.feat.map((_, k) => k).sort((a, b) => (ps[b] - ps[a]) || (a - b));
        for (let i = 0; i < 4; i++) if (pidx[i] !== r.pidx[i]) { pidxMis++; break; }
        // (2) end-to-end agreement
        const dec = dedupDecision(r.cands, score);
        const prodKey = r.lab.tx + '_' + r.lab.ty;            // logged committed-coord hex
        const ok = dec.winKey === prodKey;
        if (ok) agree++;
        marginRec.push({ margin: dec.margin === Infinity ? null : dec.margin, agree: ok, n: r.N, cell: r.cell });
    }
    console.log(JSON.stringify({
        plans: n,
        feat_mismatches: featMis, vprior_mismatches: vpriorMis, pidx_mismatches: pidxMis,
        reuse_exact: featMis === 0 && vpriorMis === 0 && pidxMis === 0,
        student_S_dec: +(agree / n).toFixed(4),
    }, null, 1));
    if (process.env.WRITE_CALIB) fs.writeFileSync(process.env.WRITE_CALIB, JSON.stringify(marginRec));
}
main();
