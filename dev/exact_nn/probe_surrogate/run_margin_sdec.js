'use strict';
// The DECISION turns on the boot DIFFERENCE between equal-catch rolled candidates
// within a plan, not the absolute boot. Common-mode boot error may cancel. This
// measures, per intermediate surrogate:
//   (a) S_dec: surrogate-as-policy (vprior with 4 rolled overridden by
//       catches_cheap+boot_cheap), committed coord vs oracle label.
//   (b) decisive-margin agreement: does the surrogate pick the SAME argmax among
//       the rolled candidates as the true rollout? (the real decision)
//   (c) regime split: S_dec and boot-err by flock density / target distance.
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const IP = require('./intermediate_probe.js');
const DATA_DIR = path.join(__dirname, '..', 'data_1e6');

const _ab = new ArrayBuffer(8), _dv = new DataView(_ab);
function f64hex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); }
function* records(file) { const txt = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8'); for (const line of txt.split('\n')) { if (!line) continue; yield JSON.parse(line); } }
function commit(cands, score) { let bi = 0, bs = -Infinity; for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; } return { bi, hx: f64hex(cands[bi][0]), hy: f64hex(cands[bi][1]) }; }

(async () => {
    await IP.init();
    const F = (c, s, a, p) => ({ incCoh: c, incSep: s, incAlign: a, incPredAvoid: p });
    const MODES = {
        short30: { kind: 'short', K: 30, fl: F(1, 1, 1, 1) },
        short45: { kind: 'short', K: 45, fl: F(1, 1, 1, 1) },
        short60: { kind: 'short', K: 60, fl: F(1, 1, 1, 1) },
        nopredavoid: { kind: 'flags', fl: F(1, 1, 1, 0) },
        full: { kind: 'full' },
    };
    const R = IP.makeRollout({});
    const N_REC = +(process.argv[2] || 1500);
    const STRIDE = +(process.argv[3] || 31);
    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (STRIDE > 1) files = files.filter((_, i) => i % STRIDE === 0);

    const acc = {}; for (const m of Object.keys(MODES)) acc[m] = {
        decMatch: 0, decTot: 0,
        // among boot-decided plans (all rolled catch equal AND winner is a rolled cand): did surrogate pick same rolled argmax?
        rollArgMatch: 0, rollArgTot: 0,
        // regime buckets: by density proxy (#boids within NEIGHBOR_DISTANCE of predator) and target dist
        sparseDec: 0, sparseTot: 0, denseDec: 0, denseTot: 0,
        sparseBootErr: [], denseBootErr: [],
    };
    const ND = 60;

    let nrec = 0;
    outer:
    for (const file of files) {
        for (const r of records(path.join(DATA_DIR, file))) {
            const s = r.s, cands = r.cands;
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            // density proxy: # alive boids within 2*ND of predator (local crowding)
            let near = 0; for (let i = 0; i < s.bx.length; i++) { const dx = s.bx[i] - s.px, dy = s.by[i] - s.py; if (dx * dx + dy * dy < (2 * ND) ** 2) near++; }
            const sparse = near <= 3;
            // true rolled
            const trueCatch = r.rolled.map(x => x[1]); const trueBoot = r.rolled.map(x => x[2] === null ? -Infinity : x[2]);
            const allEqCatch = trueCatch.every(c => c === trueCatch[0]);

            for (const m of Object.keys(MODES)) {
                const A = acc[m];
                const score = r.vprior.slice();
                const cheapBoot = [], cheapCatch = [];
                for (let rk = 0; rk < r.rolled.length; rk++) {
                    const ci = r.rolled[rk][0]; const cand = cands[ci];
                    const out = R.rollScore(s, cand[0], cand[1], MODES[m]);
                    cheapBoot.push(out.boot); cheapCatch.push(out.catches);
                    score[ci] = out.catches + out.boot;
                    if (isFinite(trueBoot[rk]) && isFinite(out.boot)) {
                        const e = Math.abs(out.boot - trueBoot[rk]);
                        if (sparse) A.sparseBootErr.push(e); else A.denseBootErr.push(e);
                    }
                }
                const cc = commit(cands, score);
                A.decTot++; if (cc.hx === r.lab.tx && cc.hy === r.lab.ty) A.decMatch++;
                if (sparse) { A.sparseTot++; if (cc.hx === r.lab.tx && cc.hy === r.lab.ty) A.sparseDec++; }
                else { A.denseTot++; if (cc.hx === r.lab.tx && cc.hy === r.lab.ty) A.denseDec++; }
                // boot-decided: all rolled equal catch -> winner is the max-boot rolled
                if (allEqCatch) {
                    let tArg = 0, tB = -Infinity; for (let k = 0; k < trueBoot.length; k++) if (trueBoot[k] > tB) { tB = trueBoot[k]; tArg = k; }
                    let cArg = 0, cB = -Infinity; for (let k = 0; k < cheapBoot.length; k++) if (cheapBoot[k] > cB) { cB = cheapBoot[k]; cArg = k; }
                    A.rollArgTot++; if (tArg === cArg) A.rollArgMatch++;
                }
            }
            if (++nrec >= N_REC) break outer;
        }
    }
    function med(a) { if (!a.length) return null; const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; }
    const rep = { nrec, modes: {} };
    for (const m of Object.keys(MODES)) {
        const A = acc[m];
        rep.modes[m] = {
            S_dec: A.decMatch / A.decTot,
            rolledArgmaxAgree_bootDecided: A.rollArgTot ? A.rollArgMatch / A.rollArgTot : null,
            rollArgTot: A.rollArgTot,
            S_dec_sparse: A.sparseTot ? A.sparseDec / A.sparseTot : null, sparseTot: A.sparseTot,
            S_dec_dense: A.denseTot ? A.denseDec / A.denseTot : null, denseTot: A.denseTot,
            bootErrMedian_sparse: med(A.sparseBootErr), bootErrMedian_dense: med(A.denseBootErr),
        };
    }
    console.log(JSON.stringify(rep, null, 2));
    fs.writeFileSync(path.join(__dirname, 'margin_sdec_result.json'), JSON.stringify(rep, null, 2));
})();
