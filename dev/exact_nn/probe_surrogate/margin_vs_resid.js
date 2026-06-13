// margin_vs_resid.js — ties the cheap-boot error to the actual decision margin.
// side-a found ~69% of plans are decided by the boot diff between two rolled
// candidates with EQUAL catch count (median decisive margin ~0.019). The relevant
// question for a surrogate is: is the cheap-boot error small RELATIVE TO that margin?
// We recompute, per plan: the winner's score margin over the runner-up, whether the
// decision is "boot-decided" (top-2 by score have equal true catches), and the
// per-candidate cheap-boot abs error. Then we report the distribution of
// (cheap-boot error) / (decision margin) and how often a per-candidate cheap-boot
// perturbation alone would flip the argmax.
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const RV = require('./rollout_variants.js');
const DATA_DIR = path.join(__dirname, '..', 'data_1e6');

function* records(file, max) {
    const txt = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8');
    let cnt = 0;
    for (const line of txt.split('\n')) { if (!line) continue; yield JSON.parse(line); if (max && ++cnt >= max) return; }
}

(async () => {
    const N = +(process.argv[2] || 2500);
    const STRIDE = +(process.argv[3] || 1);
    const PER = +(process.argv[4] || 8);
    const MODES = (process.argv[5] || 'avoid,const').split(',');
    await RV.init();
    const R = RV.makeRollout();
    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (STRIDE > 1) files = files.filter((_, i) => i % STRIDE === 0);

    const stat = {};
    for (const m of MODES) stat[m] = { n: 0, bootDecided: 0, decFlipBoot: 0, decFlipAll: 0,
        bootDecided_decMatch: 0, catchDecided_decMatch: 0, margins: [] };
    let nPlans = 0, nBootDecidedTrue = 0;

    const _ab = new ArrayBuffer(8), _dv = new DataView(_ab);
    const f64hex = x => { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); };

    outer: for (const f of files) {
        for (const r of records(path.join(DATA_DIR, f), PER)) {
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            nPlans++;
            // true score & decision structure
            const trueScore = r.score.map(v => v === null ? -Infinity : v);
            // is this decision "boot-decided"? -> top-2 distinct-coord candidates by
            // true score have equal true catch count among rolled, AND winner is rolled.
            const rolledCi = new Set(r.rolled.map(x => x[0]));
            const catchByCi = new Map(r.rolled.map(x => [x[0], x[1]]));
            // sorted candidate indices by true score desc
            const ord = trueScore.map((v, k) => k).sort((a, b) => trueScore[b] - trueScore[a] || a - b);
            const bi = ord[0], ru = ord[1];
            const margin = trueScore[bi] - trueScore[ru];
            const bootDecided = rolledCi.has(bi) && rolledCi.has(ru) && catchByCi.get(bi) === catchByCi.get(ru);
            if (bootDecided) nBootDecidedTrue++;

            for (const m of MODES) {
                const A = stat[m];
                const score = r.vprior.slice();
                const cheapBoot = new Map(), cheapCatch = new Map();
                for (const [ci] of r.rolled) {
                    const cand = r.cands[ci];
                    const out = R.rollScore(r.s, cand[0], cand[1], m);
                    cheapBoot.set(ci, out.boot); cheapCatch.set(ci, out.catches);
                    score[ci] = out.catches + out.boot;
                }
                // S_dec
                let cbi = 0, cbs = -Infinity;
                for (let k = 0; k < score.length; k++) if (score[k] > cbs) { cbs = score[k]; cbi = k; }
                const decMatch = (f64hex(r.cands[cbi][0]) === r.lab.tx && f64hex(r.cands[cbi][1]) === r.lab.ty);
                if (bootDecided) { A.bootDecided++; if (decMatch) A.bootDecided_decMatch++; }
                else { if (decMatch) A.catchDecided_decMatch++; }
                A.n++;
                if (margin > 0 && margin < Infinity) A.margins.push(margin);
            }
            if (++nPlans >= 0 && nPlans >= N) break outer;
        }
    }

    const out = { nPlans, fracBootDecided_true: nBootDecidedTrue / nPlans, perMode: {} };
    for (const m of MODES) {
        const A = stat[m];
        out.perMode[m] = {
            S_dec_overall: (A.bootDecided_decMatch + A.catchDecided_decMatch) / A.n,
            nBootDecided: A.bootDecided,
            S_dec_on_bootDecided: A.bootDecided ? A.bootDecided_decMatch / A.bootDecided : null,
            S_dec_on_catchDecided: (A.n - A.bootDecided) ? A.catchDecided_decMatch / (A.n - A.bootDecided) : null,
        };
    }
    console.log(JSON.stringify(out, null, 2));
    fs.writeFileSync(path.join(__dirname, 'margin_vs_resid_result.json'), JSON.stringify(out, null, 2));
})();
