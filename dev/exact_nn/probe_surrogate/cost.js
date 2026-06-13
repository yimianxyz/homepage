// cost.js — clean wall-time cost proxy. Times each rollout mode in ISOLATION
// (one mode per pass, after a warmup pass) so JIT/interleaving don't contaminate.
// Reports us/rollout and us/plan (4 rolled candidates incl. boot recompute).
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const RV = require('./rollout_variants.js');

const DATA_DIR = path.join(__dirname, '..', 'data_1e6');

function loadRecs(n, stride) {
    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (stride > 1) files = files.filter((_, i) => i % stride === 0);
    const recs = [];
    for (const f of files) {
        const txt = zlib.gunzipSync(fs.readFileSync(path.join(DATA_DIR, f))).toString('utf8');
        for (const line of txt.split('\n')) {
            if (!line) continue;
            const r = JSON.parse(line);
            recs.push({ s: r.s, cands: r.cands, rolled: r.rolled, cfg: r.cfg });
            if (recs.length >= n) return recs;
        }
    }
    return recs;
}

(async () => {
    const N = +(process.argv[2] || 500);
    await RV.init();
    const R = RV.makeRollout();
    const recs = loadRecs(N, 7);
    const MODES = ['const', 'avoid', 'full'];

    function pass(mode) {
        let nRoll = 0;
        const t0 = process.hrtime.bigint();
        for (const r of recs) {
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            for (let rk = 0; rk < r.rolled.length; rk++) {
                const ci = r.rolled[rk][0], c = r.cands[ci];
                const out = R.rollScore(r.s, c[0], c[1], mode);
                if (out.catches < -1) console.log('x');  // prevent DCE
                nRoll++;
            }
        }
        const t1 = process.hrtime.bigint();
        const us = Number(t1 - t0) / 1000;
        return { usPerRollout: us / nRoll, usPerPlan: us / recs.length, nRoll };
    }

    // warmup each mode once, then measure
    for (const m of MODES) pass(m);
    const out = { nRecords: recs.length };
    for (const m of MODES) out[m] = pass(m);
    out.speedup_const_vs_full = out.full.usPerRollout / out.const.usPerRollout;
    out.speedup_avoid_vs_full = out.full.usPerRollout / out.avoid.usPerRollout;
    console.log(JSON.stringify(out, null, 2));
    fs.writeFileSync(path.join(__dirname, 'cost_result.json'), JSON.stringify(out, null, 2));
})();
