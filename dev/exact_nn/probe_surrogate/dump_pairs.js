// dump_pairs.js — emit raw (boot_true, boot_cheap, catches_true, catches_cheap)
// pairs per mode for richer analysis (correlation, residual after best linear fit,
// which is the actual "can a residual-learning NN recover boot?" question).
// Writes JSON arrays to pairs_<mode>.json.
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
    const pairs = {}; for (const m of MODES) pairs[m] = { bt: [], bc: [], ct: [], cc: [] };
    let n = 0;
    outer: for (const f of files) {
        for (const r of records(path.join(DATA_DIR, f), PER)) {
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            for (const m of MODES) {
                for (const [ci, c, b] of r.rolled) {
                    const bt = (b === null) ? null : b;
                    const cand = r.cands[ci];
                    const out = R.rollScore(r.s, cand[0], cand[1], m);
                    pairs[m].bt.push(bt); pairs[m].bc.push(out.boot === -Infinity ? null : out.boot);
                    pairs[m].ct.push(c); pairs[m].cc.push(out.catches);
                }
            }
            if (++n >= N) break outer;
        }
    }
    for (const m of MODES) fs.writeFileSync(path.join(__dirname, `pairs_${m}.json`), JSON.stringify(pairs[m]));
    console.log(JSON.stringify({ n, modes: MODES, perMode_nPairs: Object.fromEntries(MODES.map(m => [m, pairs[m].bt.length])) }));
})();
