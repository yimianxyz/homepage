// baselines.js — reference S_dec baselines that frame the L1 learning problem.
// All computed on oracle decision records (no model); they bracket what any
// student can/should achieve and reveal WHY NN-alone D1 is hard.
//
//   prior-alone      : argmax(vprior) — prod's value net WITHOUT the rollout.
//                      The floor any student must beat.
//   rollout-changes  : fraction of plans where the rollout moves the committed
//                      coordinate off the prior's argmax (= 1 − prior-alone on
//                      the coordinate). High ⇒ the decision is rollout-dominated.
//   catch-oracle     : among the 4 rolled candidates, pick max TRUE catches
//                      (tiebreak true boot). Upper bound for a student that
//                      predicts the catch-count perfectly and uses catches→boot
//                      ordering — but only over rolled winners.
//   full-oracle      : exact vprior on the 12 non-rolled + true catches+boot on
//                      the 4 rolled → argmax. ≈ prod (sanity: ~100%).
//
//   node dev/exact_nn/baselines.js [--data dir]
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const dataDir = (() => { const i = process.argv.indexOf('--data'); return i > 0 ? process.argv[i + 1] : path.join(__dirname, 'data'); })();
const _dv = new DataView(new ArrayBuffer(16));
function coordKey(c) { _dv.setFloat64(0, c[0]); _dv.setFloat64(8, c[1]); return _dv.getBigUint64(0) + '_' + _dv.getBigUint64(8); }

let n = 0, prior = 0, catchO = 0, fullO = 0;
const byCell = {};
for (const f of fs.readdirSync(dataDir).sort()) {
    if (!f.endsWith('.decisions.jsonl.gz')) continue;
    for (const line of zlib.gunzipSync(fs.readFileSync(path.join(dataDir, f))).toString().split('\n')) {
        if (!line) continue;
        const r = JSON.parse(line); n++;
        const cell = r.cell; byCell[cell] = byCell[cell] || { n: 0, prior: 0, catchO: 0 };
        byCell[cell].n++;
        const win = coordKey(r.cands[r.bi]);
        // prior-alone
        let pa = 0, pb = -Infinity;
        for (let k = 0; k < r.vprior.length; k++) if (r.vprior[k] > pb) { pb = r.vprior[k]; pa = k; }
        const priorHit = coordKey(r.cands[pa]) === win;
        if (priorHit) { prior++; byCell[cell].prior++; }
        // catch-oracle (max true catches among rolled, tiebreak boot, then lowest index)
        let bc = -1, bb = -Infinity, bci = -1;
        for (const [ci, cat, boot] of r.rolled) {
            const b = boot === null ? -Infinity : boot;
            if (cat > bc || (cat === bc && b > bb)) { bc = cat; bb = b; bci = ci; }
        }
        if (bci >= 0 && coordKey(r.cands[bci]) === win) { catchO++; byCell[cell].catchO++; }
        // full-oracle: exact vprior on non-rolled + (catches+boot) on rolled → argmax
        const rolledIdx = new Map(r.rolled.map(([ci, cat, boot]) => [ci, cat + (boot === null ? -Infinity : boot)]));
        const sc = r.vprior.map((v, k) => rolledIdx.has(k) ? rolledIdx.get(k) : v);
        let fa = 0, fb = -Infinity;
        for (let k = 0; k < sc.length; k++) if (sc[k] > fb) { fb = sc[k]; fa = k; }
        if (coordKey(r.cands[fa]) === win) fullO++;
    }
}
const pct = (x, d = n) => (x / d * 100).toFixed(2) + '%';
console.log(`plans: ${n}`);
console.log(`prior-alone (argmax vprior, NO rollout):    ${pct(prior)}   ← floor a student must beat`);
console.log(`rollout changes committed coordinate:       ${pct(n - prior)}   ← decision is rollout-dominated`);
console.log(`catch-oracle (max true catches among rolled): ${pct(catchO)}   ← ceiling for perfect catch prediction (rolled winners)`);
console.log(`full-oracle (exact vprior + true rolled):    ${pct(fullO)}   ← sanity ≈ prod`);
console.log('\nper cell  (prior-alone / catch-oracle):');
for (const c of Object.keys(byCell).sort()) {
    const b = byCell[c];
    console.log(`  ${c.padEnd(16)} ${pct(b.prior, b.n)} / ${pct(b.catchO, b.n)}  (n=${b.n})`);
}
