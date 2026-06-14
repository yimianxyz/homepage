// moe_livetest.js — LIVE end-to-end self-test of the Phase-2 MoE deploy. Drives
// real held-out games with the certified oracle fork (the exact code that made the
// training data), and at every prod decision feeds the LIVE-computed structure to
// moePolicy and compares the committed target / egBoid to prod. This proves the
// full chain (live structure -> moe_features -> moeForward -> argmax) reproduces
// prod with NO fallback, and that packed-block S_dec == live-game S_dec.
//
//   node moe_livetest.js <moe_weights.json> [--games 20] [--maxFrames 8000]
'use strict';
const path = require('path');
const { createGame } = require(path.join(__dirname, '..', 'stepper.js'));
const { loadFork } = require(path.join(__dirname, '..', 'oracle_candidate.js'));
const { CELLS } = require(path.join(__dirname, '..', 'device_matrix.js'));
const { loadMoePolicy } = require(path.join(__dirname, 'moePolicy.js'));

const _dv = new DataView(new ArrayBuffer(8));
function f64hex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); }
function isVal(seed) { return (((seed >>> 0) * 2654435761) >>> 0) % 10 === 0; }
function arg(n, d) { const i = process.argv.indexOf('--' + n); return i >= 0 ? process.argv[i + 1] : d; }

const P = loadMoePolicy(process.argv[2] || 'moe_weights.json');
const NGAMES = parseInt(arg('games', '24'), 10);
const MAXF = parseInt(arg('maxFrames', '8000'), 10);

async function runGame(cell, seed) {
    const stat = { plan_ok: 0, plan_n: 0, eg_ok: 0, eg_n: 0 };
    const g = await createGame({ policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        W: cell.W, H: cell.H, seed, startBoids: cell.startBoids, frameMs: cell.frameMs, fastRender: true });
    const forkCheap = await loadFork(g);
    const numBoids = g.api.getNumBoids();
    let pend = null;
    let prevEg = null;
    g.win.__oracle = {
        planStart(s, cands, fr, vprior, cfg) {
            pend = { s: { px: s.px, py: s.py, psize: s.psize, bx: s.bx.slice() },
                cands: cands.map(c => [c.x, c.y]), feat: fr.feat.map(r => r.slice()),
                vprior: vprior.slice(), cfg: { W: cfg.W, Hc: cfg.Hc }, N: s.bx.length };
            pend.rolled = [];
        },
        roll(ci, catches, boot) { pend.rolled.push([ci, catches, boot]); },
        planEnd(pidx, score, bi) {
            // prod committed coords (canonical lowest-index of bi's coord group)
            const keys = pend.cands.map(c => f64hex(c[0]) + f64hex(c[1]));
            const win = keys[bi]; let ti = bi; for (let k = 0; k < 16; k++) if (keys[k] === win) { ti = k; break; }
            const prodTx = pend.cands[ti][0], prodTy = pend.cands[ti][1];
            // moePolicy decision from the SAME live structure
            const rec = { s: { px: pend.s.px, py: pend.s.py, psize: pend.s.psize },
                nAlive: pend.N, cands: pend.cands, feat: pend.feat, vprior: pend.vprior,
                pidx: pidx.slice(), rolled: pend.rolled };
            const d = P.chooseTarget(rec, pend.cfg);
            stat.plan_n++;
            if (f64hex(d.tx) === f64hex(prodTx) && f64hex(d.ty) === f64hex(prodTy)) stat.plan_ok++;
            pend = null;
        },
        frameEnd(target, pframe, egBoid, boids, r, pred) {
            // endgame commit = a NEW egBoid selection (sticky otherwise)
            if (egBoid && egBoid !== prevEg && boids.length >= 1 && boids.length <= 5) {
                const snap = { px: pred.position.x, py: pred.position.y, psize: pred.currentSize,
                    bx: boids.map(b => b.position.x), by: boids.map(b => b.position.y),
                    bvx: boids.map(b => b.velocity.x), bvy: boids.map(b => b.velocity.y) };
                const d = P.chooseEgBoid(snap, { W: cell.W, Hc: cell.H });
                stat.eg_n++;
                if (d.egIdx === boids.indexOf(egBoid)) stat.eg_ok++;
            }
            prevEg = egBoid;
        },
    };
    g.setForce(forkCheap.force);
    for (let f = 0; f < MAXF; f++) { g.stepFrame(); if (g.boidCount() === 0) break; }
    return stat;
}

(async () => {
    const agg = { plan_ok: 0, plan_n: 0, eg_ok: 0, eg_n: 0 };
    let done = 0;
    for (const cell of CELLS) {
        for (let s = 0; s < Math.ceil(NGAMES / CELLS.length); s++) {
            // pick a val-marked seed in the train range, disjoint per cell
            let seed = 100000 + (CELLS.indexOf(cell) * 1000) + s * 7;
            while (!isVal(seed)) seed++;
            const st = await runGame(cell, seed);
            for (const k in agg) agg[k] += st[k];
            done++;
            console.error(`[${cell.id} s${seed}] plan ${st.plan_ok}/${st.plan_n} eg ${st.eg_ok}/${st.eg_n}`);
        }
    }
    const ps = agg.plan_ok / Math.max(agg.plan_n, 1), es = agg.eg_ok / Math.max(agg.eg_n, 1);
    const pooled = (agg.plan_ok + agg.eg_ok) / Math.max(agg.plan_n + agg.eg_n, 1);
    console.log('=== LIVE end-to-end S_dec (val seeds, NO fallback) ===');
    console.log(`PLANNER ${ps.toFixed(4)} (n=${agg.plan_n})   ENDGAME ${es.toFixed(4)} (n=${agg.eg_n})   POOLED ${pooled.toFixed(4)} (n=${agg.plan_n + agg.eg_n})`);
})();
