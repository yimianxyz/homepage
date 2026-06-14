// moePolicy.js — the Phase-2 single-MoE DECISION export (SPEC_PHASE2_MOE §3).
// Pure NN, NO fallback: the MoE's argmax IS the committed decision in every boid
// case. The deterministic structure (candidates/rollout/scan) is an allowed
// feature stage; the decision→force map (prod steer/intercept) is downstream and
// owned by the harness (§3). This module exposes the DECISION:
//
//   const { loadMoePolicy } = require('./moePolicy.js');
//   const P = loadMoePolicy('moe_weights.json');
//   // N>5 planner — planRecord = prod's logged plan inputs (oracle planStart/roll):
//   P.chooseTarget({s,cands,feat,vprior,pidx,rolled}, cfg) -> {tx,ty,slot}
//   // 1<=N<=5 endgame — snapshot = the <=5 live boids:
//   P.chooseEgBoid({px,py,bx,by,bvx,bvy}, cfg)             -> {egIdx, margin}
//   // dispatch by N (planRecord required only for N>5):
//   P.decide(snapshot, cfg, planRecord?)                  -> {regime, ...} | null(N==0)
//
// The committed target is the COORDINATE-DEDUPED argmax (SPEC §3): group slots by
// bitwise-equal (x,y), max logit per group, lowest-index canonical — matches prod's
// committed-coordinate label and side-b's coord-dedup S_dec. float64 throughout.
'use strict';
const path = require('path');
const { loadMoE } = require(path.join(__dirname, 'moeForward.js'));
const F = require(path.join(__dirname, 'moe_features.js'));
const { scanT } = require(path.join(__dirname, '..', 'endgame', 'eg_scan.js'));

const NSLOT = F.NSLOT;
const ZERO_P = Array.from({ length: NSLOT }, () => new Array(F.PLANNER_DIM).fill(0));
const ZERO_E = Array.from({ length: NSLOT }, () => new Array(F.ENDGAME_DIM).fill(0));
const ALL_T = new Array(NSLOT).fill(true);
const ALL_F = new Array(NSLOT).fill(false);

// bitwise coord key (exact, per SPEC §3)
function coordKey(x, y) { const b = Buffer.allocUnsafe(16); b.writeDoubleLE(x, 0); b.writeDoubleLE(y, 8); return b.toString('latin1'); }

function loadMoePolicy(weightsPath) {
    const moe = loadMoE(weightsPath);

    // PLANNER (N>5): planRecord carries prod's exact structure outputs.
    //   s: {px,py,...,psize}  cands:[16][x,y]  feat:[16][19]  vprior:[16]
    //   pidx:[16]  rolled:[4][ci,catches,boot]
    function chooseTarget(rec, cfg) {
        const s = rec.s, n = (rec.nAlive != null ? rec.nAlive : (s.bx ? s.bx.length : rec.N));
        const pb = F.plannerSlots(rec.cands, rec.feat, rec.vprior, s.px, s.py, n, rec.pidx, rec.rolled);
        const gf = F.gateFeat(n, n / 120.0, s.psize);
        const { logit } = moe.forward(pb, ZERO_E, gf, ALL_T, ALL_T, ALL_F);
        // coordinate-deduped argmax: best logit per coord group, lowest-index canonical
        const best = {};   // key -> {logit, slot}
        for (let k = 0; k < NSLOT; k++) {
            const key = coordKey(rec.cands[k][0], rec.cands[k][1]);
            if (!(key in best) || logit[k] > best[key].logit) best[key] = { logit: logit[k], slot: k };
        }
        let win = null;
        for (const key in best) if (!win || best[key].logit > win.logit || (best[key].logit === win.logit && best[key].slot < win.slot)) win = best[key];
        return { tx: rec.cands[win.slot][0], ty: rec.cands[win.slot][1], slot: win.slot, logit };
    }

    // ENDGAME (1<=N<=5): compute the ACTUAL scan-t per boid (prod intercept geometry,
    // eg_scan == bit-identical to prod), feed it, argmin via the NN.
    function chooseEgBoid(snap, cfg) {
        const n = snap.bx.length;
        const W = cfg.W, Hc = cfg.Hc;
        const boids = new Array(n), ts = new Array(n);
        for (let i = 0; i < n; i++) {
            boids[i] = { x: snap.bx[i], y: snap.by[i], vx: snap.bvx[i], vy: snap.bvy[i] };
            ts[i] = scanT(snap.px, snap.py, snap.bx[i], snap.by[i], snap.bvx[i], snap.bvy[i], W, Hc);
        }
        const es = F.endgameSlots(snap.px, snap.py, boids, ts, W, Hc);   // [n][20]
        const eb = new Array(NSLOT); const sv = new Array(NSLOT);
        for (let k = 0; k < NSLOT; k++) { eb[k] = k < n ? es[k] : ZERO_E[k]; sv[k] = k < n; }
        const gf = F.gateFeat(n, n / 120.0, snap.psize != null ? snap.psize : 0);
        const { logit, g } = moe.forward(ZERO_P, eb, gf, sv, ALL_F, sv);
        let egIdx = 0, b1 = -Infinity, b2 = -Infinity;
        for (let k = 0; k < n; k++) { if (logit[k] > b1) { b2 = b1; b1 = logit[k]; egIdx = k; } else if (logit[k] > b2) b2 = logit[k]; }
        return { egIdx, margin: (n >= 2 && isFinite(b2)) ? (b1 - b2) : Infinity, logit: logit.slice(0, n), g };
    }

    function decide(snap, cfg, planRecord) {
        const n = snap.bx ? snap.bx.length : snap.nAlive;
        if (n === 0) return null;                       // N==0: no decision (Vector 0,0)
        if (n <= 5) return Object.assign({ regime: 'endgame' }, chooseEgBoid(snap, cfg));
        if (!planRecord) throw new Error('decide: planRecord (prod structure) required for N>5');
        return Object.assign({ regime: 'planner' }, chooseTarget(planRecord, cfg));
    }

    return { chooseTarget, chooseEgBoid, decide, moe };
}

module.exports = { loadMoePolicy };
