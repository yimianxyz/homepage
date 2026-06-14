// moe_features.js — the SINGLE shared featurizer for the Phase-2 MoE policy.
// Used by BOTH the training packer (moe_pack.js, reading the Phase-1 oracle logs)
// AND the deploy (moePolicy.js, computing features live). One function → bit-
// identical features on both sides (the Phase-1 D1/L1e parity lesson). Pure
// float64, deterministic; no RNG, no wall-clock.
//
// Two per-slot feature blocks, fed to the two experts of the MoE:
//   PLANNER block  (28-dim, slot = candidate k, valid for all 16 when N>5)
//   ENDGAME block  (20-dim, slot = boid i, valid for i<N when 1<=N<=5)
// Phase-2 key: the planner block carries the ACTUAL rollout outputs (catch+boot
// for the rolled candidates) and the endgame block carries the ACTUAL scan-t —
// the deterministic "structure" feeds the NN, so each expert reduces to learning
// argmax / argmin of visible scores (the old ~37% rollout-bound ceiling is gone).
'use strict';
const path = require('path');
const { egBoidFeatures, EG_NFEAT, TMAX } = require(path.join(__dirname, '..', 'endgame', 'eg_features.js'));

const CP_PS = 200.0, CP_VS = 6.0;   // cheap_planner CP.PS / CP.VS (predator-relative norms)

const PLANNER_DIM = 29;             // 2 + 19 + 1 + 3 + 1 + 1 + 1 + cheapScore
const ENDGAME_DIM = EG_NFEAT + 2;   // 18 + scan_t + reachable = 20
const NSLOT = 16;

// --- PLANNER per-slot block ---------------------------------------------------
// Mirrors studentScores.featurize candTok (25) + 3 Phase-2 fields (mask,catch,boot).
//   cands : [16][x,y]   feat: [16][19] (cp_features, already normalized)
//   vprior: [16]        px,py: predator pos   nAlive: live boid count
//   pidx  : [16] roll order (rolled = pidx[0:4])
//   rolled: [4][ci,catches,boot]  (aligned to pidx[0:4]; boot may be -Inf)
// Returns [16][28].
function plannerSlots(cands, feat, vprior, px, py, nAlive, pidx, rolled) {
    // map rolled slot -> {catch,boot}
    const rmap = {};
    for (let r = 0; r < rolled.length; r++) {
        const ci = rolled[r][0];
        let boot = rolled[r][2];
        if (boot === null || !isFinite(boot)) boot = -1e9;   // extermination sentinel (≈never in 1e6)
        rmap[ci] = { c: rolled[r][1], b: boot };
    }
    const out = new Array(NSLOT);
    for (let k = 0; k < NSLOT; k++) {
        const kind = k === 0 ? 0 : (k <= nAlive ? 1 : 2);   // e3d / boid / E3D-pad
        const rolledMask = rmap.hasOwnProperty(k) ? 1.0 : 0.0;
        const cval = rolledMask ? rmap[k].c : 0.0;
        const bval = rolledMask ? rmap[k].b : 0.0;
        const row = new Array(PLANNER_DIM);
        row[0] = (cands[k][0] - px) / CP_PS;
        row[1] = (cands[k][1] - py) / CP_PS;
        for (let i = 0; i < 19; i++) row[2 + i] = feat[k][i];
        row[21] = vprior[k] / 3.0;
        row[22] = kind === 0 ? 1.0 : 0.0;
        row[23] = kind === 1 ? 1.0 : 0.0;
        row[24] = kind === 2 ? 1.0 : 0.0;
        row[25] = rolledMask;
        row[26] = cval;            // raw catch count (0..~23)
        row[27] = bval;            // raw terminal boot value
        // cheapScore[k] = prod's EXACT committed-decision score = (rolled? catch+boot : vprior).
        // This is the decisive signal (vprior + rolled outputs, both allowed structure);
        // feeding it lets the shared head reconstruct prod's argmax to high precision.
        row[28] = rolledMask ? (cval + bval) : vprior[k];
        out[k] = row;
    }
    return out;
}

// --- ENDGAME per-slot block ---------------------------------------------------
//   px,py: predator pos   boids: [{x,y,vx,vy}] (the <=5 live boids, in order)
//   scanTs: [N] actual prod scan-t per boid (integer frames, or null=unreachable)
//   W,Hc: torus dims
// Returns [N][20].
function endgameSlots(px, py, boids, scanTs, W, Hc) {
    const n = boids.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        const b = boids[i];
        const fr = egBoidFeatures(px, py, b.x, b.y, b.vx, b.vy, W, Hc);   // 18
        const t = scanTs[i];
        const reachable = (t === null || t === undefined) ? 0.0 : 1.0;
        const tClip = reachable ? Math.min(t, TMAX) : TMAX;
        const row = fr.slice();
        row.push(tClip / 100.0);     // [18] ACTUAL scan-t (the Phase-2 enabler)
        row.push(1.0 - reachable);   // [19] unreachable flag
        out[i] = row;
    }
    return out;
}

// --- GATE situation summary ---------------------------------------------------
// N dominant; the learned gate places a sharp step at the N=5/6 regime boundary.
function gateFeat(N, fracAlive, psize) {
    return [N / 8.0, N / 120.0, fracAlive, (psize || 0) / 20.0];
}
const GATE_DIM = 4;

module.exports = {
    plannerSlots, endgameSlots, gateFeat,
    PLANNER_DIM, ENDGAME_DIM, GATE_DIM, NSLOT, CP_PS, CP_VS,
};
