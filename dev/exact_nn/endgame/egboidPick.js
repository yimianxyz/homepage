// egboidPick.js — the DETERMINISTIC JS L1e endgame egBoid picker (task #18, the
// D4 twin of studentScores.js). Predicts each present boid's scan-t with the
// trained per-boid MLP (float64), returns the argmin egBoid + the deduped scan-t
// margin. side-b composes L1e = commit egboidPick.egIdx iff margin >= τ, else
// prod's EXACT intercept() scan fallback → bitwise-exact by construction.
//
// Contract:
//   const { loadEgStudent } = require('./egboidPick.js');
//   const pick = loadEgStudent('eg_weights.json');
//   const r = pick(snapshot, cfg);   // {egIdx, margin, ts:[scan-t per boid]}
//     snapshot: { px, py, bx[], by[], bvx[], bvy[] }   (the <=5 live boids, in order)
//     cfg:      { W, Hc }
// Features via eg_features.js (SAME function used to pack training data → exact
// parity). MLP [NFEAT→h→h→1], GELU (prod's A-S cp_erf), float64. No RNG/wall-clock.
'use strict';
const fs = require('fs');
const path = require('path');
const { egBoidFeatures } = require('./eg_features.js');

// GELU with prod's Abramowitz-Stegun erf (bit-identical to cp_gelu / studentScores).
function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
}
function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }
function lin(W, b, x) {
    const out = new Array(W.length);
    for (let o = 0; o < W.length; o++) { let a = b[o]; const wr = W[o]; for (let i = 0; i < x.length; i++) a += wr[i] * x[i]; out[o] = a; }
    return out;
}
function geluVec(v) { const o = new Array(v.length); for (let i = 0; i < v.length; i++) o[i] = gelu(v[i]); return o; }

function loadEgStudent(weightsPath) {
    const J = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
    const W = J.weights;
    // state_dict keys: net.0.{weight,bias} (Linear), net.2.*, net.4.* (Sequential 0,2,4)
    const W0 = W['net.0.weight'], b0 = W['net.0.bias'];
    const W2 = W['net.2.weight'], b2 = W['net.2.bias'];
    const W4 = W['net.4.weight'], b4 = W['net.4.bias'];

    function scanTPred(featRow) {                 // -> predicted scan-t in FRAMES
        let h = geluVec(lin(W0, b0, featRow));
        h = geluVec(lin(W2, b2, h));
        const o = lin(W4, b4, h);                 // (1)
        return o[0] * 100.0;                      // label was scan-t/100
    }

    function egboidPick(snapshot, cfg) {
        const n = snapshot.bx.length;
        const ts = new Array(n);
        for (let i = 0; i < n; i++) {
            const fr = egBoidFeatures(snapshot.px, snapshot.py, snapshot.bx[i], snapshot.by[i],
                snapshot.bvx[i], snapshot.bvy[i], cfg.W, cfg.Hc);
            ts[i] = scanTPred(fr);
        }
        // argmin predicted scan-t (lowest-index tiebreak)
        let egIdx = 0, best = Infinity, second = Infinity;
        for (let i = 0; i < n; i++) if (ts[i] < best) { second = best; best = ts[i]; egIdx = i; }
            else if (ts[i] < second) second = ts[i];
        const margin = (n >= 2 && Number.isFinite(second)) ? (second - best) : Infinity;
        return { egIdx, margin, ts };
    }
    egboidPick.scanTPred = scanTPred;
    return egboidPick;
}

module.exports = { loadEgStudent };
