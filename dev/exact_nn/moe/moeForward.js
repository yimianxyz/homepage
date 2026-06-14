// moeForward.js — the deterministic JS forward of the Phase-2 MoE policy.
// Mirrors moe_model.py (MoEPolicy) EXACTLY: ASGELU (A-S erf == prod cp_erf),
// SlotExpert (proj→masked mean/max set-context→post→LayerNorm), Gate (sigmoid),
// shared Head. float64 throughout (canonical deploy precision; the JS artifact is
// the one S_dec is measured on). No RNG, no wall-clock.
//
//   const { loadMoE } = require('./moeForward.js');
//   const moe = loadMoE('moe_weights.json');
//   const logit = moe.forward(plannerBlock16x28, endgameBlock16x20, gateFeat4,
//                             slotValid16, pValid16, eValid16);   // number[16]
'use strict';
const fs = require('fs');

// A-S 7.1.26 erf (== cheap_planner cp_erf / moe_model as_erf).
function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
}
function gelu(x) { return 0.5 * x * (1 + erf(x * 0.7071067811865476)); }

// Linear: W [out][in], b [out], x [in] -> [out]
function lin(W, b, x) {
    const o = new Array(W.length);
    for (let r = 0; r < W.length; r++) { let a = b[r]; const wr = W[r]; for (let i = 0; i < x.length; i++) a += wr[i] * x[i]; o[r] = a; }
    return o;
}
function geluVec(v) { const o = new Array(v.length); for (let i = 0; i < v.length; i++) o[i] = gelu(v[i]); return o; }

// LayerNorm over the vector (biased var, eps=1e-5) * weight + bias.
function layernorm(x, w, b, eps) {
    const n = x.length; let mu = 0; for (let i = 0; i < n; i++) mu += x[i]; mu /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - mu; v += d * d; } v /= n;
    const inv = 1 / Math.sqrt(v + eps);
    const o = new Array(n);
    for (let i = 0; i < n; i++) o[i] = (x[i] - mu) * inv * w[i] + b[i];
    return o;
}

function loadMoE(weightsPath) {
    const J = typeof weightsPath === 'string' ? JSON.parse(fs.readFileSync(weightsPath, 'utf8')) : weightsPath;
    const W = J.weights;
    const D = J.d;
    const NSLOT = 16, LN_EPS = 1e-5;

    // SlotExpert forward over K slots. pref = 'E_p' or 'E_e'.
    function expert(pref, blocks, valid) {
        const K = blocks.length;
        // proj: Linear0 -> GELU -> Linear2 -> GELU  (per slot)
        const h = new Array(K);
        for (let k = 0; k < K; k++) {
            let a = geluVec(lin(W[pref + '.proj.0.weight'], W[pref + '.proj.0.bias'], blocks[k]));
            a = geluVec(lin(W[pref + '.proj.2.weight'], W[pref + '.proj.2.bias'], a));
            h[k] = a;                                    // (d)
        }
        // masked mean + max over valid slots
        const mean = new Array(D).fill(0); const mx = new Array(D).fill(-Infinity);
        let cnt = 0;
        for (let k = 0; k < K; k++) if (valid[k]) { cnt++; const hk = h[k]; for (let d = 0; d < D; d++) { mean[d] += hk[d]; if (hk[d] > mx[d]) mx[d] = hk[d]; } }
        const c = cnt > 0 ? cnt : 1;
        for (let d = 0; d < D; d++) { mean[d] /= c; if (mx[d] === -Infinity) mx[d] = 0; }   // nan_to_num(neginf=0)
        // post: Linear0(3d->d) -> GELU -> Linear2(d->d)  (NO LayerNorm — see moe_model.py)
        const out = new Array(K);
        for (let k = 0; k < K; k++) {
            const ctx = h[k].concat(mean, mx);           // 3d
            let p = geluVec(lin(W[pref + '.post.0.weight'], W[pref + '.post.0.bias'], ctx));
            out[k] = lin(W[pref + '.post.2.weight'], W[pref + '.post.2.bias'], p);
        }
        return out;
    }

    function gate(gf) {
        let a = geluVec(lin(W['gate.net.0.weight'], W['gate.net.0.bias'], gf));
        a = lin(W['gate.net.2.weight'], W['gate.net.2.bias'], a);   // (1)
        return 1 / (1 + Math.exp(-a[0]));
    }

    function head(e) {   // e (d) -> scalar
        const b = geluVec(lin(W['H.body.weight'], W['H.body.bias'], e));
        return lin(W['H.out.weight'], W['H.out.bias'], b)[0];
    }

    const CHEAP_COL = 28, SCANT_COL = 18;          // planner cheapScore / endgame scan_t col
    const wSkip = W['w_skip'][0];

    // planner/endgame blocks: [16][*]. For the absent regime pass a zero block.
    function forward(plannerBlock, endgameBlock, gateFeat, slotValid, pValid, eValid) {
        const ep = expert('E_p', plannerBlock, pValid);
        const ee = expert('E_e', endgameBlock, eValid);
        const g = gate(gateFeat);
        const logit = new Array(NSLOT);
        for (let k = 0; k < NSLOT; k++) {
            if (!slotValid[k]) { logit[k] = -1e9; continue; }
            const e = new Array(D);
            for (let d = 0; d < D; d++) e[d] = g * ep[k][d] + (1 - g) * ee[k][d];
            // gated decisive-signal residual skip (matches moe_model.py)
            const dec = g * plannerBlock[k][CHEAP_COL] + (1 - g) * (-endgameBlock[k][SCANT_COL]);
            logit[k] = head(e) + wSkip * dec;
        }
        return { logit, g };
    }
    forward.gate = gate;
    return { forward, D };
}

module.exports = { loadMoE };
