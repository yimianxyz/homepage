// egboidPickRaw.js — RAW-kinematics deploy (genuineness ablation). Same float64
// scan-t MLP + argmin as egboidPick.js, but the per-boid features come from
// eg_features_raw.js (NO closed-form analytic reach-time). Snapshot carries the
// predator kinematics (pvx,pvy,psize) per the lead's raw-kinematics spec.
//   const { loadEgStudentRaw } = require('./egboidPickRaw.js');
//   const pick = loadEgStudentRaw('eg_weights_raw.json');
//   const r = pick(snapshot, cfg);   // {egIdx, margin, ts}
//     snapshot: { px,py,pvx,pvy,psize, bx[],by[],bvx[],bvy[] }   cfg: { W, Hc }
'use strict';
const path = require('path');
const { egBoidFeatures } = require(path.join(__dirname, 'eg_features_raw.js'));

function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }
function lin(W, b, x) { const o = new Array(W.length); for (let r = 0; r < W.length; r++) { let a = b[r]; const wr = W[r]; for (let i = 0; i < x.length; i++) a += wr[i] * x[i]; o[r] = a; } return o; }
function geluVec(v) { const o = new Array(v.length); for (let i = 0; i < v.length; i++) o[i] = gelu(v[i]); return o; }

function loadEgStudentRaw(weightsPath) {
    const fs = require('fs');
    const J = typeof weightsPath === 'string' ? JSON.parse(fs.readFileSync(weightsPath, 'utf8')) : weightsPath;
    const W = J.weights;
    const W0 = W['net.0.weight'], b0 = W['net.0.bias'], W2 = W['net.2.weight'], b2 = W['net.2.bias'], W4 = W['net.4.weight'], b4 = W['net.4.bias'];
    function scanTPred(fr) { let h = geluVec(lin(W0, b0, fr)); h = geluVec(lin(W2, b2, h)); return lin(W4, b4, h)[0] * 100.0; }
    function pick(s, cfg) {
        const n = s.bx.length; const ts = new Array(n);
        for (let i = 0; i < n; i++) ts[i] = scanTPred(egBoidFeatures(s.px, s.py, s.pvx, s.pvy, s.psize, s.bx[i], s.by[i], s.bvx[i], s.bvy[i], cfg.W, cfg.Hc));
        let egIdx = 0, best = Infinity, second = Infinity;
        for (let i = 0; i < n; i++) { if (ts[i] < best) { second = best; best = ts[i]; egIdx = i; } else if (ts[i] < second) second = ts[i]; }
        return { egIdx, margin: (n >= 2 && Number.isFinite(second)) ? (second - best) : Infinity, ts };
    }
    pick.scanTPred = scanTPred;
    return pick;
}
module.exports = { loadEgStudentRaw };
