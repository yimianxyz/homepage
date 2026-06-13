// ibp.js — SOUND Interval Bound Propagation through cp_value's MLP
// (23 -> 48 GELU -> 48 GELU -> 1 linear), using the SAME cp_gelu (A&S erf) as
// js/cheap_planner.js so the bound is exact wrt the deployed activation.
//
// Affine layer with interval input [lo,hi]: for output neuron j with weights w,
// bias b:  center c_q=(lo_q+hi_q)/2, radius r_q=(hi_q-lo_q)/2;
//   out_center = b + sum_q w_q*c_q ; out_radius = sum_q |w_q|*r_q ;
//   [out_lo,out_hi] = [out_center-out_radius, out_center+out_radius].  (sound, tight
//   for affine maps over a box).
//
// GELU is NON-monotone (global min at x*≈-0.751677, value≈-0.169978; decreasing
// for x<x*, increasing for x>x*; lim_{x->-inf} GELU=0^-, lim_{x->+inf}=+inf).
// Sound image of [a,b] under GELU:
//   hi = max(gelu(a), gelu(b))                  (max always at an endpoint)
//   lo = (x* in [a,b]) ? gelu(x*) : min(gelu(a),gelu(b))
// This is EXACT (the tightest sound interval) because gelu is unimodal-down then
// monotone-up: on any interval the extrema are endpoints or the single trough.
'use strict';

var cp = require('../../../js/cheap_planner.js');
var gelu = cp.cp_gelu;

// Global min of THIS gelu (A&S erf approximation). Find numerically once, robustly:
// scan + golden-section refine. The trough is near -0.7517.
var GELU_XMIN = (function () {
    var lo = -2.0, hi = 0.0;
    // coarse scan
    var bestx = lo, bestv = gelu(lo);
    for (var x = lo; x <= hi; x += 1e-4) { var v = gelu(x); if (v < bestv) { bestv = v; bestx = x; } }
    // golden refine around bestx
    var a = bestx - 1e-3, b = bestx + 1e-3, gr = (Math.sqrt(5) - 1) / 2;
    var c = b - gr * (b - a), d = a + gr * (b - a);
    for (var it = 0; it < 200; it++) {
        if (gelu(c) < gelu(d)) b = d; else a = c;
        c = b - gr * (b - a); d = a + gr * (b - a);
    }
    return (a + b) / 2;
})();
var GELU_VMIN = gelu(GELU_XMIN);

function geluInterval(a, b) {
    var ga = gelu(a), gb = gelu(b);
    var hi = ga > gb ? ga : gb;
    var lo;
    if (a <= GELU_XMIN && GELU_XMIN <= b) lo = GELU_VMIN;
    else lo = ga < gb ? ga : gb;
    return [lo, hi];
}

// Propagate [lo,hi] (length net layer indim) through one affine layer L
// (L.w[j] length indim, L.b[j]). Returns pre-activation [preLo,preHi] per out j.
function affineInterval(L, lo, hi) {
    var outDim = L.b.length, inDim = L.w[0].length;
    var c = new Array(inDim), r = new Array(inDim);
    for (var q = 0; q < inDim; q++) { c[q] = (lo[q] + hi[q]) * 0.5; r[q] = (hi[q] - lo[q]) * 0.5; }
    var preLo = new Array(outDim), preHi = new Array(outDim);
    for (var j = 0; j < outDim; j++) {
        var wr = L.w[j], cen = L.b[j], rad = 0.0;
        for (q = 0; q < inDim; q++) { var w = wr[q]; cen += w * c[q]; rad += (w < 0 ? -w : w) * r[q]; }
        preLo[j] = cen - rad; preHi[j] = cen + rad;
    }
    return { lo: preLo, hi: preHi };
}

// Full IBP: standardized input interval {lo,hi}[23] -> [outLo, outHi] of cp_value.
function ibpValue(net, sLo, sHi) {
    var L0 = net.layers[0], L1 = net.layers[1], L2 = net.layers[2];
    // layer 0 affine
    var p0 = affineInterval(L0, sLo, sHi);
    // gelu
    var h0lo = new Array(48), h0hi = new Array(48);
    for (var j = 0; j < 48; j++) { var g = geluInterval(p0.lo[j], p0.hi[j]); h0lo[j] = g[0]; h0hi[j] = g[1]; }
    // layer 1 affine
    var p1 = affineInterval(L1, h0lo, h0hi);
    var h1lo = new Array(48), h1hi = new Array(48);
    for (j = 0; j < 48; j++) { var g1 = geluInterval(p1.lo[j], p1.hi[j]); h1lo[j] = g1[0]; h1hi[j] = g1[1]; }
    // layer 2 linear (output)
    var p2 = affineInterval(L2, h1lo, h1hi);
    return { lo: p2.lo[0], hi: p2.hi[0] };
}

module.exports = { ibpValue: ibpValue, geluInterval: geluInterval, affineInterval: affineInterval,
    GELU_XMIN: GELU_XMIN, GELU_VMIN: GELU_VMIN };
