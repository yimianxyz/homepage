// crown.js — TIGHTER sound bound on cp_value than plain IBP, via forward
// linear-relaxation (CROWN-lite). For each neuron we carry a SOUND linear lower
// and upper bound in terms of the 23 standardized inputs x:
//     Llam . x + Lc  <=  neuron  <=  Ulam . x + Uc
// Affine layers compose these exactly (split + and - weights onto the right
// bound). GELU is relaxed by a sound linear envelope chosen from the neuron's
// concrete pre-activation interval [l,u] (from IBP-style concretization of the
// current linear bounds). The final scalar output's [lo,hi] is obtained by
// maximizing / minimizing its linear forms over the input box.
//
// GELU envelope (sound) on [l,u], g(x)=cp_gelu(x):
//   We need lines aL*x+bL <= g <= aU*x+bU for all x in [l,u].
//   g is convex for x >= x_infl and the standard relaxations are messy because g
//   is neither convex nor monotone globally. We use a robust, provably-sound
//   construction: sample g densely on [l,u], then
//     UPPER line: the chord/secant lifted to lie ABOVE g — take the line through
//       (l,g(l)) and (u,g(u)) and raise its intercept by max_x (g(x) - line(x))
//       (>=0); this is sound (line+gap >= g everywhere on [l,u]).
//     LOWER line: same secant lowered by max_x (line(x) - g(x)); sound.
//   The secant slope a=(g(u)-g(l))/(u-l); bU = secant_intercept + maxAbove,
//   bL = secant_intercept - maxBelow. Both verified sound by the dense scan with a
//   safety margin. (For tiny intervals this collapses to ~exact; for wide ones it
//   is far tighter than IBP because the slope tracks g.)
'use strict';
var cp = require('../../../js/cheap_planner.js');
var g = cp.cp_gelu;

// sound linear envelope of gelu on [l,u]: returns {aL,bL,aU,bU}
function geluEnvelope(l, u) {
    if (u - l < 1e-12) {
        var gv = g(l);
        return { aL: 0, bL: gv, aU: 0, bU: gv };
    }
    var a = (g(u) - g(l)) / (u - l);
    var b0 = g(l) - a * l;        // secant intercept
    // dense scan for max deviation above/below the secant
    var N = 400, maxAbove = 0, maxBelow = 0;
    for (var i = 0; i <= N; i++) {
        var x = l + (u - l) * i / N;
        var line = a * x + b0;
        var gv2 = g(x);
        var dAbove = gv2 - line;   // g above secant
        var dBelow = line - gv2;   // g below secant
        if (dAbove > maxAbove) maxAbove = dAbove;
        if (dBelow > maxBelow) maxBelow = dBelow;
    }
    // safety margin against scan-grid miss (gelu is smooth; |g''| modest). Add a
    // conservative slack proportional to the grid step^2 curvature; cheap & sound.
    var step = (u - l) / N;
    var slack = 0.5 * step * step * 0.3 + 1e-9;   // 0.3 >~ max|g''| bound near trough
    return { aL: a, bL: b0 - maxBelow - slack, aU: a, bU: b0 + maxAbove + slack };
}

// concretize a linear form (lam[23], c) over box [xLo,xHi] -> [lo,hi]
function concretize(lam, c, xLo, xHi) {
    var lo = c, hi = c;
    for (var q = 0; q < lam.length; q++) {
        var w = lam[q];
        if (w >= 0) { lo += w * xLo[q]; hi += w * xHi[q]; }
        else { lo += w * xHi[q]; hi += w * xLo[q]; }
    }
    return [lo, hi];
}

// CROWN-lite forward pass. Returns {lo,hi} of cp_value over the input box.
function crownValue(net, xLo, xHi) {
    var nIn = 23;
    // neuron linear bounds: arrays of {Llam[nIn],Lc, Ulam[nIn],Uc}
    // input layer: identity
    var cur = [];
    for (var q = 0; q < nIn; q++) {
        var Ll = new Float64Array(nIn), Ul = new Float64Array(nIn);
        Ll[q] = 1; Ul[q] = 1;
        cur.push({ Ll: Ll, Lc: 0, Ul: Ul, Uc: 0 });
    }
    for (var layer = 0; layer < net.layers.length; layer++) {
        var L = net.layers[layer], outDim = L.b.length, inDim = L.w[0].length;
        var isLast = (layer === net.layers.length - 1);
        var pre = [];   // pre-activation linear bounds
        for (var j = 0; j < outDim; j++) {
            // affine combine: sum_i w[j][i]*cur[i] + b[j]
            var Ll2 = new Float64Array(nIn), Ul2 = new Float64Array(nIn);
            var Lc2 = L.b[j], Uc2 = L.b[j];
            var wr = L.w[j];
            for (var i = 0; i < inDim; i++) {
                var w = wr[i], ci = cur[i];
                if (w >= 0) {
                    for (var q2 = 0; q2 < nIn; q2++) { Ll2[q2] += w * ci.Ll[q2]; Ul2[q2] += w * ci.Ul[q2]; }
                    Lc2 += w * ci.Lc; Uc2 += w * ci.Uc;
                } else {
                    for (q2 = 0; q2 < nIn; q2++) { Ll2[q2] += w * ci.Ul[q2]; Ul2[q2] += w * ci.Ll[q2]; }
                    Lc2 += w * ci.Uc; Uc2 += w * ci.Lc;
                }
            }
            pre.push({ Ll: Ll2, Lc: Lc2, Ul: Ul2, Uc: Uc2 });
        }
        if (isLast) {
            // output neuron(s): concretize
            var rLo = concretize(pre[0].Ll, pre[0].Lc, xLo, xHi)[0];
            var rHi = concretize(pre[0].Ul, pre[0].Uc, xLo, xHi)[1];
            return { lo: rLo, hi: rHi };
        }
        // GELU relaxation per neuron
        var next = [];
        for (j = 0; j < outDim; j++) {
            var pj = pre[j];
            var preLo = concretize(pj.Ll, pj.Lc, xLo, xHi)[0];
            var preHi = concretize(pj.Ul, pj.Uc, xLo, xHi)[1];
            var env = geluEnvelope(preLo, preHi);
            // g >= aL*pre + bL ; g <= aU*pre + bU. Slopes here are equal (secant
            // slope a). Compose with pre's linear bounds. Slope a may be + or -.
            var aL = env.aL, aU = env.aU;
            var Ll3 = new Float64Array(nIn), Ul3 = new Float64Array(nIn), Lc3, Uc3;
            // lower bound of g: aL*pre + bL. If aL>=0 use pre lower bound; else pre upper.
            if (aL >= 0) { for (q2 = 0; q2 < nIn; q2++) Ll3[q2] = aL * pj.Ll[q2]; Lc3 = aL * pj.Lc + env.bL; }
            else { for (q2 = 0; q2 < nIn; q2++) Ll3[q2] = aL * pj.Ul[q2]; Lc3 = aL * pj.Uc + env.bL; }
            // upper bound of g: aU*pre + bU. If aU>=0 use pre upper; else pre lower.
            if (aU >= 0) { for (q2 = 0; q2 < nIn; q2++) Ul3[q2] = aU * pj.Ul[q2]; Uc3 = aU * pj.Uc + env.bU; }
            else { for (q2 = 0; q2 < nIn; q2++) Ul3[q2] = aU * pj.Ll[q2]; Uc3 = aU * pj.Lc + env.bU; }
            next.push({ Ll: Ll3, Lc: Lc3, Ul: Ul3, Uc: Uc3 });
        }
        cur = next;
    }
    // unreachable
    return null;
}

module.exports = { crownValue: crownValue, geluEnvelope: geluEnvelope };
