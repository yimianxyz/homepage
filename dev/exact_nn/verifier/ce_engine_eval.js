// ce_engine_eval.js — engine-agnostic value-net forward for the REAL §4c
// cross-engine check. Runs identically under node (V8) and the SpiderMonkey
// jsshell; reads a harvested-states file + value_net.json, prints one line per
// plan: the 16 vprior scores as raw float64 hex (so the comparison is bitwise,
// engine-independent of toString rounding).
//
// cp_erf / cp_gelu / cp_value are COPIED VERBATIM from js/cheap_planner.js — the
// exact arithmetic prod runs — so the only thing that can differ between engines
// is Math.exp (in cp_erf) and Math.SQRT2. That is precisely the §4c question.
//
//   node       dev/exact_nn/verifier/ce_engine_eval.js  harvest.json value_net.json > v8.txt
//   LD_LIBRARY_PATH=/tmp/sm /tmp/sm/js dev/exact_nn/verifier/ce_engine_eval.js harvest.json value_net.json > sm.txt
'use strict';

// ---- VERBATIM from js/cheap_planner.js (cp_erf, cp_gelu, cp_value) ----
function cp_erf(x) {
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
}
function cp_gelu(x) { return 0.5 * x * (1 + cp_erf(x / Math.SQRT2)); }
function cp_value(net, feat, ctx) {
    var K = feat.length, V = new Array(K);
    var L0 = net.layers[0], L1 = net.layers[1], L2 = net.layers[2];
    var cs = [(ctx[0] - net.xmu[0]) / net.xsd[0], (ctx[1] - net.xmu[1]) / net.xsd[1],
              (ctx[2] - net.xmu[2]) / net.xsd[2], (ctx[3] - net.xmu[3]) / net.xsd[3]];
    for (var k = 0; k < K; k++) {
        var x = new Array(23);
        for (var i = 0; i < 19; i++) x[i] = (feat[k][i] - net.fmu[i]) / net.fsd[i];
        x[19] = cs[0]; x[20] = cs[1]; x[21] = cs[2]; x[22] = cs[3];
        var h0 = new Array(48);
        for (var j = 0; j < 48; j++) { var a = L0.b[j], wr = L0.w[j]; for (var q = 0; q < 23; q++) a += wr[q] * x[q]; h0[j] = cp_gelu(a); }
        var h1 = new Array(48);
        for (j = 0; j < 48; j++) { a = L1.b[j]; wr = L1.w[j]; for (q = 0; q < 48; q++) a += wr[q] * h0[q]; h1[j] = cp_gelu(a); }
        var o = L2.b[0], w2 = L2.w[0]; for (j = 0; j < 48; j++) o += w2[j] * h1[j];
        V[k] = o;
    }
    return V;
}

// ---- engine-agnostic IO ----
var IS_NODE = (typeof process !== 'undefined' && process.versions && process.versions.node);
function loadJSON(path) {
    if (IS_NODE) return JSON.parse(require('fs').readFileSync(path, 'utf8'));
    return JSON.parse(read(path));                 // SpiderMonkey shell builtin
}
function emit(line) { if (IS_NODE) process.stdout.write(line + '\n'); else print(line); }  // sm print() adds newline
var ARGS = IS_NODE ? process.argv.slice(2) : scriptArgs;

// float64 -> 16 hex chars (bitwise, engine-independent)
var _ab = new ArrayBuffer(8), _f = new Float64Array(_ab), _u = new Uint32Array(_ab);
function f64hex(x) {
    _f[0] = x;
    var hi = (_u[1] >>> 0).toString(16), lo = (_u[0] >>> 0).toString(16);
    return ('00000000' + hi).slice(-8) + ('00000000' + lo).slice(-8);
}

var harvest = loadJSON(ARGS[0]);
var net = loadJSON(ARGS[1]);
emit('# engine=' + (IS_NODE ? ('node-v8 ' + process.versions.v8) : ('spidermonkey ' + (typeof version === 'function' ? version() : '?'))) + ' plans=' + harvest.length);
for (var p = 0; p < harvest.length; p++) {
    var v = cp_value(net, harvest[p].feat, harvest[p].ctx);
    var hexes = new Array(v.length);
    for (var k = 0; k < v.length; k++) hexes[k] = f64hex(v[k]);
    emit(p + ' ' + hexes.join(','));
}
