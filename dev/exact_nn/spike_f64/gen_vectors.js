// EXACT-NN float64 spike — stage 1 (JS side, ground truth).
// Generates bit-exact test vectors for every float-op class the prod policy
// uses, so check_torch.py can measure which classes torch (CPU/CUDA, float64)
// reproduces BITWISE and which diverge (and by how many ulps).
//
// Op classes (from the op inventory of js/predator_cheap.js, cheap_planner.js,
// js/predator.js, js/vector.js — policy path only):
//   ieee   : +,-,*,/ chains, sqrt(x*x+y*y), fastMag, round   (expected: exact)
//   exp    : Math.exp over the two real usage ranges          (expected: ulps)
//   pow    : Math.pow over the two real usage ranges          (expected: ulps)
//   erf    : cp_erf / cp_gelu composite (uses exp internally)
//
// Usage: node gen_vectors.js > vectors.jsonl     (~200k lines, deterministic)
'use strict';

// xorshift128 for deterministic inputs (same one the repo's evals use)
function rng(seed) {
    var x = seed >>> 0, y = 362436069, z = 521288629, w = 88675123;
    return function () {
        var t = x ^ (x << 11); x = y; y = z; z = w;
        w = (w ^ (w >>> 19)) ^ (t ^ (t >>> 8));
        return (w >>> 0) / 4294967296;
    };
}

var buf = new ArrayBuffer(8), f64 = new Float64Array(buf), u32 = new Uint32Array(buf);
function hex(v) {
    f64[0] = v;
    return ('00000000' + u32[1].toString(16)).slice(-8) + ('00000000' + u32[0].toString(16)).slice(-8);
}
function emit(op, ins, out) {
    process.stdout.write(JSON.stringify({ op: op, in: ins.map(hex), out: hex(out) }) + '\n');
}

function fastMag(x, y) {
    var ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
    return (ax > ay ? ax : ay) * 0.96 + (ax < ay ? ax : ay) * 0.398;
}
function cp_erf(x) {
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
}
function cp_gelu(x) { return 0.5 * x * (1 + cp_erf(x / Math.SQRT2)); }

var r = rng(20260611), i;
var N = 20000;

// ieee chains over position/velocity magnitudes seen in game (coords 0..2600, vel 0..6, force 0..0.05)
for (i = 0; i < N; i++) {
    var a = (r() - 0.5) * 5200, b = (r() - 0.5) * 5200, c = (r() - 0.5) * 12, d = (r() - 0.5) * 12;
    emit('mul_add', [a, c, b, d], a * c + b * d);
    emit('div', [a, b === 0 ? 1 : b], a / (b === 0 ? 1 : b));
    emit('sqrt_hyp', [a, b], Math.sqrt(a * a + b * b));
    emit('fastmag', [c, d], fastMag(c, d));
    emit('round', [a / 1700], Math.round(a / 1700));
}
// exp: usage 1 = GELU pre-activations exp(-x*x), x in roughly [0, 8]
//      usage 2 = E3D reach term exp(-dpred/reach_scale), arg in [-3, 0]
for (i = 0; i < N; i++) {
    var x1 = r() * 8;       emit('exp_negsq', [-x1 * x1], Math.exp(-x1 * x1));
    var x2 = -r() * 3;      emit('exp_reach', [x2], Math.exp(x2));
}
// pow: usage 1 = (cnt+1)^dens_pow, cnt integer 0..120; usage 2 = w^sharp, w in (0,1]
for (i = 0; i < N; i++) {
    var cnt = Math.floor(r() * 121); emit('pow_dens', [cnt + 1, 2.373], Math.pow(cnt + 1, 2.373));
    var w = r() || 1e-12;            emit('pow_sharp', [w, 9.25], Math.pow(w, 9.25));
}
// erf/gelu composite over realistic pre-activation range
for (i = 0; i < N; i++) {
    var z = (r() - 0.5) * 16;
    emit('erf', [z], cp_erf(z));
    emit('gelu', [z], cp_gelu(z));
}
