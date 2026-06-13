// EXACT-NN fdlibm port — reference vector generator (node = ground truth).
// Emits JSONL: {"op": <range tag>, "in": [hex u64 ...], "out": hex u64}
// (same style as ../spike_f64/gen_vectors.js). Deterministic.
//
// Range tags (POLICY ranges must verify 100.0000% bitexact):
//   exp_p64    POLICY  exp over [-64, 0], dense + random
//   exp_p3     POLICY  exp over [-3, 0], dense + random
//   exp_edge   POLICY  boundary/special args in the policy domain:
//                      -0, denormal args, exact powers of two, branch
//                      boundaries (0.5ln2 / 1.5ln2 / 2^-28 high-word cuts,
//                      k-transition multiples of 0.5ln2)
//   pow_dens   POLICY  (cnt+1) ** 2.373 for cnt = 0..120 (ALL values)
//   pow_sharp  POLICY  w ** 9.25 for w in (0,1] incl. subnormal w and
//                      subnormal-result region
//   exp_subn   stress  exp args in [-745.2, -708.3] (subnormal results)
//   exp_stress stress  full double range incl. NaN/Inf/±0/denormals
//   pow_mid    stress  z = y*log2(x) swept across [-1080, 1030]
//   pow_edge   stress  special-case cross product (NaN/Inf/±0/±1/neg base/
//                      huge |y|/yisint parity/overflow-underflow boundaries)
//   pow_stress stress  random full-range bit patterns for x and y
//
// Usage: node gen_ref.js > ref.jsonl
'use strict';

// xorshift128 — deterministic inputs (same generator as the repo's evals)
function rng(seed) {
    var x = seed >>> 0, y = 362436069, z = 521288629, w = 88675123;
    return function () {
        var t = x ^ (x << 11); x = y; y = z; z = w;
        w = (w ^ (w >>> 19)) ^ (t ^ (t >>> 8));
        return (w >>> 0) / 4294967296;
    };
}

const buf = new ArrayBuffer(8);
const f64 = new Float64Array(buf);
const u64 = new BigUint64Array(buf);
function hex(v) { f64[0] = v; return u64[0].toString(16).padStart(16, '0'); }
function fromU64(big) { u64[0] = big & 0xFFFFFFFFFFFFFFFFn; return f64[0]; }
function bitsOf(v) { f64[0] = v; return u64[0]; }

let chunk = [];
function emit(op, ins, out) {
    chunk.push(JSON.stringify({ op: op, in: ins.map(hex), out: hex(out) }));
    if (chunk.length >= 4096) { process.stdout.write(chunk.join('\n') + '\n'); chunk = []; }
}
function flush() { if (chunk.length) { process.stdout.write(chunk.join('\n') + '\n'); chunk = []; } }

function expCase(tag, x) { emit(tag, [x], Math.exp(x)); }
function powCase(tag, x, y) { emit(tag, [x, y], Math.pow(x, y)); }
// bit-neighbors: u64 +/- k (valid for non-boundary finite values)
function nbrs(v, k) {
    const b = bitsOf(v), out = [];
    for (let d = -k; d <= k; d++) out.push(fromU64(b + BigInt(d)));
    return out;
}

// optional: node gen_ref.js [seed] [scale]  (defaults: 20260611, 1 —
// the deliverable corpus; other values for extra torture runs)
const SEED = parseInt(process.argv[2] || '20260611', 10);
const SCALE = parseFloat(process.argv[3] || '1');
const r = rng(SEED);
function N(n) { return Math.round(n * SCALE); }
function randU64() {
    const hi = BigInt(Math.floor(r() * 0x100000000));
    const lo = BigInt(Math.floor(r() * 0x100000000));
    return (hi << 32n) | lo;
}

// ---------------------------------------------------------------- exp_p64
// dense sweep of [-64, 0]
for (let i = 0; i <= 120000; i++) expCase('exp_p64', -64 * i / 120000);
// random
for (let i = 0; i < N(200000); i++) expCase('exp_p64', -64 * r());

// ----------------------------------------------------------------- exp_p3
for (let i = 0; i <= 60000; i++) expCase('exp_p3', -3 * i / 60000);
for (let i = 0; i < N(200000); i++) expCase('exp_p3', -3 * r());

// --------------------------------------------------------------- exp_edge
// -0, +0, smallest/denormal magnitudes, exact powers of two in [-64, 0]
expCase('exp_edge', -0);
expCase('exp_edge', 0);
expCase('exp_edge', fromU64(0x8000000000000001n));            // -minsub
expCase('exp_edge', fromU64(0x800FFFFFFFFFFFFFn));            // -maxsub
expCase('exp_edge', fromU64(0x8010000000000000n));            // -DBL_MIN
for (let i = 0; i < N(2000); i++) {                              // random -denormals
    expCase('exp_edge', fromU64(0x8000000000000000n | (randU64() & 0x000FFFFFFFFFFFFFn)));
}
for (let e = -45; e <= 6; e++) expCase('exp_edge', -Math.pow(2, e));
// high-word branch cuts: |x| ~ 0.5ln2 (0x3FD62E42|43), 1.5ln2 (0x3FF0A2B1|B2),
// 2^-28 (0x3E300000) — sample both sides of each cut, negative args
for (const hw of [0x3FD62E42n, 0x3FD62E43n, 0x3FF0A2B1n, 0x3FF0A2B2n,
                  0x3E2FFFFFn, 0x3E300000n]) {
    for (const lo of [0x00000000n, 0x00000001n, 0x80000000n, 0xFEFA39EFn, 0xFFFFFFFFn]) {
        expCase('exp_edge', fromU64(0x8000000000000000n | (hw << 32n) | lo));
    }
}
// k-transition multiples of 0.5ln2 down to -64, with ulp neighbors + jitter
const HALF_LN2 = 0.34657359027997264;
for (let j = 1; j <= 184; j++) {
    const c = -j * HALF_LN2;
    if (c < -64) break;
    for (const v of nbrs(c, 2)) expCase('exp_edge', v);
    for (let i = 0; i < 8; i++) expCase('exp_edge', c + (r() - 0.5) * 1e-13);
}

// --------------------------------------------------------------- exp_subn
// args whose results are subnormal or near the underflow boundary
for (let i = 0; i < N(30000); i++) expCase('exp_subn', -745.2 + r() * (745.2 - 708.3));
for (const v of nbrs(-745.1332191019411, 50)) expCase('exp_subn', v);  // u_threshold
for (const v of nbrs(-708.3964185322641, 50)) expCase('exp_subn', v);  // ln(2^-1022)

// ------------------------------------------------------------- exp_stress
// full-range random bit patterns (hits NaN/Inf/denormal/huge automatically)
for (let i = 0; i < N(200000); i++) expCase('exp_stress', fromU64(randU64()));
// positive range incl. overflow boundary
for (let i = 0; i < N(20000); i++) expCase('exp_stress', r() * 720);
for (const v of nbrs(709.782712893384, 50)) expCase('exp_stress', v);   // o_threshold
expCase('exp_stress', 1);                                               // exp(1) == E
for (const v of nbrs(1, 4)) expCase('exp_stress', v);
expCase('exp_stress', Infinity);
expCase('exp_stress', -Infinity);
for (const nan of [0x7FF8000000000000n, 0xFFF8000000000000n, 0x7FF8000000000123n,
                   0xFFF8000000000123n, 0x7FF0000000000001n, 0xFFF0000000000001n,
                   0x7FF4000000000000n, 0x7FFFFFFFFFFFFFFFn]) {
    expCase('exp_stress', fromU64(nan));
}
for (let i = 0; i < N(5000); i++) expCase('exp_stress', (r() - 0.5) * 6e-8); // |x|<2^-28 region
for (let i = 0; i < N(5000); i++) expCase('exp_stress', (r() - 0.5) * 2.2);  // mid/k=0/k=±1

// ---------------------------------------------------------------- pow_dens
for (let cnt = 0; cnt <= 120; cnt++) powCase('pow_dens', cnt + 1, 2.373);

// --------------------------------------------------------------- pow_sharp
for (let i = 0; i < N(200000); i++) powCase('pow_sharp', r() || 1, 9.25);
powCase('pow_sharp', 1, 9.25);
powCase('pow_sharp', fromU64(0x3FEFFFFFFFFFFFFFn), 9.25);   // 1 - 2^-53
powCase('pow_sharp', 0.5, 9.25);
powCase('pow_sharp', 0.25, 9.25);
powCase('pow_sharp', fromU64(0x0010000000000000n), 9.25);   // DBL_MIN
powCase('pow_sharp', fromU64(0x000FFFFFFFFFFFFFn), 9.25);   // max subnormal
powCase('pow_sharp', fromU64(0x0000000000000001n), 9.25);   // min subnormal
for (let i = 0; i < N(20000); i++) {                           // random subnormal w
    powCase('pow_sharp', fromU64(randU64() & 0x000FFFFFFFFFFFFFn || 1n), 9.25);
}
for (let i = 0; i < 30000; i++) {                           // log-uniform w over (0,1]
    powCase('pow_sharp', Math.pow(2, -1074 * r()), 9.25);
}
for (let i = 0; i < N(20000); i++) {                           // subnormal-result band
    powCase('pow_sharp', Math.pow(2, -(105 + 13 * r())), 9.25);
}

// ----------------------------------------------------------------- pow_mid
// sweep z = y*log2(x) across all result magnitudes incl. sub/overflow edges
for (let i = 0; i < N(50000); i++) {
    const x = Math.pow(2, (r() - 0.5) * 60) * (1 + r());
    const z = -1080 + r() * 2110;
    const y = z / Math.log2(x);
    if (!isFinite(y)) continue;
    powCase('pow_mid', x, y);
}

// ---------------------------------------------------------------- pow_edge
const xs = [0, -0, 1, -1, Infinity, -Infinity, NaN,
            fromU64(0x7FF8000000000123n), fromU64(0x7FF0000000000001n),  // NaN payload/sNaN
            2, -2, 0.5, -0.5, 1.5, -1.5, 3, -3,
            fromU64(0x3FF0000000000001n),   // 1 + 2^-52
            fromU64(0x3FEFFFFFFFFFFFFFn),   // 1 - 2^-53
            fromU64(0xBFF0000000000001n), fromU64(0xBFEFFFFFFFFFFFFFn),
            1.0000000001, 0.9999999999, -1.0000000001, -0.9999999999,
            fromU64(0x0000000000000001n), fromU64(0x000FFFFFFFFFFFFFn),  // subnormals
            fromU64(0x8000000000000001n), fromU64(0x0010000000000000n),
            fromU64(0x7FEFFFFFFFFFFFFFn), fromU64(0xFFEFFFFFFFFFFFFFn),  // ±DBL_MAX
            1e300, -1e300, 1e-300, 10, -10];
const ys = [0, -0, 1, -1, 2, -2, 0.5, -0.5, 3, -3, 4, 5, 2.5, -2.5, 9.25, 2.373,
            Infinity, -Infinity, NaN, fromU64(0xFFF8000000000456n),
            0.3333333333333333, 1023.9999999999999, 1024, 1025, -1074, -1075, -1076.5,
            2147483647, 2147483648, 2147483648.5, 2147483649,            // 2^31 cut
            -2147483649, 3000000000, 3000000001,
            1048576, 1048577, 1048576.5,                                  // 2^20 cut
            9007199254740992, 9007199254740991, 4503599627370497,         // 2^53, odd 2^52+1
            18446744073709551616, 1.8446744073709552e19 * 1.0000001,      // 2^64 cut
            1e300, -1e300, 1e16 + 1, 52.5, -52.5];
for (const a of xs) for (const b of ys) powCase('pow_edge', a, b);
// y*log2(x) exactly at the 1024 / -1075 cut checks
for (const [a, b] of [[2, 1024], [2, 1023.9999999999999], [2, -1074], [2, -1075],
                      [2, -1074.9999999999999], [4, 512], [4, -537.5], [0.5, 1075],
                      [0.5, 1074], [0.5, -1024], [32, 204.8], [32, -215]]) {
    powCase('pow_edge', a, b);
}
// dense |1-x| tiny with huge |y| (the log-series branch), both yisint parities
for (let i = 0; i < N(5000); i++) {
    const x = fromU64(0x3FEFFFFF00000000n + (randU64() & 0x1FFFFFFFFn));  // ix in {0x3FEFFFFF,0x3FF00000}
    const y = (r() < 0.5 ? -1 : 1) * Math.floor(r() * 1e18 + 2147483649);
    powCase('pow_edge', r() < 0.25 ? -x : x, y);
    powCase('pow_edge', x, y + 0.5);
}
// negative base, integer y of both parities incl. k=20/52 cut exponents
for (let i = 0; i < N(5000); i++) {
    const xb = -(r() * 4 + 0.25);
    const yi = Math.floor(r() * 4e6) - 2e6;
    powCase('pow_edge', xb, yi);
    powCase('pow_edge', xb, yi + 0.5);
    powCase('pow_edge', xb, (Math.floor(r() * 1e15) + 4503599627370496));
}

// y == 0.5 fast path (= libm sqrt; exercises the correctly-rounded-sqrt
// emulation): random magnitudes, subnormals, perfect squares +- ulps
for (let i = 0; i < N(30000); i++) {
    powCase('pow_edge', fromU64(randU64() & 0x7FFFFFFFFFFFFFFFn), 0.5);
}
for (let i = 0; i < N(10000); i++) {
    powCase('pow_edge', fromU64(randU64() & 0x000FFFFFFFFFFFFFn || 1n), 0.5);
}
for (let i = 0; i < N(10000); i++) {
    const s = fromU64((randU64() & 0x000FFFFFFFFFFFFFn) | 0x3FF0000000000000n) *
              Math.pow(2, Math.floor(r() * 120) - 60);
    for (const v of nbrs(s * s, 1)) powCase('pow_edge', v, 0.5);
}
powCase('pow_edge', fromU64(0x7FEFFFFFFFFFFFFFn), 0.5);   // DBL_MAX
powCase('pow_edge', fromU64(0x0010000000000000n), 0.5);   // DBL_MIN
powCase('pow_edge', fromU64(0x0000000000000001n), 0.5);   // min subnormal
powCase('pow_edge', 4 - Math.pow(2, -51), 0.5);           // top of [1,4) band

// -------------------------------------------------------------- pow_stress
for (let i = 0; i < N(200000); i++) powCase('pow_stress', fromU64(randU64()), fromU64(randU64()));
// finite-ish stress: moderate magnitudes both signs
for (let i = 0; i < N(50000); i++) {
    const x = fromU64(randU64() & 0x47FFFFFFFFFFFFFFn);  // |exp field| capped
    const y = (r() - 0.5) * 2400;
    powCase('pow_stress', r() < 0.5 ? -x : x, y);
}

flush();
