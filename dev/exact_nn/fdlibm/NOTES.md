# fdlibm port — bitwise v8 Math.exp / Math.pow in torch float64

Deliverable of the EXACT-NN GPU-replica track: `js_exp(x)` / `js_pow(x, y)`
in `fdlibm_torch.py` reproduce node's `Math.exp` / `Math.pow` **bit-for-bit**
(float64 in/out, fully vectorized, CPU + CUDA, no per-element python).

## Source of the algorithm

- **v8 `src/base/ieee754.cc`, tag `11.3.244.8`** — functions `exp(double)`
  (line 1447) and `pow(double, double)` (line 2645):
  `https://raw.githubusercontent.com/v8/v8/11.3.244.8/src/base/ieee754.cc`
- That tag matches our ground truth, **node v20.20.2** (`process.versions.v8 =
  11.3.244.8-node.38`). The vendored copy
  `https://raw.githubusercontent.com/nodejs/node/v20.20.2/deps/v8/src/base/ieee754.cc`
  is **byte-identical** to the v8 tag (verified by `diff`; both fetched copies
  are kept here as `ieee754_v8_11.3.cc` / `ieee754_node_v20.20.2.cc`).
- The code is Sun's fdlibm `e_exp.c` / `e_pow.c` (netlib) with V8
  modifications; constants cross-checked against the hex pairs in the source
  comments. Notable V8 deviation from classic fdlibm: `exp(1.0)` is
  special-cased to return `E = 0x4005BF0A8B145769` exactly.
- **Caveat for future engines:** v8 `main` (2026) has since moved `exp`/`pow`
  to `third_party/llvm-libc` (`LIBC_NAMESPACE::shared::exp/pow`) — a
  *different* implementation. This port is bit-exact for the fdlibm-based
  engines (node 20/22-era v8, current production browsers we test against),
  which is what the spike's engine-portability corollary requires. If ground
  truth ever moves to a v8 with llvm-libc math, the port must be redone.

## Porting approach

- Every float intermediate is plain IEEE-754 float64 (`+ - * /`, sqrt) in the
  same order as the C source; each such op is correctly rounded, hence
  bit-identical on any IEEE backend (CPU and CUDA float64 — proven per op
  class in `../spike_f64/REPORT.md`). Eager torch only — do **not** run under
  `torch.compile` (operation fusion/reassociation would break exactness).
- HI/LO word macros (`EXTRACT_WORDS`, `GET/SET_HIGH/LOW_WORD`,
  `INSERT_WORDS`) emulated with int64 bit ops on `.view(torch.int64)`;
  signed-int32 wraparound reproduced where the C code relies on it
  (`_wrap32`, masked adds for `twopk`, `j += (int)((u32)n << 20)`).
- Branches vectorized as `torch.where` chains applied in **reverse priority**
  so the final select order equals the C early-return order. Data-dependent
  shift amounts are clamped only in lanes whose results are discarded.
- Constants constructed from the exact hex in the source comments
  (`_hexf`). Two constants exist only as decimal literals in C (`ovt`,
  the 1/3 in pow's |y|-huge series); python parses decimals correctly
  rounded exactly like the compiler — their bit patterns are `assert`ed
  at import (`0x3C971547652B82FE`, `0x3FD5555555555555`).

## Branch coverage — FULL (no unsupported inputs)

`exp`: NaN (payload-preserving), ±inf, overflow (`x > 709.78...`), underflow
(`x < -745.13...`), `|x| < 2^-28` (returns `1+x`), `x == 1` (returns E),
0.5ln2/1.5ln2 reduction cuts, `k == 1024` (`y*2.0*0x1p1023`), `k < -1021`
subnormal-result scaling (`y*twopk*2^-1000`).

`pow`: y==±0 → 1 (before NaN!); NaN x/y; y = ±inf / ±1 / 2 / 0.5 fast paths;
x = ±0/±1/±inf special table incl. `(-1)^non-int`; `yisint` parity logic with
both the k>20 and k≤20 cuts; `(x<0)^non-int` → sNaN; |y| > 2^31 and > 2^64
huge-y paths incl. the |1−x| ≤ 2^-20 log series; subnormal x (`ax *= 2^53`
renormalization); z = y·log2(x) overflow (≥1024) / underflow (≤−1075) checks
incl. the exact-boundary `ovt`/`p_l` tie-breaks; subnormal results via
`scalbn`; negative-base sign via `s`.

Helpers the C code delegates to:

- `base::Divide(1, x)` (v8 `overflowing-math.h`) is plain IEEE division —
  torch `/` matches.
- `scalbn` (libm, subnormal pow results): the operation is exactly defined
  (x·2^n, one rounding); ported as fdlibm `s_scalbn` — any conforming
  scalbn returns identical bits.
- `sqrt` (libm, `pow(x, 0.5)` path): IEEE requires correct rounding, but
  **this torch build's CPU sqrt (MKL vector path) is 1 ulp off** on some
  inputs (e.g. `sqrt(1+2^-52)`; the spike's footnote-1 anomaly). `_ieee_sqrt`
  re-rounds exactly: normalize to `m·2^E2` (E2 even, m ∈ [1,4)), seed with
  `torch.sqrt`, then decide among seed±3 ulp candidates by the exact sign of
  `(midpoint² − m)` — Dekker two-product splits it into float64 terms that
  are integer multiples of 2^-106 below 2^63·2^-106, so they convert
  losslessly to int64 and the int64 sum's sign is exact (ties impossible:
  a 53-bit double can't equal a midpoint's ≥105-bit square).

## NaN bit semantics (measured against node on x86-64)

- `Float64Array` in node preserves NaN payloads, so they are observable and
  the verifier checks them bit-exactly.
- `exp`/`pow` C paths `return x + x` / `x + y`: x86 SSE returns the **first**
  NaN operand with the quiet bit set (payload+sign preserved). Emulated as
  `bits | 0x0008000000000000` (CUDA arithmetic would canonicalize NaNs).
- pow, |x| == 1 and y = ±inf: `return y - y` = inf − inf = x86 "indefinite"
  QNaN **0xFFF8000000000000**.
- pow, `(x<0)^non-int` and `(-1)^non-int`:
  `std::numeric_limits<double>::signaling_NaN()` = **0x7FF4000000000000**,
  handed to JS unquieted (measured: `Math.pow(-2, 0.5)`).
- Caveat: these exact NaN bits are properties of node/v8 **on x86-64** (our
  VMs, our ground truth). A non-x86 engine may produce different NaN bits in
  these paths; all non-NaN outputs are engine-portable IEEE results.

## Verification

`gen_ref.js` (node = ground truth, deterministic xorshift128, optional
`[seed] [scale]` args) → `ref.jsonl` (1,511,640 vectors); `verify.py` runs
js_exp/js_pow and compares bits. **CPU results (this container):**

| range | class | n | bitexact% | maxulp |
|---|---|---|---|---|
| exp_p64 — exp over [−64,0] dense+random | POLICY | 320,001 | **100.0000** | 0 |
| exp_p3 — exp over [−3,0] dense+random | POLICY | 260,001 | **100.0000** | 0 |
| exp_edge — −0/denormals/pow2/branch cuts | POLICY | 4,479 | **100.0000** | 0 |
| pow_dens — (cnt+1)^2.373, cnt=0..120 ALL | POLICY | 121 | **100.0000** | 0 |
| pow_sharp — w^9.25, w∈(0,1] incl. subnormal | POLICY | 270,007 | **100.0000** | 0 |
| exp_subn — subnormal-result band | stress | 30,202 | 100.0000 | 0 |
| exp_stress — full double range + specials | stress | 230,121 | 100.0000 | 0 |
| pow_mid — z swept across [−1080,1030] | stress | 50,000 | 100.0000 | 0 |
| pow_edge — special-case cross product | stress | 96,708 | 100.0000 | 0 |
| pow_stress — random full-range bits | stress | 250,000 | 100.0000 | 0 |

Plus three torture passes (`seeds 1, 424242, 987654321`, scale 4 — 5.4M
vectors each, 16.2M total): **100.0000% bitexact, every range, max ulp 0.**
Zero mismatches over ~17.7M vectors overall.

CUDA: not testable in this container (CPU-only torch); `verify.py` prints a
cuda section automatically when run on a GPU VM. Expected exact: every
operation used is either correctly-rounded IEEE float64 on CUDA (mul/add/div
— spike-proven 100%), exact integer/bit ops, or the backend-independent
`_ieee_sqrt`. No torch `exp/pow/sqrt` result is trusted anywhere.

## Files

- `fdlibm_torch.py` — `js_exp`, `js_pow` (+ internal `_scalbn`, `_ieee_sqrt`).
- `gen_ref.js` — reference-vector generator (node): `node gen_ref.js > ref.jsonl`.
- `verify.py` — `python3 verify.py ref.jsonl`; exits 1 if any POLICY range
  is not 100% bitexact on any tested device.
- `ieee754_v8_11.3.cc`, `ieee754_node_v20.20.2.cc`, `ieee754_v8_main.cc` —
  fetched source references (v8 tag, node vendored copy, current main showing
  the llvm-libc migration).
- `ref.jsonl` — generated, gitignored.

Throughput (this container's CPU, single run, 1e6 elements): js_exp ~2.3 s,
js_pow ~9.9 s — irrelevant for the GPU use case (tens of elementwise kernels,
all parallel on the L4s).
