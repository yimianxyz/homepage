# float64 spike — can the GPU be a bitwise engine for the prod policy?

20k deterministic vectors per op class, generated in node (v8) with exact bit
patterns (`gen_vectors.js`, md5-stable), recomputed in torch float64 on
ml-forecast-1 CPU and CUDA (`check_torch.py`). Bit-exact match rates:

| op class | usage in policy | CPU | CUDA | maxulp |
|---|---|---|---|---|
| mul_add, div, fastmag | everything | 100% | **100%** | 0 |
| sqrt(x²+y²) | flock/steer/intercept | 99.13%¹ | **100%** | 1 |
| exp(−x²) | value-net GELU (cp_erf) | 89.4% | 89.8% | 1 |
| exp(−d/reach) | E3D candidate gen | 89.5% | 89.5% | 1 |
| pow | E3D dens/sharp weights | 90.1% | 72.5% | 1–2 |
| erf / gelu composite | value net | 99.2% | 98.4–99.3% | 4–512² |
| round (as floor(x+.5)) | intercept wrap | 83.7%³ | 83.7%³ | −0 artifacts |

¹ torch-CPU-only 1-ulp anomaly (likely a vectorized-libm path); CUDA clean, so
irrelevant for the GPU plan. ² relative blowup where the polynomial cancels;
inherited from exp. ³ JS `Math.round` returns −0 on (−0.5, −0]; `floor(x+0.5)`
returns +0. In the only prod usage (`d − PX*Math.round(d/PX)`) the −0 washes
out in the multiply+subtract, and away from exact .5 boundaries the integer
values agree — replicable with a faithful jsround.

## Conclusions

1. **Rollout physics, steer, intercept = bitwise-replicable on CUDA float64**
   (pure IEEE chains; preserve op order; emulate the grid's linked-list
   iteration order exactly).
2. **The only bit-divergent ops are `Math.exp` / `Math.pow`** (and the erf/gelu
   built on exp): v8 uses its own fdlibm port; torch uses libm/CUDA intrinsics;
   they differ in the last 1–2 ulp on ~10–27% of calls. These sit in (a) the
   E3D candidate point (continuous coordinates → rollouts diverge) and (b) the
   value-net forward (scores → argmax can flip near ties).
3. **Path to a fully bitwise GPU replica: port v8's fdlibm `exp`/`pow`
   (incl. `log`) as float64 arithmetic + bit twiddles in torch.** ~tens of
   flops each, vectorizable. Verify the port bitwise over the actual usage
   ranges (exp args ∈ [−64, 0]; pow: (cnt+1)^2.373 cnt∈[0..120],
   w^9.25 w∈(0,1]). Until that port passes, GPU results are SCREENING-ONLY
   and all labels/verification stay in JS (node).

## Engine-portability corollary (spec amendment)

Prod runs in several browser engines whose `Math.exp/pow` differ in ulps, so
"bit-exact" is only well-defined *within* an engine. The shipped candidate
must therefore make **the same Math calls in the same order** as prod — then
it is bit-exact in *every* engine, including ones we never test. Any design
that precomputes/replaces a transcendental with a different approximation
breaks T1 portability even if it matches in node. (The NN fast path is exempt
where its role is purely discrete: picking the same candidate index leads to
the same downstream calls regardless of engine.)
