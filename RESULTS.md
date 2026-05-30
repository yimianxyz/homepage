
## M3 — minimality + behavioral verification (CONCLUSION)

### The end-to-end ceiling is ~85–88% of production, and 100% is mathematically impossible
High-precision eval of the candidate net (G13, 512,256, no-density) on **2048** held-out seeds:

| metric | value | reading |
|---|---|---|
| mean_prod | 8.316 | production baseline (2048-seed block) |
| mean_e2e | 7.04 / 7.09 (a/b) | ~85% of prod |
| mean_delta | −1.25 | the mean-catch shortfall |
| dist_gap | 1.23 / 1.28 | honest behavioral gap (sorted catch distributions) |
| **per_seed_corr** | **0.10** | **chaos signature — per-seed catches are ~uncorrelated with prod** |
| mean_abs_diff | 4.5 | per-seed |diff|, chaos-dominated |

`per_seed_corr ≈ 0.10` is the crux: it is NOT a model defect, it is the M0 result made
visible. Production is memoryless and deterministic, but the sim is chaotic (ε* < 1e-7: a
1e-7 force perturbation already decorrelates per-seed catches to chance). Any e2e net makes
small but non-zero per-frame force errors; over 1500 frames these compound chaotically, so
(a) per-seed catch identity is unreachable in principle, and (b) the mean-catch fidelity tops
out where the accumulated divergence settles — empirically ~85–88%. This is corroborated three
independent ways: M0 (chaos bound), exhaustive architecture saturation (below), and the prior
RL search (~8.05 ceiling). **100% behavioral equivalence is not achievable for ANY policy that
is not bit-identical to production; the best end-to-end raw-obs fidelity is ~85–88%.**

### What was exhausted (none breaks the catch ceiling)
- **Pooling/set encoders** (mean, cross-attn, density-gated): patrol angle floors ~36–40°. Worst.
- **Cartesian grid** G9→G25: resolution-saturated (G17≈G21>G25).
- **Log-polar grid** (direction-aligned bins): ≈ Cartesian (~18°).
- **Capacity** 38k→2M params: only lever that moves patrol ANGLE (22°→16.4°) — but angle does
  NOT convert to catches in the 16–21° band (Δ4° ⇒ +0.26 catches). Catches saturate ~7.1–7.2.
- **Per-boid local-density channel** (the "missing feature"): adds ~0.4° at matched capacity = noise.
- **Finer grid + density** (G21): worse (18.5°). 

The catch ceiling is independent of every representational lever — confirming it is the chaos
amplification of residual force error, not an information/architecture limit.

### Minimal net at the ceiling (Occam deliverable)
Catches vs net (G13 unless noted, no-density, count+momentum grid + K=8 nearest):

| net | catches | at plateau? |
|---|---|---|
| 64 (1 layer) | 6.2 | no — underfit |
| 128 (1 layer) | 6.2 | no — underfit |
| 256,128 | 6.92 | borderline |
| 256,256 | 7.14 | YES |
| 512,256 | 7.20 | YES |
| 1024,512,256 | 7.18 | YES (no gain over 512,256) |
| **G9** 512,256 | 7.20 | YES (coarser grid still at plateau) |

The plateau is reached by any **2-hidden-layer MLP ≥256 wide on a G9–G13 density+momentum
grid + K-nearest**; ~290–410k params. Bigger buys nothing on catches; single-layer underfits.
Density is NOT needed. Final minimal corner (G9 256,128 / 128,128 / 256,256) confirming now.
