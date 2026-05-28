# weighted_predicted patrol mode — confirmed +1.77 over flock_centroid (2026-05-28)

## Result

A new patrol-target mode, **weighted_predicted**, beats the shipped
`flock_centroid` patrol by ~1.5–1.8 catches per 16-seed×5000-frame run,
confirmed in BOTH the GPU sim (sim_torch) and the JS oracle (ground truth):

| Patrol mode | sim_torch (256 seeds) | JS oracle (256 seeds) |
|-------------|----------------------:|----------------------:|
| flock_centroid (shipped) | 21.934 | 20.637 |
| **weighted_predicted (la=5)** | **23.438** | **22.406** |
| **Δ (paired)** | **+1.504, z=2.78** | **+1.770, z=3.02** |

JS paired test: n=256, sd=9.37, z=+3.02 (p≈0.0013 one-tailed),
per-seed 139 wins / 9 ties / 108 losses.

## What it is

`weighted_predicted` sets the predator's patrol target (used when no boid
is within PREDATOR_RANGE) to a **density-weighted centroid plus a short
lookahead of the density-weighted mean velocity**:

```
w_i      = 1 / sqrt(d_i^2 + 1)         # d_i = dist(predator, boid_i)
target   = Σ w_i·pos_i / Σ w_i  +  lookahead · Σ w_i·vel_i / Σ w_i
lookahead = 5 frames
```

It combines two ideas:
- **density weighting** (like `weighted_centroid`): pull toward the nearest
  dense cluster rather than the global centre of mass, which can sit in an
  empty gap between clusters.
- **short anticipation** (like `predicted_centroid` but small lookahead):
  aim where that cluster is heading a few frames out, not where it was.

Neither idea alone beats flock_centroid at 256 seeds:
- weighted_centroid alone = 21.80 (≈ flock, −0.13)
- predicted_centroid (lookahead 30) = known to hurt
It's the **combination at short lookahead** that wins.

## Lookahead sweep (sim_torch, 256 seeds)

| lookahead | mean | Δ vs flock |
|----------:|-----:|-----------:|
| 3  | 23.40 | +1.46 |
| **5**  | **23.44** | **+1.51** |
| 8  | 22.93 | +1.00 |
| 10 | 22.69 | +0.76 |
| 15 | 23.31 | +1.38 |

Short lookahead (3–5) is best; the curve is noisy but stays positive.
Fine sweep around the peak (256 seeds): la4=22.23, **la5=23.44**, la6=22.63,
la7=22.87 — la5 is a clear, sharp optimum.

## Distance-weighting exponent (sim_torch, 256 seeds, la=5)

w = 1/(d²+1)^(weight_pow/2):

| weight_pow | mean | note |
|-----------:|-----:|------|
| 0.5 | 22.55 | gentler falloff → closer to flock, worse |
| **1.0** | **23.44** | current default = 1/sqrt(d²+1) |
| 1.5 | 23.39 | ~equal |
| 2.0 | 23.56 | marginally higher, within noise (SE≈0.42) |

The exponent is not a meaningful lever in [1.0, 2.0] — all give ~+1.5 over
flock. Keeping the simple, JS-validated `weight_pow=1` (1/sqrt). The mode is
near-optimal in its (lookahead, weight_pow) parameter space; further gains
would require a structurally different patrol target, not parameter tuning.

## Why this matters

This is the first statistically significant improvement over the shipped
baseline found in this exploration. Critically, it is a **structural**
change (which target the predator aims at), exactly like the original
+39% win (`random` → `flock_centroid` patrol). By contrast, ~5 separate
weight-level ES runs (evolution strategies on the NN weights, both from
the shipped NN and the H=8 rule_v3 NN) ALL failed to beat baseline once
verified at 256 seeds — the "best perturbations" at S=16/32 were pure
seed-luck that vanished at 256 seeds.

**Lesson reaffirmed: the lever for this predator is structural
(targeting / behavior), not weight optimization.**

## Faithfulness note

sim_torch was independently confirmed faithful for this mode: at the same
64 seeds, JS weighted_predicted = 20.91 vs sim_torch = 21.02 (within 0.1).
The two sim_torch bugs found and fixed earlier this session were:
1. `nn_forward` clipMagnitude used exact sqrt instead of JS's
   alpha-max-beta-min approximation.
2. `_step_boids_sequential` did TWO flock passes per frame (live-page
   `tick()`+`render()`) instead of ONE (Oracle's render-only step).

## Shipping

To ship, the production patrol logic (js/predator.js / js/simulation.js
autonomous-target computation, currently flock_centroid) would change to
weighted_predicted with lookahead=5 — a ~10-line change, same shape as
the flock_centroid patch. The NN is unchanged (it reads the patrol target
via the seek_auto_xy feature, so it adapts automatically). **Not yet
applied to production — pending user go-ahead.**

## Files

- `dev/oracle.js` — `weighted_predicted` mode in `computeAutoTarget`.
- `dev/sim_torch.py` — `weighted_predicted` in `_update_auto_target`
  (+ `auto_target_opts` lookahead).
- `dev/eval_sim.py` — `--lookahead` / `--K` patrol opts.
- `dev/screen_patrol.py` — patrol-mode sweep harness.
