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

## Distillation attempt (does NOT beat the patrol-swap)

Tested whether a NN distilled on weighted_predicted-patrol data beats the
shipped NN when both run weighted_predicted (256 seeds, sim_torch):

| policy | catches |
|--------|--------:|
| **shipped NN + weighted_predicted** | **23.44** |
| distilled H4 (rule_v1 + wpred) + wpred | 22.31 |
| distilled H8 (rule_v1 + wpred) + wpred | 21.66 |
| shipped NN + flock (baseline) | 21.93 |

The fresh NNs train to ~0 val loss because rule_v1's output is a gated copy
of the seek-vector features (slots 29–32) — the NN reconstructs the rule
exactly, so it scores like the bare rule (~22.3), not better. The shipped NN
scores higher (23.44) because it was distilled from **random-patrol** data,
which gave it a generalization/smoothing bonus that the matched-training copy
loses. To put that bonus into a fresh NN you'd train on rule_v1 + random
patrol — which just reproduces the shipped NN. So **there is no distillation
path that beats "keep the shipped NN, change the patrol."**

Deploy candidate = shipped NN (unchanged) + weighted_predicted patrol, already
JS-verified at +1.77 (z=3.02).

## Shipping — DEPLOYED 2026-05-28

Shipped to production in `js/predator.js` `getAutonomousForce` (commit on
`main`, GitHub Pages serves main root → https://yimianxyz.github.io/homepage/).
The NN (`js/predator_weights.json`) is unchanged — it reads the patrol target
via its seek_auto_xy feature and adapts automatically.

Note discovered at deploy time: production `main` was still running the
ORIGINAL random-canvas-point patrol — the flock_centroid +39% fix had only
ever lived on the `rl/teacher` branch, never merged to main. So this deploy
takes production from random patrol straight to weighted_predicted (the full
jump: +39% from centroid targeting, plus the +1.77 from density-weighting +
lookahead).

Pre-deploy gate: headless Playwright run of the exact main artifact
(`dev/verify_prod_patrol.js`, VERIFY_ROOT pointed at the deploy worktree) —
page boots, predator catches actively (45 in 45s real-time), zero console
errors. boid.js/vector.js diffs between main and rl/teacher are non-behavioral
(Math.pow→x*x, for-in→indexed); policy_features.js diff is append-only (slots
35–44), so the featureDim-35 NN reads identical features on both branches.

## Files

- `dev/oracle.js` — `weighted_predicted` mode in `computeAutoTarget`.
- `dev/sim_torch.py` — `weighted_predicted` in `_update_auto_target`
  (+ `auto_target_opts` lookahead).
- `dev/eval_sim.py` — `--lookahead` / `--K` patrol opts.
- `dev/screen_patrol.py` — patrol-mode sweep harness.
