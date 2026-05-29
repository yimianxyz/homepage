# nearest_cluster patrol — densest-cluster targeting (2026-05-29)

## Result (DEPLOYED to production)

A GPU-searched patrol policy that aims at the **centroid of the densest local
cluster** of boids — led forward by the predator's own travel time — beats the
previous flock-centroid patrol by **~+41% catches/eval**, validated on fresh
held-out seeds never used during tuning.

GPU sim (sim_torch, 512 seeds 5000–5511, paired vs flock_centroid):

| patrol policy | catches | vs flock |
|---|---:|---:|
| flock_centroid (original +39% fix) | 21.75 | — |
| weighted_predicted (prev deploy) | 22.36 | +2.8% |
| **nearest_cluster (deployed)** | **30.68** | **+41.0% (z=22.9)** |

NN steering weights are unchanged throughout — every patrol policy here only
changes the target the net reads via its `seek_auto_xy` feature.

## The policy

When no boid is within hunting range (PREDATOR_RANGE=80):

```
densest = boid with the most live neighbors within CLUSTER_R (=150px)
cluster = densest + all its neighbors within CLUSTER_R
C  = centroid(cluster)            # uniform mean of cluster member positions
Vc = mean velocity(cluster)
dcent = ||C - predator||
lead  = min(dcent / PREDATOR_MAX_SPEED * LEAD_SCALE, LEAD_MAX)   # frames
target = C + lead * Vc
LEAD_SCALE = 0.4, LEAD_MAX = 120, PREDATOR_MAX_SPEED = 2.5
```

Two ideas combine:
1. **Densest-cluster anchoring.** Aiming at the global centre of mass is bad —
   it lands in the empty gap between separate clusters. Anchoring on the boid
   with the most neighbors and taking that neighborhood's centroid keeps the
   target on the dense core. (Head-to-head: a global uniform centroid with the
   same adaptive lead scored 22.6 vs 25–30 for the cluster centroid.)
2. **Adaptive travel-time lead.** The predator is slower than the boids
   (max 2.5 vs 6.0), so the right amount to lead depends on how far it must
   travel. Leading by `dist/speed` (scaled, capped) beats any fixed lookahead.

## How it was found (GPU coordinate-ascent, ~12 rounds)

Lineage of structural wins, each held-out confirmed:
- random patrol → flock_centroid: +39% (pre-existing)
- flock → weighted_predicted (density-weighted centroid + fixed 5-frame lookahead): +1.8
- → adaptive travel-time lead: +1.4
- → densest-cluster anchoring (nearest_cluster): +1.9 (r80), rising with radius
- joint re-tuning of (radius, lead_scale, lead_max) after each coupled change

Converged optima (each a broad plateau; jagged points = per-sample noise):
- cluster_r: 120–200 flat (~30); declines past 250 (→ global centroid, bad)
- lead_scale: 0.4 peak (coupled with lead_max)
- lead_max: 120–140 peak; sharp — uncapped (∞) collapses to 21.6 (overshoot)
- centroid weighting within cluster: uniform (density-weighting no better)

## Key lessons reaffirmed

- **The lever is the predator's targeting structure, not its weights.** Five ES
  runs on the NN weights all failed to beat baseline; the entire >40% gain came
  from where the predator aims.
- **GPU eval (sim_torch) is faithful** (matched JS at 256 seeds for shipped, and
  at 64 seeds for weighted_predicted), so the whole search ran on GPU at ~5 min
  /256-seed-eval instead of ~70 min in JS — ~14× faster, enabling ~12 rounds.
- **Hold out fresh seeds.** Rounds 4–11 reused seeds 3000–3511, so the champion
  was re-validated on never-seen seeds 5000–5511 (+41%, z=22.9) to rule out
  overfitting to the tuning set.

## Deploy

- `js/predator.js` `getAutonomousForce`: patrol target = nearest_cluster.
- `dev/oracle.js`: matching `nearest_cluster` in `computeAutoTarget`.
- `dev/sim_torch.py`: `nearest_cluster` in `_update_auto_target` (+ cluster_r /
  lead_scale / lead_max / centroid_pow opts; `eval_sim.py` CLI flags).
- Production O(N²) cluster scan (120²) runs only on patrol frames — negligible
  in-browser at 120 boids; headless check showed 52–55 catches/45s, no errors.
- Shipped to `main` (commit 3ad9cb1); GitHub Pages built OK → live.

## Distillation note

As with weighted_predicted, "distilling into a NN" cannot improve this: the
steering net is unchanged, and the densest-cluster target needs global density
(all 120 boids) which a K=4-nearest-boid net cannot compute internally — so the
target stays a JS computation feeding seek_auto_xy. A fresh NN distilled from
the rule reconstructs it and loses the shipped net's generalization bonus
(confirmed previously). Distillation cross-check on nearest_cluster data was run
for parity; deployed artifact = shipped NN + nearest_cluster patrol.
