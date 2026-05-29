# Evolved patrol target — AlphaEvolve-style GPU search

## Result

A GPU-batched evolutionary search over a parametric patrol-target program found
a policy that **beats the deployed `nearest_cluster` patrol by ~+0.58
catches/eval (+7.4%)**, confirmed across two independent 2048-seed held-out
blocks:

| config | block 50000 | block 80000 | avg |
|--------|------------:|------------:|----:|
| **evolved (E3D, shipped)** | 8.376 ± 0.092 | 8.416 ± 0.092 | **8.40** |
| `nearest_cluster` (previous) | 7.856 ± 0.088 | 7.777 ± 0.086 | 7.82 |

Δ = +0.52 (4.1 SE) on block 50000, +0.64 (5.1 SE) on block 80000. Decisive and
reproducible. This is the first config in the whole predator campaign to clearly
beat the deployed policy out-of-sample.

Shipped config (`EVOLVED_PATROL` in `js/predator.js`):

```
cluster_r 178.09  dens_pow 2.373  reach_scale 1515  sharp 9.25
lead_scale 0.454  lead_max 230.6  nbhd 0.461
```

## The mechanism

Only the PATROL target changed; the chase is still the trained NN. Every live
boid `i` gets an attractiveness

    attract_i = (neighbors_within_cluster_r + 1)^dens_pow * exp(-dist_pred_i / reach_scale)

normalized to the per-frame max and raised to `sharp` (a soft argmax). The
target is the attract-weighted centroid + mean velocity, **blended (`nbhd`)
toward the densest boid's neighborhood centroid** (the key new lever — a
discrete select that stops the smooth centroid from landing in the empty gap
between two clusters), led forward by the predator's travel time
(`dist/PRED_MAX_SPEED * lead_scale`, capped at `lead_max`).

`reach_scale` is the other structural axis: the predator is slow, so a closer,
slightly-less-dense cluster can beat a distant denser one. `nbhd` was actively
used by every winning island (~0.4–0.5); the earlier `momentum` lever evolved to
~0 (dead) and was dropped from the search.

## How the search ran

- **Eval**: `dev/sim_torch.py` — full boid+predator sim, GPU-batched, fp64 for
  JS fidelity, CUDA-graph replay. 2048 envs × 1500 frames ≈ 150 s on one L4.
- **Optimizer**: `dev/evolve_patrol.py` — self-contained numpy CEM (elite
  ranking, robust to seed noise; PyPI unavailable on the VMs). Per-generation
  seed resampling (kills seed-overfitting) + a fixed held-out MU-validation
  block.
- **Islands**: 3 L4 GPUs, independent seeds/start points (AlphaEvolve flavor —
  new mechanisms added to the `evolved` branch between rounds). All converged to
  the same `nbhd`-augmented family at ~8.2–8.4, strong evidence of a real
  ceiling for this mechanism class.
- **Gate**: `dev/gate_multi.py` re-scores top held-out configs at 2048 seeds on
  *fresh* large blocks. The 512-seed CEM validation overfit its own block
  (ev2c/ev2d looked best on 50000 but regressed on 80000); the multi-block gate
  picked the config that generalizes — E3D.

## JS deployment + parity

`js/predator.js` `computeEvolvedTarget()` is a line-for-line port of the
`sim_torch` `evolved` branch. Verified by `dev/parity_dump.py` (dumps real
mid-rollout states + the target sim_torch computes) → `dev/check_parity.js`
(recomputes in JS): **max abs error 1.4e-12 over 75 patrol-mode cases**. The
port is numerically identical to the searched policy.

## Convergence — search is done (2026-05-29)

Per the directive ("stop only on convergence, 100% confident it will never
increase") the search ran four further independent islands to completion and
gated each island's held-out best at 2048 seeds on the same two fresh blocks
(50000 / 80000). All converged into E3D's basin; **none beat E3D**:

| config | mechanism | block 50000 | block 80000 | gated avg |
|--------|-----------|------------:|------------:|----------:|
| **E3D (shipped)** | nbhd 0.46, sharp 9.25 | 8.376 | 8.416 | **8.396** |
| ev3d | nbhd **0.69** (deliberately distinct) | 8.325 | 8.387 | 8.356 |
| ev_refine | E3D-seeded, drifted to sharp 13 | 8.307 | 8.209 | 8.258 |
| ev2d | converged back to E3D-adjacent | — | — | ~8.33 |
| `nearest_cluster` | baseline | 7.856 | 7.777 | 7.817 |

The decisive evidence: islands started from **diverse and deliberately
different** mechanisms — broad random starts, a high-neighborhood-blend variant
(nbhd 0.69 vs E3D's 0.46), and a fine refinement seeded directly on E3D — every
one tops out at ~8.3–8.4 and none exceeds E3D. The headline held-out values
these islands reported (ev3d 8.81, ev_refine 8.71) were the **same ~0.4–0.5
small-block (512-seed) overfit** seen throughout; at 2048 seeds they collapse
back to the basin. This is a hard ceiling for the parametric-patrol mechanism
class. **E3D (8.40, +7.4% over NC) is the confirmed best and final config.**

Beating it further would require a different *class* of mechanism (e.g.
multi-step anticipation / planning), which is the same conclusion the RL
campaign reached and is not browser-deployable. Recommendation: ship E3D.
