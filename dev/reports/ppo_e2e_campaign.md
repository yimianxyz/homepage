# End-to-end / PPO predator-policy campaign

Goal: replace or improve on the hand-crafted **nearest_cluster** patrol policy
(deployed, ~7.6 catches @1500 frames, 64-seed holdout) with a learned steering
policy — ideally end-to-end. Constraint: only the predator policy may change;
the boid policy is fixed. Eval is GPU-only (sim_torch), gated on held-out seeds.

## Dynamics facts (so the numbers make sense)

- Predator MAX_SPEED 2.5, boids 6.0 — the predator is **slower** than its prey.
- Steering per frame is capped to **0.05** (`clipMagnitude` in the deployed
  weights; `PREDATOR_MAX_FORCE=0.05` is a dead constant, the cap is enforced by
  the NN's clipMagnitude). So the predator turns slowly: ~50 frames to reach max
  speed, ~100 to reverse. This sluggishness is shared by every policy and is the
  core difficulty — catching requires anticipation, not raw chase.
- Catch radius ≈ size×0.7 ≈ 8.4; feed cooldown 100 ms ≈ 8 frames (a hidden state
  the egocentric obs does not expose).

## What was tried, in order

| Approach | Obs | Method | Holdout @1500f | Verdict |
|---|---|---|---|---|
| nearest_cluster (deployed) | hand-crafted | — | ~7.6 | baseline to beat |
| weight-ES on distilled net | 35-d production | ARS-V1 | no gain | dead end (5+ runs) |
| e2e from random | polar grid 75 | ES | central ~5.4 | sample-inefficient |
| e2e from random | polar grid 75 | PPO | best ~6.0, plateau | local optimum |
| e2e augmented | grid + cluster/nearest 83 | PPO | best ~5.6–5.8, plateau | **target-as-feature did not help** |
| **residual** | grid+cluster/nearest 83 | PPO on top of deployed | **7.86 ± 0.18** | = base, no gain |
| residual+cooldown | + cooldown 84 | PPO on top of deployed | **7.70–7.74** | = base, no gain |
| from-scratch+shape | grid 75 | PPO + dense shaping | in progress | testing sparse-reward |
| target-residual | grid+cluster/nearest 83 | PPO offsets aim, base steers | in progress | testing aim-control |

### Decisive 512-seed gates (seedStart 5000, 1500f)

The 64-seed training holdout has SE ≈ 0.4, so the **max** over ~20 eval points is
inflated ~1.5–2 SE by selection bias — "best holdout 8.5–9.3" was a mirage. Gating
the saved best.pt at 512 seeds (SE ≈ 0.17) against the same policy with
`--resid_scale 0` (= exact base) is the honest test:

| policy | base @512 | trained @512 | Δ |
|---|---|---|---|
| residual (scale 0.05) | 7.838 | 7.859 | +0.02 |
| residual+cooldown (VM1, 0.05) | 7.838 | 7.697 | −0.14 |
| residual+cooldown (VM2, 0.08) | 7.838 | 7.738 | −0.10 |

Every learned policy converges to ≈ base (the residual learns ≈0). Giving it the
cluster target, the nearest boid, and the feed-cooldown state — info the base
ignores — produced **no** improvement.

### Key negative result

From-scratch learned steering — ES or PPO, raw grid or augmented with the exact
nearest_cluster target handed to it as a feature — **plateaus at ~5.5–6.0**,
well below the hand-crafted seek law's 7.6. Giving the policy the cluster target
for free did **not** close the gap, so the bottleneck is not *where to go* but
*how to steer/pursue* under the sluggish dynamics. PPO consistently settles into
a worse local optimum than the hand-crafted controller.

### Residual RL (current)

Rather than relearn pursuit from scratch, ride on the deployed policy and learn
only a correction:

    total_steer = clip( base_production_steer + resid_scale · π(obs), 0.05 )

With an untrained policy (π≈0) this reproduces the deployed controller exactly —
**confirmed: iter-0 holdout ≈ 7.5–8.7**. PPO can then only redirect within the
0.05 force budget, so the floor is the champion and any learning is upside.
Three configs running (resid_scale ∈ {0.03, 0.05, 0.08}, lr ∈ {2e-4, 3e-4}).

## Infra notes

- `sim_torch` parallel + CUDA-graph is the eval/vec-env; faithful to JS at 256
  seeds (Spearman 1.0 vs sequential Oracle).
- VM background launches must detach via setsid+nohup+**disown**; inline
  `pkill -f e2e_ppo` self-matches the launcher shell — use `scripts/launch_ppo.sh`.
