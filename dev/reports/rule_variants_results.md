# Rule-variant structural exploration (2026-05-23)

## Motivation

The +39% improvement that the shipped predator has over the original
baseline came from a **structural change to PATROL** (flock_centroid
target instead of random points), not from any weight-level training.
We tried analogous structural changes to the **ATTACK** branch (which
boid to chase, and where to aim) hoping for similar wins.

## Variants tested

All evaluated on 16 seeds × 5000 frames with `flock_centroid` patrol.

| Variant | What it does |
|---------|--------------|
| `rule_v1` | Original: chase nearest boid within PREDATOR_RANGE; else patrol. |
| `rule_v2(α)` | Same target, but aim α frames ahead via `dx + α·(bvx - vx)`. |
| `rule_v3(mode, distW, α)` | Pick best of K=4 nearest within range by `score`. Modes: |
|         | `score_minus_dist`: closing_speed − distW · d |
|         | `closing_only`:     closing_speed |
|         | `time_to_catch`:    −d / max(closing, ε) |
|         | Optional α-lookahead on each candidate's position. |
| `rule_v4` | Solve perfect-intercept quadratic for each candidate; pick smallest t. |
| `rule_v5(T)` | Multi-step prediction with boid AVOIDANCE acceleration accounted for. |

## sim_torch sequential+graph results (16 seeds × 5000 frames)

Reference: **shipped NN sim_torch = 22.94**.

| Variant | sim_torch mean | Δ vs shipped |
|---------|---------------:|-------------:|
| `v3_smd_a5_w05`  | **23.5625** | **+0.63** ✓ |
| `v5_T8`          | **23.4375** | **+0.50** ✓ |
| shipped NN       | 22.94       | baseline |
| `v3_tte_a5`      | 22.75       | −0.19 |
| `v3_tte_a0`      | 22.125      | −0.82 |
| `v5_T5`          | 21.9375     | −1.01 |
| `v5_T5_w005`     | 21.9375     | −1.01 |
| `v1`             | 21.25       | −1.69 |
| `v4` (intercept) | 21.125      | −1.82 |
| `v2_a5`          | 20.5        | −2.44 |
| `v3_smd_a0_w05`  | 19.9375     | −3.00 |
| `v3_closing_a5`  | 19.8125     | −3.13 |
| `v5_T12`         | 19.5        | −3.44 |
| `v3_closing_a0`  | 18.8125     | −4.13 |
| `v2_a8`          | 18.625      | −4.32 |
| `v3_smd_a8_w05`  | 17.8125     | −5.13 |
| `v5_T2`          | 15.4375     | −7.50 |

Two sim_torch winners: `v3_smd_a5_w05` (+0.63) and `v5_T8` (+0.50).

## JS verification (16 seeds × 5000 frames, flock_centroid)

| Variant | JS mean | Δ vs shipped (24.25) |
|---------|--------:|---------------------:|
| shipped NN (reference) | 24.25 | — |
| `rule_v3_smd_a5_w05`   | **23.4** | **−0.85** |
| `rule_v2_a5` (prior)   | 23.25 | −1.00 |
| `rule_v1` + flock_centroid (prior) | ~22 | ~−2.25 |
| `rule_v5_T8` | **19.8** | **−4.45** — sim_torch artifact (forward-sim model error) |

`v3_smd_a5` gives the smart-target structural change a clean test:
+0.15 catches over `rule_v2_a5` in JS (23.4 vs 23.25), confirming
that picking-by-closing-speed beats picking-by-nearest at the
margin. But the gap to shipped NN (−0.85) is the same ~1-catch gap
between the rule and its NN-distilled version that we've seen all
along — the NN learns useful smoothing that no simple rule captures.

## What this tells us

1. **The "smart target selection" structural fix is real but small.**
   Picking by closing speed instead of nearest improves rule_v2 by
   ~0.15 catches in JS — marginal, not the +6.81 we got from
   `flock_centroid` patrol.

2. **`sim_torch` is a noisy proxy for the rule comparison.**
   `v3_smd_a5` looked +0.63 over shipped in sim_torch but JS-verified
   to −0.85. The rank-order is broadly preserved (v3 > v1 in both)
   but absolute magnitudes diverge.

3. **The NN-distillation bonus dominates the structural gain.** Both
   `rule_v2_a5` and `rule_v3_smd_a5` are below their NN-distilled
   versions by ~1 catch. The NN smooths the rule's discrete branch
   decisions in ways no closed-form rule captures.

4. **Lookahead α=5 is the sweet spot.** α=8 overshoots in every
   variant (v2 and v3 both degrade); α=0 misses the predator's own
   forward motion. This α=5 finding now also holds for the smart-
   target case, confirming the prior `rule_v2 α=5 best` result.

## Files

- `dev/policy_spec.js` — `rulePolicy_v3`, `rulePolicy_v4`, `rulePolicy_v5`.
- `dev/eval_tte.js` — `--policy rule_v3 [--mode … --distW … --alpha …]`,
  `--policy rule_v4`, `--policy rule_v5 [--steps …]`.
- `dev/rule_torch.py` — sim_torch Python ports (graph-safe with
  pre-allocated buffers via `make_rule_buffers`).
- `dev/eval_rule_torch.py` — GPU rule eval harness.

## Next direction

Train a fresh NN on the `rule_v3_smd_a5` dataset. The NN smoothing
bonus that lifted rule_v2 (23.25) to shipped (24.25) should also lift
rule_v3_smd_a5 (23.4) to ~24.4 if the bonus is universal — that would
finally beat the +39% ceiling.
