# Predator-NN RL improvement — research trail summary

## Final result so far

**NN + flock_centroid patrol = 24.25 catches** (vs original baseline 17.44).

- Improvement: **+6.81 catches per seed (+39%, z=3.55)**
- Holdout (seeds 200..215): **+5.75 (+36%, z=3.59)** — generalizes.
- Shipped to `js/predator.js` (5-line patch); NN weights unchanged.

## Method trail

| step | catches | comments | result |
|------|---------|----------|--------|
| Random null policy | ~6 | upper bound for "no policy" | baseline |
| Distilled NN, random patrol | 17.44 | shipped weights, original deployment | initial baseline |
| ES σ=0.05 (8 seeds, 1·SE) | 17.5 | try-1 looked +17%, verify@16 said noise | failed |
| Eval framework `dev/eval_tte.js` | - | single source of truth; 16-seed, 5000-frame | infrastructure |
| **flock_centroid patrol** | **24.25** | structural fix to autoTarget logic | **SHIPPED** |
| nearest_boid patrol | 21.44 | beat random but lost to centroid | rejected |
| farthest_in_K patrol | 22.25 | speculative, worse than centroid | rejected |
| predicted_centroid patrol | 21.38 | lookahead overshoots flock | rejected |
| weighted_centroid patrol | 24.81 | tied with flock (z=0.52), more code | rejected |
| ES σ=0.05 on flock baseline | 24.25 | all 20 tries strictly worse than baseline | confirmed local opt |
| Feature lookahead N∈{1..12} | ≤25.25 | NN can't interpret without retrain | rejected |
| Feature: velocity slots 35..42 | - | append-only addition, backward compat | infrastructure |
| rule_v2 α-sweep (α∈{1..8}) | ≤23.4 | best α=5..8 only marginal vs rule | partial |
| v4: H=4 + rule_v2 α=8 distill | 19.4 | H=4 can't fit rule_v2 (val_loss 0.0345) | failed |
| v5: H=4 + rule + flock data | 21.4 | perfect fit but loses lucky errors | failed |
| Feature: seek_boid_v2 slot 43,44 | - | precomputed rule_v2 hunt for trivial fit | infrastructure |
| v6: H=4 + rule_v2 α=8 + seek_v2 | 22.6 | val_loss 5.9e-10, ≈ rule_v2 catches | failed |
| Seed search K=10 (rule data) | 22.6 (top) | seed=6 looked +1.25 on 8 seeds, regressed on 16 | failed |

## What we learned

1. **Structural ≫ weights.** A 5-line change to the patrol-target
   logic dwarfed everything we could find in weight space. The
   flock-centroid lift (+6.81) is bigger than any improvement ES,
   retraining, or seed search produced on top of it (all in the
   noise).

2. **Eval noise is the bottleneck.** Per-seed catch std ≈ 6, so even
   with 16 seeds the SE on the mean is ~1.5. We can only reliably
   detect Δ ≥ 3 catches/seed (≥12% improvement). Many candidates that
   *look* better are just sampling noise.

3. **The shipped NN is "lucky."** Its 24.25 score under flock_centroid
   is the result of distilling against rule + random-patrol data,
   then deploying under flock_centroid. The feature-distribution
   mismatch creates residual NN errors that *help*. Retraining to
   match deployment removes the luck (v5: 21.4 ≈ rule). Random
   re-inits mostly land below shipped (only 1/10 above on filter,
   regressed on full eval).

4. **Smarter rules don't auto-win.** rule_v2 with velocity-aware
   prediction (α=8) only gets 23.4 — *below* the shipped NN's lucky
   24.25. Perfect distillation of rule_v2 produces 22.6, still below.

5. **Capacity isn't the bottleneck for the rule.** H=4 fits rule and
   rule_v2 (with seek_boid_v2 feature) to val_loss ~5e-10. Bigger H
   has no supervised slack to exploit.

## What's not tried yet

- **Mixed-mode distillation**: train on a union of (rule + random
  patrol) and (rule + flock patrol) so the NN learns both regimes.
- **Direct policy gradient RL**: skip the rule altogether; reward =
  catches, optimize end-to-end. Likely needs the simulation ported
  to vectorized NumPy / PyTorch for batched rollouts.
- **Large seed search (K=50–100)**: lucky-init rate ≤10% observed at
  K=10; more samples could turn up bigger lifts at the cost of
  K×eval_time wall.
- **H=8 trained the same way as shipped**: started, in progress.
- **GPU-accelerated batched eval**: port the JS simulation to NumPy
  for parallel rollouts → much lower SE per wall minute → smaller
  detectable improvements.

## Branches

| branch | tip | summary |
|--------|-----|---------|
| `main` | dd2bbd5 | original homepage, original random-patrol predator |
| `rl/auto-target-v2` | 274844c | ships flock_centroid in `js/predator.js` |
| `rl/sigma-0.05-flock` | 90aed63 | the +39% baseline + ES null result |
| `rl/lookahead` | 29c0390 | everything since (features, rule_v2, seed search) |
