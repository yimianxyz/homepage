# RL improvement research — running plan

## Current state

- **Baseline**: NN + flock_centroid patrol = **24.25 catches** / 16 seeds × 5000 frames
- Shipped to js/predator.js on branch `rl/auto-target-v2` (pushed to origin).
- Eval framework: `dev/eval_tte.js` is the single source of truth.

## Progress trail (current branch `rl/sigma-0.05-flock`)

| step                             | catches | Δ vs prior  | z     |
|----------------------------------|---------|-------------|-------|
| Random baseline (null policy)    | ~6      | -           | -     |
| Distilled NN (random patrol)     | 17.44   | (baseline)  | -     |
| ES σ=0.05 (8 seeds, 1·SE)        | 17.5    | NOISE       | 0.89  |
| flock_centroid patrol            | **24.25** | **+6.81** | **3.55** |
| weighted_centroid                | 24.81   | +0.56       | 0.52  |

## In-flight

**ES σ=0.05 on flock_centroid baseline** (~4hr). Hill climb 20 tries
at z>2 threshold. Tests whether weight-space search can find further
wins on top of the structural patrol fix.

## Queue (ordered by EV / effort)

1. **Smarter hunt rule + redistill**. The rule currently steers to
   nearest-boid.position. Modify to steer to predicted position
   (position + α × velocity), redistill NN against it. The rule
   doesn't use velocity today; teaching it to does adds anticipation
   to the hunt branch (where flock_centroid only helped the patrol
   branch). Effort: ~1 day. Likely lift: 10–20%.

2. **Feature expansion: boid velocities**. Append (vx, vy) of K=4
   nearest boids to FEATURE_DIM (35→43). Retrain via supervised
   distillation against a velocity-aware rule, OR pure RL from the
   current weights with the new feature weights init to 0. Lets the
   network learn anticipation directly, not via rule. Effort: ~1 day.
   Likely lift: 15–30%.

3. **Architecture: H=4 → H=8**. Tiny network has ceiling on what it
   can express beyond the rule. Retrain w/ supervised distill first
   (loss should remain ~1e-13), then ES on the larger param space
   (~300 params). Effort: ~half day. Likely lift: 5–15%.

4. **Hyperparameter R sweep**. POLICY_R=80 is the hunt-radius
   threshold; if R is wrong (too small/too large), the patrol/hunt
   branch is misaligned. Sweep R∈{60,80,100,120} would need feature
   pipeline retuning. Effort: ~1 day. Likely lift: 5–10%.

5. **Multi-step planning**. Predator looks ahead N=3 frames using a
   learned dynamics model. Big change. Effort: ~3 days. Likely
   lift: 30%+.

## Things explicitly tried and rejected

- σ=0.05 hill climb (8 seeds, 1·SE accept): false positives only.
- predicted_centroid (centroid + 30·mean_velocity): worse than
  flock_centroid. Lookahead overshoots — boids flock with high
  alignment but low net translation.
- nearest_boid patrol: +22.9% but loses head-to-head to centroid.
- farthest_in_K patrol: +27.6%; loses to centroid.
- rule policy + flock_centroid: rule's bang-bang overshoots centroid;
  NN's softer response beats it. So distilling more aggressively
  isn't the right move — current distillation has useful smoothing.
