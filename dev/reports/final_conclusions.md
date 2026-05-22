# Predator-NN RL exploration — final conclusions (2026-05-22)

## TL;DR

**+39% improvement (flock_centroid patrol) is the achievable ceiling for
this problem under the current sim/architecture/feature space.** All
attempts to improve beyond shipped's 24.25 catches in JS have failed.

## Result summary

| approach | best catches in JS | vs shipped |
|----------|--------------------|----|
| **Shipped (NN + flock_centroid)** | **24.25** | — |
| Original baseline (random patrol) | 17.44 | -28% |
| K=20 supervised random-init seeds | 22-24 | ≤0 |
| ES σ=0.05 on shipped | nothing accepted | ≤0 |
| ES σ=0.10 on shipped (20 tries) | nothing accepted | ≤0 |
| ES σ=0.20 on shipped (20 tries) | nothing accepted | ≤0 |
| ES σ=0.30 on shipped (20 tries) | nothing accepted | ≤0 |
| H=4 supervised (rule + flock data) | 21.4 | -2.8 |
| H=8 supervised (rule + random data) | 20.0 | -4.3 |
| H=64 supervised (rule + random data) | 20.56 | -3.7 |
| rule_v2 α=5 (predicted target) | 23.3 | -1 |
| rule_v2 α=8 + NN distill | 22.6 | -1.6 |
| H=4 + lookahead N=8/12 features | 24.5 ≤ 25.3 (z≤0.4) | ~0 |
| Set Transformer (47k params, pretrained) | 18.6 (sim_torch) | — |
| Set Transformer (177k params, pretrained) | 17.0 (sim_torch) | — |
| Sim_torch ES (40+ gens, MLP+SetXF) | doesn't transfer to JS | ≤0 |

## What we learned

1. **The structural fix dwarfed everything else.** The 5-line change to
   `js/predator.js` (random patrol target → flock centroid) gave
   +6.81 catches per seed. No subsequent weight-level optimization
   found more than ~1 catch of additional improvement, and even those
   collapsed in 16-seed verification.

2. **Sim_torch ≠ JS.** The vectorized NumPy/PyTorch port runs the
   boid update in parallel; the JS sim runs it sequentially. This
   difference makes sim_torch a poor proxy for JS rank-ordering of
   candidate policies — rank correlation ρ≈0.17 across K=10 random NNs.
   ES trained on sim_torch finds policies that look good in sim_torch
   but regress in JS (e.g., VM2 sim_torch 10.84 catches → JS 18.4,
   below supervised init's 20.6).

3. **Shipped is at a hard local optimum.** ES at four different sigma
   values in JS (the ground truth env) found NO improvement across 60+
   tries. Closest candidates were Δ ≈ -2 to -3.5 catches (z ≈ -1 to -2
   below shipped). The neighbourhood is genuinely worse-on-average in
   every direction we probed.

4. **Random-init supervised NNs cluster below shipped.** K=20 random
   seeds trained against the rule produced JS catches 22.3 → 26.1 on
   the 8-seed filter (seed 6 looked best) but regressed to 22.56 on
   16-seed verification. None statistically beat shipped (z=-0.86).

5. **Set Transformer doesn't help here.** Permutation-invariant
   attention is theoretically the right inductive bias for boid sets,
   but it's hard to train from random init via ES (stuck at ~4
   catches), and supervised pretrain against the rule only mimics
   imperfectly (val_mse ~5e-4, catches 17–18 in sim_torch). ES on
   pretrained init oscillates and drifts down. ~12 hours of GPU time
   on 2 VMs produced no policy better than the pretrained baseline.

## Branches and shipping state

- **`main`** (`dd2bbd5`): original homepage, random-patrol predator.
- **`rl/auto-target-v2`** (`274844c`): ships the +39% flock_centroid
  patch in `js/predator.js`. NN weights unchanged. *This is the
  recommended ship target.*
- **`rl/sigma-0.05-flock`** (`90aed63`): +39% baseline + ES null result.
- **`rl/lookahead`** (`...`): all the lookahead/rule_v2 work.
- **`rl/teacher`** (`...`): GPU ES, supervised pretrain, Set Transformer.

## What would actually help (out of current scope)

1. Sequential boid update in sim_torch — would make GPU ES transfer to
   JS. Substantial port effort.
2. Game-design changes (smaller boid avoidance, shorter feed cooldown,
   bigger catch radius) — would bump scores without policy change.
3. Replacing the rule entirely with planning / search.
4. Adding stochasticity to the policy + risk-shaped objectives.

## Recommendation

Ship the **`rl/auto-target-v2`** branch as the production update. It's
a 5-line change to `js/predator.js` that delivers +39% (z=3.55 train,
z=3.59 holdout) over the prior baseline with no architectural risk
and no weight retraining required. The shipped NN weights remain
identical.
