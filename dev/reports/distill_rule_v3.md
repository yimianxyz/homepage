# Distilling rule_v3 into NN (2026-05-23)

## What we found so far

Best non-RL: shipped NN @ 16-seed × 5000-frame JS eval = 24.25.

| Model | Train data | JS @ 16 seeds | JS @ 32 seeds | Notes |
|-------|------------|--------------:|--------------:|-------|
| **shipped NN (H=4)** | rule + RANDOM patrol | **24.25** | **22.13** | baseline |
| rule_v3_smd_a5 (no NN) | — | 23.4 | — | pure rule, no NN |
| **H=8 NN, rule_v3 + FLOCK** | rule_v3_smd_a5 + flock_centroid | **24.7** | **22.94** | tentative +0.81 win @ 32, z=0.53 |
| H=4 NN, rule_v3 + FLOCK | same | 20.1 | — | underfit; rule's discrete branch hard for tiny NN |
| H=16 NN, rule_v3 + FLOCK | same | 21.1 | — | overfit |
| H=8 NN, rule_v3 + FLOCK + EMA + edgeOversample | same + better hyper | — | 21.8 | EMA hurts (averaging across discrete branches) |
| H=8 NN, rule_v3 + RANDOM patrol | dataset gen in progress | (pending) | (pending) | The "shipped pattern" — random-patrol training, flock_centroid eval |

## Hypothesis: random-patrol training matters

The shipped NN was trained on `dataset_v3` (rule + RANDOM patrol). At eval
time, the patrol mode is `flock_centroid` (the +39% structural fix).
So the NN saw diverse training situations but is evaluated in a
"better" patrol mode. The +2.25-catch lift it has over the pure rule
likely comes from this generalization.

My current H=8 distillation uses rule_v3 + flock_centroid for BOTH
training and evaluation. That's narrower training data. Might explain
why H=8 only gives +0.81 over shipped at 32 seeds (less than the +2.25
the original NN got).

If we replicate the shipped pattern with rule_v3 — train on rule_v3 +
RANDOM patrol, eval on flock_centroid — we expect:
  - Rule_v3 base (~23.4 in flock_centroid eval)
  - + NN-generalization bonus (~+2.25 as for rule_v1)
  - = ~25.65 catches

That would be a clear, statistically significant win over shipped.

## What's running now

- VM 2: generating `rule_v3_smd_a5_RANDOMpatrol.bin` (80 seeds × 5000
  frames, RANDOM auto_target). ~50% done.
- After gen: train H=8 NN on it (~5 min), JS-verify with
  flock_centroid (~10 min).

## Update 2026-05-23: random-patrol training tested

Hypothesis: train rule_v3 NN on RANDOM patrol (like shipped did with
rule_v1) to get a +2.25 generalization bonus over pure rule.

Result: WORSE than flock-patrol training.

| Model | Train data | JS @ 32 seeds |
|-------|------------|--------------:|
| H=8 NN, rule_v3 + RANDOM patrol | rule_v3 + random | 21.75 |
| H=8 NN, rule_v3 + FLOCK patrol  | rule_v3 + flock  | **22.94** |

The "random patrol generalizes" pattern doesn't replicate for rule_v3
— suggesting that the +2.25 bonus shipped got was specific to rule_v1
(smoothing its sharper discrete choice), not a universal "diverse data"
benefit.

## Update 2026-05-27: 128-seed verify and scale-up

The H=8 rule_v3 + FLOCK NN was JS-verified at 128 seeds:

| Model | JS @ 128 seeds | Δ vs shipped | z (paired) |
|-------|---------------:|-------------:|-----------:|
| shipped NN | 21.20 | — | — |
| H=8 NN, rule_v3 + FLOCK | **22.34** | **+1.14** | **+1.36** |

Trending toward significance (p≈0.087 one-tailed). At 256 seeds the
expected z if effect is real ≈ 1.92 (p≈0.027). Currently running both
at 256 seeds for the verification.

In parallel:
- 8-seed RNG ensemble training (H=8 rule_v3 + flock) — pick best of 8.
- Patrol-mode screening (`weighted_centroid`, `weighted_predicted`,
  `nearest_K_centroid`) — looking for a better-than-flock_centroid
  structural change to compound with the distillation bonus.

## What's the elegant insight

The shipped NN's success isn't just NN smoothing — it's specifically
NN generalization from a diverse training distribution to a
favorable evaluation distribution. The patrol-mode mismatch is a
feature, not a bug.

To replicate this with a smarter base rule, we need to train on the
SAME diverse-patrol distribution. Only then does the NN have a chance
to compound the rule's structural improvement with the generalization
bonus.

If the random-patrol distillation works: we have the recipe for any
future rule improvement. Iterate rule → train NN on random patrol →
eval on flock_centroid → measure.
