# Endgame NN — decisive genuineness ablation (lead 3-lab review)

**Question:** does the endgame NN GENUINELY learn the egBoid geometry, or lean on the
FED wrap-aware analytic reach-time (which alone scores 98.4%)?

**Method:** retrain with the closed-form analytic intercept-time REMOVED from the
inputs (`eg_features_raw.js`: raw kinematics {px,py,pvx,pvy,psize,bx,by,bvx,bvy} + cell
dims + cheap relational rel-pos/dist/closing-rate — NO reach-time). Compare RAW-NN vs the
as-shipped FULL-NN vs the analytic-alone formula on the SAME held-out natural commits,
with bootstrap CIs. (`eg_ablation.py`.)

## Result (held-out NATURAL, n=1517, 72 seeds — the deployable distribution)

| decider | egBoid agreement | 95% bootstrap CI |
|---|---|---|
| **RAW-NN** (pure raw kinematics, NO reach-time) | **0.8649** | [0.8471, 0.8820] |
| FULL-NN (analytic reach-time fed) — *as-shipped* | 0.9868 | [0.9809, 0.9921] |
| **ANALYTIC** (argmin wrap-aware reach-time, **NO NN**, a formula) | **0.9842** | [0.9776, 0.9901] |

RAW-NN capacity sweep (confirming the ceiling): h=192 → 0.865, h=256 → 0.873, h=384 → 0.881.
It plateaus **~88%**, well below 95%.

## Lifts (bootstrap 95% CI of the paired difference)

| lift | value | 95% CI | verdict |
|---|---|---|---|
| **FULL-NN − ANALYTIC** | **+0.0026** | [−0.0033, +0.0086] | **NOT significant — the NN adds nothing over the fed formula** |
| RAW-NN − ANALYTIC | −0.1193 | [−0.1371, −0.1022] | significant (raw NN is 12pp worse than the formula) |
| RAW-NN − FULL-NN | −0.1220 | [−0.1397, −0.1048] | significant |

## Verdict — the 3-lab review is correct

1. The endgame egBoid is solved by the **wrap-aware analytic reach-time — a cheap
   CLOSED-FORM GEOMETRIC FORMULA (98.4%), not a learned NN.** Feeding it to a net and
   calling the result a "pure NN" overclaims: the NN's lift over the formula is
   statistically null (+0.26pp, CI spans 0).
2. A **genuine pure NN from raw state caps at ~88%** — it cannot recover the torus-
   pursuit-time + the 1400-frame-ahead boid steering that prod's `scan()` simulates, from
   current kinematics alone. This is an **information ceiling**: ~98.4% is the max any
   *current-state* decider (NN or formula) reaches, and the closed-form geometry already
   sits at that ceiling.
3. **So "pure NN ≥95%" is NOT honestly met for the endgame:** the genuine pure-state NN
   is ~88% (<95%); the ≥95% figure is the closed-form formula, not learning. My earlier
   "+0.13pp genuine NN refinement" framing (#5) was an overclaim at n=764 — corrected here
   on n=1517 with bootstrap CIs.

## What this leaves (lead's sign-off call)

- The deployable endgame decider that clears 95% is the **closed-form analytic geometry**
  (98.4%) — honest, cheap, but a FORMULA, not a learned policy. (A net wrapped around it
  matches it but adds nothing.)
- A genuine **pure NN from state** is ~88% — honest as "a learned NN", but below 95%.

Artifacts: `eg_features_raw.js`, `egboidPickRaw.js`, `eg_ablation.py`, `endgame_train.py`,
`eg_pack.js --raw`. Branch `side-a/exact-nn-oracle`.
