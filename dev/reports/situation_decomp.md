# Situation-decomposition predator search

Goal (user): break the game into sub-situations, find the optimal policy for
each via rule-based search, distill into ONE general end-to-end NN that beats
the production predator by **≥50%**. The deliverable must stay a reactive,
browser-deployable net.

## North star

Full closed-loop game: 120 boids, 1500 frames, mean catches over a held-out
seed block (GPU `sim_torch.py`). Higher is better.

| policy | mean catches | n seeds | SE |
|---|---|---|---|
| **production (E3D evolved-patrol + chase)** | **8.3447** | 2048 | 0.090 |
| target (≥50% better) | **≥12.52** | — | — |

Production on `main` is the 10,352-param radial set-net, a ~0.987-cos distill of
exactly this E3D policy, so E3D-evolved is the faithful baseline policy.

Prior context (memory `predator-rl-ceiling`): a multi-day reactive/parametric
search topped out at ~8.4; the note concluded beating it "needs non-reactive
multi-step anticipation (planning/MPC)." The new ingredient here is exactly
that — per-situation **planned** optima distilled into a reactive net (amortized
planning / expert iteration), which the prior search never tried.

## Experiment 1 — phase-conditional evolved params (headroom probe)

The dominant "situation" variable is how many boids remain (herding → small-N →
endgame). Production uses ONE evolved-patrol param set for all phases. We CMA-ES
a SEPARATE param set per live-count phase (`sim_torch.py` `_phase_params` hook,
selected per-frame by live count) — a strict superset of global E3D, so it can
only help. Magnitude of the gain = the headroom that phase-specialization (the
cheapest situation decomposition) reveals before investing in the full
planner+distill pipeline.

Invariant checked: phase-conditional with identical E3D params in every phase
reproduces global E3D exactly (parity match, CPU).

3 islands (different phase granularities), one per VM, CMA-ES over P×7 params.
Results recorded below as they land.

### Launch (2026-06-01)

All 3 islands running, pop=24 × 160 seeds × 1500 frames, 60 gens,
sigma0=0.15. Eager (phase hook is not CUDA-graph-safe) → ~271 s/gen,
~4.5 h/island.

| island | VM | live-count edges | P | dim | seedStart | seed |
|---|---|---|---|---|---|---|
| ph1 | ml-forecast-1 | 2,8,25,70 | 5 | 35 | 30000 | 1 |
| ph2 | ml-forecast-2 | 1,4,12,40,90 | 6 | 42 | 50000 | 2 |
| ph3 | ml-forecast-3 | 8,40 | 3 | 21 | 70000 | 3 |

ph1 gen-0 best 8.41 (≈E3D, since x0 starts every phase at E3D — the
search is a strict superset, so gen-0 ≈ baseline confirms wiring). Each
island writes `~/situ/ckpt/phX/best.json` + `log.jsonl`; held-out
re-scoring of any winner on a fresh seed block (n≥2048) before any claim.

### Result — NEGATIVE (held-out, 2026-06-01)

All 3 islands finished 60 gens. Their on-search 160-seed bests
(9.24 / 9.25 / 9.36) looked promising, but those are the **max over
pop×gens on a single 160-seed block — selection-bias-inflated**. The
honest test is `phase_rescore.py`: the winning phase params vs global
E3D on the SAME fresh held-out block (seedStart 200000, n=2048, paired).

| island | P | search 160-seed | held-out phase | held-out E3D | paired gain | gain σ |
|---|---|---|---|---|---|---|
| ph1 | 5 | 9.24 | 8.489 ± .094 | 8.350 ± .093 | **+0.139** | 1.15 |
| ph2 | 6 | 9.25 | 8.401 ± .093 | 8.350 ± .093 | **+0.052** | 0.45 |
| ph3 | 3 | 9.36 | 8.267 ± .089 | 8.350 ± .093 | **−0.083** | −0.71 |

All three paired gains are **statistically insignificant (<2σ)**, and ph3 is
actually negative — its 9.36 on-search best collapsed to 8.27 held-out, below
E3D. Held-out E3D re-measures at 8.3496 ≈ the 8.3447 north-star baseline
(sanity OK). The 160-seed→2048-seed collapse (9.2–9.4 → 8.27–8.49) is dramatic
and identical to the prior `predator-rl-ceiling` small-block overfit signature.

**Conclusion:** phase-conditional reactive-parameter specialization (the
cheapest situation decomposition) buys ~0–1.7%, within noise — it does
NOT break the ~8.4 reactive ceiling. This independently reconfirms
`predator-rl-ceiling`: reactive/parametric tweaks are saturated. The
160→held-out collapse is the key lesson — never trust a CMA-ES winner's
own-block mean; always paired-re-score on fresh seeds.

**Implication for the deliverable.** The remaining lever is the planner
+ distill pipeline (non-reactive multi-step anticipation). But the
binding constraint is that the deliverable must stay a *reactive*,
memoryless browser net. A lookahead planner can score higher in closed
loop, yet distilling it back to a reactive function should regress toward
the same reactive ceiling — a reactive net only ever sees the current
state, which is exactly what E3D already optimizes. So the planner's
edge that *survives distillation* is limited to finding better
*instantaneous* target choices than the E3D heuristic; the phase result
suggests that heuristic is already near-optimal for reactive play. The
+50% (≥12.52) target via a reactive net is therefore likely infeasible;
the planner phase will quantify how much (if any) instantaneous-choice
headroom remains.

## Experiment 2 — planner ceiling probe (`planner_probe.py`)

The decisive test of feasibility. If a non-reactive, multi-step-anticipating
**expert** that controls only the distillable lever (the patrol-target choice)
cannot reach ≥12.52, then a *reactive* net distilled from it certainly cannot,
and the user's ≥50% goal is infeasible for the required deliverable.

Design: greedy receding-horizon **target-commitment** planner. Steering is
production's exact analytic decomposition (M5): `force = nearest-in-POLICY_R ?
seek(nearest) : seek(target)` — NN-free, so any gain is purely from better
*instantaneous target choice* (the part that survives distillation). Every `D`
frames the planner branches over `K` candidate targets (E3D's own target as
candidate 0, plus the K−1 nearest live boids lead-adjusted), rolls the TRUE
dynamics forward `H` frames committed to each candidate, picks the candidate
with the most catches over the horizon, holds it `D` frames, re-plans. Because
candidate 0 is E3D's target and rollout is on the true dynamics, the planner is
≥ analytic-E3D by construction (up to rollout-horizon truncation).

Baselines on the same held-out block: `--controller e3d` (analytic E3D, no
planning) isolates the NN-vs-analytic gap; `--controller planner` measures the
ceiling. Results recorded below as they land.

### Results (held-out seedStart=200000, n=512)

| controller | K | H | D | seedStart | mean | SE | vs 8.34 |
|---|---|---|---|---|---|---|---|
| e3d (analytic, no planning) | – | – | – | 200000 | 8.139 | 0.177 | — |
| planner | **1** | 60 | 15 | 200000 | **8.176** | 0.179 | +0% (control) |
| planner | 2 | 60 | 15 | 200000 | 10.656 | 0.184 | +28% |
| planner | 8 | 60 | 15 | 200000 | **13.998** | 0.194 | **+68%** |
| planner | 8 | 60 | 15 | 300000 | 14.303 | 0.200 | +71% (fresh block) |
| planner | 10 | 90 | 10 | 200000 | 18.258 | 0.203 | +119% |
| planner | 16 | 120 | 8 | 200000 | **21.871** | 0.212 | **+162%** |

**POSITIVE — and large.** The planner, controlling ONLY the patrol-target choice
via true-dynamics rollout, beats the +50% target (≥12.52) at every K≥8 and scales
monotonically with candidate count and lookahead. Two decisive controls:

1. **K=1 (only E3D's target as candidate) = 8.18 ≈ the 8.14 e3d baseline** — the
   planning machinery with no real choice reproduces baseline exactly, so the gain
   is entirely from *better target selection*, not a counting/leakage bug. (The
   rollout runs on a separate B·K-env sim; the real env advances one true frame
   per step; catches ≤ ~180/run by the feed cooldown — 14–22 is well within bounds.)
2. **Fresh seed block reproduces 14.0 → 14.3** — not seed-specific.

**This overturns the prior `predator-rl-ceiling` conclusion.** That conclusion
("instantaneous target choice is saturated; +50% via a reactive net is infeasible")
was an artifact of only ever searching E3D's *7-parameter family* (the patrol-evolve
islands AND the phase probe above). A richer target-selection *function* is
dramatically better. Crucially, the planner's chosen target is a deterministic
function of the CURRENT state (the future it rolls out is itself determined by the
present), so this is **distillable into a reactive, memoryless net**: keep
production's architecture (net → target point → analytic seek + in-range chase),
just replace E3D's hand-derived target function with one learned from the planner.

## Experiment 3 — distill the planner's target function into a reactive net

Plan: (1) generate (state → planner target) pairs across many seeds; (2) train a
reactive net mapping production-style features → 2D target; (3) closed-loop eval
with net-target + analytic seek/chase; (4) DAgger to fix distribution shift. The
open question is how much of the 14–22 planner ceiling survives reactive
distillation. Even partial recovery clears the ≥12.52 target. Results below.

### Distillation results — the planner edge does NOT survive reactive distillation

| approach | what it learns | closed-loop mean | vs baseline |
|---|---|---|---|
| v1 `distill_planner.py` | MSE regress planner's 2D target | 7.5–7.8 | below (multimodal averaging) |
| v2 `distill_planner2.py` | pointer-net, CE on argmax(gain), pick argmax | 8.24–8.44 | ≈ baseline |
| v3 `distill_planner3.py` all | v2 repro | 8.24 (pick0=0.92) | −1.3% |
| v3 decisive (train only decisive frames) | — | 7.77 (pick0=0.44) | −6.9% |
| v3 margin-weighted CE | — | 7.21 (pick0=0.22) | −13.6% |

**Root cause (decisive, K8/H60/D15, n=38400 decisions):** the per-candidate gain is
"catches in the next H=60 frames", and with ~8 catches/1500 frames it is **86.2%
all-tie** (every candidate gives equal gain over the horizon) → `argmax` defaults to
index 0 = the E3D candidate → **91.7% of labels are "pick E3D"**. CE is minimised by
always picking E3D, which IS baseline (v3-all pick0=0.92, mean 8.24). Only **13.8%**
of frames are decisive, and E3D is still best in 40% of those.

**The killer: decisive-frame TRAIN accuracy is only 44.7%.** On the training set,
restricted to frames where the choice matters, the net barely beats "always pick the
majority class." So the planner's decisive pick is **not a learnable function of the
M=16-nearest observation** — it depends on the full 120-boid future the planner rolls
out, information absent from the local obs. Forcing the net to deviate from E3D
(decisive/weighted modes drop pick0 to 0.44/0.22) makes catches **worse** (7.77/7.21):
its deviations are mostly wrong guesses.

**Implication:** this is an INSUFFICIENT-OBSERVATION failure, not distribution shift,
so **DAgger cannot fix it** (DAgger relabels visited states, but the labels still
aren't a function of the obs). The fix must be richer information at decision time.
Two routes: (a) per-candidate cheap-lookahead FEATURES, or (b) skip the net entirely
and replace the planner's expensive true-dynamics rollout with a CHEAP O(N) rollout
the browser can run every frame, keeping the argmax-over-candidates structure
(`cheap_planner.py`). Route (b) is the more promising deployable answer. Results next.
