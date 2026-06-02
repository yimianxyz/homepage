# Planner→NN distillation — rethink (2026-06-02)

User redirect: prod reverted to the NN; planner is now the *target to
distill*. New directive: **data prep is the key**, decompose into expert
sub-problems (MoE), keep experiments small/fast/high-ROI, smallest possible
net for the browser, use all 3 GPUs. "Reproduce the same **result**" (catches),
no hard theoretical bound — iterate step by step.

## Label audit (ds1024_dense08.pt, 192,512 planner decisions, K=16/H=120/D=8)

Zero new rollouts — audited the existing teacher data directly:

| metric | value |
|--------|------:|
| planner intercepts a boid (dense-gain winner ≠ E3D) | **72.1%** |
| E3D patrol wins | 27.9% |
| integer-catch ALL-TIE decisions | 39.7% |
| winner margin (dense) < 0.05 | **65.2%** |
| clean integer-catch UNIQUE intercept | 15.0% |
| committed target >10px from E3D | 72.0% |
| winner distribution across the 15 boid candidates | ~uniform (4–7% each) |
| top1–top2 target separation on low-margin frames | median ~100px (only 15.8% <30px) |

## Root cause of every prior distillation failure

The planner's **per-frame target choice is multimodal and near-arbitrary**:
on ~65% of decisions the winner beats the runner-up by <0.05 gain, the two
top targets are ~100px apart, and *which* boid wins is nearly uniform. So all
three prior framings are ill-posed for the **same** reason:

- **classify the candidate index** (#87/#89, 0.55 wall) — label flips between
  equivalent indices.
- **regress the 2D committed coordinate** (deepsets 33%) — averages far-apart
  equally-good targets into a meaningless midpoint.
- **regress the 16-dim gain vector** (#92/#94) — gains differ by <0.05 = fitting
  tie-break noise.

This is the classic multimodal-imitation pathology (MPC-Net, Carius et al.
2019). Action-cloning a multimodal expert by regression/classification cannot
work; their fix was **mixture-of-experts** + a value-structured loss.

## The reframe

Goal = **outcome-equivalence (same catches), NOT action-equivalence (same
per-frame target).** Since most boid choices are near-equivalent, the student
may pick *any* good intercept and still catch the same. This dissolves the
ill-posedness and is exactly what the user asked ("reproduce the same result").

## Plan (small, parallel, data-first)

- **E1** reactive-vs-lookahead reference ladder: e3d / always-nearest-intercept
  / nearest-within-R-else-E3D gate / full planner. Tells us how much of the
  edge is reactive (→ how small the net can be).
- **E2** tiny coordinate-regression net trained on **decisiveness-filtered**
  frames (consensus targets only; E3D prior on ambiguous), selected by
  **closed-loop catches** not supervised loss.
- **E3** MoE: small gate (patrol-E3D vs pursue-cluster) + smooth intercept head.
- **E4** DAgger on the student's own closed-loop states + short ES/PG polish.

Selection metric is always closed-loop catches in sim_torch, never argmax acc.
(sim_torch is rank-misleading at the ES tail — ρ=0.55 — so JS-verify finalists.)

## Results — E1 ladder + E2 tiny nets (n=256, 5000 frames, sim_torch)

E1 reference ladder (catches, se ~0.4–0.5):

| controller | mean |
|------------|-----:|
| e3d (production patrol)      | 34.18 |
| nearest-only (always chase)  | 21.54 |
| gateR_60 (chase if <60px)    | **34.46** (best reactive, +0.8% over e3d) |
| gateR_90 / 120 / 160 / 220   | 33.29 / 31.71 / 29.38 / 25.73 (monotone worse) |
| planner (K16/H120 ceiling)   | _pending_ |

**Reactive gating barely beats E3D** (best +0.8%). Aggressive chasing hurts.
So whatever edge the planner has is NOT a simple reactive gate.

E2 tiny coordinate nets (2,146 params, 31→32→32→2):

| variant (ambiguous-frame prior) | val SmoothL1 | closed-loop |
|---------------------------------|-------------:|------------:|
| e3d   (margin 0.10)             | 0.366 | **33.52** (≈ e3d) |
| committed (raw, no filter)      | —     | 27.20 (multimodal blur) |
| nearest (margin 0.10)           | 0.186 | 19.85 (myopic) |

The raw-committed net confirms the blur prediction (27.2 < e3d). The e3d-prior
net just reproduces E3D (33.5). **A 2,146-param net already ≈ E3D in closed
loop** — so reproducing E3D is trivial; the open question is the planner ceiling.
If planner ≈ e3d here, sim_torch does not exhibit the planner's browser edge and
selection must move to the real-JS regime; if planner >> e3d, the decisive-frame
intercepts are the target and we escalate to E3 (MoE) / E4 (DAgger).
