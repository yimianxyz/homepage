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
