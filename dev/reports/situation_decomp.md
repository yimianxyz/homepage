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
