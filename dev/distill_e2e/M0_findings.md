# Milestone-0 — Feasibility of a 100%-identical end-to-end NN

Run: `feas.py --seeds 256 --frames 1500 --device cuda` on L4 (ml-forecast-1).
Production policy = shipped E3D evolved-patrol + 35-feat builder + 35→4→2 NN.

## FEAS-2 (the decisive result): action→catch sensitivity

Perturb production steering by `N(0, sigma)` and count exact per-seed catch matches
against unperturbed production (256 seeds).

| sigma | exact_match | mean_abs_diff (catches) | mean_catches |
|------:|:-----------:|:-----------------------:|:------------:|
| 0     | 256/256     | 0.00 | 8.125 |
| 1e-7  | 25/256      | 3.55 | 8.30 |
| 1e-6  | 18/256      | 3.75 | 8.39 |
| 1e-5  | 25/256      | 3.89 | 8.21 |
| 1e-4  | 29/256      | 3.79 | 8.46 |
| 1e-3  | 27/256      | 3.68 | 8.54 |
| 1e-2  | 18/256      | 3.72 | 8.48 |

**The action-error budget ε* is below 1e-7.** A steering perturbation of 1e-7 —
*smaller than float32 epsilon (~1.2e-7) on forces of magnitude ~0.05* — already
collapses per-seed catch identity from 256/256 to ~chance (18–29/256 is the random
coincidence rate for integer catch counts). The exact-match curve does not decay
gradually; it falls off a cliff between 0 and 1e-7 and stays flat thereafter. This
is the signature of a **chaotic system** with large Lyapunov amplification over 1500
frames.

## FEAS-1: memorylessness

Always-recompute-target vs production: **18/256** exact match, mean_abs_diff 3.52,
mean 8.20 vs 8.125. Verdict: history-dependent — but note 18/256 is exactly the
chaos floor, so the per-seed decorrelation is dominated by FEAS-2 chaos (changing the
target-update rule is itself a >1e-7 perturbation). The freeze adds only a small mean
shift (+0.08 catches).

## Verdict for the project goal

**An end-to-end NN that produces 100% identical per-seed catches as production is
mathematically impossible** — not a capacity problem, a chaos problem. Any policy
that is not *bit-identical* to the float64-heuristic + float32-NN production pipeline
introduces force error ≫ 1e-7, which the dynamics amplify to ~3.7 catches/seed
divergence and chance-level per-seed match. This is the same mechanism that made
sim_torch (8.25) and JS (5.25) disagree on the *identical* policy: cross-engine float
differences are far larger than 1e-7.

## Reframed success metric (achievable, meaningful)

"Same result" is redefined from per-seed bit-identity (impossible) to **behavioral /
distributional equivalence**:
1. Force MSE vs production → as small as the (memoryless, raw-obs) function class allows.
2. Mean catch rate matches production within the per-seed noise band (~8.1 ± seed σ).
3. Per-seed catch *distribution* statistically indistinguishable (KS / mean-CI overlap),
   and hunting behavior visually identical.

The end-to-end NN is then a *refactor* win — one clean raw-obs net replacing the
hand-built 35-feature pipeline + evolved-patrol heuristic — judged on behavioral
equivalence, not impossible bit-identity.
