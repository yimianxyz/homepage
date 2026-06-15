# Elegant unification test — does prod's rollout-planner reproduce the N≤5 endgame?

**Idea (user):** un-gate prod's `planCheap` rollout-planner to run for ALL N≥1 (the
rollout is the future-sim the current-frame endgame net lacked; value-net NN genuinely
in-loop everywhere; no new training). **Decisive measurement on existing infra.**

## Measurement 1 — decision agreement S_dec(N≤5)

On the logged N≤5 endgame-commit states, ran prod's `planCheap` (exposed by source-
injection from the certified oracle fork; cands+bi captured via the oracle hooks), mapped
its committed target → the boid it commits to, vs prod's actual `intercept()` egBoid.

| held-out NATURAL (n=1517, 72 seeds) | scatter (n=240 sample) |
|---|---|
| slot-mapped S_dec **0.2136** / nearest-boid **0.6124** | slot 0.254 / nearest 0.583 |
| **E3D-patrol quirk rate 43.0%** | 36.7% |

- The honest per-state range is ~21% (strict: a "patrol" target = not a boid) to ~61%
  (lenient: map any target to its nearest boid). **Both far below 95%.** The range is wide
  because planCheap commits a *target coordinate*, not a *boid identity* — and for few boids
  the E3D patrol point often sits near a boid (so the strict count under-reports; the lenient
  count over-reports). The unambiguous measure is the full-game outcome (Measurement 2).
- **The E3D-patrol quirk is real and load-bearing:** on ~43% of natural N≤5 states planCheap
  commits to the patrol candidate, not a boid — the 90-frame rollout + value-net bootstrap
  prefers positioning over a chase the 2.4×-slower predator can't win in 90 frames.

## Measurement 2 — full-game clear outcome (the decisive, mapping-free test)

Ran the UNIFIED policy (planCheap for ALL N, intercept gate removed, commit-and-hold every
D=16 frames) vs prod (planner N>5 + intercept N≤5), same seeds, endgame games (startBoids=5
scatter, 30 seeds × 4 cells, to 6000 frames).

| | prod | UNIFIED (planCheap all-N) |
|---|---|---|
| **cleared (caught all boids)** | **120/120 (100%)** | **63/120 (52.5%)** |
| 390×844 | 30/30 | 24/30 |
| 1024×768 | 30/30 | 25/30 |
| **1680×1050** | 30/30 | **8/30** |
| **2560×1440** | 30/30 | **6/30** (avg 1.77 boids left uncaught) |

**Clear regression: 47.5pp overall; 73–80% failure on big screens.** (Scatter is the *easy*
endgame distribution; natural fast-survivor endgames would regress at least as much.)

## Verdict — the unification FAILS; prod's two mechanisms exist for a real reason

This is exactly what prod's own code documents. `js/predator_cheap.js` `intercept()` comment:
> "ENDGAME INTERCEPTOR — used for the last ≤5 boids. A lone survivor has no flockmates, so it
> flies a near-straight line and the lookahead [planner] degrades into a **tail-chase the
> 2.4×-slower predator can never win (it gets stuck and never clears the last boid 12-18% of
> the time on big screens)**. The slow predator's one structural edge is the torus WRAP:
> scan the target's straight track ... for the EARLIEST point the predator can reach in time."

So **prod added `intercept()` PRECISELY because the rollout-planner fails in the endgame.**
Un-gating the planner re-introduces that failure: it tail-chases (47.5pp clear regression,
73-80% on big screens) and commits to the patrol point 43% of the time. The planner's
**90-frame rollout horizon + value-net objective** cannot find the **torus-wrap catches** that
take far more than 90 frames — the slow predator's only edge, which `intercept()`'s
**1400-frame torus scan** is built to exploit. The two deciders optimize genuinely different
objectives over genuinely different horizons; they diverge in the endgame by design.

**The real trade for the user:** the rollout-planner is an elegant single mechanism, but it
is the *wrong* objective for the endgame and would lose ~half the games' clears (most on big
screens). prod's intercept (torus scan) is not a redundant special-case — it is the
load-bearing fix for the slow predator's structural endgame weakness.

Artifacts: `unify_measure.js` (decision agreement), `unify_traj.js` (full-game clear),
`oracle_policy.js` (planCheap exposed). Pure prod machinery, no NN of ours. Branch
`side-a/exact-nn-oracle`.
