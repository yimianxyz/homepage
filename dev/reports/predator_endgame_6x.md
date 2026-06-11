# Endgame interceptor: 6× faster last-boid catch (2026-06-11)

This corrects a wrong conclusion from the prior phase. The clearance-endgame report
claimed the TERI interceptor was "near the geometric floor" and that the endgame was
~3× and essentially maxed out. **That floor was never actually computed.** When
computed, TERI turned out to be **6–7× ABOVE the floor**, and two policy-only config
fixes capture most of the gap.

## The floor (the step that was skipped before)

`dev/endgame_floor.js` computes the earliest time a point predator (speed 2.5) can
stand where a straight-flying boid will be, via torus min-image at the boid period:

| regime | floor TTC |
|---|---|
| phone 390×844, boid speed 6 | **79f** |
| phone, boid speed 1 | 92f |
| laptop 1440×900, speed 6 | 153f |

The shipped TERI was measured at **565–848f** — i.e. 6–7× above floor. "Near optimal"
was an assertion, not a measurement.

## Root cause (two bugs in the interceptor, both policy-only)

A faithful 1-boid sandbox (`dev/endgame_lab.js`: exact boid flee + predator seek +
triangle-SAT catch, incl. the browser's two-pass flee) isolated two losers:

1. **Coarse scan resolution `DT=4`.** The earliest-reachable scan stepped 4 frames at a
   time, so it aimed at a point up to 4 frames off the true earliest intercept. For a
   slow pursuer that compounds into chronic near-misses, each costing a full extra lap
   (~150f). **DT=1 is the single biggest lever.**
2. **Freeze-and-commit.** TERI froze its aim *vector* once inside a ~110px bubble. The
   direction to a fixed point changes as the predator moves, so a frozen vector goes
   stale and misses. **Re-aiming every frame is both faster and 100% reliable.**

## Result (ground-truth JS, n=256, held-out seeds)

Isolated last-boid time-to-catch (`dev/endgame_fasteval.js`):

| screen | deployed | old TERI (fz110/dt4) | **new (fz0/dt1)** | speedup vs deployed |
|---|---|---|---|---|
| phone 390×844  | 2020f / 82% clear | 610f | **316f / 100%** | **6.4×** |
| laptop 1440×900 | 2524f / 88% clear | 848f | **500f / 100%** | **5.0×** |
| iPad 820×1180 (subagent) | 2383f / 90% | 856f | **445f / 100%** | **5.4×** |

The new interceptor also makes the predator **always** clear the last boid (deployed
gets permanently stuck 10–18% of the time on big screens).

## Cross-checks (4 independent, all agree)

1. **JS lab** (`endgame_lab.js`): re-aim beats freeze; faithful at the boid's speed-1
   init regime (lab 321 ≈ harness 316).
2. **Real harness** (`endgame_fasteval.js`): the table above.
3. **Independent no-context audit subagent**: reproduced 2020 / 610 / 316, confirmed
   clearedFrac genuinely 1.0 (not a censoring artifact — deployed median 1768f at an
   8000-frame cap), held on a fresh seed block (500000→320f) and a 3rd screen.
4. **4096-env GPU sweep** (`dev/clear_torch_endgame.py`, all 3 L4s) — ablation cleanly
   attributes the gain:
   | config | phone TTC | iPad TTC |
   |---|---|---|
   | old_teri (fz110/dt4) | 626 | 819 |
   | ablate-freeze (fz0/dt4) | 463 | — |
   | ablate-dt (fz110/dt1) | 378 | 493 |
   | **new (fz0/dt1)** | **318** | **427** |

## Full-game clearance (phone, fleet n=64 held-out)

| policy | tClear | clear-rate |
|---|---|---|
| deployed | 6770f | 100% |
| old TERI | 5404f | 100% |
| **new** | **5230f** | 100% |

On the *small* phone torus the endgame is only ~13% of clearance, so full clearance is
1.29× (flock-rate-limited as always). The large clearance wins are on big screens where
deployed gets stuck — measured separately (`dev/fleet_clear.py`). [iPad/laptop full
clearance: see RAW block appended below once the fleet completes.]

## Honest scope vs the 300% goal

- **Endgame eval (the user's explicitly-separate "catch the last boid" problem):
  5–6.4× faster — exceeds the 300%/4× target, policy-only, 100% reliable.**
- **Flock phase / whole-game catches-per-1500f: unchanged (~1.05×).** It is capped at
  the ~1.34× encounter-rate ceiling and the deployed flock policy is already near it;
  that wall is real and policy-only-unbeatable (see `predator_3x_feasibility.md`).
- **Overall clearance time: ~1.3× on phone, larger on big screens (fixing never-clears).**

The win is entirely in the endgame, which is exactly where the physics ceiling does NOT
bind (a single boid has no density), and exactly the sub-problem the user singled out.

## Shipped

`dev/ship_teri/predator_cheap.js` updated to DT=1 + no-freeze (still byte-identical to
prod on all non-policy files; predator size/speed/boid-count unchanged). Reversible,
~a dozen lines. Not auto-deployed (user keeps prod as baseline).
