# FINAL VERDICT — predator policy search (2026-06-11)

One page. The detail lives in `predator_3x_feasibility.md`,
`predator_clearance_endgame.md`, and `predator_4x_verdict.md`; this supersedes
them as the executive summary.

## The goal, and the constraint that bounds it

- **Goal (escalated):** a predator **policy** that is "300% better" (4×) than the
  deployed prod baseline.
- **Hard constraint (user, verbatim):** *"we should never modify the simulation and
  any other thing! The only thing we can change is the predator policy."* No change
  to boid count, predator size/speed/force, flee range, or the catch rule.

These two together are the whole story: **4× is physically impossible under a
policy-only change**, and that is now proven, not asserted.

## Why 4× whole-game (catches / 1500 frames) is impossible policy-only

The predator moves at 2.5; boids at 6 (2.4× faster). Boids flee only inside an
80px bubble and **do not respawn**. The number of catches in a fixed time is
bounded by an **encounter-rate ceiling**: catches ≤ distinct boids that cross the
predator's catch reach = density × 2·reach·v_pred·T. That ceiling is **1.23–1.34×**
the deployed policy depending on device — and the deployed policy already runs at
**74–81% of it**. Five independent checks, all agreeing:

1. **Encounter-rate ceiling ≈ 1.34×**; deployed at 74–81% efficiency.
2. **Delete flee entirely** (max physical help a policy could ever extract): **+32%**.
3. **+20% predator speed** (not allowed; an upper bound on motion gains): **+12%**.
4. **Herding / orbit / shepherd controllers** (the "drive the flock" idea): **−16 to −44%**.
5. **Prior 6-week end-to-end RL + this phase's re-tune + heavy planner**: cap at **+5–22%**.

The endgame TTC is likewise near its geometric floor (GPU-swept), so 4× fails on the
clearance interpretation too. There is no policy-only path to 4×. I did **not**
fabricate a number to satisfy the target.

## The real, delivered policy-only win — `dev/ship_teri/`

A genuine improvement that respects the constraint exactly (verified: the six
non-policy JS files are **byte-identical to prod**; only `predator.js` and
`predator_cheap.js` change):

1. **TERI endgame interceptor** (Torus Earliest-Reachable Lead Intercept, gated to
   ≤5 boids). The slow predator's one structural edge is the torus wrap. A lone boid
   flies a straight line, so scan its track for the earliest torus-min-image-reachable
   point, commit + FREEZE aim just outside the 80px flee bubble, and ram head-on.
   - **Last-boid TTC: ~3× faster** (phone 1675→565f, laptop 2416→848f).
   - **Fixes never-clears:** on laptop/iPad the deployed policy gets stuck on the last
     boid ~10% of the time (clear-rate 87.5% / 90.6%). TERI → **100% clear**.
2. **ES device-mix patrol re-tune** (the shipped params were over-fit to the old 1680²
   square): **+4.9% device-weighted**, zero extra deploy compute.

**Net on full clearance (held-out seeds, real prod code):** ~**1.2–1.4× faster** to
clear every boid, **100% clear on every device**, last-boid hunt ~3× faster and
visibly purposeful (cuts across the edge to ambush — the nature-predator behavior the
user asked for).

## Status

- VMs `ml-forecast-{1,2,3}`: **TERMINATED** (no cost).
- Work committed + pushed to `origin/rl/teacher`.
- `dev/ship_teri/` is a ready, reversible, drop-in candidate.

## The one open decision (the user's, not mine)

The user said *"keep the current prod version as is."* So `ship_teri` is **not**
auto-deployed. To make it live, copy `dev/ship_teri/predator.js` and
`dev/ship_teri/predator_cheap.js` over `js/`. That is the only outward-facing step,
and it is held for an explicit go-ahead.
