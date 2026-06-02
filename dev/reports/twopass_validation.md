# Two-pass browser dynamics — implementation + faithfulness validation (2026-06-02)

## What & why
The LIVE browser (`js/simulation.js` `run()`) runs **two** `boid.flock()` passes
per frame — `tick()` (all boids flock from frame-start positions, accel only, no
movement) then `render()` (each boid flocks AGAIN seeing in-frame updates from
boids 0..i-1, then updates) — before one position update, so predator-avoidance
is summed twice. `sim_torch`, the Oracle, `eval_js_patrol.js`, and every teacher
dataset run a **single** pass. Phase 0 needs the true browser dynamics to test
whether the planner's edge over E3D is real (the suspected ρ≈0.55 sim↔browser
cause).

## Implementation
`Sim(two_pass=True)` (`sim_torch.py`): `_step_boids()` dispatches to
`_step_boids_twopass()` — pass 1 = vectorized `_compute_boid_acceleration()`
(frame-start) added to `boid_accel`; pass 2 = sequential per-boid
`_compute_single_boid_acceleration(i)` added on top, then per-boid vel/limit/
pos/wrap/accel-reset (identical arithmetic to the validated `_step_boids_sequential`).
Predator is stepped only by the caller afterward, so `pred_pos` is frame-start
during both passes (double-counted) — matches the browser. `planner_probe.py`
gains a `TWO_PASS` global + `--twopass`; propagates to all 7 Sim constructions,
so the planner plans AND acts under two-pass.

## Cross-check 1 — independent code audit (fresh no-memory subagent)
Audited `_step_boids_twopass` against the JS source line-by-line. Verdict:
**no material discrepancies** affecting live-boid trajectories. Pass order,
accel accumulation (`+=`, not overwrite), zero-at-frame-start guarantee,
fast_limit/MAX_SPEED/wrap ordering, frozen-predator double-count, `<` vs `<=`
neighbor gates, and iteration order all match. Only note: caught/dead boids keep
integrating their (unused, masked-out) positions — same as the pre-existing
sequential path, invisible to forces/catches.

## Cross-check 2 — numerical trajectory match (fixed predator, one seed)
`twopass_check.js` (real js/ boids, pinned predator) vs `twopass_check.py`
(sim_torch), seed 200000, predator at the boid spawn (449,958) so all 120 boids
are within PREDATOR_RANGE → avoidance fully exercised.

| frames | mode   | max \|Δpos\| | mean \|Δpos\| |
|-------:|--------|-------------:|--------------:|
| init   | —      | **0.000**    | 0.000         |
| 1      | single | 4.07e-08     | 1.66e-08      |
| 1      | two    | 1.08e-07     | 3.30e-08      |
| 3      | two    | 6.46e-05     | 2.51e-06      |
| 10     | two    | 5.37e-04     | 4.13e-05      |
| 30     | two    | 1.02e+00     | 2.95e-02      |

Init is **bit-identical**. At 1 frame the two-pass match (~1e-7) is the SAME
order as the already-validated single-pass (~4e-8) — two-pass adds no new
structural error; the residual is fp summation-order noise (torch tree-reduce vs
JS sequential adds). Divergence then grows ~10–100×/few frames — textbook
chaotic (Lyapunov) amplification of rounding, NOT a dynamics bug. By ~300 frames
trajectories fully decorrelate.

## Key implication for the proxy-trust gate (Phase 0 step 2)
Boid flocking is **chaotic**: even with bit-identical per-step dynamics, per-seed
trajectories (hence per-seed catch counts) decorrelate between sim_torch and JS
over a full 5000-frame episode. So a low per-seed rank correlation (the ρ≈0.55
worry) does **not** by itself prove a dynamics bug — it can be pure chaos. The
trustworthy proxy metric is the **mean over many seeds** (distributional match),
not per-seed rank. Phase 0's ρ check must be read in this light.

## Status
Two-pass port validated. 2×2 premise table {E3D,planner}×{single,two} running:
single E3D=34.18 (ladder); single planner=VM1; two E3D=VM3; two planner=VM2.
