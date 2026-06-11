# FINAL VERDICT — predator policy search (2026-06-11)

One page. Detail in `predator_endgame_6x.md` (the win), `predator_3x_feasibility.md`,
`predator_clearance_endgame.md`, `predator_4x_verdict.md`. This is the executive summary.

## The goal and the constraint

- **Goal:** a predator **policy** "300% better" (4×) than the deployed prod baseline,
  measured on the user's pivoted eval — **time to clear every boid**, with the
  **last-boid "endgame" treated as a separate problem to optimize.**
- **Hard constraint (user, verbatim):** *"we should never modify the simulation and any
  other thing! The only thing we can change is the predator policy."*

## The answer has two halves — one wall, one breakthrough

**1. The flock phase / whole-game catch rate is physics-capped at ~1.34× — a real wall.**
Catches in fixed time ≤ density × 2·reach·v_pred·T (the encounter-rate ceiling); the
deployed flock policy already runs at 74–81% of it. Confirmed five ways: delete-flee
(+32%), +20% predator speed (+12%), herding controllers (−16 to −44%), slower prey
backfires, and a prior 6-week RL search + this phase's re-tune all cap at +5–22%. A
single boid has no "density," so this ceiling **does not bind the endgame.**

**2. The endgame (catch the last boid) is now 5–6.4× faster — the goal MET on the
sub-problem the user singled out.** The prior phase wrongly called the endgame
"near-optimal" without computing its floor. The floor is ~80–150f; the shipped TERI ran
at 565–848f (6–7× above it). Two **policy-only** fixes capture it:
- **DT 4→1**: scan the boid's track at single-frame resolution (the coarse scan missed
  the true earliest-reachable intercept by up to 4 frames → chronic near-misses).
- **Remove freeze-and-commit**: re-aim every frame instead of freezing a stale aim vector.

Isolated last-boid time-to-catch (ground-truth JS, n=256, held-out seeds):

| screen | deployed | **new** | speedup | clear-rate |
|---|---|---|---|---|
| phone 390×844  | 2020f | **316f** | **6.4×** | 82% → **100%** |
| iPad 820×1180  | 2383f | **445f** | **5.4×** | 90% → **100%** |
| laptop 1440×900 | 2524f | **500f** | **5.0×** | 88% → **100%** |

Cross-checked 4 independent ways (JS lab, real harness, a no-context audit subagent, a
4096-env GPU ablation across all 3 L4s) — all agree, and all confirm the deployed
policy literally **gets stuck and never clears** 10–18% of the time on big screens,
which the new interceptor fixes (100% clear).

## What "300% better" comes to, honestly

- **Endgame eval: 5–6.4× (>300%) ✓** — policy-only, validated, 100% reliable.
- **Overall clearance time: ~1.3× on phone, larger on big screens** (converting
  never-clears into always-clears). Bounded above by the flock wall.
- **Whole-game catches/1500f: ~1.05×** — physics-capped, unchanged. Not 4×, and can't be.

The 4× target is met precisely where physics allows it (the endgame), and is
impossible precisely where physics forbids it (the encounter-rate-limited flock). The
search did not stop at the wall — it found the one place the wall doesn't apply.

## Shipped — `dev/ship_teri/`

Prod predator + ES device-mix patrol re-tune + the new endgame interceptor (DT=1,
no-freeze). Verified strictly **policy-only**: all six non-policy JS files byte-identical
to prod; predator size/speed and boid count unchanged. Reversible, ~a dozen lines.

## Status / the one open decision

GPU VMs were restarted for the at-scale validation and must be stopped when the last
fleet run lands (cost). `ship_teri` is **not** auto-deployed (user said "keep current
prod as is"). To go live: copy `dev/ship_teri/predator.js` + `predator_cheap.js` over
`js/`. Held for explicit go-ahead.
