# Predator "3× better policy" — feasibility & search findings (2026-06-10)

Goal (user): find a predator policy ≥3× the deployed cheap baseline, optimised for
real devices (phone/laptop/iPad), exploiting the endgame and screen edges/wrap.

## Eval harness (rebuilt; the foundation)

- **`dev/fasteval.js`** — fast faithful headless eval on the REAL JS engine in pure
  node (~5 s/seed/1500f, ~75× faster than the vm-sandbox path). Search metric.
  - Two-pass loop (`simTick; tick; render`) + the one-time pre-loop `tick()`.
  - Per-episode policy rebuild (the cheap policy leaks IIFE state across seeds).
  - `PREDATOR_RANGE` locked to 80 (prod bakes it before isMobileDevice exists).
  - **Independently audited** (fresh subagent): after the pre-loop-tick fix, bit-
    identical to a faithful browser-loop reconstruction (maxAbsDiff=0, 40 seeds).
    The omitted pre-tick had biased catches ~+6% high — fixed.
- **`dev/eval_device_browser.js`** — chromium ground-truth spot-check (stub
  setInterval so the page's background sim doesn't pollute the shared RNG).
- **`dev/fleet_eval.py` + `~/eval/` on the 3 VMs** — fan eval jobs across 12 vCPUs.
- node v20.18.1 is bit-identical local ↔ all 3 VMs. fasteval (node V8) and chromium
  agree at the MEAN; per-seed they diverge a few catches (V8-version float in a
  chaotic system) — fasteval for search, chromium to confirm finalists.

## Hard physical bounds (these frame everything)

- **Predator max speed 2.5 vs boid 6** — the predator is 2.4× SLOWER than every
  prey. It cannot win a straight chase; catches come from interception geometry,
  flock dynamics, and boundary/wrap effects.
- **Boids do NOT respawn.** Catches/episode are capped at the starting count N
  (60 mobile / 120 desktop). On a phone the deployed policy already catches ~26/60
  (43%), so a literal 3× (78) is mathematically impossible there.
- **Within 1500 frames the predator is catch-RATE limited, not depletion limited**
  (phone: ~26 caught, ~34 boids still alive at the end). The endgame (running out of
  boids) only binds in the *live* page, which runs forever and depletes.

## Deployed-policy baselines (faithful, fasteval, n≈96)

| device (W×H, N)        | catches/1500f |  vs 1680² |
|------------------------|---------------|-----------|
| phone 390×844, N=60    | ~25.7         | 2.0×      |
| iPad-P 820×1180, N=120 | ~24.7         | 1.9×      |
| laptop 1440×900, N=120 | ~20.4         | 1.6×      |
| 1680² square, N=120    | ~12.7         | 1.0×      |

**Key reframing:** the deployed policy's quoted ~12/1500f was measured on an
unrealistic 1680² square. On real devices the SAME policy already catches ~1.6–2.0×
more, purely from screen geometry (smaller torus → prey can't escape the slow
predator; higher density). So "≈2× the quoted number" is already shipped.

## Corrected lever table (FIXED harness, n=32–96, the numbers to trust)

Baselines: phone 24.7, iPad-P 25.8, laptop 21.1 (catches/1500f).

| lever (vs baseline)                         | phone  | iPad-P | laptop | deploy cost |
|---------------------------------------------|--------|--------|--------|-------------|
| wrap-aware                                  | +4.9%  | −0.8%  | +1.4%  | ~free       |
| search ×2 (K_roll6/Hs90/D12)                | +6.6%  | +5.1%* | +11.9%*| 2× rollout  |
| search ×4.4 (K_roll8/Hs120/D10)             | +12.0% | —      | —      | 4.4×        |
| search ×18 ceiling (K16/Hs150/D6, offline)  | +22.2% | +17.2% | +21.9% | 18× (NO)    |
(* the iPad/laptop ×2 rows also had wrap on.)

**Wrap is essentially neutral** once the harness bug is removed (the earlier
"+6–10% on big screens" was the +6% harness bias + noise). The one robust lever is
**search depth** — catch rate rises smoothly with rollout compute, +5–12% at 2–4×,
plateauing ~+20% at 18× — bounded by what a phone can run synchronously per frame.

## Levers tested (detail)

- **Wrap-aware (toroidal min-image) distances** in the policy (steering, candidate
  generation, ballistic predict, density/centroid): clean A/B (exp{wrap} vs exp{}):
  phone −1%, iPad-P +9.6%, laptop +8.5%, square +5.8%. Helps on LARGER screens
  (distant-boid chases more often have a shorter seam path), neutral on phone.
  Confounded by the value net going OOD on wrap-aware features. **HURTS the 1-boid
  endgame** (TTC 1594 vs 1370, cleared 71% vs 79%).
- **Stronger patrol search** (K_roll 4→16, Hs 90→150, D 16→6, ~10× compute):
  phone 25.4 → 29.8 (+17%). Real but modest headroom; far from 3×; too heavy to
  deploy on phones as-is.
- **1-boid endgame**: baseline clears ~79% in ~1370 frames; neither wrap nor more
  search helps (both ≈ or worse). A lone boid 2.4× faster on an open torus is
  genuinely near-uncatchable by these policy families.
- **Ambush** (loiter on the flock flow-line outside PREDATOR_RANGE): ~neutral (the
  added candidates are scored by the same charge-in rollout, so they rarely change
  the committed steering).
- **The "feint to the far edge" idea** (chase a boid to commit it to max speed, then
  cut across the wrap to intercept where it re-enters): defeated by the flee mechanic.
  A boid only flees within PREDATOR_RANGE (80px); the moment the predator backs off
  past 80px the boid STOPS fleeing and flocking/wander takes over, so it does not
  travel predictably to the far seam — there is nothing to intercept there. The
  reactive version (wrap-aware "always take the short seam path") IS implemented and
  measured neutral.

## Red-team cross-check (independent subagent) and its reconciliation

A fresh red-team ran physics-sensitivity probes and argued the catch RATE is gated by
the boids' FLEE response, not predator speed: turning flee OFF = +32%, halving the
turn factor = +26%, late detection (range 80→40) = +27%, vs only +12% from a 20%
predator-speed boost. Catches are cluster events (mean 4.1 boids within 60px at the
moment of catch; only 8% lone boids). It proposed herding/compression + strike-timing
as untested levers with large upside.

**Reconciliation — most of that headroom is NOT policy-reachable:**
- Flee triggers purely on `distance < PREDATOR_RANGE` (`js/boid.js:180`), with NO
  approach-speed term — so you cannot "sneak up"/induce late detection. TURN_FACTOR
  (0.3) is a fixed game constant. So the +26-32% flee-sensitivity is the response to
  changing GAME RULES we cannot change, not to a reachable policy.
- The only policy-reachable part is approach GEOMETRY that makes flee compress the
  flock — and the patrol family ALREADY does this (catches are already cluster events).
  Three cheap attempts to push it further — ambush candidates, a terminal-density
  "compression" reward (best λ=0.3 → +4% noisy, larger λ hurts), and lead-past-boid —
  all came up ~neutral. The +17% heavy-search ceiling is the real reachable ceiling.

So the red-team correctly identified the mechanism but its big numbers are physics-knob
sensitivities, not deployable gains. Verdict stands.

## Evolutionary param search (AlphaEvolve-style, the user's "policy explore")

`dev/evolve_policy.py` — CMA-lite ES over the 7 patrol params + POLICY_R, scored on
the phone/laptop/iPad mix via the fleet, starting from the shipped (square-evolved)
params. 4 generations × 6 candidates × 3 devices × 48 seeds. Result: the best score
per generation was 25.48 → 24.94 → 24.85 → 25.14 while the recombined mean stayed at
the deployed-params score (~24.0). The gen-0 "+5.5%" was a best-of-6 winner's-curse
outlier at n=48; the search **converged back to the shipped params**. Conclusion:
the patrol params are already near-optimal on the device mix — re-tuning yields only
noise-level differences. (Held-out re-validation of the ES "best" on fresh seeds:
<FILL>.)

## Feasibility verdict

**A literal 3× over the deployed policy is not physically reachable as a policy
improvement.** Every independent line of evidence agrees:
- The no-respawn **N cap** forbids it on phones (deployed already catches 43% of 60).
- The predator is **2.4× slower**; the heavy-search ceiling is only +17–22%.
- **Flee is distance-triggered + fixed turn factor**, so the large flee-sensitivity
  headroom a red-team found is a game-rule sensitivity, not policy-reachable.
- **Wrap-aware = neutral**, **ambush = neutral**, **compression-reward = neutral**,
  **param-ES = neutral**, **endgame = uncrackable** by these families.

The one robust, policy-reachable lever is **search depth**, which trades catches for
per-frame compute (+5–7% at 2×, +12% at 4.4×, +22% at 18×) and is bounded by phone
smoothness (deployed plan spike ~21ms; 2× ≈ 28ms, 4.4× ≈ 47ms — dep2+ risks hitches).

**What "3×" actually means here, honestly:** the deployed policy's quoted ~12/1500f
was a 1680²-square number; on real phones/iPads/laptops the *same shipped policy*
already catches ~20–26 (≈1.6–2.0×) because a smaller torus traps the faster prey.
That ~2× is already live. Beyond it, the realistic best deployable policy is ≈ the
current one + an optional 2× deeper search for +5–7% (if the user accepts the spike).

## Recommendation

1. **Keep the deployed policy** as-is — it is near the practical ceiling; param-tuning
   and wrap give nothing reliable, and deeper search costs phone smoothness for +5–7%.
2. If the user wants the marginal gain, ship the `{K_roll:6,Hs:90,D:12}` (2×) config —
   validated +5–7%, plan spike ~28ms (likely fine on modern phones; A/B on a low-end
   device first). This is the only change with a real, low-risk gain.
3. The one untried high-effort lever is **AlphaZero-style distillation of the +22%
   heavy-search ceiling into the cheap config via a retrained value net** — but prior
   distillation already plateaued at ~90% of the planner, so expected marginal upside
   is small; available on request.
