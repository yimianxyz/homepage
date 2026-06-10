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

## Levers tested

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
- **Ambush** (loiter on the flock flow-line outside PREDATOR_RANGE): under test.

## Emerging feasibility verdict (pending ambush + cross-check)

Literal 3× as a single-device policy improvement is **not physically reachable**:
the N cap forbids it on phones, and the patrol-family ceiling is only ~+17% over
deployed. The achievable win is (a) the screen-size reframing (already ~2×), plus
(b) a modest stack of deployable improvements (stronger-but-cheap search + wrap on
large screens + per-device tuning ± ambush) likely totalling +20–40% over deployed.
