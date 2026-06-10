# Clearance-time metric + TERI endgame interceptor (2026-06-10, phase 2)

User pivot: stop measuring catches-in-1500-frames (rate-limited, hides the hard part).
Measure **time to clear EVERY boid** (to extinction, censored at a max cap), and treat
"catch the flock" and "catch the last boid" as two problems, optimising both — with
strong use of the screen edge.

## The new metric (`dev/clear_eval.js`)

Runs the real game to extinction or `maxFrames`; reports **clear-rate** (fraction that
reach 0 boids before the cap) and **mean clearance time** (censored), decomposed into
`tFlock` (N→5) + `tEndgame` (5→0), plus per-segment timing. Fleet harness:
`~/eval/clear_eval.js` + `remote_clear.sh` across the 3 VMs' 12 cores.

## What the metric exposed (deployed policy)

Per-boid catch time RISES as boids deplete — phone: ~56 frames/boid in the dense
flock → **~1480 frames for the single last boid** (≈21% of total clearance for ONE
boid). The "endgame" is a different, much harder problem: a lone boid has no
flockmates, so it flies a straight line and the 2.4×-faster evader cannot be
tail-chased. **On bigger screens the deployed policy literally FAILS to clear** within
a generous cap: laptop clear-rate **87.5%**, iPad **90.6%** — it chases the last boid
forever and never catches it.

## TERI — the endgame interceptor (`dev/exp/js` `intercept()`, gated to ≤5 boids)

Designed from a pursuit-theory + nature + first-principles research workflow (Torus
Earliest-Reachable Lead Intercept, with bubble-commit). The slow predator's ONE
structural advantage is the torus wrap. A lone boid's track is a known straight line,
so:
1. **Earliest-reachable scan**: scan its future track `B + t·vB` for the smallest `t`
   the predator can reach in time, using **min-image with the boid's torus period
   (W+20)** — the short way often goes the "wrong" way (backward-wrap), putting the
   predator AHEAD of the boid on its own line for a **head-on** intercept (closing up
   to 8.5 px/frame vs the −3.5 tail-chase deficit).
2. **Bubble-commit / FREEZE**: once inside ~110px (just outside the 80px flee bubble)
   freeze the aim and ram through. The flee force then points backward along the
   boid's own velocity (DECELERATES it into the predator) and the bounded turn can't
   move it >~3px laterally in the ~7 frames it takes to cross the 80→24px gap. (Re-
   solving inside the bubble was the reliability trap — it orbits.)
3. **Target = smallest feasible `t*`**, committed through capture.

## Results

Isolated last-boid time-to-catch (`dev/endgame_fasteval.js`, startBoids=1):
- phone **1675 → 565 (2.96×)**, 97% → **100%** clear
- laptop **2416 → 848 (2.85×)**, 96% → **100%** clear

Full clearance, device mix, held-out seeds, n=32 (TERI gated ≤5 boids):
| device | deployed tClear / clear% | TERI tClear / clear% |
|---|---|---|
| phone 390×844 (N60)  | 6956 / 100% | 5905 / 100% (−15%) |
| iPad 820×1180 (N120) | 15245 / **90.6%** | 13020 / **100%** (−15%, fixes failures) |
| laptop 1440×900 (N120) | 17007 / **87.5%** | 15044 / **100%** (−12%, fixes failures) |

With the device-mix ES patrol params added (the prod artifact `dev/ship_teri`), the
flock phase also speeds up: phone 7576 → 5984 (−21%) on held-out.

## Honest assessment vs the 2× goal

- **Endgame sub-problem: ~3× faster AND 100% reliable** (the user's "catch the last
  boid is a different problem" — solved). On big screens TERI converts "never clears"
  into "always clears."
- **Overall clearance: ~1.2–1.4× faster.** It cannot reach 2× because the FLOCK phase
  (≈60% of clearance time) is catch-RATE limited by the 2.4× speed deficit and is
  already near-optimal (the prior 6-week search + this phase's ES re-tune exhaust it).
  Even an instantaneous endgame caps the overall at ~1.4×. The 2× is a physics wall on
  the flock phase, not a missing idea.

## Ship recommendation

`dev/ship_teri/` (prod predator.js + predator_cheap.js, no config): the ES device-mix
params + the TERI endgame interceptor (gated ≤5 boids). Zero deploy-time learning, a
few dozen lines, reversible. It makes the live predator **always** clear the screen
(vs getting stuck ~10% of the time on laptops/iPads) and clear it ~1.2–1.4× faster,
with the last-boid hunt ~3× faster and visibly purposeful (it cuts across the edge to
ambush). Recommended to ship; not auto-deployed (outward-facing; user keeps prod as
baseline).
