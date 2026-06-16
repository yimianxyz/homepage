# Dynamic split threshold T(screen, N0) — JS-authoritative findings (side-b #6 DEEP)

Goal: find the best **dynamic** planner/endgame split threshold `T(screen)` that beats
the fixed `T=8` recommendation, optimised for the screens users actually have (mobile +
common desktop, esp. **1920×1080** and **390/414-class mobile**).

Metric = **THROUGHPUT** = catches / frames-to-extinction, on **deployment-faithful**
games: boids spawn at the prod clustered point and the predator plays to extinction.
Clear-rate is ~100% everywhere (the endgame interceptor never gets stuck), so throughput
is the sole objective. Paired seeds (same seed → identical game until the gate first
diverges at N≤T) give a Wilcoxon/​bootstrap signal with the shared planner prefix cancelled.

## REGIME LOCK (agreed with lead + side-a)
- The formula's **N0 input = the DEPLOY boid count**: mobile (UA→) **60**, desktop **120**.
- Throughput is measured at those deploy counts. Forced boid counts appear **only** in the
  confound cross (to fit the exponents), never in a deployed-throughput number.
- Mobile faithfully = UA-mobile: 60 boids, predator range **80** (prod's load-order bake —
  UA changes NUM_BOIDS/REFRESH only, not range; confirmed with side-a), frameMs=12 (team corpus).

## Method
1. **Fork-based T\*-runner** (`verifier/endgame_fork.js`). The gate only acts at N≤T (T≤12),
   so the expensive high-N planner PREFIX (N0→13) is identical across all T. Run it once
   per (screen, seed); snapshot the full sim (deep-clone graph + exact mulberry32 RNG
   restore + virtual clock + policy closure); then fork the cheap low-N endgame per T.
   **Validated BITWISE** vs the trusted full-game harness (`diff_harness.runGame`):
   frames/eaten/cleared + FNV-64 trajDigest identical, incl. forced-N0 cross. ~5–10×.
2. **Grid** (`verifier/tstar_farm.js`), T∈{1..12}, ≥200 paired held-out seeds:
   - **Deployment cells** (real N0): mobile 360×800/390×844/393×852/412×915/414×896 @60;
     laptop 1280×720/1366×768/1440×900/1536×864/1512×982 @120;
     desktop 1600×900/1680×1050/1920×1080/2560×1440 @120.
   - **Confound cross** (separate area-effect from N0-effect): {390, 820, 1366, 1920, 2560}
     each at **both 60 and 120** boids (ua=0, forced count) — breaks the area↔N0 collinearity.
3. **Decomposition** (`verifier/tstar_fit.js`): fit `log T* = log c + α·log(N0) + q·log(area)`
   (OLS on the robust plateau-center) → the **area exponent q** and **N0 exponent α** each
   controlling for the other. Combine to `T = round(c·N0^α·A^q)`; also grid-search (c,α,q) to
   maximise prevalence-weighted throughput at deploy counts. Compare to fixed-8/fixed-5 and a
   device-class step. Prevalence weights ≈ StatCounter 2025 (1920×1080 top desktop; 360×800,
   390×844 top mobile), mobile ≈55%; a uniform-within-class sensitivity check is reported too.
4. **Anti-overfit verdict**: fit on held-out (seed≥270000), **confirm on a SEALED p2 block**
   (HMAC salt, never revealed). Per KEY screen (1920×1080, 1366×768, 390×844, 414×896):
   throughput + bootstrap CI + Wilcoxon p vs fixed-8 AND fixed-5 + %gain, at deploy counts.
5. **COMPLEXITY BAR**: if the combined formula beats fixed-8 by **<2% on 1920 AND mobile**,
   ship a **3–4 row device-class step / lookup**, not a continuous formula.

## RESULTS
_(held-out farm running on local 4 cores + VM3 g2-standard-4, bitwise-identical; pending)_

<!-- to fill: surface table, α/q decomposition, winning formula, sealed per-screen CIs -->
