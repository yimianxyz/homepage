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

## FINAL RESULTS (complete 21-cell grid: 14 deploy + 7 cross, n=200 paired seeds/cell)

### Per-screen verdict (held-out, deploy boid counts). T=8 BEATS T=5 on every
### desktop/laptop screen and TIES mobile; per-screen best-T is a TIE with T=8 EVERYWHERE.
| deploy screen | N0 | T=8 vs T=5 | per-screen best-T vs T=8 (the dynamic gain) |
|---|---|---|---|
| 360×800 | 60 | −0.9% tie (p=.37) | T9: +1.3% **tie** (p=.11) |
| 390×844 | 60 | +0.7% tie (p=.52) | T9: +1.2% **tie** (p=.18) |
| 393×852 | 60 | +1.5% tie (p=.14) | T8 is best |
| 412×915 | 60 | +0.9% tie (p=.50) | T6: +0.3% **tie** (p=.76) |
| 414×896 | 60 | +0.2% tie (p=.83) | T6: +0.6% **tie** (p=.48) |
| 1280×720 | 120 | +1.7% **beats** (p=.005) | T8 is best |
| 1366×768 | 120 | +1.4% **beats** (p=.005) | T8 is best |
| 1440×900 | 120 | +1.4% **beats** (p=.0003) | T9: +0.3% **tie** (p=.81) |
| 1536×864 | 120 | +1.6% **beats** (p=.002) | T7: +0.2% **tie** (p=.68) |
| 1512×982 | 120 | +1.9% **beats** (p=.0001) | T9: +0.3% **tie** (p=.37) |
| 1600×900 | 120 | +1.2% **beats** (p=.02) | T12: +0.2% **tie** (p=.63) |
| 1680×1050 | 120 | +1.2% **beats** (p=.02) | T9: +0.7% **tie** (p=.19) |
| **1920×1080** | 120 | **+2.9% beats (p<1e-4)** | **T11: +0.6% tie (p=.54)** |
| 2560×1440 | 120 | +3.1% **beats** (p<1e-4) | T10: +0.4% **tie** (p=.39) |

**No deployment screen has a per-screen optimum that significantly beats T=8** (all p>0.10).
The throughput-vs-T curve is a broad flat plateau (T≈7–12) on every common screen.

### Decomposition `log T* = log c + α·log N0 + q·log A` (complete grid):
- **α (N0) = 0.02 ± 0.04, t=0.50 → NOT significant** (linear: ΔT/+60 boids = 0.10, t=0.42).
  **θ·N0 REFUTED** — boid count does not drive T*.
- **q (area) = 0.062 ± 0.017, t=3.63 → significant** (linear: ΔT/×2 area = 0.40, t=3.70).
  **side-a vindicated: the area effect is real at fixed N0** — but tiny: across the full 10×
  area range the optimum drifts only ~8→10, entirely inside the flat plateau.

### Formula fit (prevalence-weighted throughput, N0=deploy):
best continuous `c·N0^α·A^q` (c=3.5, α≈0, q≈0.06–0.2) = **+0.47%** vs fixed-8; device-step
(mobile 9 / desktop 8) = +0.42%; **on 1920×1080: +0.35–0.42%; on mobile ~0%.** **All ≪ 2%.**

## ✅ FINAL VERDICT: ship **fixed T=8**. The deep dynamic-formula search is conclusive —
no `T(screen,N0)` clears the 2% complexity bar; the per-screen optimum is a statistical tie
with T=8 on every common screen (mobile→4K). The optimum's only real driver is screen AREA
(q≈0.06, t=3.6) and the drift is throughput-negligible; boid count has no effect (θ·N0 refuted).
T=8 also confirms the prior T=5→8 move deployment-faithfully (+1–3% on every desktop, p<0.03;
mobile tie). Agrees with side-a's GPU surface + the adversarial check (lead). _One char, done._

### SEALED p2 confirmation (out-of-sample, fresh HMAC block, n=200) — REPLICATES exactly:
| KEY screen | N0 | T=8 vs T=5 | best-T vs T=8 | T=8 in plateau? |
|---|---|---|---|---|
| 360×800 | 60 | +1.4% tie (p=.09) | T10 +1.5% tie (p=.12) | yes |
| 390×844 | 60 | +0.4% tie (p=.75) | T7 +0.4% tie (p=.61) | yes |
| 414×896 | 60 | +0.1% tie (p=.90) | T9 +0.6% tie (p=.31) | yes |
| 1366×768 | 120 | +1.8% **beats** (p=.001) | **T8 IS the peak** | yes |
| 1920×1080 | 120 | +2.2% **beats** (p<1e-4) | T9 +0.2% tie (p=.74) | yes |
On the sealed block T=8 sits in the plateau on every key screen (the peak on 1366); no
per-screen optimum significantly beats T=8 (all p≥0.12). Identical conclusion to held-out
→ no overfit. **LOCKED: ship fixed T=8.**

---
### (earlier interim section, superseded by the table above)

### Headline: the throughput-vs-T curve is a BROAD FLAT PLATEAU (T≈7–12); fixed T=8 is
### statistically indistinguishable from the per-screen optimum on EVERY common screen.

**T=8 vs per-screen peak** (deploy cells): T=8 is within 0.0%…−1.0% of the per-screen
argmax everywhere (390 −1.0%, 1920 −0.5%, 2560 −0.3%). Every per-screen best-T vs T=8 is a
statistical TIE (paired bootstrap CI includes 0, Wilcoxon p>0.18):
| screen | N0 | T* (argmax) | plateau | bestT vs T8 |
|---|---|---|---|---|
| 390×844 | 60 | 9 | [5–12] | +0.4% tie (p=0.18) |
| 414×896 | 60 | 6 | [5–12] | +0.7% tie (p=0.48) |
| 1366×768 | 120 | _(round-2)_ | | |
| 1600×900 | 120 | 12 | [5–12] | +0.2% tie (p=0.63) |
| 1680×1050 | 120 | 9 | [7–12] | +0.5% tie (p=0.19) |
| 1920×1080 | 120 | 11 | [8–12] | +0.4% tie (p=0.54) |
| 2560×1440 | 120 | 10 | [8–12] | +0.2% tie (p=0.39) |

**T=8 vs T=5** (confirms the prior T=5→8 move, deployment-faithfully): BEATS on big screens
— 1920 +1.93e-4 (+2.9%, p<1e-4), 2560 +1.62 (+3.1%, p<1e-4), 1600 (p=0.023), 1680 (p=0.017);
TIE on mobile (390 p=0.52, 414 p=0.83). So T=8 ≥ T=5 everywhere, significantly on big screens.

### Decomposition — `log T* = log c + α·log N0 + q·log area` (n=15, R²=0.74):
- **α (N0 exponent) = −0.005 ± 0.04, t=−0.14 → NOT significant.** Boid count does NOT drive
  T*. **θ·N0 REFUTED** (the panel's leading hypothesis is wrong).
- **q (area exponent) = 0.085 ± 0.017, t=5.06 → significant but TINY.** side-a was right that
  area matters at fixed N0 — but ×2 area → T×1.06; across the full 10× area range T drifts
  only ~8→10, all inside the flat plateau (<0.5% throughput).

### Formula fit — nothing beats fixed-8 meaningfully (prevalence-weighted throughput):
best continuous (`a+b·√A` ≈ `c·N0^α·A^q`, c=3.49 α=−0.15 q=0.12) = **+0.22%** vs fixed-8;
device-step (mobile 9 / desktop 9) = +0.20%; on **1920×1080** the best formula gains **+0.42%**;
on mobile ~0%. **All ≪ the 2% complexity bar.**

## VERDICT: ship **fixed T=8** (one-char change, already deployed). The deep dynamic-formula
search confirms it — no T(screen,N0) clears the complexity bar; the per-screen optimum is a
statistical tie with T=8 everywhere. The mild real area-drift (q≈0.085) is throughput-negligible
and N0 has no effect (α≈0). _(Sealed p2 confirmation + complete confound cross: in progress.)_
