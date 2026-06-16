## side-a → #5/#6: GPU fine-T surface → the best dynamic gate is a MINIMAL 2-tier step (not √area, not θ·N0)

Full GPU sim_torch surface done: fine T={1..12} × 14 REAL common screens (mobile 360-414 / laptop 1280-1600 / desktop 1680-2560) × 48 paired seeds, startBoids=28 late-game scatter, + a confound-cross at 1280×800 (N0=28 vs 60). Then a 3-way adversarial review (formula-advocate / step-advocate / skeptic + synthesis) to stress-test the fit. Data + fitter committed (`reports/gate_surface/`, `fit_formula.py`).

### What the surface says
- **throughput-vs-T is RISING-then-FLAT** (knee ~T4-5, flat plateau to T12). The gate has a *small* effect above the knee — curves vary only ~5-12% across T at 48 seeds (noisy).
- **θ·N0 (fraction rule) is REFUTED.** Confound-cross: throughput LEVEL rises with N0 but the optimal-T STRUCTURE is ~N0-INVARIANT (knee ~T5 at both N0=28 and 60). So the deploy "T* rises mobile→desktop" is **driven by SCREEN AREA, not boid count** — the 60/120 was collinear with area (spurious). `frac_N0` fit *worse* than fixed-8.
- best continuous form is **√area**: `lin_sqrtA T=clamp(round(5+0.004·√((W+20)(Hc+20))),1,12)`, meanGap 1.72% vs fixed-8 3.99%, prevalence-weighted **+2.73%**, **+4.67% on 1920×1080**. (power_A fits p≈0.30, NOT 0.5 — but √A-with-offset is the better form.)

### ⚠️ But the continuous formula is FRAGILE (adversarial review) — recommend a minimal step
The skeptic's robustness tests on the continuous fit:
- hold out 1920×1080 → gain collapses 2.73% → **2.06%**; ±1-unit coeff perturbation swings **−3.35% to +2.47%** (noisy local optimum).
- it MISPREDICTS mid screens (1280: formula T=9 vs measured T*=10-11) and LOSES on a few (1536×864 −3.5%, 9% share).
- the durable, sign-stable signal is ONLY **"bump large screens"**; mobile+laptop are flat and fixed-8 is already near-optimal there (the 1536 cell loses −7% if bumped to 9 — do NOT touch mid-size).

**RECOMMENDATION (the only sign-stable intervention, +1.62% GPU-weighted):**
```js
// screenArea = (W + 20) * (Hc + 20)
T = (screenArea >= 1.8e6) ? 11 : 8;   // large desktop (1680/1920/2560-class) -> 11; else keep fixed-8
```
Rationale: a genuine flat-top plateau on the dominant 1920×1080 (22% share) where *every* T in 9-11 beats T8 by ~+4%; the confound shows T* rises with N0, so a higher gate on large screens is the SAFE direction at deploy counts. Simpler than a continuous formula or a 3-tier step, isolates the gain to where the signal is real, no float edge cases.

### Caveats + what side-b must seal on JS (decisive)
- **GPU is RELATIVE** (~1.5× absolute offset; clearRate=1.0 GPU==JS, endgame NN bit-exact). All %gains are relative; magnitude is side-b's call. The offset's T-dependence is unverified → the *continuous* formula's fine T-pick wouldn't survive a T-dependent offset, but the coarse large-screen bump would.
- 48 seeds → flat/noisy curves. The exact large-screen value (10 vs 11 vs 12) is plateau-noise-bound (T12 went negative in some variants) — needs ≥100 seeds on 1920×1080 to pin.
- Fit at N0=28; deploy is 60/120. Confound supports extrapolation (N0-invariant structure) but **side-b must verify at deploy counts**: (1) large screens still prefer ~10-12 over 8 at N0=120; (2) the exact value; (3) the magnitude; (4) that mobile/laptop lose nothing staying at 8.
- **GPU deploy-count verification:** 414×896 @ N0=60 (mobile, realistic count) → flat/noisy (T8=12.2 mid-pack, argmax T*=6=noise) → **fixed-8 fine at deploy count, no formula benefit**. (1920×1080@120 desktop run was in progress; killed once side-b's held-out JS conclusively settled fixed-8.)

### FINAL (2026-06-16): side-b's decisive held-out JS + my GPU cross-check AGREE → fixed-8, no formula
side-b JS: α~0, q~0.085 (negligible), no dynamic formula beats fixed-8. side-a GPU (relative): α≈0 (confound knee=T5 N0-invariant), q≈0.21 (noise-bound), and the adversarial review found the best continuous form too fragile to ship. **VERDICT: keep the production gate near a fixed mid-value (~8); a dynamic T(screen) formula is NOT worth it.** GPU work done; VM1+VM2 stopped. Sealed-confirm is side-b on VM3/CPU.
