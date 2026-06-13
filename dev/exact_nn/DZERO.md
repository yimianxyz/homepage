# Deliverable Zero — coordinate-dedup top1−top2 margin CDF

**74748** plan decisions (116 games), 6 device cells, **none-profile (no spawns), planner regime (N≥6 / D1) only.** Margin = top1−top2 over coordinate-deduped candidate groups (SPEC §3); every value replay-verified at log time AND independently re-derived (independent_margin_check.js, 0/74748 mismatches).

## What this number is — and is NOT
- **P(dmargin > τ)** = the maximum NN-alone **COVERAGE** of a prod-margin-gated hybrid (fraction the NN may decide alone); its complement is the **fallback load**. It is **NOT** an S_dec accuracy bound: `dmargin` is *prod's* margin, but L1h gates on the *student's* margin (SPEC §4c), so this CDF structurally cannot see L1h's only failure mode (student-confident-and-wrong).
- The **genuine σ-independent S_dec floor** is the **exact-tie rate**: when prod's own top groups tie bitwise (distinct coordinates, identical score), the committed target is set by the index tiebreak — unlearnable by a continuous net. A score/pointer student (L1s/L1p) misses ≥~½ of these.
- A real student with score-reconstruction error σ has S_dec ≈ 1 − ∫ φ(m)·Φ(−m/(√2·σ)) dm over the margin density φ — an **integral**, not a τ-lookup. The decisive unknown is the student's achievable σ; measure it on a small L1r run, then integrate against this CDF.

## Overall
- exact ties (dmargin=0, distinct coords) → **NN-alone S_dec floor 99.46%**: exact-tie 0.54% (407)
- winner is a ROLLED candidate (pidx[0:4]): **86.96%** — so ~13.04% of decisions commit a non-rolled vprior candidate (L1r scores those exactly)
- **L1r exact-for-free** (committed coord immune to ANY rolled-score perturbation): 0.01%
- single-coordinate-group plans: 0.00%; plans/game 644.4

CDF P(dmargin ≤ τ):

| τ | 0 | 1 | 2 | 5 | 1e-12 | 1e-9 | 0.000001 | 0.0001 | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P(≤τ) | 0.54% | 92.84% | 98.54% | 99.99% | 0.54% | 0.54% | 0.56% | 1.84% | 8.05% | 30.76% | 56.32% | 67.40% | 79.67% | 85.43% |

NN-alone coverage ceiling = P(dmargin > τ) (NOT S_dec — see above); fallback load = P(≤τ):

- τ=0.000001: coverage **99.44%**, fallback load 0.56%
- τ=0.001: coverage **91.95%**, fallback load 8.05%
- τ=0.01: coverage **69.24%**, fallback load 30.76%
- τ=0.1: coverage **32.60%**, fallback load 67.40%

## By device cell × N bucket (SPEC §6.3 stratification)

exact-tie shown as rate (count), or `0 (≤rule-of-three ceiling)` when none observed.

| cell | N | n | games | exact-tie → S_dec floor | P(≤1e-3) | P(≤1e-2) | P(≤0.1) | median dmargin | winner-rolled |
|---|---|---|---|---|---|---|---|---|---|
| desk_1024x768 | 6-14 | 1778 | 16 | 0 (≤0.17%) → 100.00% | 9.22% | 38.92% | 76.77% | 0.01850 | 82.56% |
| desk_1024x768 | 15+ | 7994 | 16 | 1.43% (114) → 98.57% | 6.96% | 25.29% | 60.72% | 0.05234 | 91.28% |
| desk_1512x982 | 6-14 | 2378 | 16 | 0 (≤0.13%) → 100.00% | 9.80% | 38.86% | 75.23% | 0.02193 | 81.67% |
| desk_1512x982 | 15+ | 11118 | 16 | 0.34% (38) → 99.66% | 6.22% | 26.21% | 62.23% | 0.04813 | 87.94% |
| desk_1680x1050 | 6-14 | 2849 | 16 | 0 (≤0.11%) → 100.00% | 12.81% | 42.08% | 81.85% | 0.01566 | 84.84% |
| desk_1680x1050 | 15+ | 12161 | 16 | 0.35% (43) → 99.65% | 6.55% | 27.04% | 63.73% | 0.04434 | 86.65% |
| desk_2560x1440 | 6-14 | 2883 | 12 | 0 (≤0.10%) → 100.00% | 14.39% | 45.75% | 83.18% | 0.01301 | 78.84% |
| desk_2560x1440 | 15+ | 13612 | 12 | 0.07% (10) → 99.93% | 6.66% | 28.37% | 65.74% | 0.03906 | 84.25% |
| ipad_820x1180 | 6-14 | 2896 | 24 | 0 (≤0.10%) → 100.00% | 10.98% | 42.13% | 79.90% | 0.01502 | 83.70% |
| ipad_820x1180 | 15+ | 7923 | 24 | 0.19% (15) → 99.81% | 7.55% | 30.71% | 67.90% | 0.03240 | 88.35% |
| iphone_390x844 | 6-14 | 2669 | 32 | 0.04% (1) → 99.96% | 10.15% | 38.10% | 75.31% | 0.01968 | 88.09% |
| iphone_390x844 | 15+ | 6487 | 32 | 2.87% (186) → 97.13% | 10.79% | 32.37% | 66.61% | 0.03232 | 93.19% |

## L1r risk surface — flip margin (only the 4 rolled scores can move the decision)

Excludes exact-for-free plans (no rolled candidate can flip the committed coordinate). l1rMargin is a CONSERVATIVE proxy (uses current group maxes → over-states risk).

| cell | N | exact-for-free | movable plans | P(≤1e-3) | P(≤1e-2) | P(≤0.1) | median |
|---|---|---|---|---|---|---|---|
| desk_1024x768 | 6-14 | 0.00% | 1778 | 8.55% | 34.98% | 74.02% | 0.02415 |
| desk_1024x768 | 15+ | 0.00% | 7994 | 6.81% | 23.81% | 57.71% | 0.06381 |
| desk_1512x982 | 6-14 | 0.04% | 2377 | 9.59% | 36.60% | 73.12% | 0.02574 |
| desk_1512x982 | 15+ | 0.00% | 11118 | 5.97% | 24.00% | 58.14% | 0.06239 |
| desk_1680x1050 | 6-14 | 0.00% | 2849 | 12.32% | 38.96% | 79.78% | 0.01891 |
| desk_1680x1050 | 15+ | 0.00% | 12161 | 6.32% | 24.96% | 59.21% | 0.05879 |
| desk_2560x1440 | 6-14 | 0.07% | 2881 | 13.50% | 41.31% | 80.56% | 0.01860 |
| desk_2560x1440 | 15+ | 0.00% | 13612 | 6.36% | 25.96% | 60.43% | 0.05405 |
| ipad_820x1180 | 6-14 | 0.07% | 2894 | 10.68% | 39.32% | 77.40% | 0.01976 |
| ipad_820x1180 | 15+ | 0.00% | 7923 | 7.26% | 28.13% | 63.97% | 0.04322 |
| iphone_390x844 | 6-14 | 0.04% | 2668 | 9.75% | 35.79% | 73.61% | 0.02305 |
| iphone_390x844 | 15+ | 0.00% | 6487 | 10.51% | 31.17% | 64.84% | 0.03720 |

## Caveats (scope of this estimate)
- **Optimistic lower bound on tie/near-tie density:** none-profile only. Spawn / 5→6→5 re-crossing / same-coord double-tap states (SPEC §5, §7 top risk) contribute ZERO plans here and plausibly carry a heavier tail; measured next via `shard_runner --spawnFrac`.
- **D1 only:** the N≤5 intercept/egBoid-commit decision (D4 / L1e) has its own, still-unmeasured margin distribution.
- **Autocorrelation:** plans within a game-to-extinction are correlated; effective n < n, especially the thin 6-14 cells. Per-cell tail percentiles below ~3/n are counting-noise dominated.
- **Train seeds only** (100000–160000, all < 270000): the sealed verification set (≥270000) is untouched.

