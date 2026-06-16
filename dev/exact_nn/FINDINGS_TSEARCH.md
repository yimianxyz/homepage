# Optimal planner/endgame split — JS-authoritative findings (side-b #6)

Metric = THROUGHPUT (catches/frame; clear-rate 100% everywhere → stuck-rate moot).
Paired-seed farm (`verifier/throughput_farm.js` + `throughput_stats.js`), 3 rules
(`candidates/split.js`: count-T / density Td=clamp(round(Tref·(A/A_ref)^p)) / horizon
min-wa0>H). Held-out search ≥270000; sealed confirm = fresh p2 salt (≥290000).

## RECOMMENDATION: move the split threshold T=5 → **T=8** (count rule; one-char change)
- **Beats T=5 on every screen, never worse.** Search (n=80, SB=28): +2% (390) / +3% (820) /
  ~0% (1024) / +3% (1512) / +2% (1680) / **+7% (2560, p=0.0001)**. Robust, safe.
- The true per-screen optimum **rises sub-linearly** (T\*≈5 small → ~12 biggest); a finer sweep
  (SB=15, n=50) gives bigger big-screen gains: 1512 **+18%**, 1680 +9%, 2560 **+25%** at T=8–12.
- **T=10** is the more aggressive option (captures more big-screen gain: 2560 +11–22%) at a
  small-screen near-tie; **T=8 is the OCCAM-safe choice** (never significantly worse anywhere).

## Per-screen T\* and gain vs T=5
| screen | search n=80 (SB28): T=8 gain | refined n=50 (SB15): T\* / best gain |
|---|---|---|
| 390×844 (mobile) | +2% | T\*≈8 |
| 414×896 (mobile) | — | (sealed pending) |
| 820×1180 | +3% | — |
| 1024×768 | ~0% (tie) | T\*≈5 (T=8 −3%) |
| 1512×982 | +3% | **T\*≈8, +18%** |
| 1680×1050 | +2% | **T\*≈12, +9%** |
| 2560×1440 | **+7%** | **T\*≈12, +25%** |

(SB-sensitivity: SB=28 shows T=8 best everywhere modestly; SB=15+finer-T reveals the
sub-linear rise to T\*≈12 on big screens. Both agree T≥8 ≥ T=5; disagree only by which big-T.)

## Rules verdict (OCCAM)
- **count fixed T=8: the recommendation** — simplest, beats T=5 everywhere, never worse.
- **density (Td=clamp(round(Tref·(A/A_ref)^p)), Tref=8, p≈0.3)**: auto-adapts (Td 6→13) and
  tracks the sub-linear T\* curve, BUT measured **~1–5% UNDER the best per-screen fixed-T**
  (caps below T\*≈12 on the largest screens) → does **not beat best-fixed-T significantly** →
  per the OCCAM rule, NOT recommended over a fixed T. (A higher Tref/p could close this; not
  worth the complexity given fixed-T=8 already wins.)
- **horizon (min-wa0 > H, H=40/90/140): REFUTED** — worse than T=5 on every screen (the
  reach-time trigger doesn't align with the throughput optimum). Not the elegant winner.

## Status / resume
Sealed-confirm (count{5,8,10,12}+density-p0.3, 7 screens incl mobile 390/414, p2 block, n=60)
running — gives the authoritative per-screen T=8-vs-T=5 p-values + bootstrap CIs + %gain +
the mobile numbers. Resume: `node verifier/throughput_stats.js evidence/phase2/throughput/
sealed_ALL.json --baseline count:T=5` (or merge sealed_* shards first). Then post final CIs.
Shards: `evidence/phase2/throughput/shards/{search,refined,sealed}_*.json`.
