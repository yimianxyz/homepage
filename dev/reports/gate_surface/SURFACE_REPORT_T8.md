## side-a → GPU THROUGHPUT SURFACE complete: T*-vs-screen + the auto-capture verdict

Full batched GPU sweep done (sim_torch port of the deployed gated policy, 10 screens × {count, area-scaled density, reach-time horizon} × 32 paired seeds, startBoids=28 late-game scatter, packed). **Reminder (cross-check): GPU is decisive only for RELATIVE comparisons** — clearRate=1.00 GPU==JS, endgame NN bit-exact, but GPU throughput runs ~1.5× the JS level (late-game sim drift). T*-vs-screen trend + rule-vs-rule are robust (offset cancels); side-b's JS pins absolute T*.

### 1. count-T* rises only WEAKLY with screen (~A^0.2 GPU / ~A^0.3 JS) and the curves are FLAT
Per-screen count-T* (GPU argmax; plateaus are flat so argmax is noisy): 390→5, 414→8, 768→8, 820→5, 1280→7, 1440→7, 1512→8, 1680→8, 1920→7, **2560→12**. Throughput varies only ~8–18% across T=3..12 → broad optima. **T\* ≈ 6–8 for typical screens, ~12 only at 2560×1440.** prod T=5 sits at/below the low edge everywhere.

### 2. ⇒ NO adaptive scaling rule is needed or helps — a single FIXED T≈8 captures the optimum
Best single setting across ALL 10 screens (gap vs per-screen count-T*):
- **fixed count T=8: mean gap 1.9%, worst 5.6%** ← winner, no per-screen tuning
- fixed T=5 (prod): mean 6.5%, worst 11.9%  •  fixed T=10: mean 3.5%, worst 7.8%
- **area-scaled density** (best Tref=5): mean 5.0%, worst 11.9% — WORSE than fixed T=8. Its optimal Tref *decreases* with screen (10 on small → 3 on big) = the signature of **OVER-correction** (linear-area scaling too aggressive vs the true ~A^0.2–0.3).
- **reach-time horizon** (wa0>H): mean ~20%, worst ~28%, screen-dependent h* — clearly worst; does NOT auto-capture.

### 3. Answer to "does the horizon/adaptive rule auto-capture with one setting?"
**No — and none is needed.** Because T* scales so weakly with area and the throughput-vs-T curves are flat, the "one setting" that works is simply a **fixed count threshold ≈ 8** (within ~2% mean of per-screen-optimal across mobile→desktop, beating prod T=5). Adaptive rules (area density, reach-time horizon) do NOT improve on it — density over-scales, horizon underperforms. Only the very largest screen (2560) mildly prefers higher (T≈12), so if one extra knob is ever wanted, it's a tiny bump at the top end, not a full scaling law.

### Caveat + handoff
GPU understates big-screen T* (1.5× late-game offset compresses the tail) — JS gave higher T* (7→10→12), so the **robust fixed value is likely T≈8–10**, not exactly 8. **side-b's paired farm decides the exact T + significance.** Net recommendation for the decider: **a single fixed T in ~8–10 beats prod T=5 and obviates per-screen/adaptive tuning.** Raw surface JSONL + analyzer committed (side-a/exact-nn-oracle).
