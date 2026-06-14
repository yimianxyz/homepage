# Phase-2 held-out ablation baselines (pre-handoff) — the NN-vs-raw-argmax frame

Run on **held-out calibration [270000,280000)** (NO sealed touch) with the
verdict_moe harness, before side-a's MoE landed. These are the deterministic
**ceiling** (oracle = perfect argmax of the visible scores) and **floor**
(raw_prior = no rollout / no NN) that side-a's `nn` S_dec will be read against.
They also validate verdict_moe at scale on the real distribution (not just the
buffer-seed self-test).

| mode | distribution | S_dec planner | S_dec endgame | S_dec pooled | gate | n |
|---|---|---|---|---|---|---|
| **oracle** | scatter (endgame) | — | **100.000%** | — | PASS | 5661 egCommits |
| **oracle** | natural (all-N) | **100.000%** | 100.000% | **100.000%** | PASS | 5449 plans |
| **raw_prior** | scatter (endgame) | — | **99.329%** | — | PASS | 5661 egCommits |
| **raw_prior** | natural (all-N) | **25.069%** | 100.000% | 25.601% | FAIL | 5449 plans |

oracle scatter is 100% on **all 6 device cells** (892–973 egCommits each).

## What this establishes (the honest NN-vs-raw-argmax story)
- **Ceiling = 100% (oracle).** The committed target/egBoid is *fully determined*
  by the scores/features the NN is fed: a perfect argmax/argmin of them reproduces
  prod exactly (5661 endgame + 5449 planner decisions, 0 disagreements). So a
  perfect MoE-NN can reach S_dec=100%; the only residual is the NN's continuous
  approximation flipping on near-ties (bounded by the Phase-1 margin CDF).
- **Planner: the rollout feature is load-bearing.** Prior-only argmax (vprior,
  NO rollout) matches prod just **25%**. Feeding the actual rollout outputs is what
  lifts the planner toward the 100% ceiling — this is the Phase-2 enabler and the
  reason the old ~37% rollout-bound ceiling is gone. The NN's job here is genuinely
  "argmax over the rollout-augmented scores" (per SPEC §1.4, allowed by design).
- **Endgame: the analytic feature does the heavy lifting.** The wrap-aware analytic
  intercept-time argmin (NO NN) already hits **99.3%** — above the 95% gate. So the
  endgame is structurally easy; the NN consumes this feature and refines it
  marginally (the L1e NN was ~98.2% standalone, i.e. slightly *below* the raw
  analytic — capacity doesn't obviously help here). Honest framing: the endgame
  "NN-driven" claim is real but the geometric prior is most of it.

## At handoff
Re-run `oracle` + `raw_prior` on the SAME held-out seeds alongside side-a's `nn`
and report all three side-by-side (the ablation). Then the one-shot SEALED nn
verdict @ offset 60 is the gate. JSONs: `abl_{oracle,rawprior}_{scatter,natural}_calib.json`.
