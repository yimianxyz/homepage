# Phase-2 (pure MoE-NN) verdict — side-b resume contract

> Cold-resume note (post-/clear): this is the EXACT pipeline + run sequence for
> the Phase-2 S_dec verdict. Goal/spec: `SPEC_PHASE2_MOE.md`. Thread: issue #6.
> Branch: `side-b/exact-nn-moe` (worktree `/workspace/.team/wt-exact-nn-moe`).

## What Phase 2 is (vs Phase 1)
Phase 1 (L0+L1h+L1e) shipped a **bitwise-exact** policy with a deterministic
**fallback** (NN decided only when a certificate fired / margin≥τ). Phase 2 is a
**pure single MoE-NN, NO fallback**: the unified NN's argmax IS the committed
decision in every boid case (N≥1). Success = **S_dec ≥ 95%** output similarity to
prod, **pooled AND per-regime (planner/endgame) AND per device cell**. Not bitwise
(a continuous NN isn't bit-equal); behavioral decision-agreement is the metric.

## The instruments (all on this branch, self-validated)
- `diff_harness.js` — the certified lockstep instrument (Phase-1). Phase-2 added an
  **additive, default-off** `opt.forceSim` accumulator (per-frame cosine + rel-mag,
  per regime) — every prior metric + the bitwise path are untouched; `--selftest`
  green incl. the new `S_force(identity) cos≈1 & rel==1` assertion.
- `candidates/moe.js` — the **no-fallback unified MoE candidate**. Injects at BOTH
  prod decision anchors (planner: in `planCheap` AFTER the K_roll rollouts, before
  prod's argmax; endgame: `intercept`'s `if(!egBoid)`, before prod's argmin-scan).
  Modes (env `EXACTNN_MOE_MODE`): `nn` (side-a's model) | `oracle` (commits prod's
  exact pick — the ablation CEILING, S_dec must be 100%) | `raw_prior` (argmax of
  vprior alone / argmin of the wrap-aware analytic time — the ablation FLOOR, no
  rollout/no NN) | `perturb` (deterministic flip fraction — calibration check).
  Model contract (confirm at handoff; harness is agnostic):
  `loadMoePolicy(weights)(state,cfg) → {slot}` where planner slot∈0..15→cands[slot],
  endgame slot∈0..n-1→boids[slot]. Env: `EXACTNN_MOE_STUDENT`, `EXACTNN_MOE_WEIGHTS`.
- `verifier/verdict_moe.js` — the Phase-2 orchestrator: S_dec pooled/per-regime/
  per-cell + S_traj (fork) + S_force (lockstep & fork), the ≥95% gate, sealed/
  calibration seed selection. `--selftest` proves it on oracle (→100%) + perturb
  (→≈1−p) over BUFFER seeds [280000,290000) (no calib/sealed touch).
- Sealed machinery (Phase-1, unchanged): `verifier/seal_seeds.js`, salt at
  `~/.exactnn_seal_salt` (chmod 600), commitment `verifier/seal_commitment.json`.
  **Phase-2 uses a FRESH disjoint slice `--sealOffset 60`** (Phase-1 used 0/20/40).

## Handoff trigger
side-a delivers `moePolicy.js` + `moe_weights.json` (single unified NN) to
`/workspace/.team/exact-nn-moe-student/` (or pings #5/#6). The egboid/cands feature
record side-a logs already matches the adapter's packed record (cands+cp_features+
vprior+rolled catch/boot+rolled-mask | endgame scan-t feats; deduped committed-slot).

## THE RUN (one-shot, at handoff)
```
cd /workspace/.team/wt-exact-nn-moe/dev/exact_nn
S=/workspace/.team/exact-nn-moe-student            # side-a's drop

# 0. sanity: the model loads + routes (single forward pass, no hidden fallback)
EXACTNN_MOE_MODE=nn EXACTNN_MOE_STUDENT=$S/moePolicy.js EXACTNN_MOE_WEIGHTS=$S/moe_weights.json \
  node diff_harness.js --candidate candidates/moe.js --W 1024 --H 768 --seedStart 280500 --seeds 2 --json

# 1. CALIBRATION read (held-out [270000,280000), NOT sealed) — sets expectations
node verifier/verdict_moe.js --mode nn --student $S/moePolicy.js --weights $S/moe_weights.json \
  --calibration --natural --seeds 40 --out evidence/moe_calib_natural.json
node verifier/verdict_moe.js --mode nn --student $S/moePolicy.js --weights $S/moe_weights.json \
  --calibration --scatter --seeds 200 --out evidence/moe_calib_scatter.json

# 2. SEALED one-shot verdict @ offset 60 (THE gate). natural = deployable all-N pooled.
node verifier/verdict_moe.js --mode nn --student $S/moePolicy.js --weights $S/moe_weights.json \
  --natural --seeds 128 --sealOffset 60 --out evidence/moe_sealed_natural.json
# high-power endgame (scatter) for the per-regime endgame number
node verifier/verdict_moe.js --mode nn --student $S/moePolicy.js --weights $S/moe_weights.json \
  --scatter --seeds 512 --sealOffset 60 --out evidence/moe_sealed_scatter.json
# (shard the natural run across VM3's 4 cores + container cores by seed slices;
#  node v20.18.1 on VM3 is md5-identical ground truth.)

# 3. ABLATION (honesty: NN vs raw-argmax) — on CALIBRATION (don't spend sealed)
for M in oracle raw_prior; do
  node verifier/verdict_moe.js --mode $M --calibration --natural --seeds 40 --out evidence/moe_abl_${M}_natural.json
  node verifier/verdict_moe.js --mode $M --calibration --scatter --seeds 200 --out evidence/moe_abl_${M}_scatter.json
done
# report S_dec(nn) vs oracle [ceiling=argmax of visible scores] vs raw_prior [floor].

# 4. 4-angle adversarial audit (Workflow) + reveal salt (audit trail) + post #6.
node verifier/seal_seeds.js --reveal > evidence/moe_seal_reveal.json   # at verdict time only
```
Gate: `S_dec_pooled ≥ 95% AND S_dec_planner ≥ 95% AND S_dec_endgame ≥ 95%`
(verdict_moe exits 0 = PASS / 2 = FAIL). Report `PHASE2_VERDICT.md` + post to #6.

## Anti-Goodhart discipline (unchanged from Phase 1)
Sealed seeds never revealed to side-a, never touched during their training/arch-
select. The one-shot sealed run is the verdict. Held-out [270000,280000) is the
only set used for calibration/ablation. Ground truth is JS float64 (node) only.
Verify side-a's model is a genuine single forward pass with NO hidden fallback
(read the inference export; an `nn` vs `oracle` gap that exactly matches prod's
argmax would be a tell the "NN" is just calling prod's argmax).
