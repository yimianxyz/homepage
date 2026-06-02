# RL v2 — batched-ES training on GPU sim_torch (2026-05-22/23)

## Setup

Three parallel ES variants on three L4 spot VMs, all using the new
sim_torch sequential+graph eval (B = K*S batched into one CUDAGraph
per frame):

| VM | init | sigma | K | top_k | max_step_norm | P |
|----|------|-------|---|-------|---------------|---|
| 1  | shipped (JS 24.25)        | 0.05 | 128 | 8  | 0.02 | 154 |
| 2  | H=16-embed-shipped        | 0.05 | 128 | 8  | 0.02 | 610 |
| 3  | v6 = rule_v2 distill (JS 22.6) | 0.05 | 128 | 8 | 0.02 | 194 |

K=128 batched candidates + baseline = B=(K+1)·S = 2064 elements in a
single sim_torch run per gen. ~4 min/gen on L4. 300 gens planned each.

## What happened

After 50 gens (~3.5 hours of training per VM), JS-verifying ten
representative checkpoints reveals a sharp pattern:

### Sim_torch baseline drifts but doesn't climb

| VM | gen 0 | gen 10 | gen 20 | gen 40 | best_so_far peaks (sim_torch) |
|----|-------|--------|--------|--------|--------------------------------|
| 1  | 6.06  | 5.81   | -      | 5.06   | 8.25 (single perturbation) |
| 2  | 5.00  | 5.5    | -      | 5.875  | 8.06 |
| 3  | 5.69  | 5.94   | 5.94   | 6.06   | 8.13 |

### JS verification of best candidates ("highest sim_torch perturbation")

| Source | sim_torch (1500 frames) | JS (16 seeds × 5000 frames) |
|--------|------------------------:|----------------------------:|
| shipped (reference)        | 6.06 | 24.25 |
| v6 init (reference)        | 5.69 | 22.6 |
| VM 1 best.json (gen 1 pert)| 7.875 | 19.94 – 22.1 |
| VM 3 best.json (gen 14 pert)| 8.125 | **18.25** |
| VM 3 best.json (earlier)   | 7.69  | 21.19 |

The sim_torch tail (top perturbation) systematically scores *below*
shipped in JS. Higher sim_torch ≠ higher JS at the tail — the ρ=0.55
matters most where it hurts.

### JS verification of CENTRAL theta (smoothed ES iterate)

| Checkpoint | What it is | JS |
|------------|------------|-----|
| v6 (init)            | start of training | 22.6  |
| VM 3 ckpt_gen0010    | central theta after 10 gens | **23.19** (+0.6 vs init) |
| VM 3 ckpt_gen0020    | after 20 gens | 22.69 (drifted back) |
| VM 3 ckpt_gen0030    | after 30 gens | 20.38 (degraded below init) |

The central theta DID find a real JS improvement of +0.6 catches over
the v6 init in the first 10 gens. Then sim_torch's noise overwhelmed
the signal — by gen 30 the policy is *worse* in JS than the starting
point.

## What this tells us

1. **Sim_torch ES works some of the time.** Gen 10 of VM 3 hit a real
   JS-positive direction. The trainer just doesn't *recognize* it
   because the selection criterion is sim_torch, not JS.

2. **Sim_torch is rank-misleading at the tail.** The single best
   perturbation per gen — what naïve ES would "save" — JS-verifies
   poorly. The *mean* of many perturbations (central theta) is more
   reliable than the *max*.

3. **The +39% ceiling (shipped JS 24.25) holds.** Best RL result so
   far: **23.19** (v6 ES gen 10), still 1.06 below shipped.

4. **The signal is real but fragile.** ES is finding JS-improving
   directions but then drifting past them.

## What I'm doing about it

- **Killed VM 3's degraded run** (gen 30 onwards drifting away from
  the gen-10 peak).
- **Restarted VM 3** with the gen-10 ckpt as init, finer hyperparams
  (sigma=0.02, max_step_norm=0.01) and `ckpt_every=1` so every gen is
  saved and can be JS-verified externally. Goal: find the JS peak
  precisely.
- Wrote `dev/rl_train_jsaware.py` — a trainer that JS-verifies the
  central theta every N gens and saves the JS-best as `best.json`.
  Ready to deploy once we know how often to JS-check (the watcher is
  doing this manually now).

## What else might break the ceiling

- **Structural changes**: richer features (predicted catch
  interception points, flock cohesion, cooldown-aware planning),
  recurrent net, attention-over-boids. The +39% itself was structural
  (flock_centroid patrol target); the next breakthrough likely is too.
- **Multi-init random restart**: run many short ES runs from many
  inits and JS-verify peaks. The "gen-10 phenomenon" might be more
  common than current data shows.
- **JS-aware ES**: use JS as the selection signal directly (slow but
  exact); `rl_train_jsaware.py` is the framework.

## 2026-06-02 — cval+cls distillation is a dead end; ES sweep relaunched

**Distillation of the planner's per-frame centered gain (task #94) failed
across three architectures.** distill_v5 cval+cls trained to convergence
(80 epochs) on `ds1024_dense08.pt`, then closed-loop-evaluated:

| Encoder      | params | closed-loop mean | % of planner | vs shipped base |
|--------------|-------:|-----------------:|-------------:|----------------:|
| deepsets     | big    | 7.107 ± 0.17     | 33.2%        | −14.8%          |
| transformer  | 6L/192 | 6.068 ± 0.18     | 28.4%        | −27.3%          |
| crossattn    | 6L/192 | 5.875 ± 0.17     | 27.5%        | −29.6%          |

All three converge to the same wall (val decisive-choice acc ~0.18–0.24,
vs E3D-const 0.138) and all land *below* the shipped baseline in closed
loop. **Supervised distillation of the teacher's choices does not
transfer to closed-loop catches** — the per-frame argmax target is too
noisy/ill-posed (consistent with the earlier 0.55 classification wall,
tasks #86/#89). Conclusion: stop pursuing choice/value distillation.

**Pivot: direct closed-loop ES is the only line that has ever produced a
JS-positive direction (the gen-10 +0.6 above).** Relaunched a 3-VM ES
sweep (rl_train_v2, P=154 radial, frames=5000, rotate_seeds):

| VM | sigma | K  | lr  | top_k | seed | out |
|----|-------|----|-----|-------|------|-----|
| 1  | 0.02  | 64 | 0.5 | 8     | 1234 | es_stage2_shipped |
| 2  | 0.04  | 64 | 0.5 | 8     | 7777 | es_vm2_sig04 |
| 3  | 0.02  | 96 | 0.8 | 12    | 4242 | es_vm3_K96 |

**CRITICAL — do not trust sim_torch best_so_far** (e.g. VM1's 25.15): the
ρ=0.55 tail-misleading result above still holds. The reliable signal is
JS-verifying the *central theta* per gen (ckpt_every=1 saves all). Next:
JS-verify the central-theta peaks of these three runs; if none beats
shipped 24.25, the small radial net is capacity-bound and the real move
is a **deeper net** (user's explicit ask) with JS-aware selection.
