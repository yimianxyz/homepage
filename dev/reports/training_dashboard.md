# Predator RL — current training dashboard

Last updated: 2026-05-21 22:14 UTC.

## Best policy in JS (deployment ground truth)
- **shipped** (c3v3_H4, dataset_v3 random-patrol training): **24.25 catches** (16 seeds × 5000 frames, flock_centroid patrol).
- +39% over original baseline (17.44); z=3.55 train, z=3.59 holdout.

## Pivot — sim_torch ES doesn't transfer to JS

VM 2 ARS reached sim_torch baseline 10.84 (gen 3) → JS only 18.44 (vs
supervised init 20.56 and shipped 24.25). Even though sim_torch
catches improved, JS performance **regressed**.

Confirmed the rank-discordance: sim_torch is unsafe as an optimization
target for JS-deployed policies. Killed VM 1 (stuck in greedy mode)
and VM 2 (oscillating, not transferring). VM 3 keeps running (random
seeds, exploring widely).

## Running experiments (current)

| machine | experiment | progress | est. wall |
|---------|-----------|----------|-----------|
| VM 1 (us-central1-a) | **JS-native** rl_es σ=0.10, K=20 tries on shipped | tries 0/20 | 4 hr |
| VM 2 (us-central1-a) | **JS-native** rl_es σ=0.30, K=20 tries on shipped | tries 0/20 | 4 hr |
| VM 3 (us-central1-c) | GPU ARS H=64, σ=0.10, random seeds (kept running) | gen 1/200 | 50 hr |
| local CPU | JS rl_es σ=0.20, K=20 tries on shipped | tries 0/20 | 4 hr |
| local watcher | every 30 min: pull VM3 best.pt → JS verify | continuous | indefinite |

All GPU runs are **warm-started** from a supervised H=64 distillation of
the rule (dev/checkpoints/supervised_H64_init.pt; 23.56 catches in
sim_torch on seeds 100..115, 20.56 in JS).

## Sim_torch caveat

sim_torch (parallel boid updates) gives different per-NN rank than JS
(sequential updates). Spearman ρ ≈ 0.17 between the two sims on the
K=20 random-init NNs we earlier checked. Mean catches for the SHIPPED
weights coincide within 1 catch (sim_torch ~22, JS ~24), so
optimization in sim_torch is a noisy proxy for JS — top picks need
JS verification before declaring victory.

## Queued for next launches (ready code, not running)

1. **Set Transformer + ES** (`dev/rl_train_set_transformer.py`): per-boid
   permutation-invariant attention. 178k params vs 2434 MLP. Will
   require supervised pretrain too. Push code already on all 3 VMs.
2. **PPO with actor-critic**: not implemented yet.
3. **CMA-ES**: not implemented; standard library.
4. **Curriculum learning**: easier env → scale up.

## Idea backlog
See `dev/reports/research_summary.md` and the catalog of ~135 ideas
covered in conversation. Categories: arch, activations, output heads,
training algorithms, loss/objective, features, transfer/pretrain,
optimizer, regularization, domain randomization, sequence/memory,
larger archs, smarter exploration, trajectory training, ensembles,
curriculum, special-purpose archs, RL refinements, smaller-but-better,
multi-task, transformer transfer, hybrid algorithmic, misc.

## Cost watch

3 spot L4 VMs × $0.21/hr × 50 hr ≈ $32 for full ARS run. Watcher
runs on local. JS ES on local CPU (no $).
