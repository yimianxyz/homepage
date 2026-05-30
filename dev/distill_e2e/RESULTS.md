# End-to-end predator NN — distillation search log

Goal (user): replace the production predator policy (E3D evolved patrol + 35-feature
builder + 35->4->2 chase NN) with ONE simple raw-observation end-to-end NN that
reproduces production behaviour. Stop only when both 100% match and minimality hold.

## M0 — feasibility (see M0_findings.md)
Exact per-seed catch identity is **physically impossible** (chaos: a 1e-7 steering
perturbation already decorrelates per-seed catches to chance; ε* < 1e-7). Goal reframed
to **behavioral equivalence**: match mean/distribution of catches (~8.1) and hunting.

## M1 — raw-obs encoding + on-policy dataset
`raw_obs.py`: predator-centric, torus-wrapped. pred_vel (2) + K-nearest boids rel
pos/vel (4K) + GxG soft density grid + GxG*2 momentum field. `gen_dataset_e2e.py`
captures (obs, production steering force, d1, pred_auto) on-policy from the shipped E3D
policy. Train 512–1024 seeds, val held-out.

## M2 — architecture / objective search (GPU, L4)
Behavioural eval = `eval_e2e.py` (E2ESim plugs net in place of the whole pipeline);
512 held-out seeds 70000+; production baseline 8.19.

| step | finding | catches |
|---|---|---|
| force-head MLP, density grid | learns chase (~20° ang) but FAILS patrol (~66°, capacity-independent) | 4.3 |
| + velocity/momentum field | did NOT fix patrol; HURT chase (signal dilution) | — |
| diag: why? | production force is Reynolds (desired_vel − cur_vel): 56°/105° off dir-to-target; force is an ill-conditioned regression target. dir-to-target itself is learnable at ~24°. | — |
| **Reynolds output head** (net predicts desired velocity, fixed `clip(desired−vel,0.05)` head) | unlocks patrol | **5.7–5.9** |
| capacity sweep (1.9k→44k params) | catches flat 5.5→5.7 → not capacity-bound | ~5.7 |
| direct dir-label training | worse (crude chase label, no lead) | 4.2 |
| grid G9 vs G13 (+2× data) | dir-to-target 23°→21° — resolution barely helps | — |
| **regime decomposition** (hybrid_diag.py) | **chase_e2e 7.86 (96% of prod), patrol_e2e 6.24 (76%)** → CHASE is ~solved, PATROL is the whole gap | — |

### Current read
The KNN chase encoding reproduces production interception almost exactly. The gap is
PATROL: the density-weighted, reach/nbhd cluster-SELECTION centroid floors at ~22°
direction error for an MLP on a predator-centric grid — likely because production weights
each boid by its OWN local (boid-boid) neighborhood density, a structure a predator-centric
histogram only approximates. Open question (in progress): is this a data/capacity limit or
a real raw-obs ceiling? (cf. memory: prior raw-obs PPO also capped ~6.0.)
