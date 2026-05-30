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
| **scale-up** G13 grid + 170k params (256,128) + 1024 seeds + 250 ep | full **6.48**, patrol_e2e **6.77**, chase_e2e **8.49** (>prod) | 6.48 |

### Current read (post scale-up)
The "~6.0 raw-obs ceiling" from prior PPO is **NOT** a hard wall. Jointly raising grid
resolution (G9→G13), capacity (44k→170k), data (512→1024 seeds) and epochs (→250) moved
**both** regimes up: chase_e2e 7.86→8.49 (now exceeds prod — chase is saturated/solved),
patrol_e2e 6.24→6.77. The full e2e net is 6.48 (errors in both regimes compound). The whole
remaining gap is still PATROL (6.77 vs prod 8.19 = 83%), but it is responding to scale —
so we are capacity/resolution/data-bound, not at a representational ceiling.

The patrol computation is a density-weighted cluster-SELECTION centroid. A predator-centric
density histogram approximates it and finer resolution helps (G13>G9). Open: which of the
four levers drove the +0.53 patrol gain, and where does the grid curve plateau. Next:
isolate by pushing grid resolution + data further (one decisive higher-res run), and in
parallel test a set/attention encoder that sees exact boid positions (the architecture that
can in principle compute the density-weighted centroid without histogram quantization).
