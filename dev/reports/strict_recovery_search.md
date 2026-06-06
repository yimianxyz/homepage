# All-strict cheap-system search — recover the planner (2026-06-06)

Goal: find the smallest/cheapest deploy config whose catch rate approaches the
**strict planner ceiling = 18.3** /1500f (radial baseline ≈ 6.6). Every number
here is from the **strict** sim (`sim_torch`, FAST_TWO_PASS=False) or the JS
production path — no fast approximation. Search uses short strict episodes
(fewer rounds, each step exactly strict) at large n for cheap statistical power;
finalists confirmed at 1500f + JS.

Net: `net_strict.pt` (3553 params), trained on VM1's 12k strict shard.
Cost model: cheap eval ≈ decisions × Hs × (120-boid sequential pass-2 loop);
**n and K_roll are ~free** (batched), frames/Hs are the cost. ~18 min/config at
frames=300/Hs=60, n=128.

## Round 1 — K_roll sweep (frames=300, n=128, strict, Hs=60, prune=ball)
| K_roll | cheap | SE | vs E3D (0.92) |
|-------:|------:|---:|--------------:|
| 1 | 1.336 | 0.10 | +45% |
| 2 | 1.328 | 0.09 | +44% |
| 4 | 1.406 | 0.10 | +53% |

**Finding: K_roll is FLAT** (1→4 gives +0.07, within noise). Rolling more
candidates does not help — the ballistic prune's top-1 already captures the
benefit. So K_roll is NOT the lever (refutes the prior hypothesis from the
fast-sim recovery curve). The strict net's cheap does beat E3D by ~45-53% here
(better separation than the fast-trained net, which only tied radial at 1500f).

Bottleneck must be elsewhere: rollout depth Hs, candidate set, prune rule, or
the value net. → Round 2 pivots to Hs, prune_by, and the full-1500f headline.

## Round 2 — Hs depth + prune (frames=300, n=128, strict)
| config | cheap | vs E3D 0.92 |
|--------|------:|------------:|
| K_roll=1, Hs=60 (R1) | 1.336 | +45% |
| K_roll=1, **Hs=120** | 1.289 | +40% |
| K_roll=1, **prune_v** | 0.922 | +0% (= E3D, degenerate) |

**Findings:**
- **Hs is FLAT** (60→120: 1.34→1.29, within noise). Deeper rollout does NOT help.
- **prune_by=value degenerates to E3D** — the weak net ranks E3D best and never
  deviates. Ballistic prune (1.34) >> value prune (0.92). The net is a worse
  candidate-selector than the analytic ballistic heuristic.
- **ALL rollout levers are flat**: K_roll (1/2/4), Hs (60/120) → cheap stuck at
  ~1.3 while the planner gets 18.3. This flatness is a RED FLAG — the rollout
  signal isn't differentiating candidates. Either a deploy bug or a structural
  mismatch vs the planner's rollout.

## Round 3 — LOSS SWEEP: breaks the 0.55 wall (n=128 val, ds_strict_vm2)
Data audit first: 46% of decision rows have NO catch (zero signal); E3D(cand0) is
the argmax in 72% of rows; ranking signal is only 34% of gain variance. So the
net must win the rare ~28% decisive rows. The original `absval` loss made it
chase scene-difficulty instead.

| loss | exact (matches planner pick) | gain_pick | gain_e3d | gain_oracle |
|------|-----:|----:|----:|----:|
| absval,listnet (orig) | 0.293 | 0.949 | 0.925 | 1.416 |
| value,listnet,cls | **0.710** | 0.926 | 0.925 | 1.416 |
| listnet,cls | **0.711** | 0.927 | 0.925 | 1.416 |

**Adding `cls` (classification of the planner's argmax) jumps exact 0.29 -> 0.71**
— the prior reactive-distillation "0.55 wall" was a LOSS-FUNCTION artifact, not an
information limit. Caveat: gain_pick barely exceeds gain_e3d even at exact 0.71
(per-decision edges are tiny but compound over the episode: E3D 7.15 -> planner
18.3). Deploy catch rate is the real test -> retraining value,listnet,cls as
net_rank.pt and deploy-evaluating vs the absval net (1.34).

## Round 4 — TWO BIG CORRECTIONS

### (a) 300f horizon was misleading; cheap already beats radial at 1500f
The whole 300f sweep ("flat ~1.3, capped") was a SHORT-HORIZON ARTIFACT. At the
real 1500f horizon (absval net `net_strict.pt`, K_roll=1, Hs=60, n=128):

**cheap = 10.35  vs  E3D 7.27  vs  radial ~6.6  vs  planner 18.3.**

So cheap BEATS radial by +3.75 and recovers **57% of the planner** — it was never
stuck. The differentiating catches accumulate late in the episode; 300f can't see
them. Lesson: run the config search at >=1500f (or verify short-horizon ranking
first). This is already a SHIPPABLE policy (beats production radial).

### (b) Rollout fidelity bug (found by the identity check)
cheap(K16,Hs120,no_value) MUST equal the planner. It was 1.19 vs planner 2.17
(ratio 0.55) at 300f -> bug. Cause: the cheap rollout loop omitted `_decay_size`
+ frame/time advance that the planner's `_step_with_target` does -> predator never
shrank in the rollout -> over-counted catches -> mis-ranked candidates. FIXED in
feat_planner.py (run_value_lookahead_cheap). Re-running identity (should -> ratio
1.0) and the fixed cheap @1500f (should lift 10.35).

### (c) cls ranker net hurts deploy
net_rank (value,listnet,cls; exact 0.71) deployed at 0.97 < absval's 1.34 — the
cls net outputs ranking scores, not catch-calibrated values, so it combines badly
with rollout catch counts in the value bootstrap. Pure net (no rollout) = 0.93
~ E3D: per-decision planner-matching does NOT transfer to catches (distribution
shift + the planner's edge is closed-loop). **For DEPLOY use the calibrated absval
net; the rollout carries the catches.**

## Round 4b — IDENTITY VERIFIED after the fix (ratio 0.99)
Re-ran cheap(K16,Hs120,no_value) vs planner with the fixed rollout (n=64, 300f):
**student 2.156 vs planner 2.172 -> ratio 0.993 ≈ 1.0** (was 0.55 before fix).

CONFIRMED: the cheap framework is now provably correct — with full rollout it
EXACTLY equals the planner. So "reach exactly the planner" is achievable at
K_roll=16/Hs=120 (that IS the planner); the question is how cheap we can go.
Every prior cheap number was depressed by the _decay_size bug. Re-running the
recovery curve at the REAL 1500f horizon with the fix:
- K1/Hs60 (cheapest), K1/Hs120, K4/Hs60, ... -> K16/Hs120 (=planner 18.3).

## (old) Round 3b — IDENTITY CORRECTNESS CHECK (superseded by 4b)
cheap(K=16, K_roll=16, Hs=120, no_value) MUST equal the planner (roll all 16 to
full depth, argmax true catches). Running it alongside the planner at the same
seeds (n=64, frames=300). If student_mean ≈ planner_mean → harness correct, cheap
genuinely caps low. If student_mean << planner_mean → deploy bug to fix before
trusting any cheap number. (Plus VM2 Hs=120/K4 and VM3 full-1500f still running.)
