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

## Round 2 — (running)
- VM1: Hs=120 (match planner horizon), K_roll=1, frames=300.
- VM2: prune_by=value, K_roll=1, Hs=60, frames=300.
- VM3: K_roll=1, Hs=60, frames=1500 (headline vs radial 6.6 / planner 18.3).
