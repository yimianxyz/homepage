# Replica throughput gate — VERDICT: **KILL the bitwise-replica track**

SPEC §6: *"build the batched sequential-order rollout skeleton with stock
transcendentals and measure plan-decisions/sec vs the ~30-core node farm. **Kill
the bitwise-replica track if GPU < ~10× the farm** — VM3 then reverts to
screening-only with stock ops (labels stay JS regardless)."*

Measured on ml-forecast-3 (NVIDIA L4, 23 GB; torch 2.9.1+cu129; node v20.18.1),
state cell 1512×982 / N=120 (desktop) and N=30 (mid/late game). Raw numbers in
`results/node_baseline.json`, `results/gpu_l4.json`.

## Numbers

| | N=120 | N=30 |
|---|---|---|
| node **single core** (plans/s) | 16.09 | 134.43 |
| node **~30-core farm** (×30) | ~483 | ~4033 |
| **GPU best** (CUDA graphs, B=16384) | **55.11** | **743.62** |
| GPU best ÷ farm | **0.114×** | **0.184×** |
| GPU best ÷ single core | 3.4× | 5.5× |
| GPU **eager** (exactness-legal; B=4096/256) | 21.27 / 1.67 | 6.43 |

## Verdict — KILL (decisive, every reading)

- **vs the farm (the SPEC's bar):** GPU is **0.11–0.18× the farm** — not below
  10×, below **1×**: the L4 is ~6–9× **slower** than the 30-core node farm,
  ~50–90× short of the 10×-faster keep threshold.
- **vs a single core (most GPU-favorable framing):** 3.4× (N=120) / 5.5×
  (N=30) — still under 10×.
- **The exactness-legal ceiling is worse.** A *bitwise* replica must run **eager**
  (CUDA graphs fuse/reassociate float ops → breaks bit-exactness; fdlibm NOTES
  §"Porting approach" requires eager). The eager column is **21.3 plans/s at
  N=120 (0.04× farm)** and **1.67 plans/s** at small batch. Add the fdlibm
  `js_exp`/`js_pow` cost (tens of extra elementwise int64/bit kernels per
  transcendental, ~10–50× a stock `torch.exp`) and the realistic bitwise-replica
  rate is **far below even the stock-eager number**. There is no batch size that
  reaches 10× the farm.

## Root cause (why GPU loses here)

`planCheap`'s rollout is **sequential over boids**: pass 2
(`predator_cheap.js:216`) updates boids in place so boid *i* reads already-moved
neighbours *j<i*. Per plan that is N×Hs in-place steps (≈120×90) on a hard
dependency chain the L4 cannot parallelize; GPU parallelism only spans the
independent-rollout batch dimension, which saturates fast (N=120: 53.7→55.1
plans/s from B=4096→16384 — flat). The node farm runs each plan as tight
straight-line float64 in V8 and scales linearly with cores. Structure, not
constant factors, decides this.

## Consequences

1. **No GPU bitwise replica is built.** The op-level groundwork stays as a
   verified, documented asset: `dev/exact_nn/fdlibm/` (`js_exp`/`js_pow`,
   100.0000% bitexact vs node v20.20.2 over ~17.7M CPU vectors; CUDA exactness
   re-checkable on a GPU VM). It is available for a small offline cross-check if
   ever needed, but is not wired into a full-policy replica.
2. **Ground truth / labels were always JS regardless** (SPEC §5) — unaffected.
3. **VM3 reverts to screening-only with stock ops.** Still useful: the §4c
   **student-attack adversarial search** (GPU screens candidate boundary states
   where a student's deduped top-2 margin is large but argmax≠prod; stock-op
   `max|dV|≈2e-7` is fine for *screening* — every hit is re-proven in JS), and
   GPU training of L1r/L1s/L1p students. VM3 is **stopped** now (no student
   delivered yet → no queued GPU job); restart in ~1 attempt when side-a ships a
   checkpoint.

## Parity (skeleton structure is faithful — recorded, not a gate input)

The skeleton reproduces prod's *structure* exactly where it must: catches and
committed plan target **bitwise-identical** to JS on all three parity dumps
(`max|dcand|≈1e-13`, `max|dpos|≤2e-10` from stock transcendentals; `max|dV|≈2e-7`
is the stock-`erf`/`exp` drift — exactly the divergence that mandates JS labels
and would require the fdlibm port for a true replica). This confirms the gate
measured the *right* computation; it just isn't fast enough on GPU.
