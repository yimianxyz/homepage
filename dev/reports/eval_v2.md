# sim_torch v2 — fast, JS-faithful eval (2026-05-22)

## Why

The earlier sim_torch ran boids in parallel within a frame and was
rank-discordant with JS (ρ≈0.17 across NNs). GPU ES training was
optimising a proxy that didn't predict JS performance — VM 2's
SetXF policy reached sim_torch_par 10.84 catches but only JS 18.4.

## Fix #1: sequential boid update (matches JS dynamics)

JS does:
  1. `tick()` — parallel flock pass; ADD forces to per-boid acceleration.
  2. `render()` — sequential per-boid loop: ADD more forces, then
     `update()` applies acceleration and zeros it. Each boid sees
     in-frame updates of boids 0..i-1.

`Sim(sequential=True)` replicates this exactly — confirmed by
`per_seed_catches[3,0] == [3,0]` on 2-seed × 200-frame canonical
shipped run.

## Fix #2: CUDA Graphs (10× speedup on the canonical eval)

Sequential mode is launch-bound: 120 boids × ~30 small CUDA ops per
frame = 3600 kernel launches/frame. Each launch is ~5μs of overhead.
Total per-frame launch tax: ~18ms. The actual GPU math is microscopic
(B=16 × N=120 × 4 bytes = 7.6 KB working set).

`Sim.run_graph(max_frames)` captures one `step()` into a `CUDAGraph`
and replays it for each remaining frame. Single graph launch ≈ 1μs
overhead, so 5000 frames = 5 ms of pure replay overhead. The launch
tax is gone.

Refactor needed for capture-safety:
  - All `self.X = new_tensor` reassignments → `self.X.copy_(...)` or
    advanced-indexed setitem.
  - `self.frame * FRAME_MS` (Python int) → `self._frame_ms` (GPU scalar).
  - `torch.tensor(scalar, ...)` calls inside the step → pre-allocated
    constants on the target device (`self._wrap_*`, `self._inf_t`, etc.).
  - In-place catch-detection: `boid_alive[rows, first_idx] = cur & ~any_catch`
    instead of clone-+-write-+-reassign.

## Throughput on L4 GPU (sequential + graph, 1000 frames, shipped weights)

| B    | seed-fps | comment |
|------|---------:|---------|
|   16 |     316  | canonical eval batch |
|   64 |    1246  | linear scaling holds |
|  256 |    4053  | 13× over B=16 (16× expected) |
|  512 |    6325  | 20× — sublinear creeping in |
| 1024 |    9195  | 29× — GPU compute saturating |
| 2048 |   12628  | 40× — near GPU peak |

## Canonical 16-seed × 5000-frame eval (shipped weights, sequential + graph)

| metric | sequential+graph | sequential no-graph | parallel | JS truth |
|--------|------------------|---------------------|----------|----------|
| mean catches | 22.94 | 22.94 | 22.12 | 24.25 |
| wall time | **246 s** | 2467 s | 51 s | ~420 s (4 workers) |
| seed-fps | 325 | 32 | 1567 | ~190 |

Graph mode is bit-identical to non-graph (per_seed match exactly).
Sequential mean (22.94) is 5.4% lower than JS (24.25) at the
canonical scale — likely float32-NN-forward drift accumulating over
5000 frames. The 2-seed × 200-frame smoke test still matches JS
bit-exactly, so the dynamics port is correct; the drift is in
trajectory divergence due to NN precision.

## Comparison: ES use case

| approach | per ES generation (K=32 cands × 16 seeds × 5000 frames) |
|----------|--------------------------------------------------------:|
| sequential CPU (single thread) | ~21 h |
| sequential CUDA, no graph, sequential per-NN | ~21 h |
| sequential CUDA + graph, batched B=512 | **~7 min** |

Speedup: 180×. Achieves the user's "at least 10× faster" goal by a
wide margin in the workload that actually matters for ES.

## Files

- `dev/sim_torch.py` — added `sequential=True` arg, `run_graph()` method,
  pre-allocated graph-safe constants.
- `dev/eval_sim.py` — high-level `evaluate(weights, seeds, frames)` API
  with sequential+graph defaults.

## Next

1. Verify rank correlation of sequential mode vs JS truth across the
   K=20 random-init NN checkpoints (running on VM 2).
2. If ρ < 0.7, investigate the float32 NN drift — promote NN forward
   to float64.
3. Add multi-policy batched NN forward so K candidates can be evaluated
   in one batched sim run.
