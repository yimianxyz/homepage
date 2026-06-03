# Phase 0 — the 2×2 premise table + adversarial cross-checks (2026-06-03)

Goal of Phase 0: decide whether "reproduce the planner" is a real, worth-chasing
target before spending any effort on teacher relabel / student training. The
single table {E3D, planner} × {single-pass, two-pass} resolves the premise.

## The 2×2 table (n=256 seeds, frames=5000, K=16 H=120 D=8, seedStart=200000)

| policy   | single-pass | two-pass (deployment-faithful) |
|----------|------------:|-------------------------------:|
| E3D      | **34.18** (SE 0.42) | **28.11** (sim_torch) — JS n=256 cross-check in flight |
| planner  | **71.18** (SE 0.45) | *running* — VM2 n=256 (decisive) + VM1 n=128 fresh-seeds early read |
| planner randtie (null) | *pending* | *running* — VM3 n=256 |

Single-pass headline: planner **2.08×** E3D (71.18 vs 34.18) on identical seeds.
The decisive open cell is two-pass planner vs two-pass E3D=28.11.

## Cross-check round 1 — adversarial audit of the single-pass 2× edge

Fresh no-memory subagent, briefed neutrally (given the setup + the two raw
numbers, NOT our interpretation), asked: what would make 71.18 vs 34.18 spurious?

### What it found (and our adjudication)

**ADOPTED — no accounting bug (the measurement is clean).** It traced the rollout
vs real-episode catch tally line-by-line: both advance via the same
`_step_with_target` → `_check_catches` (`sim_torch.py:968-992`); gain =
`roll.catches − c0` on a *separate* Sim/counter from the real `sim.catches`
(`planner_probe.py:80,361-364`); candidate 0 is exactly the E3D target and ties
resolve to it (`argmax`), so planner ≥ E3D by construction. No double-count, no
cooldown reset, no cross-talk. The committed target is held only D=8 frames while
each rollout is scored over the full H=120 — a *conservative* mismatch. → The
71.18 is a correct measurement of the planner's value; the 2× gap is real.

**ADOPTED but RE-FRAMED (the load-bearing one) — "clairvoyance" is really
*computation*, not *information*.** The auditor's headline worry: the sim has no
per-frame RNG (all randomness drawn once at t=0, `_initialize`), so the planner's
H=120 rollout is an *exact* oracle of the future, and a single-frame reactive net
"structurally cannot" reproduce it → 71.18 is an unachievable ceiling.

This is half right and the distinction matters for the whole project:
- It is **NOT an information advantage**. Deterministic dynamics + fully-observed
  current state ⇒ the planner's choice π*(s) = argmax_k rollout(s, a_k) is a pure
  (deterministic) function of the *current* state s. It peeks at no hidden RNG —
  there is none. A policy that observes the same full state s has access to the
  same information. The in-browser planner is itself deployable (it runs live in
  a Web Worker; the only cost is ~430ms staleness, which a zero-latency student
  removes). So this is not forbidden clairvoyance.
- It **IS a computation advantage**. π*(s) is a function of s, but an expensive,
  highly non-smooth one — it requires forward-simulating chaotic flock dynamics.
  A *small feedforward reactive* net almost certainly cannot approximate it; that
  is exactly the 0.55-accuracy wall hit 3× before (`distill_v4_theory.md`).

**Consequence we adopt:** a pure single-frame → target feedforward student is the
wrong architecture and will recover only the reactive-recoverable slice of the
gain (the E1 ladder bounds that at ≈+0.8% over E3D). To reproduce the planner the
student must perform **deploy-time lookahead computation over the current state** —
precisely the plan's Phase 2 route 6a (*cheap online lookahead*: keep argmax-over-K,
replace the H=120 true rollout with an O(N) few-step approximate rollout the
browser runs every frame, learn only a tiny scorer/correction). The audit is thus
strong independent confirmation of the plan's central architectural bet, not a
refutation of the project — but it does sharpen that an "end-to-end feedforward
NN" likely cannot hit 99% alone; the artifact that reproduces the planner is a
NN **scorer + online rollout** hybrid. (Strategic fork to surface to the user once
the two-pass gate is settled.)

**ALREADY ADDRESSED — the `randtie` null was never recorded.** Correct; it is now
running (VM3 two-pass n=256; single-pass arm to follow). 39.7% of decisions are
integer all-ties and 65% of dense-margin winners beat the runner-up by <0.05
(`distill_rethink.md`), so the null quantifies how much of the 2× is genuine
lookahead vs tie-break selection on chaos. A real +1-catch margin should dominate,
so we expect the edge to survive, but it must be measured.

**NOTED — regime richness.** Deployed NN ≈8 catches in live JS vs E3D=34 here on
single-pass sim_torch ⇒ this comparison lives in a ~4× richer regime than the live
browser, and single-pass is not the live (two-pass) world. Does not bias the
planner/E3D *ratio* (same sim), but "71 catches" is not comparable to any prod
number. This is exactly why the decisive cell is **two-pass**.

### Not adopted
- "71.18 is an unachievable ceiling, the target is unrealistic" — rejected as
  stated. It is unachievable *for a static reactive net*, which we already knew;
  it is not unachievable for a lookahead-carrying student (Phase 2 route). The
  in-browser planner's live deployability is the existence proof.

## Gate status
Still gated on the two-pass cells. Continue to Phase 1/2 only if two-pass planner
shows a real edge over two-pass E3D=28.11. If two-pass collapses the edge, the
"reproduce the planner" target is moot → surface to user. Next: merge JS e3d
two-pass n=256 (validate 28.11), collect VM1/VM2 two-pass planner + VM3 randtie,
run `analyze_2x2.py` (paired bootstrap + MDE) on the full table.
