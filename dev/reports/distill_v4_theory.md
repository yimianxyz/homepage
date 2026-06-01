# Can the lookahead planner be distilled into a reactive NN? — first principles

The user challenged the Experiment-3 conclusion ("the planner's edge does NOT
survive reactive distillation; insufficient observation, DAgger can't fix it")
and asked three sharp questions:

1. Is there a **theoretical boundary** that makes the planner non-distillable?
2. If not — is the problem the **data generation / coverage**?
3. Is the **loss correctly passed** to the net?

This note answers all three from first principles, identifies the root cause,
and specifies the architecture (`distill_v4.py`) built to reach **>99% of the
best planner's catches**.

## 0. What the planner actually computes

Every `D` frames the planner forms `K` candidate patrol targets `c_1..c_K`
(deterministic functions of the current state `s`: the E3D target + the K−1
nearest live boids, lead-adjusted), rolls the **true dynamics** forward `H`
frames committed to each, and commits to

```
pi(s) = c_{ argmax_k  gain(s, c_k) },   gain(s,c) = catches over next H frames if target=c
```

Two facts about `pi`:

- `gain(s, ·)` is produced by iterating the simulator — a **deterministic,
  finite, mostly-continuous** map from the full state `s` (120 boids × {pos,vel}
  + predator + sizes + alive mask + feed-cooldown ≈ 489 reals) and a candidate
  `c ∈ R²` to an integer count.
- `pi(s)` is therefore a deterministic function of `s` **alone** (the future it
  rolls out is itself determined by the present). It is *reactive* in the only
  sense that matters for deployment: no hidden memory is required — the D-frame
  commitment is a free design choice, not an information dependency.

## 1. Theoretical boundary? — NO

Claim: there is **no** fundamental obstruction to approximating `pi` with a
feed-forward net that reads the full state.

Argument (first principles):

- **Determinism + finiteness.** `gain(s,c)` is computed by a fixed-length program
  (H simulator steps, each a composition of arithmetic, `min`/`max`, `sqrt`,
  comparisons). It is a well-defined function `R^d × R² → ℤ`. There is no
  stochasticity, no hidden state, no infinite recursion.
- **Universal approximation.** A function that is bounded and measurable on a
  compact domain is approximable to arbitrary `L²` accuracy by a sufficiently
  large MLP (Hornik). Attention/Deep-Sets layers are universal approximators of
  *permutation-invariant* functions on sets (Zaheer Deep Sets 2017; Lee et al.,
  Set Transformer 2019) — and `gain` **is** permutation-invariant in the boid
  index (relabelling boids does not change the physics).
- **The only non-smoothness is harmless.** `pi` is discontinuous exactly on the
  measure-zero set where two candidates' gains tie (`argmax` flips). But on that
  set, *by construction both choices yield equal catches* — so any approximation
  error there costs **zero** closed-loop performance. This is why we target
  >99% of the planner's *catches*, not >99% exact-action agreement: the policy
  is allowed to disagree with the planner precisely where disagreement is free.

So the Experiment-3 wall is **not** a theorem about distillability. It must be an
artifact of the specific distiller. It is.

## 2. Root cause — two design bugs, both fixable

### Bug A — input aliasing (the "data coverage" hypothesis, sharpened)

v1–v3 fed the net only the **M=16 nearest boids**. But `gain(s,c)` depends on the
**full 120-boid interacting future**: cohesion drags far boids toward the flock,
predator-avoidance scatters near ones, and which candidate is best is decided by
how the *whole* swarm reorganises over H frames. Two states that share an
identical 16-nearest view but differ in the other 104 boids can have **different**
best candidates.

Consequence: the training target is **not a function of the observation**. No
architecture and no amount of data can fit a relation that isn't a function —
the irreducible error shows up as the **44.7% decisive-frame TRAIN accuracy** in
v3 (it cannot even memorise its own training set). This is *not* distribution
shift (so DAgger alone cannot fix it, correctly noted in Exp-3) — it is an
**observability** defect. The fix is to **stop truncating the input**: feed all
120 boids through a permutation-invariant set encoder.

This reframes the user's "data generation / coverage" intuition precisely: the
data did not *cover* the information the label depends on — not because we
sampled the wrong *states*, but because we **projected away** the deciding
variables before the net ever saw them.

### Bug B — the loss does not pass the signal (the "loss" hypothesis, confirmed)

v2/v3 minimised **hard cross-entropy on `argmax_k gain`**. Two failures:

- **Tie degeneracy.** With ~8 catches / 1500 frames, the per-candidate gain over
  the horizon is **~86% all-tie**; `argmax` of an all-tie defaults to index 0 =
  the E3D candidate ⇒ **~92% of labels are "pick E3D"**. CE is globally minimised
  by the constant predictor "always E3D" — which *is* the baseline (v3-all
  pick0=0.92, mean 8.24 ≈ 8.34). The loss actively rewards collapsing to E3D.
- **Discarded magnitudes.** CE on the argmax throws away *how much* better the
  winner is and gives **no gradient** to rank the K−1 losers. The rich, dense
  teacher signal (the value of every candidate) is compressed to one bit and
  then dominated by ties.

Fix: regress the **per-candidate value** directly (SmoothL1 on `gain`), optionally
with a **listwise soft-ranking** loss (KL to `softmax(gain/τ)`). This (a) gives a
dense gradient on *every* candidate *every* frame, (b) is **tie-agnostic** —
tied candidates simply receive equal predicted value and `argmax` among equals is
harmless, and (c) matches the literature: value/Q-style distillation of planners
(TD-MPC; Bootstrapped MPC, arXiv:2503.18871) consistently beats action-cloning.

### Bug C — teacher strength & on-policy coverage

v3 distilled the **weak** K8/H60/D15 teacher (~14). We distill the **verified best**
config and run **DAgger** (relabel the *net's own* closed-loop states) so the
train distribution equals the eval distribution — the standard MPC→NN recipe
(Ahn et al. 2023, on-policy imitation of MPC; Tao et al. 2023, tube-guided
augmentation). DAgger is now *appropriate* because, with the full-set input, the
label finally *is* a function of the observation — so relabelling visited states
adds the missing coverage instead of chasing an unlearnable relation.

## 3. The v4 architecture (`distill_v4.py`)

Permutation-invariant **pointer/scorer** over the same K candidates the planner
ranks; reactive and browser-deployable (compute K candidates cheaply, score,
argmax — no rollout at deploy time).

- **Input:** all 120 boids as a set `(dx,dy,vx,vy,alive)` predator-relative +
  predator state `(vx,vy, frac_alive, feed-cooldown)`. No M-truncation.
- **Encoder (3 choices, ablated):**
  - `deepsets` — per-boid MLP → masked mean+max pool (O(N), cheapest, ~production-class).
  - `transformer` — ISAB×L (inducing-point attention, O(N·m)) → masked pool (Set Transformer, Lee 2019).
  - `crossattn` — ISAB×L boid tokens; each **candidate is a query** that
    cross-attends to {boid tokens, predator token} → per-candidate context.
    Most expressive: directly models "given I commit to this target, which boids
    matter." (= candidate-conditioned PMA.)
- **Loss:** `value` (SmoothL1 on gain) | `listnet` (KL to softmax(gain/τ)) | `both`.
- **Teacher:** verified best planner; **DAgger** iterations for coverage.
- **Decisive TRAIN accuracy is the diagnostic.** If full-set input lifts it well
  above v3's 44.7%, Bug A is confirmed and (by §1) closed-loop should track the
  teacher up to tie-cost.

## 4. Verification of the teacher (held-out, fresh seeds)

(Best config K16/H120/D8 measured +162% on seedStart 200000, n=512. Re-checking
on fresh blocks 300000 / 600000 and probing K24 for the true ceiling.)

| run | K | H | D | seedStart | n | mean | SE | vs 8.3447 |
|---|---|---|---|---|---|---|---|---|
| original | 16 | 120 | 8 | 200000 | 512 | 21.871 | 0.212 | +162% |
| verA2 | 16 | 120 | 8 | 300000 | 512 | _pending_ | | |
| verB2 | 16 | 120 | 8 | 600000 | 512 | _pending_ | | |
| verK24 | 24 | 120 | 8 | 300000 | 320 | _pending_ | | |

## 5. Results (distill_v4)

_pending — runs land here._

Sources / inspiration:
- Lee et al., *Set Transformer*, ICML 2019 (arXiv:1810.00825) — ISAB/PMA, permutation invariance.
- Zaheer et al., *Deep Sets*, NeurIPS 2017 — permutation-invariant universal approximation.
- Hansen et al., *TD-MPC* — value-based learning married to planning.
- *Bootstrapped Model Predictive Control*, arXiv:2503.18871 — policy imitates MPC + model-based value estimation.
- Ahn et al., *MPC via On-Policy Imitation Learning*, PMLR v211, 2023 — DAgger-style MPC distillation.
- Tao et al., *Robust Policies from MPC via Imitation + Tube-Guided Augmentation*, arXiv:2306.00286 — coverage/augmentation.
