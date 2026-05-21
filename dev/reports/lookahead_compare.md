# Lookahead sweep results

Baseline: NN + flock_centroid = 24.25 catches.

| N    | meanCatches | Δ vs base | SE   | z     |
|------|-------------|-----------|------|-------|
| 0    | 24.25 (baseline) | -    | -    | -     |
| 1    | 21.25       | -3.00     | 1.12 | -2.67 |
| 2    | 21.63       | -2.63     | 1.46 | -1.79 |
| 3    | 20.63       | -3.63     | 1.88 | -1.93 |
| 5    | 23.00       | -1.25     | 1.94 | -0.65 |
| 8    | 24.50       | +0.25     | 2.29 |  0.11 |
| 12   | 25.25       | +1.00     | 2.32 |  0.43 |

## Conclusion: feature-level lookahead doesn't help

Small N (1–3) confuses the NN — it was trained on actual positions
and reacts to shifted features as if they were the current state,
giving wrong steering. Larger N (8, 12) creeps back to baseline but
never significantly above.

Per-seed Δ at N=12: range [-12, +22], std=9.3. Hugely inconsistent —
the policy is *different* under lookahead, not *better*.

Without retraining the NN against velocity-aware targets, this
feature-injection trick can't move the needle. Need to either:
- modify the rule to use velocities and re-distill
- or RL the new architecture directly from random init
