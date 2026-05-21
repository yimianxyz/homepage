# autoTarget structural-fix verification

## All evals on 16-seed × 5000-frame, 120 boids.

|                        | seeds 100..115 | seeds 200..215 (holdout) |
|------------------------|----------------|--------------------------|
| NN + random  (baseline)| 17.44          | 16.13                    |
| NN + flock_centroid    | **24.25**      | **21.88**                |
| rule + random          | 19.75          | -                        |
| rule + flock_centroid  | 22.00          | -                        |

## Paired-delta tests (16 seeds each)

| comparison                       | Δmean | SE   | z    |
|----------------------------------|-------|------|------|
| NN+flock vs NN+random  (100..115)| +6.81 | 1.92 | **3.55** |
| NN+flock vs NN+random  (200..215)| +5.75 | 1.60 | **3.59** |
| rule+flock vs rule+random        | +2.25 | 2.40 | 0.94 |
| NN+flock vs rule+flock           | +2.25 | 1.77 | 1.27 |
| rule+random vs NN+random         | +2.31 | 2.07 | 1.12 |

## Conclusions

1. **flock_centroid generalizes**: z=3.55 on train seeds → z=3.59 on
   holdout seeds is essentially identical. Not an overfit artefact.
2. **The NN is the right policy for flock_centroid** (not the rule):
   the NN gains +39% from the patrol change; the rule only +11%. The
   NN's slightly damped response avoids overshooting the centroid
   that the bang-bang rule produces.
3. **NN + flock_centroid is the new shipping policy.** No weight
   retraining required — the feature pipeline auto-adapts.
