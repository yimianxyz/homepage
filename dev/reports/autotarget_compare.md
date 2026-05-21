# autoTarget sweep — 16 seeds × 5000 frames

Baseline (random patrol target): meanCatches=17.44

| mode             | meanCatches | Δmean  | paired SE | z     | %      |
|------------------|-------------|--------|-----------|-------|--------|
| random (base)    | 17.44       |  -     |  -        | -     | -      |
| nearest_boid     | 21.44       | +4.00  | 2.06      | 1.94  | +22.9% |
| flock_centroid   | **24.25**   | +6.81  | 1.92      | **3.55**  | **+39.1%** |
| farthest_in_K    | 22.25       | +4.81  | 2.12      | 2.27  | +27.6% |

**flock_centroid wins** with z=3.55 (p<0.0002 one-sided). Almost every
seed improved (only 4 minor losses, max -6 catches).

## Why it works

The rule's original autoTarget is a random canvas point regenerated
every 5 s (predator.js:43-49). When no boid is within R=80, the
predator spent a lot of time wandering toward random points instead
of where the boids actually are. flock_centroid pegs the autoTarget
to the centre of mass of the live boid swarm, so the predator
naturally moves toward density. Once a boid enters R, the hunt
branch takes over and the policy steers to the nearest individual.

`nearest_boid` is a tempting alternative — but produces *oscillation*
when nearest flips between two equidistant boids, and underperforms
flock_centroid by a wide margin (z=1.94 vs z=3.55). The centroid is
smoother and aligns the predator with where the next boids will be.

## The NN policy already supports it without retraining

Because the structural change only updates which target the predator
aims at when in patrol mode, the NN's seek_auto_xy feature
auto-updates with no weight retraining required. Shipping this would
be a one-line change in js/predator.js (replace the random-target
block with the centroid computation).

## Variant sweep — all modes A/B'd on seeds 100..115

| mode               | meanCatches | Δ vs random | z    |
|--------------------|-------------|-------------|------|
| random (base)      | 17.44       | -           | -    |
| nearest_boid       | 21.44       | +4.00       | 1.94 |
| predicted_centroid | 21.38       | +3.94       | 1.91 |
| farthest_in_K      | 22.25       | +4.81       | 2.27 |
| flock_centroid     | 24.25       | +6.81       | 3.55 |
| **weighted_centroid** | **24.81** | **+7.38**   | **4.26** |

`weighted_centroid` edges `flock_centroid` by +0.56 (z=0.52, tied
within noise). `predicted_centroid` (centroid + 30·mean_velocity)
*hurt* slightly — boids flock so mean velocity is correlated but
small relative to position spread; the lookahead pushes the target
past where the flock actually is.

Choosing **flock_centroid** for production: simpler code, same
performance within noise. Five-line patch in js/predator.js.
