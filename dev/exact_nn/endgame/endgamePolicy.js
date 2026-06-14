// endgamePolicy.js — the Phase-2 (simplified) PURE endgame NN decision export.
// Final policy = prod planner (N>5, UNCHANGED) + this pure NN for the N≤5 endgame.
//
// The endgame decision (prod intercept()) = egBoid = argmin over present boids of
// scan(boid).t. This NN predicts each boid's scan-t from RAW kinematics / cheap
// closed-form geometry (eg_features, 18-dim: wrap-aware analytic reach time,
// closing/radial rates, distances, raw rel pos/vel, cell dims) — it is NOT fed
// prod's exact O(N·TMAX) scan-t. argmin of the prediction IS the committed egBoid.
//
// PURE, NO FALLBACK: there is no scan/cert fallback in the decision path — the NN's
// argmin is always committed. (For the all-unreachable case prod uses a nearest-
// distance tie-break; that branch ~never fires at TMAX=1400, and the NN ranks those
// by its learned scan-t anyway — still a pure NN decision.)
//
//   const { loadEndgamePolicy } = require('./endgamePolicy.js');
//   const endgamePolicy = loadEndgamePolicy('./eg_weights.json');
//   const egIdx = endgamePolicy(snapshot, cfg);            // index into the live boids
//     snapshot : { px,py, pvx,pvy,psize, bx[],by[],bvx[],bvy[] }  (1..5 live boids)
//     cfg      : { W, Hc }
// Returns the committed egBoid index (lowest-index tie-break on equal predictions).
// `.detail(snapshot,cfg)` returns { egIdx, margin, ts } for verification/calibration.
//
// Float64, deterministic; GELU via prod's Abramowitz-Stegun erf (cp_erf) so the
// forward is bit-faithful to the trainer (endgame_train.py ASGELU). No RNG/clock.
'use strict';
const path = require('path');
const { loadEgStudent } = require(path.join(__dirname, 'egboidPick.js'));

function loadEndgamePolicy(weightsPath) {
    const pick = loadEgStudent(weightsPath);   // {egIdx, margin, ts} = pure NN argmin scan-t
    function endgamePolicy(snapshot, cfg) {
        return pick(snapshot, cfg).egIdx;       // the committed egBoid — no fallback
    }
    endgamePolicy.detail = (snapshot, cfg) => pick(snapshot, cfg);
    return endgamePolicy;
}

module.exports = { loadEndgamePolicy };
