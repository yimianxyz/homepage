// eg_features_raw.js — RAW-KINEMATICS endgame features for the genuineness ablation
// (lead 3-lab review). Identical to eg_features.js EXCEPT the closed-form analytic
// intercept-TIME estimates are REMOVED from the inputs: no no-wrap analytic time
// (atClip), no wrap-aware analytic time (wa0/wa1 — the heuristic that scores 98.82%
// alone), no wrap-gain, no reach-time-derived flags. The NN must learn the torus
// pursuit-time argmin from RAW geometry alone.
//
// Inputs (lead spec): raw kinematics {px,py,pvx,pvy,psize, bx,by,bvx,bvy} + cell dims
// + purely-relational cheap features (rel-pos, distance, closing/tangent rate). The
// min-image wrapped rel-pos (rwx,rwy) + torus dims still give the NN the wrap geometry;
// it just isn't handed the closed-form answer. Pure float64, deterministic.
'use strict';
const SM = 2.5, BORDER_OFFSET = 10, BMAX = 6, POS_S = 200.0, VS = 6.0;

function wrap(d, P) { return d - P * Math.round(d / P); }

// px,py,pvx,pvy,psize : predator;  bx,by,bvx,bvy : one boid;  W,Hc : torus dims
function egBoidFeatures(px, py, pvx, pvy, psize, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const rwx = wrap(bx - px, PX), rwy = wrap(by - py, PY);     // min-image displacement (the wrap geometry)
    const rawx = bx - px, rawy = by - py;                       // raw (un-wrapped) — signals wrap-relevance
    const dist0 = Math.sqrt(rwx * rwx + rwy * rwy);
    const dsafe = dist0 < 1e-6 ? 1e-6 : dist0;
    const radial = (rwx * bvx + rwy * bvy) / dsafe;             // closing rate (+ = receding)
    const tangent = (rwx * bvy - rwy * bvx) / dsafe;            // cross rate
    const bspeed = Math.sqrt(bvx * bvx + bvy * bvy);
    return [
        rwx / POS_S, rwy / POS_S,
        dist0 / POS_S,
        rawx / POS_S, rawy / POS_S,
        bvx / BMAX, bvy / BMAX, bspeed / BMAX,
        radial / BMAX, tangent / BMAX,
        W / 2560.0, Hc / 1440.0,                                // torus period (lets the NN learn the wrap)
        pvx / VS, pvy / VS, (psize || 0) / 20.0,                // predator kinematics (lead spec; causally light)
    ];
}
const EG_NFEAT = 15;

module.exports = { egBoidFeatures, wrap, EG_NFEAT, SM, BORDER_OFFSET, BMAX, POS_S, VS };
