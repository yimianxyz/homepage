// eg_features.js — per-boid feature vector for the L1e scan-t regressor.
// ONE JS function used by BOTH the training packer AND the deploy egboidPick, so
// features are bit-identical on both sides (the D1 parity lesson). Pure float64.
//
// scan-t(boid) depends only on (relpos, boid vel, torus dims, sM=2.5). The torus
// makes it non-convex (a wrap-around shortcut can beat the head-on chase — the
// slow predator's only edge). Key feature: the analytic NO-WRAP 2-body intercept
// time (closed-form); the NN learns the wrap correction on top of it.
'use strict';
const SM = 2.5, BORDER_OFFSET = 10, BMAX = 6, POS_S = 200.0, TMAX = 1400;

function wrap(d, P) { return d - P * Math.round(d / P); }

// Smallest t>=0 with |relMin + vel*t| = sM*t (head-on, no torus). null if none.
function analyticT(rwx, rwy, bvx, bvy) {
    const a = bvx * bvx + bvy * bvy - SM * SM;
    const b = 2 * (rwx * bvx + rwy * bvy);
    const c = rwx * rwx + rwy * rwy;
    if (Math.abs(a) < 1e-12) {            // linear: b t + c = 0
        if (Math.abs(b) < 1e-12) return c < 1e-12 ? 0 : null;
        const t = -c / b; return t >= 0 ? t : null;
    }
    const disc = b * b - 4 * a * c;
    if (disc < 0) return null;
    const sq = Math.sqrt(disc);
    const t1 = (-b - sq) / (2 * a), t2 = (-b + sq) / (2 * a);
    let best = Infinity;
    if (t1 >= 0 && t1 < best) best = t1;
    if (t2 >= 0 && t2 < best) best = t2;
    return Number.isFinite(best) ? best : null;
}

// Returns a fixed-length feature array for one boid relative to the predator.
function egBoidFeatures(px, py, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const rwx = wrap(bx - px, PX), rwy = wrap(by - py, PY);     // t=0 min-image displacement
    const rawx = bx - px, rawy = by - py;                       // raw (un-wrapped) displacement
    const dist0 = Math.sqrt(rwx * rwx + rwy * rwy);
    const dsafe = dist0 < 1e-6 ? 1e-6 : dist0;
    const radial = (rwx * bvx + rwy * bvy) / dsafe;             // + = boid receding
    const tangent = (rwx * bvy - rwy * bvx) / dsafe;            // signed cross / dist
    const bspeed = Math.sqrt(bvx * bvx + bvy * bvy);
    const at = analyticT(rwx, rwy, bvx, bvy);                   // no-wrap intercept time (THE feature)
    const atClip = at == null ? TMAX : Math.min(at, TMAX);
    return [
        rwx / POS_S, rwy / POS_S,
        dist0 / POS_S,
        rawx / POS_S, rawy / POS_S,                            // raw vs wrapped → signals wrap-relevance
        bvx / BMAX, bvy / BMAX, bspeed / BMAX,
        radial / BMAX, tangent / BMAX,
        atClip / 100.0,                                        // analytic intercept time (~scan-t prior)
        at == null ? 1.0 : 0.0,                                // no-wrap-unreachable flag
        W / 2560.0, Hc / 1440.0,                               // cell identity (torus period)
    ];
}
const EG_NFEAT = 14;

module.exports = { egBoidFeatures, analyticT, wrap, EG_NFEAT, SM, BORDER_OFFSET, BMAX, POS_S, TMAX };
