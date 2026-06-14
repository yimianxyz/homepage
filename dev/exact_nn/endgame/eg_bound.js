// eg_bound.js — SOUND, zero-risk certificate that a proposed egBoid k is prod's
// TRUE argmin scan-t WITHOUT the full 1400-step scan (the L1e analogue of D1's
// exact cheap-bound gate; the lead's rigor requirement). Bounds (all integer-safe):
//   * sound LOWER bound  L(j) = ceil(dist0(j)/(sM+bspeed_j)): closing rate of the
//     predator (sM) toward ANY image of boid j is <= sM+bspeed_j, and the closest
//     image at t=0 is the min-image distance dist0(j); so scan-t(j) >= L(j). Holds
//     even for unreachable j (scan-t=inf >= L). PROVEN (0 violations on the data).
//   * sound UPPER bound  U(k): a VERIFIED reachable integer — probe the exact scan
//     condition |minImage(b+vt-p)| <= sM*t at integers near the analytic intercept
//     time; the first hit is a sound upper bound (scan-t(k) <= U(k)). A few evals,
//     not the full TMAX scan. null if no probe hits (cannot certify → fall back).
// Certificate: U(k) finite and U(k) < min_{j!=k} L(j)  ⇒  k is the UNIQUE argmin
// (scan-t(k) <= U(k) < L(j) <= scan-t(j)) = prod's egBoid, lowest-index tiebreak
// moot. Sound by construction → zero residual risk, no τ.
'use strict';
const { analyticT, wrap } = require('./eg_features.js');
const SM = 2.5, BORDER_OFFSET = 10, PROBE = 12, TMAX = 1400;   // TMAX mirrors intercept()'s scan horizon

function soundLowerT(px, py, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const rwx = wrap(bx - px, PX), rwy = wrap(by - py, PY);
    const dist0 = Math.sqrt(rwx * rwx + rwy * rwy);
    const bsp = Math.sqrt(bvx * bvx + bvy * bvy);
    return Math.ceil(dist0 / (SM + bsp));
}

// verified reachable integer t (sound upper bound on scan-t), probing near analytic-t.
// Mirrors prod's scan horizon EXACTLY: only integers t in [0,TMAX] count as reachable
// (prod returns null for scan-t>TMAX). The explicit t<=TMAX guard makes U match prod's
// reachability DEFINITION — so the certificate's soundness no longer rests on the
// implicit numeric coincidence maxL(593)<TMAX(1400) (L1e audit, soundness lens).
function soundUpperT(px, py, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const rwx = wrap(bx - px, PX), rwy = wrap(by - py, PY);
    const at = analyticT(rwx, rwy, bvx, bvy);
    const start = at == null ? 0 : Math.max(0, Math.floor(at));
    if (start > TMAX) return null;                     // beyond prod's scan horizon
    for (let t = start; t <= start + PROBE && t <= TMAX; t++) {
        const ddx = wrap(bx + bvx * t - px, PX), ddy = wrap(by + bvy * t - py, PY);
        if (Math.sqrt(ddx * ddx + ddy * ddy) <= SM * t) return t;
    }
    return null;       // could not cheaply verify reachability within [start, start+PROBE]∩[0,TMAX]
}

// certify proposed egIdx k (boids: [{x,y,vx,vy}]). Returns true iff PROVABLY argmin.
function certify(px, py, boids, W, Hc, k) {
    const U = soundUpperT(px, py, boids[k].x, boids[k].y, boids[k].vx, boids[k].vy, W, Hc);
    if (U == null) return false;
    for (let j = 0; j < boids.length; j++) {
        if (j === k) continue;
        const L = soundLowerT(px, py, boids[j].x, boids[j].y, boids[j].vx, boids[j].vy, W, Hc);
        if (!(U < L)) return false;
    }
    return true;
}

module.exports = { soundLowerT, soundUpperT, certify, SM, BORDER_OFFSET, PROBE };
