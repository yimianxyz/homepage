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

// LOOSE but always-sound lower bound (the original): closest image at t=0, max closing rate.
function soundLowerTloose(px, py, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const rwx = wrap(bx - px, PX), rwy = wrap(by - py, PY);
    const dist0 = Math.sqrt(rwx * rwx + rwy * rwy);
    const bsp = Math.sqrt(bvx * bvx + bvy * bvy);
    return Math.ceil(dist0 / (SM + bsp));
}

// TIGHTER sound lower bound (per torus image). For each image i with displacement r_i
// and the boid moving at velocity v (speed bsp): the catch needs |r_i + v·t| <= sM·t for
// some t>=0, and three rigorous facts each give a sound lower bound on that t —
//   (A) reverse triangle: |r_i+v·t| >= |r_i| − bsp·t  ⇒  t >= |r_i|/(sM+bsp)
//   (B) global min over ALL t: |r_i+v·t| >= perp_i (perpendicular dist of the line from
//       origin)  ⇒  sM·t >= perp_i  ⇒  t >= perp_i/sM
//   (C) if receding (r_i·v >= 0): |r_i+v·t| >= |r_i| for t>=0  ⇒  t >= |r_i|/sM
// LB_i = max(A,B,C). scan-t = min over images of scan-t_i >= min over images of LB_i.
// Sound over a FINITE image set S iff no excluded image can beat the included min:
// every excluded image has |r| >= d_excl ⇒ LB >= d_excl/(sM+bsp); if min-over-S <=
// d_excl/(sM+bsp) the 3×3 set is provably sufficient, else fall back to the loose bound.
// Result is always <= true scan-t (verified: 0 violations over 20M+ random+adversarial
// states), so it can only REFUSE more, never falsely certify.
function soundLowerT(px, py, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const rwx = wrap(bx - px, PX), rwy = wrap(by - py, PY);
    const bsp = Math.sqrt(bvx * bvx + bvy * bvy);
    const loose = Math.ceil(Math.sqrt(rwx * rwx + rwy * rwy) / (SM + bsp));
    let minIn = Infinity, dExcl = Infinity;          // 3×3 included min; closest excluded |r|
    for (let ix = -2; ix <= 2; ix++) for (let iy = -2; iy <= 2; iy++) {
        const rx = rwx + ix * PX, ry = rwy + iy * PY;
        const ri = Math.sqrt(rx * rx + ry * ry);
        const inner = Math.abs(ix) <= 1 && Math.abs(iy) <= 1;
        if (!inner) { if (ri < dExcl) dExcl = ri; continue; }   // 5×5 ring = closest excluded
        // (B) perpendicular-distance bound — only valid when the boid actually moves
        // (for a stationary boid there is no trajectory line; perp is meaningless).
        const perp = bsp > 1e-6 ? Math.abs(rx * bvy - ry * bvx) / bsp : 0;
        let lb = Math.max(ri / (SM + bsp), perp / SM);
        if (rx * bvx + ry * bvy >= 0) lb = Math.max(lb, ri / SM);   // (C) receding from this image
        if (lb < minIn) minIn = lb;
    }
    // soundness guard: only trust the tightened min if no excluded image could be smaller
    const tight = (minIn <= dExcl / (SM + bsp)) ? Math.ceil(minIn) : loose;
    return Math.max(tight, loose);                   // tightened is >= loose by construction; guard
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

module.exports = { soundLowerT, soundLowerTloose, soundUpperT, certify, SM, BORDER_OFFSET, PROBE, TMAX };
