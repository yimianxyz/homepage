// eg_scan.js — standalone EXACT reimplementation of intercept()'s scan() + the
// egBoid argmin, for: (a) recomputing the per-boid scan-t LABEL, (b) the L1e
// exact-scan FALLBACK, (c) cross-checking the oracle logger (independent code
// path, per the adversarial-audit discipline). Bit-identical to predator_cheap.js
// intercept(): PX=W+2·BORDER_OFFSET(=W+20), PY=Hc+20, sM=PREDATOR_MAX_SPEED=2.5,
// TMAX=1400, wx(d)=d−PX·Math.round(d/PX). Pure float64, deterministic.
'use strict';
const SM = 2.5, BORDER_OFFSET = 10, TMAX = 1400;

function scanT(px, py, bx, by, bvx, bvy, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    for (let t = 0; t <= TMAX; t++) {
        const ddx = (bx + bvx * t - px) - PX * Math.round((bx + bvx * t - px) / PX);
        const ddy = (by + bvy * t - py) - PY * Math.round((by + bvy * t - py) / PY);
        if (Math.sqrt(ddx * ddx + ddy * ddy) <= SM * t) return t;
    }
    return null;
}

// prod's egBoid commit: argmin scan-t (lowest-index tiebreak); nearest-wrapped-
// distance fallback if NONE reachable. Returns {egIdx, ts:[t per boid], nearestFallback}.
function egPick(px, py, boids, W, Hc) {
    const PX = W + 2 * BORDER_OFFSET, PY = Hc + 2 * BORDER_OFFSET;
    const ts = boids.map(b => scanT(px, py, b.x, b.y, b.vx, b.vy, W, Hc));
    let bestT = Infinity, egIdx = -1;
    for (let i = 0; i < boids.length; i++) if (ts[i] != null && ts[i] < bestT) { bestT = ts[i]; egIdx = i; }
    let nearestFallback = false;
    if (egIdx < 0) {                          // none reachable -> nearest wrapped distance
        nearestFallback = true;
        let nd2 = Infinity;
        for (let i = 0; i < boids.length; i++) {
            const dx = (boids[i].x - px) - PX * Math.round((boids[i].x - px) / PX);
            const dy = (boids[i].y - py) - PY * Math.round((boids[i].y - py) / PY);
            const d2 = dx * dx + dy * dy;
            if (d2 < nd2) { nd2 = d2; egIdx = i; }
        }
    }
    return { egIdx, ts, nearestFallback };
}

module.exports = { scanT, egPick, SM, BORDER_OFFSET, TMAX };
