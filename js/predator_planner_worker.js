// Receding-horizon planner rollout — runs OFF the main thread.
//
// This is the ~21-catch lookahead "teacher" (dev/planner_probe.py run_planner),
// ported to the browser. It is a TEMPORARY, URL-flag-gated policy for visual
// inspection only (?policy=planner); production is unaffected.
//
// Fidelity: the hot rollout (rolloutFlat) is a flat-typed-array port that
// mirrors the REAL sim math in vector.js + boid.js + predator.js *exactly* —
// same alpha-max-beta-min fast magnitude, same double-flock-per-frame, same
// sequential render-phase position updates, same catch/cooldown/decay. We also
// keep rolloutRef, which drives the ACTUAL imported Boid class, and a 'selftest'
// message that asserts the flat port reproduces rolloutRef to <1e-6. The flat
// port exists because the object-based rollout is ~1.5s/plan (too slow for
// real-time); flat is ~10-20x faster so the live predator re-plans on fresh
// state instead of steering toward a stale target.
//
// Each rollout frame reproduces the live frame exactly:
//   simTick()-> clock += 12ms
//   tick()   -> flock all boids (accumulate accel, no move)
//   render() -> per boid, IN ORDER: flock again (accel) + update (move); so
//               boid i sees boids <i already moved this phase. Then predator
//               moves (chase nearest-in-POLICY_R else seek committed target);
//               then one catch check gated by the 100ms feed cooldown; decay.
// Decisions branch K candidate patrol targets (cand0 = E3D evolved target,
// cand1..K-1 = nearest live boids lead-adjusted), roll H frames committed to
// each, return the target with the most catches (ties -> earliest = E3D).
'use strict';

importScripts('vector.js', 'boid.js', 'predator.js');

// Sim constants. From imports: PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE,
// PREDATOR_SIZE, MAX_SPEED, MAX_FORCE, NEIGHBOR_DISTANCE, DESIRED_SEPARATION,
// EPSILON, PREDATOR_RANGE, PREDATOR_TURN_FACTOR, EVOLVED_PATROL.
var FRAME_MS = 12;
var FEED_COOLDOWN = 100;
var GROWTH = 1.2, DECAY = 0.002;
var BASE_SIZE = PREDATOR_SIZE;          // 12
var MAX_SIZE = BASE_SIZE * 1.8;         // 21.6
var CATCH_FACTOR = 0.7;
var LEAD_SCALE = EVOLVED_PATROL.lead_scale, LEAD_MAX = EVOLVED_PATROL.lead_max;
var SEP_MULT = 2, COH_MULT = 1, ALIGN_MULT = 1;

var cfg = { K: 16, H: 120, POLICY_R: 80, W: 1680, Hc: 1680, predRange: 80 };

// alpha-max-beta-min fast magnitude (vector.js getFastMagnitude, byte-identical).
function fastMag(x, y) {
    var ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
    return (ax > ay ? ax : ay) * 0.96 + (ax < ay ? ax : ay) * 0.398;
}

// ---------------------------------------------------------------------------
// FLAT rollout: scalar/typed-array mirror of boid.js + predator.js.
// ---------------------------------------------------------------------------
// Scratch buffers (grown on demand, reused across candidates within a plan()).
var _px, _py, _vx, _vy, _ax, _ay, _alive, _cap = 0;
function ensureCap(n) {
    if (n <= _cap) return;
    _px = new Float64Array(n); _py = new Float64Array(n);
    _vx = new Float64Array(n); _vy = new Float64Array(n);
    _ax = new Float64Array(n); _ay = new Float64Array(n);
    _alive = new Uint8Array(n); _cap = n;
}

// Uniform spatial hash so flock neighbor queries are O(local) not O(n). Cell
// size == NEIGHBOR_DISTANCE (the largest interaction radius), so EVERY boid
// within 60px is guaranteed to lie in the queried 3x3 cell block — the neighbor
// SET is byte-identical to brute force (verified: verify_planner_parity == 0).
// Distances are raw euclidean (boid.js does NOT wrap distance), so the grid
// likewise does NOT wrap — edge boids don't see across the toroidal seam, same
// as the live sim. Doubly-linked buckets give O(1) move when a boid changes
// cell during the sequential render phase. One grid is shared (single-threaded
// worker, one rollout/episode at a time).
var GCELL = NEIGHBOR_DISTANCE;            // 60
var GOFF = 120;                            // > boid border (10) so indices >= 0
var g_ncx = 0, g_ncy = 0, g_ncells = 0;
var g_head, g_next, g_prev, g_cellOf, g_ccap = 0, g_gcap = 0;
function gridConfig() {
    g_ncx = ((cfg.W + 2 * GOFF) / GCELL | 0) + 2;
    g_ncy = ((cfg.Hc + 2 * GOFF) / GCELL | 0) + 2;
    g_ncells = g_ncx * g_ncy;
    if (g_ncells > g_ccap) { g_head = new Int32Array(g_ncells); g_ccap = g_ncells; }
}
function gridEnsure(n) {
    if (n > g_gcap) { g_next = new Int32Array(n); g_prev = new Int32Array(n); g_cellOf = new Int32Array(n); g_gcap = n; }
}
function cellIdx(x, y) {
    var cx = (x + GOFF) / GCELL | 0; if (cx < 0) cx = 0; else if (cx >= g_ncx) cx = g_ncx - 1;
    var cy = (y + GOFF) / GCELL | 0; if (cy < 0) cy = 0; else if (cy >= g_ncy) cy = g_ncy - 1;
    return cy * g_ncx + cx;
}
function gridReset() { g_head.fill(-1, 0, g_ncells); }
function gridInsert(i, x, y) {
    var c = cellIdx(x, y); g_cellOf[i] = c; g_prev[i] = -1;
    var h = g_head[c]; g_next[i] = h; if (h >= 0) g_prev[h] = i; g_head[c] = i;
}
function gridRemove(i) {
    var c = g_cellOf[i]; if (c < 0) return;
    var pr = g_prev[i], nx = g_next[i];
    if (pr >= 0) g_next[pr] = nx; else g_head[c] = nx;
    if (nx >= 0) g_prev[nx] = pr;
    g_cellOf[i] = -1;
}
function gridMove(i, x, y) {
    var nc = cellIdx(x, y); if (nc === g_cellOf[i]) return;
    gridRemove(i);
    g_cellOf[i] = nc; g_prev[i] = -1;
    var h = g_head[nc]; g_next[i] = h; if (h >= 0) g_prev[h] = i; g_head[nc] = i;
}
function gridBuild(px, py, alive, n) {
    gridConfig(); gridEnsure(n); gridReset();
    for (var i = 0; i < n; i++) if (alive[i]) gridInsert(i, px[i], py[i]);
}

// Accumulate one flock() contribution onto (ax[i], ay[i]), reading current
// positions of live neighbors via the spatial grid. Mirrors Boid.flock add-order
// exactly. Arrays are passed in so the same verified code drives both rollouts
// (scratch buffers) and the closed-loop episode eval (separate buffers).
function accumulateFlock(px, py, vx, vy, ax, ay, alive, i, n, predX, predY) {
    var pix = px[i], piy = py[i];
    var cx = 0, cy = 0, cn = 0;        // cohesion: sum neighbor positions
    var sx = 0, sy = 0, sn = 0;        // separation: sum normalized/dist deltas
    var alx = 0, aly = 0, an = 0;      // alignment: sum neighbor velocities
    var ci = g_cellOf[i], cyi = (ci / g_ncx) | 0, cxi = ci - cyi * g_ncx;
    var gylo = cyi > 0 ? cyi - 1 : 0, gyhi = cyi < g_ncy - 1 ? cyi + 1 : g_ncy - 1;
    var gxlo = cxi > 0 ? cxi - 1 : 0, gxhi = cxi < g_ncx - 1 ? cxi + 1 : g_ncx - 1;
    for (var gy = gylo; gy <= gyhi; gy++) {
        var rowbase = gy * g_ncx;
        for (var gx = gxlo; gx <= gxhi; gx++) {
            for (var j = g_head[rowbase + gx]; j >= 0; j = g_next[j]) {
                if (j === i) continue;
                var rx = px[j] - pix, ry = py[j] - piy;
                var dist = Math.sqrt(rx * rx + ry * ry) + EPSILON;
                if (dist <= NEIGHBOR_DISTANCE) { cx += px[j]; cy += py[j]; cn++; }
                if (dist > 0 && dist < DESIRED_SEPARATION) {
                    var ddx = -rx, ddy = -ry;                 // pos - boid.pos
                    var m = Math.sqrt(ddx * ddx + ddy * ddy); // iNormalize (TRUE mag)
                    if (m > 0) { ddx /= m; ddy /= m; }
                    ddx /= dist; ddy /= dist;                 // iDivideBy(distance)
                    sx += ddx; sy += ddy; sn++;
                }
                if (dist > 0 && dist < NEIGHBOR_DISTANCE) { alx += vx[j]; aly += vy[j]; an++; }
            }
        }
    }
    var vxi = vx[i], vyi = vy[i], fm, s;
    // cohesion = seek(avgPos)
    var cohx = 0, cohy = 0;
    if (cn > 0) {
        var dx = cx / cn - pix, dy = cy / cn - piy;
        fm = fastMag(dx, dy); if (fm > 0) { s = MAX_SPEED / fm; dx *= s; dy *= s; }
        dx -= vxi; dy -= vyi;
        fm = fastMag(dx, dy); if (fm > MAX_FORCE) { s = MAX_FORCE / fm; dx *= s; dy *= s; }
        cohx = dx; cohy = dy;
    }
    // separation
    var sepx = 0, sepy = 0;
    if (sn > 0) { sepx = sx / sn; sepy = sy / sn; }
    fm = fastMag(sepx, sepy);
    if (fm > 0) {
        s = MAX_SPEED / fm; sepx *= s; sepy *= s;
        sepx -= vxi; sepy -= vyi;
        fm = fastMag(sepx, sepy); if (fm > MAX_FORCE) { s = MAX_FORCE / fm; sepx *= s; sepy *= s; }
    }
    // alignment
    var algx = 0, algy = 0;
    if (an > 0) {
        var avx = alx / an, avy = aly / an;
        fm = fastMag(avx, avy); if (fm > 0) { s = MAX_SPEED / fm; avx *= s; avy *= s; }
        avx -= vxi; avy -= vyi;
        fm = fastMag(avx, avy); if (fm > MAX_FORCE) { s = MAX_FORCE / fm; avx *= s; avy *= s; }
        algx = avx; algy = avy;
    }
    // apply multipliers, then add in boid.js order: cohesion, separation, alignment
    sepx *= SEP_MULT; sepy *= SEP_MULT;
    cohx *= COH_MULT; cohy *= COH_MULT;
    algx *= ALIGN_MULT; algy *= ALIGN_MULT;
    ax[i] += cohx; ay[i] += cohy;
    ax[i] += sepx; ay[i] += sepy;
    ax[i] += algx; ay[i] += algy;
    // predator avoidance
    var qx = pix - predX, qy = piy - predY;
    var pdist = Math.sqrt(qx * qx + qy * qy) + EPSILON;
    if (pdist > 0 && pdist < PREDATOR_RANGE) {
        fm = fastMag(qx, qy); if (fm > 0) { qx /= fm; qy /= fm; }   // iFastNormalize
        var str = (PREDATOR_RANGE - pdist) / PREDATOR_RANGE * PREDATOR_TURN_FACTOR;
        qx *= str; qy *= str;
        var lim = MAX_FORCE * 1.5;
        fm = fastMag(qx, qy); if (fm > lim) { s = lim / fm; qx *= s; qy *= s; }
        ax[i] += qx; ay[i] += qy;
    }
}

function updateBoid(px, py, vx, vy, ax, ay, i) {
    vx[i] += ax[i]; vy[i] += ay[i];
    var fm = fastMag(vx[i], vy[i]);
    if (fm > MAX_SPEED) { var s = MAX_SPEED / fm; vx[i] *= s; vy[i] *= s; }
    px[i] += vx[i]; py[i] += vy[i];
    var B = BORDER_OFFSET;                       // boid border = 10
    if (px[i] > cfg.W + B) px[i] = -B;
    if (px[i] < -B) px[i] = cfg.W + B;
    if (py[i] > cfg.Hc + B) py[i] = -B;
    if (py[i] < -B) py[i] = cfg.Hc + B;
    ax[i] = 0; ay[i] = 0;
}

// One predator step toward (tx,ty), chase-overriding to the nearest live boid
// within POLICY_R. Mutates the 1-element state object p={x,y,vx,vy}. Shared by
// rolloutFlat and the episode eval.
function predatorStepFlat(px, py, alive, n, p, tx, ty) {
    var bestD2 = Infinity, nx = 0, ny = 0;
    for (var i = 0; i < n; i++) {
        if (!alive[i]) continue;
        var dx = px[i] - p.x, dy = py[i] - p.y;
        var d2 = dx * dx + dy * dy;
        if (d2 < bestD2) { bestD2 = d2; nx = dx; ny = dy; }
    }
    var desx, desy;
    if (bestD2 < cfg.POLICY_R * cfg.POLICY_R) { desx = nx; desy = ny; }
    else { desx = tx - p.x; desy = ty - p.y; }
    var fm = fastMag(desx, desy);
    if (fm > 0) { var sc = PREDATOR_MAX_SPEED / fm; desx *= sc; desy *= sc; }
    var stx = desx - p.vx, sty = desy - p.vy;
    fm = fastMag(stx, sty);
    if (fm > PREDATOR_MAX_FORCE) { var sc2 = PREDATOR_MAX_FORCE / fm; stx *= sc2; sty *= sc2; }
    p.vx += stx; p.vy += sty;
    fm = fastMag(p.vx, p.vy);
    if (fm > PREDATOR_MAX_SPEED) { var sc3 = PREDATOR_MAX_SPEED / fm; p.vx *= sc3; p.vy *= sc3; }
    p.x += p.vx; p.y += p.vy;
    var B = 20;
    if (p.x > cfg.W + B) p.x = -B; if (p.x < -B) p.x = cfg.W + B;
    if (p.y > cfg.Hc + B) p.y = -B; if (p.y < -B) p.y = cfg.Hc + B;
}

function rolloutFlat(s, tx, ty, H) {
    var n = s.bx.length;
    ensureCap(n);
    for (var i = 0; i < n; i++) {
        _px[i] = s.bx[i]; _py[i] = s.by[i];
        _vx[i] = s.bvx[i]; _vy[i] = s.bvy[i];
        _ax[i] = 0; _ay[i] = 0; _alive[i] = 1;
    }
    gridBuild(_px, _py, _alive, n);
    var p = { x: s.px, y: s.py, vx: s.pvx, vy: s.pvy };
    var size = s.psize, lastFeed = s.lastFeed, now = s.nowMs, catches = 0;
    for (var f = 0; f < H; f++) {
        // tick(): flock all live boids (no move)
        for (i = 0; i < n; i++) if (_alive[i]) accumulateFlock(_px, _py, _vx, _vy, _ax, _ay, _alive, i, n, p.x, p.y);
        // render(): per live boid in order, flock again then move immediately
        for (i = 0; i < n; i++) if (_alive[i]) { accumulateFlock(_px, _py, _vx, _vy, _ax, _ay, _alive, i, n, p.x, p.y); updateBoid(_px, _py, _vx, _vy, _ax, _ay, i); gridMove(i, _px[i], _py[i]); }
        predatorStepFlat(_px, _py, _alive, n, p, tx, ty);
        // catch check (gated by feed cooldown), TRUE distance
        if (now - lastFeed >= FEED_COOLDOWN) {
            var cr = size * CATCH_FACTOR;
            for (i = 0; i < n; i++) {
                if (!_alive[i]) continue;
                var ex = p.x - _px[i], ey = p.y - _py[i];
                if (Math.sqrt(ex * ex + ey * ey) < cr) {
                    size = Math.min(size + GROWTH, MAX_SIZE);
                    lastFeed = now; _alive[i] = 0; gridRemove(i); catches++; break;
                }
            }
        }
        if (size > BASE_SIZE) size = Math.max(size - DECAY, BASE_SIZE);
        now += FRAME_MS;
    }
    return catches;
}

// ---------------------------------------------------------------------------
// Zero-staleness closed-loop episode eval (diagnostic). Drives the flat sim for
// `frames` steps; every D frames it RE-PLANS from the CURRENT state (no main-
// thread lag) and commits the chosen target. controller 'e3d' commits the
// evolved-patrol target; 'planner' commits plan()'s argmax target. This is the
// JS mirror of planner_probe.py and tells us the planner's true closed-loop
// ceiling here, isolating policy quality from real-time staleness.
var e_px, e_py, e_vx, e_vy, e_ax, e_ay, e_alive, e_cap = 0;
function ensureECap(n) {
    if (n <= e_cap) return;
    e_px = new Float64Array(n); e_py = new Float64Array(n);
    e_vx = new Float64Array(n); e_vy = new Float64Array(n);
    e_ax = new Float64Array(n); e_ay = new Float64Array(n);
    e_alive = new Uint8Array(n); e_cap = n;
}
// Build a plan-snapshot object from current LIVE episode boids (alive only).
function episodeSnapshot(n, p, size, lastFeed, now) {
    var bx = [], by = [], bvx = [], bvy = [];
    for (var i = 0; i < n; i++) {
        if (!e_alive[i]) continue;
        bx.push(e_px[i]); by.push(e_py[i]); bvx.push(e_vx[i]); bvy.push(e_vy[i]);
    }
    return { bx: bx, by: by, bvx: bvx, bvy: bvy, px: p.x, py: p.y, pvx: p.vx, pvy: p.vy, psize: size, lastFeed: lastFeed, nowMs: now };
}
function evalClosedLoop(s, frames, D, controller) {
    var n = s.bx.length;
    ensureECap(n);
    for (var i = 0; i < n; i++) {
        e_px[i] = s.bx[i]; e_py[i] = s.by[i]; e_vx[i] = s.bvx[i]; e_vy[i] = s.bvy[i];
        e_ax[i] = 0; e_ay[i] = 0; e_alive[i] = 1;
    }
    var p = { x: s.px, y: s.py, vx: s.pvx, vy: s.pvy };
    var size = s.psize, lastFeed = s.lastFeed, now = s.nowMs, catches = 0;
    var tx = p.x, ty = p.y;
    for (var f = 0; f < frames; f++) {
        if (f % D === 0) {
            var snap = episodeSnapshot(n, p, size, lastFeed, now);
            if (!snap.bx.length) { tx = p.x; ty = p.y; }
            else if (controller === 'planner') { var r = plan(snap); tx = r.x; ty = r.y; }
            else {
                var lite = new Array(snap.bx.length);
                for (var q = 0; q < snap.bx.length; q++) lite[q] = { position: { x: snap.bx[q], y: snap.by[q] }, velocity: { x: snap.bvx[q], y: snap.bvy[q] } };
                var e = computeEvolvedTarget({ x: p.x, y: p.y }, lite, EVOLVED_PATROL, null) || { x: p.x, y: p.y };
                tx = e.x; ty = e.y;
            }
        }
        // Rebuild from current episode state: plan() above reuses the shared grid
        // for its scratch rollouts, so it must be reset here regardless.
        gridBuild(e_px, e_py, e_alive, n);
        for (i = 0; i < n; i++) if (e_alive[i]) accumulateFlock(e_px, e_py, e_vx, e_vy, e_ax, e_ay, e_alive, i, n, p.x, p.y);
        for (i = 0; i < n; i++) if (e_alive[i]) { accumulateFlock(e_px, e_py, e_vx, e_vy, e_ax, e_ay, e_alive, i, n, p.x, p.y); updateBoid(e_px, e_py, e_vx, e_vy, e_ax, e_ay, i); gridMove(i, e_px[i], e_py[i]); }
        predatorStepFlat(e_px, e_py, e_alive, n, p, tx, ty);
        if (now - lastFeed >= FEED_COOLDOWN) {
            var cr = size * CATCH_FACTOR;
            for (i = 0; i < n; i++) {
                if (!e_alive[i]) continue;
                var ex = p.x - e_px[i], ey = p.y - e_py[i];
                if (Math.sqrt(ex * ex + ey * ey) < cr) {
                    size = Math.min(size + GROWTH, MAX_SIZE); lastFeed = now; e_alive[i] = 0; gridRemove(i); catches++; break;
                }
            }
        }
        if (size > BASE_SIZE) size = Math.max(size - DECAY, BASE_SIZE);
        now += FRAME_MS;
    }
    var aliveCount = 0; for (i = 0; i < n; i++) aliveCount += e_alive[i];
    return { catches: catches, aliveEnd: aliveCount };
}

// ---------------------------------------------------------------------------
// REFERENCE rollout: drives the ACTUAL imported Boid class. Used only by the
// 'selftest' message to prove rolloutFlat is faithful.
// ---------------------------------------------------------------------------
function makeFakeSim(predPosVec) {
    return {
        separationMultiplier: SEP_MULT, cohesionMultiplier: COH_MULT, alignmentMultiplier: ALIGN_MULT,
        obstacles: [], predator: { position: predPosVec }, canvasWidth: cfg.W, canvasHeight: cfg.Hc
    };
}
function rebuildBoids(s, sim) {
    var n = s.bx.length, boids = new Array(n);
    for (var i = 0; i < n; i++) {
        var b = Object.create(Boid.prototype);
        b.position = new Vector(s.bx[i], s.by[i]);
        b.velocity = new Vector(s.bvx[i], s.bvy[i]);
        b.acceleration = new Vector(0, 0);
        b.simulation = sim; b.render_size = 10; b.death_throws = 0; b.sabateur = false;
        boids[i] = b;
    }
    return boids;
}
function rolloutRef(s, tx, ty, H) {
    var predPos = new Vector(s.px, s.py);
    var sim = makeFakeSim(predPos);
    var boids = rebuildBoids(s, sim);
    var pvx = s.pvx, pvy = s.pvy, predX = s.px, predY = s.py;
    var size = s.psize, lastFeed = s.lastFeed, now = s.nowMs, catches = 0;
    var R2 = cfg.POLICY_R * cfg.POLICY_R, B = 20;
    for (var f = 0; f < H; f++) {
        var i, n = boids.length;
        for (i = 0; i < n; i++) boids[i].flock(boids);
        for (i = 0; i < n; i++) { boids[i].flock(boids); boids[i].update(); }
        var bestD2 = Infinity, nx = 0, ny = 0;
        for (i = 0; i < boids.length; i++) {
            var dx = boids[i].position.x - predX, dy = boids[i].position.y - predY;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; nx = dx; ny = dy; }
        }
        var desired = (bestD2 < R2) ? new Vector(nx, ny) : new Vector(tx - predX, ty - predY);
        desired.iFastSetMagnitude(PREDATOR_MAX_SPEED);
        var steer = desired.subtract(new Vector(pvx, pvy));
        steer.iFastLimit(PREDATOR_MAX_FORCE);
        pvx += steer.x; pvy += steer.y;
        var pv = new Vector(pvx, pvy); pv.iFastLimit(PREDATOR_MAX_SPEED); pvx = pv.x; pvy = pv.y;
        predX += pvx; predY += pvy;
        if (predX > cfg.W + B) predX = -B; if (predX < -B) predX = cfg.W + B;
        if (predY > cfg.Hc + B) predY = -B; if (predY < -B) predY = cfg.Hc + B;
        sim.predator.position.x = predX; sim.predator.position.y = predY;
        if (now - lastFeed >= FEED_COOLDOWN) {
            var cr = size * CATCH_FACTOR;
            for (i = 0; i < boids.length; i++) {
                if (Math.sqrt(Math.pow(predX - boids[i].position.x, 2) + Math.pow(predY - boids[i].position.y, 2)) < cr) {
                    size = Math.min(size + GROWTH, MAX_SIZE); lastFeed = now;
                    boids.splice(i, 1); catches++; break;
                }
            }
        }
        if (size > BASE_SIZE) size = Math.max(size - DECAY, BASE_SIZE);
        now += FRAME_MS;
    }
    return catches;
}

// ---------------------------------------------------------------------------
function candidates(s) {
    var n = s.bx.length, i;
    var lite = new Array(n);
    for (i = 0; i < n; i++) lite[i] = { position: { x: s.bx[i], y: s.by[i] }, velocity: { x: s.bvx[i], y: s.bvy[i] } };
    var e3d = computeEvolvedTarget({ x: s.px, y: s.py }, lite, EVOLVED_PATROL, null) || { x: s.px, y: s.py };
    var cands = [{ x: e3d.x, y: e3d.y }];
    var order = [];
    for (i = 0; i < n; i++) { var dx = s.bx[i] - s.px, dy = s.by[i] - s.py; order.push([dx * dx + dy * dy, i]); }
    order.sort(function (a, b) { return a[0] - b[0]; });
    for (var k = 0; k < cfg.K - 1; k++) {
        if (k < order.length) {
            var j = order[k][1];
            var bx = s.bx[j], by = s.by[j], bvx = s.bvx[j], bvy = s.bvy[j];
            var ddx = bx - s.px, ddy = by - s.py;
            var dcent = Math.sqrt(ddx * ddx + ddy * ddy);
            var lead = Math.min(Math.max(dcent / PREDATOR_MAX_SPEED * LEAD_SCALE, 0), LEAD_MAX);
            cands.push({ x: bx + lead * bvx, y: by + lead * bvy });
        } else {
            cands.push({ x: e3d.x, y: e3d.y });
        }
    }
    return cands;
}

function plan(s) {
    var cands = candidates(s);
    var bestGain = -1, best = cands[0];
    for (var k = 0; k < cands.length; k++) {
        var g = rolloutFlat(s, cands[k].x, cands[k].y, cfg.H);   // strict > keeps earliest (E3D) on ties
        if (g > bestGain) { bestGain = g; best = cands[k]; }
    }
    return { x: best.x, y: best.y, gain: bestGain };
}

onmessage = function (e) {
    var m = e.data;
    if (m.type === 'config') {
        cfg.K = m.K; cfg.H = m.H; cfg.POLICY_R = m.POLICY_R;
        cfg.W = m.W; cfg.Hc = m.Hc;
        if (m.predRange) PREDATOR_RANGE = m.predRange;   // match live boid avoidance range
        return;
    }
    if (m.type === 'selftest') {
        var s = m.snapshot, cands = candidates(s);
        var maxDiff = 0, t0 = Date.now(), tFlat = 0, tRef = 0, rows = [];
        for (var k = 0; k < cands.length; k++) {
            var a0 = Date.now(); var gf = rolloutFlat(s, cands[k].x, cands[k].y, cfg.H); tFlat += Date.now() - a0;
            var b0 = Date.now(); var gr = rolloutRef(s, cands[k].x, cands[k].y, cfg.H); tRef += Date.now() - b0;
            if (Math.abs(gf - gr) > maxDiff) maxDiff = Math.abs(gf - gr);
            rows.push([gf, gr]);
        }
        postMessage({ type: 'selftest', maxCatchDiff: maxDiff, tFlatMs: tFlat, tRefMs: tRef, rows: rows });
        return;
    }
    if (m.type === 'eval') {
        var t0 = Date.now();
        var res = evalClosedLoop(m.snapshot, m.frames || 5000, m.D || 8, m.controller || 'planner');
        postMessage({ type: 'eval', controller: m.controller || 'planner', catches: res.catches, aliveEnd: res.aliveEnd, ms: Date.now() - t0 });
        return;
    }
    if (m.type === 'plan') {
        if (!m.snapshot.bx.length) { postMessage({ type: 'target', x: m.snapshot.px, y: m.snapshot.py, gain: 0 }); return; }
        var r = plan(m.snapshot);
        postMessage({ type: 'target', x: r.x, y: r.y, gain: r.gain });
    }
};
