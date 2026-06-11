// Cheap ballistic predator policy — the SHIPPED production patrol policy.
//
// Distilled from the receding-horizon planner "teacher" (dev/planner_probe.py).
// Every D=16 frames it picks a patrol target by: generate 16 candidates (E3D
// patrol + 15 nearest-boid lead-adjusted intercepts), rank them by a cheap 2-body
// ballistic catchability sim, roll the K_roll=4 most-catchable candidates Hs=90
// frames through a faithful flat copy of the flock, and argmax (rolled catches +
// value-net terminal bootstrap) vs the other candidates' value-net prior. Then
// each frame it steers with the SAME analytic chase/seek as the old CHASE branch.
//
// This recovers ~90% of the receding-horizon planner's catch rate at ~1/4 its
// rollout cost (sim_torch: K4/Hs90/D16 = 16.88 vs planner 18.75; the deep rollout
// lets the committed target hold for D=16 frames). Cheap enough to run
// SYNCHRONOUSLY on the main thread — no Web Worker, no staleness.
//
// The page boot gate (window.__predatorReady) waits on the value-net load before
// boids.js starts the sim, so this policy drives the predator from frame 1.
//
// The flat rollout (fastMag, grid, accumulateFlock, updateBoid, predatorStepFlat,
// rolloutFlatState) is the byte-identical mirror of boid.js + predator.js; the
// scoring (cp_*) lives in cheap_planner.js. Both are verified against the GPU sim
// (dev/test_cheap_parity.js < 1e-6; dev/eval_cheap_production.js end-to-end).
// Constants come from boid.js + predator.js.
'use strict';

(function () {
    var FRAME_MS = 12, FEED_COOLDOWN = 100, GROWTH = 1.2, DECAY = 0.002;
    var BASE_SIZE = PREDATOR_SIZE, MAX_SIZE = BASE_SIZE * 1.8, CATCH_FACTOR = 0.7;
    var SEP_MULT = 2, COH_MULT = 1, ALIGN_MULT = 1;
    var LEAD_SCALE = EVOLVED_PATROL.lead_scale, LEAD_MAX = EVOLVED_PATROL.lead_max;
    var cfg = { K: 16, K_roll: 4, Hs: 90, D: 16, POLICY_R: 43.157, W: 1680, Hc: 1680 };

    function fastMag(x, y) {
        var ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
        return (ax > ay ? ax : ay) * 0.96 + (ax < ay ? ax : ay) * 0.398;
    }

    // ---- flat scratch sim + uniform spatial hash (verbatim from the worker) ----
    var _px, _py, _vx, _vy, _ax, _ay, _alive, _cap = 0;
    function ensureCap(n) {
        if (n <= _cap) return;
        _px = new Float64Array(n); _py = new Float64Array(n);
        _vx = new Float64Array(n); _vy = new Float64Array(n);
        _ax = new Float64Array(n); _ay = new Float64Array(n);
        _alive = new Uint8Array(n); _cap = n;
    }
    var GCELL = NEIGHBOR_DISTANCE, GOFF = 120;
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

    function accumulateFlock(px, py, vx, vy, ax, ay, alive, i, n, predX, predY) {
        var pix = px[i], piy = py[i];
        var cx = 0, cy = 0, cn = 0, sx = 0, sy = 0, sn = 0, alx = 0, aly = 0, an = 0;
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
                        var ddx = -rx, ddy = -ry;
                        var m = Math.sqrt(ddx * ddx + ddy * ddy);
                        if (m > 0) { ddx /= m; ddy /= m; }
                        ddx /= dist; ddy /= dist;
                        sx += ddx; sy += ddy; sn++;
                    }
                    if (dist > 0 && dist < NEIGHBOR_DISTANCE) { alx += vx[j]; aly += vy[j]; an++; }
                }
            }
        }
        var vxi = vx[i], vyi = vy[i], fm, s;
        var cohx = 0, cohy = 0;
        if (cn > 0) {
            var dx = cx / cn - pix, dy = cy / cn - piy;
            fm = fastMag(dx, dy); if (fm > 0) { s = MAX_SPEED / fm; dx *= s; dy *= s; }
            dx -= vxi; dy -= vyi;
            fm = fastMag(dx, dy); if (fm > MAX_FORCE) { s = MAX_FORCE / fm; dx *= s; dy *= s; }
            cohx = dx; cohy = dy;
        }
        var sepx = 0, sepy = 0;
        if (sn > 0) { sepx = sx / sn; sepy = sy / sn; }
        fm = fastMag(sepx, sepy);
        if (fm > 0) {
            s = MAX_SPEED / fm; sepx *= s; sepy *= s;
            sepx -= vxi; sepy -= vyi;
            fm = fastMag(sepx, sepy); if (fm > MAX_FORCE) { s = MAX_FORCE / fm; sepx *= s; sepy *= s; }
        }
        var algx = 0, algy = 0;
        if (an > 0) {
            var avx = alx / an, avy = aly / an;
            fm = fastMag(avx, avy); if (fm > 0) { s = MAX_SPEED / fm; avx *= s; avy *= s; }
            avx -= vxi; avy -= vyi;
            fm = fastMag(avx, avy); if (fm > MAX_FORCE) { s = MAX_FORCE / fm; avx *= s; avy *= s; }
            algx = avx; algy = avy;
        }
        sepx *= SEP_MULT; sepy *= SEP_MULT;
        cohx *= COH_MULT; cohy *= COH_MULT;
        algx *= ALIGN_MULT; algy *= ALIGN_MULT;
        ax[i] += cohx; ay[i] += cohy;
        ax[i] += sepx; ay[i] += sepy;
        ax[i] += algx; ay[i] += algy;
        var qx = pix - predX, qy = piy - predY;
        var pdist = Math.sqrt(qx * qx + qy * qy) + EPSILON;
        if (pdist > 0 && pdist < PREDATOR_RANGE) {
            fm = fastMag(qx, qy); if (fm > 0) { qx /= fm; qy /= fm; }
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
        var B = BORDER_OFFSET;
        if (px[i] > cfg.W + B) px[i] = -B;
        if (px[i] < -B) px[i] = cfg.W + B;
        if (py[i] > cfg.Hc + B) py[i] = -B;
        if (py[i] < -B) py[i] = cfg.Hc + B;
        ax[i] = 0; ay[i] = 0;
    }

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

    function snapAlive(px, py, vx, vy, alive, n, p, size, lastFeed, now) {
        var bx = [], by = [], bvx = [], bvy = [];
        for (var i = 0; i < n; i++) {
            if (!alive[i]) continue;
            bx.push(px[i]); by.push(py[i]); bvx.push(vx[i]); bvy.push(vy[i]);
        }
        return { bx: bx, by: by, bvx: bvx, bvy: bvy, px: p.x, py: p.y, pvx: p.vx, pvy: p.vy,
            psize: size, lastFeed: lastFeed, nowMs: now };
    }

    // Roll one target H frames; return catches + terminal snapshot (for the bootstrap).
    function rolloutFlatState(s, tx, ty, H) {
        var n = s.bx.length;
        ensureCap(n);
        for (var i = 0; i < n; i++) {
            _px[i] = s.bx[i]; _py[i] = s.by[i]; _vx[i] = s.bvx[i]; _vy[i] = s.bvy[i];
            _ax[i] = 0; _ay[i] = 0; _alive[i] = 1;
        }
        gridBuild(_px, _py, _alive, n);
        var p = { x: s.px, y: s.py, vx: s.pvx, vy: s.pvy };
        var size = s.psize, lastFeed = (s.lastFeed != null ? s.lastFeed : -1e9), now = s.nowMs || 0, catches = 0;
        for (var f = 0; f < H; f++) {
            for (i = 0; i < n; i++) if (_alive[i]) accumulateFlock(_px, _py, _vx, _vy, _ax, _ay, _alive, i, n, p.x, p.y);
            for (i = 0; i < n; i++) if (_alive[i]) { accumulateFlock(_px, _py, _vx, _vy, _ax, _ay, _alive, i, n, p.x, p.y); updateBoid(_px, _py, _vx, _vy, _ax, _ay, i); gridMove(i, _px[i], _py[i]); }
            predatorStepFlat(_px, _py, _alive, n, p, tx, ty);
            var cr = size * CATCH_FACTOR;
            for (i = 0; i < n; i++) {
                if (!_alive[i]) continue;
                var ex = p.x - _px[i], ey = p.y - _py[i];
                if (Math.sqrt(ex * ex + ey * ey) < cr) {
                    size = Math.min(size + GROWTH, MAX_SIZE); lastFeed = now; _alive[i] = 0; gridRemove(i); catches++; break;
                }
            }
            now += FRAME_MS;
        }
        return { catches: catches, term: snapAlive(_px, _py, _vx, _vy, _alive, n, p, size, lastFeed, now) };
    }

    // 16 candidate targets: cand0 = E3D evolved patrol; cand1..15 = nearest live
    // boids lead-adjusted. Identical to the planner worker's candidates().
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

    // The kr=1 ballistic decision. NET = value_net.json (loaded below). vizModel
    // mirrors the value net for activation_viz.js (the "predator's brain").
    var NET = null, vizModel = null;
    function planCheap(s) {
        var cands = candidates(s);
        var st = { px: s.px, py: s.py, pvx: s.pvx, pvy: s.pvy, psize: s.psize,
            bx: s.bx, by: s.by, bvx: s.bvx, bvy: s.bvy, nAlive: s.bx.length };
        var fr = cp_features(st, cands, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE);
        var vprior = cp_value(NET, fr.feat, fr.ctx);
        // Roll the top-K_roll candidates by ballistic pscore (caught - tCatchNorm);
        // ties (the common case for the slow predator) break by lowest index, so
        // this rolls the E3D patrol + the nearest live boids. Each rolled candidate
        // is scored by (rollout catches + value-net terminal bootstrap); the rest
        // keep their value-net prior. argmax commits.
        var pidx = []; for (var pk = 0; pk < cands.length; pk++) pidx.push(pk);
        pidx.sort(function (a, b) { var pa = fr.feat[a][18] - fr.feat[a][16], pb = fr.feat[b][18] - fr.feat[b][16]; return (pb - pa) || (a - b); });
        var score = vprior.slice();
        for (var rk = 0; rk < cfg.K_roll && rk < pidx.length; rk++) {
            var ci = pidx[rk];
            var rr = rolloutFlatState(s, cands[ci].x, cands[ci].y, cfg.Hs);
            var t = rr.term;
            var tcands = candidates(t);
            var tst = { px: t.px, py: t.py, pvx: t.pvx, pvy: t.pvy, psize: t.psize,
                bx: t.bx, by: t.by, bvx: t.bvx, bvy: t.bvy, nAlive: t.bx.length };
            var tfr = cp_features(tst, tcands, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE);
            var tv = cp_value(NET, tfr.feat, tfr.ctx);
            var boot = -Infinity; for (var j = 0; j < tv.length; j++) if (tv[j] > boot) boot = tv[j];
            score[ci] = rr.catches + boot;
        }
        var bi = 0, bs = -Infinity;
        for (var k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; }
        // Record the chosen candidate's value-net forward for the brain viz.
        if (vizModel) cp_value_viz(NET, fr.feat[bi], fr.ctx, vizModel);
        return { x: cands[bi].x, y: cands[bi].y };
    }

    // ---- per-frame driver ----
    var target = null, frame = 0, configured = false;

    function configure(sim) {
        cfg.W = sim.canvasWidth; cfg.Hc = sim.canvasHeight;
        configured = true;
    }

    function snapshot(pred, boids) {
        var n = boids.length;
        var bx = new Array(n), by = new Array(n), bvx = new Array(n), bvy = new Array(n);
        for (var i = 0; i < n; i++) {
            bx[i] = boids[i].position.x; by[i] = boids[i].position.y;
            bvx[i] = boids[i].velocity.x; bvy[i] = boids[i].velocity.y;
        }
        return { bx: bx, by: by, bvx: bvx, bvy: bvy,
            px: pred.position.x, py: pred.position.y, pvx: pred.velocity.x, pvy: pred.velocity.y,
            psize: pred.currentSize, lastFeed: pred.lastFeedTime, nowMs: simNow() };
    }

    // Chase nearest within POLICY_R, else seek the committed target — the same
    // analytic step as the production CHASE branch + the worker predatorStep.
    function steer(pred, boids) {
        var px = pred.position.x, py = pred.position.y;
        var bestD2 = Infinity, nx = 0, ny = 0;
        for (var i = 0, n = boids.length; i < n; i++) {
            var dx = boids[i].position.x - px, dy = boids[i].position.y - py;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; nx = dx; ny = dy; }
        }
        var desired;
        if (bestD2 < cfg.POLICY_R * cfg.POLICY_R) desired = new Vector(nx, ny);
        else if (target) desired = new Vector(target.x - px, target.y - py);
        else desired = new Vector(nx, ny);
        desired.iFastSetMagnitude(PREDATOR_MAX_SPEED);
        var s = desired.subtract(pred.velocity);
        s.iFastLimit(PREDATOR_MAX_FORCE);
        return s;
    }

    // ENDGAME INTERCEPTOR — earliest-reachable torus lead-intercept. For the last
    // <=5 boids, a lone boid flies a straight line; scan its track at single-frame
    // resolution for the EARLIEST torus-min-image-reachable point (boid period W+20,
    // so the short backward-wrap that puts the predator AHEAD wins — a head-on close
    // up to 8.5/frame vs the -3.5 tail-chase deficit) and re-aim there EVERY frame.
    // ~6x faster + 100% reliable on the last boid vs the tail-chasing lookahead
    // (deployed gets stuck and never clears 12-18% of the time on big screens).
    var egBoid = null, egAimX = 0, egAimY = 0, egFrozen = false;
    function intercept(pred, boids) {
        var px = pred.position.x, py = pred.position.y, sM = PREDATOR_MAX_SPEED;
        var PX = cfg.W + 2 * BORDER_OFFSET, PY = cfg.Hc + 2 * BORDER_OFFSET;
        // DT=1: scan the boid's track at single-frame resolution. A coarse DT=4
        // misses the true earliest-reachable point by up to 4 frames, which on a slow
        // pursuer compounds into chronic near-misses (each forcing a full extra lap).
        // Single-frame scan + re-aiming EVERY frame is ~2x faster to the catch and
        // 100% reliable; validated on JS (lab + real harness), an independent audit,
        // and a 4096-env GPU sweep — all agree freeze-and-commit was a net loss.
        var SLACK = 1.0, TMAX = 1400, DT = 1;
        function wx(d) { return d - PX * Math.round(d / PX); }
        function wy(d) { return d - PY * Math.round(d / PY); }
        function scan(B) {
            var bx = B.position.x, by = B.position.y, bvx = B.velocity.x, bvy = B.velocity.y;
            for (var t = 0; t <= TMAX; t += DT) {
                var ddx = wx(bx + bvx * t - px), ddy = wy(by + bvy * t - py);
                var d = Math.sqrt(ddx * ddx + ddy * ddy);
                if (d <= sM * t * SLACK) return { t: t, ax: ddx, ay: ddy };
            }
            return null;
        }
        var n = boids.length;
        if (egBoid) { var found = false; for (var i = 0; i < n; i++) if (boids[i] === egBoid) { found = true; break; } if (!found) { egBoid = null; egFrozen = false; } }
        if (!egBoid) {
            var bestT = Infinity, bestB = null;
            for (var i = 0; i < n; i++) { var s = scan(boids[i]); if (s && s.t < bestT) { bestT = s.t; bestB = boids[i]; } }
            if (!bestB) { var nd2 = Infinity; for (i = 0; i < n; i++) { var dx = wx(boids[i].position.x - px), dy = wy(boids[i].position.y - py); var d2 = dx * dx + dy * dy; if (d2 < nd2) { nd2 = d2; bestB = boids[i]; } } }
            egBoid = bestB; egFrozen = false;
        }
        var B = egBoid;
        var cdx = wx(B.position.x - px), cdy = wy(B.position.y - py);
        var curdist = Math.sqrt(cdx * cdx + cdy * cdy);
        // Re-aim EVERY frame at the earliest-reachable intercept point (no freeze).
        var s = scan(B);
        if (s) { egAimX = s.ax; egAimY = s.ay; }
        else {                                        // perpendicular cut-off onto the boid's line
            var bs = Math.sqrt(B.velocity.x * B.velocity.x + B.velocity.y * B.velocity.y) || 1e-6;
            var ux = B.velocity.x / bs, uy = B.velocity.y / bs;
            var along = cdx * ux + cdy * uy;
            egAimX = cdx - along * ux; egAimY = cdy - along * uy;
        }
        var desired = new Vector(egAimX, egAimY);
        desired.iFastSetMagnitude(sM);
        var st = desired.subtract(pred.velocity);
        st.iFastLimit(PREDATOR_MAX_FORCE);
        return st;
    }

    window.__cheap = {
        force: function (pred, boids) {
            if (boids.length === 0) return new Vector(0, 0);
            if (!configured && pred.simulation) configure(pred.simulation);
            if (boids.length <= 5) return intercept(pred, boids);   // ENDGAME: TERI head-on intercept
            if (frame === 0 || frame >= cfg.D) { target = planCheap(snapshot(pred, boids)); frame = 0; }
            frame++;
            return steer(pred, boids);
        }
    };

    // Own the page boot gate: boids.js waits on window.__predatorReady before
    // starting the sim, so the value net is loaded before frame 1.
    if (typeof window !== 'undefined' && typeof fetch !== 'undefined') {
        window.__predatorReady = fetch('js/value_net.json', { cache: 'no-cache' })
            .then(function (r) { if (!r.ok) throw new Error('value_net fetch failed: ' + r.status); return r.json(); })
            .then(function (net) { NET = net; vizModel = cp_viz_model(net); window.__predatorModel = vizModel; });
    }
})();
