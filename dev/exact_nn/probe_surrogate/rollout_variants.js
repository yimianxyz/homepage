// rollout_variants.js — standalone faithful + cheap rollout reproductions for the
// "cheap surrogate rollout as an NN feature" probe (side-a EXACT-NN).
//
// Loads ALL prod globals via dev/fasteval.js buildHarness (constants,
// computeEvolvedTarget/EVOLVED_PATROL, cp_features/cp_value), then re-implements
// JUST the rollout loop (rolloutFlatState/accumulateFlock/updateBoid/
// predatorStepFlat/grid) copied byte-for-byte from js/predator_cheap.js, with a
// `flockMode` toggle:
//   'full'  : the shipped rollout (double accumulateFlock per frame) — must match
//             the logged oracle r.rolled[rk] exactly (validation gate).
//   'const' : skip accumulateFlock entirely — boids integrate at constant velocity
//             (torus wrap), predator step + catch detection unchanged.
//   'avoid' : ONLY the predator-avoidance qx/qy term in accumulateFlock; no
//             cohesion/separation/alignment. predator step + catch unchanged.
//
// boot for a terminal state = max_j cp_value(NET, cp_features(term, candidates(term))).
'use strict';
const fs = require('fs');
const path = require('path');
const { buildHarness } = require('../../fasteval.js');

const JS_DIR = path.join(__dirname, '..', '..', '..', 'js');

let G = null;           // prod globals snapshot
let NET = null;

async function init() {
    const built = buildHarness({ policyDir: JS_DIR, W: 1024, H: 768 });
    if (built.win.__predatorReady && built.win.__predatorReady.then) await built.win.__predatorReady;
    G = {
        cp_features: global.cp_features,
        cp_value: global.cp_value,
        computeEvolvedTarget: global.computeEvolvedTarget,
        EVOLVED_PATROL: global.EVOLVED_PATROL,
        PREDATOR_MAX_SPEED: global.PREDATOR_MAX_SPEED,
        PREDATOR_MAX_FORCE: global.PREDATOR_MAX_FORCE,
        PREDATOR_SIZE: global.PREDATOR_SIZE,
        PREDATOR_RANGE: global.PREDATOR_RANGE,
        PREDATOR_TURN_FACTOR: global.PREDATOR_TURN_FACTOR,
        MAX_SPEED: global.MAX_SPEED,
        MAX_FORCE: global.MAX_FORCE,
        NEIGHBOR_DISTANCE: global.NEIGHBOR_DISTANCE,
        DESIRED_SEPARATION: global.DESIRED_SEPARATION,
        BORDER_OFFSET: global.BORDER_OFFSET,
        EPSILON: global.EPSILON,
    };
    NET = JSON.parse(fs.readFileSync(path.join(JS_DIR, 'value_net.json'), 'utf8'));
    return { G, NET };
}

// ---- local constants pulled from prod (must match the gate) -----------------
// Bound at init() time via makeRollout(); kept as a factory so the closure sees
// the loaded globals.
function makeRollout() {
    const PREDATOR_MAX_SPEED = G.PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE = G.PREDATOR_MAX_FORCE;
    const PREDATOR_RANGE = G.PREDATOR_RANGE, PREDATOR_TURN_FACTOR = G.PREDATOR_TURN_FACTOR;
    const MAX_SPEED = G.MAX_SPEED, MAX_FORCE = G.MAX_FORCE;
    const NEIGHBOR_DISTANCE = G.NEIGHBOR_DISTANCE, DESIRED_SEPARATION = G.DESIRED_SEPARATION;
    const BORDER_OFFSET = G.BORDER_OFFSET, EPSILON = G.EPSILON;
    const computeEvolvedTarget = G.computeEvolvedTarget, EVOLVED_PATROL = G.EVOLVED_PATROL;

    // --- predator_cheap.js IIFE-local constants (lines 27-31) ---
    const FRAME_MS = 12, GROWTH = 1.2;
    const BASE_SIZE = G.PREDATOR_SIZE, MAX_SIZE = BASE_SIZE * 1.8, CATCH_FACTOR = 0.7;
    const SEP_MULT = 2, COH_MULT = 1, ALIGN_MULT = 1;
    const LEAD_SCALE = EVOLVED_PATROL.lead_scale, LEAD_MAX = EVOLVED_PATROL.lead_max;
    const cfg = { K: 16, K_roll: 4, Hs: 90, D: 16, POLICY_R: 80, W: 1024, Hc: 768 };

    // fastMag — verbatim
    function fastMag(x, y) {
        var ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
        return (ax > ay ? ax : ay) * 0.96 + (ax < ay ? ax : ay) * 0.398;
    }

    // ---- flat scratch sim + uniform spatial hash (verbatim) ----
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

    // accumulateFlock — verbatim, with a `mode` parameter:
    //   mode==='full'  : everything (cohesion+sep+align+predator-avoid)
    //   mode==='avoid' : ONLY the predator-avoid qx/qy term (skip coh/sep/align loop work too)
    // (mode==='const' never calls this fn at all.)
    function accumulateFlock(px, py, vx, vy, ax, ay, alive, i, n, predX, predY, mode) {
        var pix = px[i], piy = py[i];
        if (mode === 'full') {
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
        }
        // predator-avoidance term (present in 'full' and 'avoid')
        var qx = pix - predX, qy = piy - predY;
        var pdist = Math.sqrt(qx * qx + qy * qy) + EPSILON;
        if (pdist > 0 && pdist < PREDATOR_RANGE) {
            var fm2 = fastMag(qx, qy); if (fm2 > 0) { qx /= fm2; qy /= fm2; }
            var str = (PREDATOR_RANGE - pdist) / PREDATOR_RANGE * PREDATOR_TURN_FACTOR;
            qx *= str; qy *= str;
            var lim = MAX_FORCE * 1.5;
            fm2 = fastMag(qx, qy); if (fm2 > lim) { var s2 = lim / fm2; qx *= s2; qy *= s2; }
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

    // Roll one target H frames; mode in {'full','avoid','const'}.
    function rolloutFlatState(s, tx, ty, H, mode) {
        var n = s.bx.length;
        ensureCap(n);
        for (var i = 0; i < n; i++) {
            _px[i] = s.bx[i]; _py[i] = s.by[i]; _vx[i] = s.bvx[i]; _vy[i] = s.bvy[i];
            _ax[i] = 0; _ay[i] = 0; _alive[i] = 1;
        }
        var useGrid = (mode === 'full');   // grid only needed for neighbor scan (full)
        if (useGrid) gridBuild(_px, _py, _alive, n);
        var p = { x: s.px, y: s.py, vx: s.pvx, vy: s.pvy };
        var size = s.psize, lastFeed = (s.lastFeed != null ? s.lastFeed : -1e9), now = s.nowMs || 0, catches = 0;
        for (var f = 0; f < H; f++) {
            if (mode !== 'const') {
                // two-pass accumulate, mirroring the engine's tick()/render() split.
                for (i = 0; i < n; i++) if (_alive[i]) accumulateFlock(_px, _py, _vx, _vy, _ax, _ay, _alive, i, n, p.x, p.y, mode);
                for (i = 0; i < n; i++) if (_alive[i]) {
                    accumulateFlock(_px, _py, _vx, _vy, _ax, _ay, _alive, i, n, p.x, p.y, mode);
                    updateBoid(_px, _py, _vx, _vy, _ax, _ay, i);
                    if (useGrid) gridMove(i, _px[i], _py[i]);
                }
            } else {
                // V_const: constant-velocity integration (ax=ay=0), torus wrap.
                for (i = 0; i < n; i++) if (_alive[i]) updateBoid(_px, _py, _vx, _vy, _ax, _ay, i);
            }
            predatorStepFlat(_px, _py, _alive, n, p, tx, ty);
            var cr = size * CATCH_FACTOR;
            for (i = 0; i < n; i++) {
                if (!_alive[i]) continue;
                var ex = p.x - _px[i], ey = p.y - _py[i];
                if (Math.sqrt(ex * ex + ey * ey) < cr) {
                    size = Math.min(size + GROWTH, MAX_SIZE); lastFeed = now; _alive[i] = 0;
                    if (useGrid) gridRemove(i); catches++; break;
                }
            }
            now += FRAME_MS;
        }
        return { catches: catches, term: snapAlive(_px, _py, _vx, _vy, _alive, n, p, size, lastFeed, now) };
    }

    // candidates — verbatim from predator_cheap.js (uses prod computeEvolvedTarget).
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

    // boot for a terminal snapshot: max value over candidates(term).
    function bootOf(term) {
        var tcands = candidates(term);
        var tst = { px: term.px, py: term.py, pvx: term.pvx, pvy: term.pvy, psize: term.psize,
            bx: term.bx, by: term.by, bvx: term.bvx, bvy: term.bvy, nAlive: term.bx.length };
        var tfr = G.cp_features(tst, tcands, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE);
        var tv = G.cp_value(NET, tfr.feat, tfr.ctx);
        var boot = -Infinity; for (var j = 0; j < tv.length; j++) if (tv[j] > boot) boot = tv[j];
        return boot;
    }

    // Score one rolled candidate (catches + boot) for a given mode.
    function rollScore(s, tx, ty, mode) {
        var rr = rolloutFlatState(s, tx, ty, cfg.Hs, mode);
        return { catches: rr.catches, boot: bootOf(rr.term) };
    }

    return { rolloutFlatState, candidates, bootOf, rollScore, cfg };
}

module.exports = { init, makeRollout, getNet: () => NET, getG: () => G };
