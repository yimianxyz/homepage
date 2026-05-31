// E3D "predator's brain" diagram — the parametric-E3D patrol net rendered in
// the exact visual language of the production activation viz (js/activation_viz.js):
// vertical dot-columns, gray ladder for the body, the predator's muted red for the
// output column, sparse top-K "firing" edges, per-stage EMA normalization, a mono
// "predator's brain" label.
//
// The difference from production is conceptual, not cosmetic: production draws a
// generic MLP's weight-matrix layers. This net has NO weight matrices to draw — it
// is the E3D computation graph with 7 learnable scalars. So the columns here are the
// PIPELINE STAGES of computeEvolvedTarget (js/predator.js), computed live each frame
// from sim.boids + sim.predator:
//
//   boids   one node per live boid; brightness = local neighbour count (density)
//   attract brightness = a_i = (cnt+1)^dens_pow * exp(-distPred/reach_scale)
//   select  brightness = w_i = (a_i / amax)^sharp   <- the peaky SELECTION op
//   aim     5 nodes: cx, cy, vx, vy, lead  (weighted centroid + velocity lead)
//   steer   2 red nodes: force x, y        (analytic seek toward the aimed target)
//
// Boids are sorted by attractiveness so the chosen densest cluster groups at the top
// of the first three columns — you can watch the "select" column collapse onto it.

(function () {
    'use strict';

    var MAXSPEED = 2.5, MAXFORCE = 0.05;

    function clamp(min, v, max) { return v < min ? min : v > max ? max : v; }

    // 7 learnable constants of the parametric-E3D net (= EVOLVED_PATROL).
    var K = {
        cluster_r: 178.09, dens_pow: 2.373, reach_scale: 1515.0,
        sharp: 9.25, lead_scale: 0.454, lead_max: 230.6, nbhd: 0.461
    };

    // Recompute every intermediate of computeEvolvedTarget, keeping the per-boid
    // arrays we want to visualize (cnt, attract, weight) plus the assembled aim.
    function computeStages(predPos, boids) {
        var n = boids.length;
        var R2 = K.cluster_r * K.cluster_r;
        var px = predPos.x, py = predPos.y;
        var cnt = new Array(n), attract = new Array(n);
        var amax = 1e-12, bestIdx = 0, bestA = -1;
        for (var i = 0; i < n; i++) {
            var bxi = boids[i].position.x, byi = boids[i].position.y, c = 0;
            for (var j = 0; j < n; j++) {
                var ex = boids[j].position.x - bxi, ey = boids[j].position.y - byi;
                if (ex * ex + ey * ey < R2) c++;
            }
            cnt[i] = c;
            var ddx = bxi - px, ddy = byi - py;
            var dpred = Math.sqrt(ddx * ddx + ddy * ddy);
            var a = Math.pow(c + 1, K.dens_pow) * Math.exp(-dpred / K.reach_scale);
            attract[i] = a;
            if (a > amax) amax = a;
            if (a > bestA) { bestA = a; bestIdx = i; }
        }
        var weight = new Array(n), wsum = 0, cx = 0, cy = 0, vx = 0, vy = 0;
        for (var k = 0; k < n; k++) {
            var w = Math.pow(attract[k] / amax, K.sharp);
            weight[k] = w; wsum += w;
            cx += w * boids[k].position.x; cy += w * boids[k].position.y;
            vx += w * boids[k].velocity.x; vy += w * boids[k].velocity.y;
        }
        if (wsum < 1e-12) wsum = 1e-12;
        cx /= wsum; cy /= wsum; vx /= wsum; vy /= wsum;
        if (K.nbhd > 0 && n > 0) {
            var bx = boids[bestIdx].position.x, by = boids[bestIdx].position.y;
            var nsum = 0, ncx = 0, ncy = 0, nvx = 0, nvy = 0;
            for (var m = 0; m < n; m++) {
                var gx = boids[m].position.x - bx, gy = boids[m].position.y - by;
                if (gx * gx + gy * gy < R2) {
                    ncx += boids[m].position.x; ncy += boids[m].position.y;
                    nvx += boids[m].velocity.x; nvy += boids[m].velocity.y; nsum++;
                }
            }
            if (nsum < 1e-12) nsum = 1e-12;
            ncx /= nsum; ncy /= nsum; nvx /= nsum; nvy /= nsum;
            cx = (1 - K.nbhd) * cx + K.nbhd * ncx;
            cy = (1 - K.nbhd) * cy + K.nbhd * ncy;
            vx = (1 - K.nbhd) * vx + K.nbhd * nvx;
            vy = (1 - K.nbhd) * vy + K.nbhd * nvy;
        }
        var dx2 = cx - px, dy2 = cy - py;
        var dcent = Math.sqrt(dx2 * dx2 + dy2 * dy2);
        var lead = clamp(0, dcent / MAXSPEED * K.lead_scale, K.lead_max);
        var tx = cx + lead * vx, ty = cy + lead * vy;
        // steer = analytic seek toward the aimed target (the net's output)
        var sdx = tx - px, sdy = ty - py;
        var sm = Math.sqrt(sdx * sdx + sdy * sdy) || 1e-12;
        var desx = sdx / sm * MAXSPEED, desy = sdy / sm * MAXSPEED;
        // predator velocity isn't in predPos; caller passes it on predPos.vel
        var pvx = predPos.vx || 0, pvy = predPos.vy || 0;
        var fx = desx - pvx, fy = desy - pvy;
        var fm = Math.sqrt(fx * fx + fy * fy);
        if (fm > MAXFORCE) { fx = fx / fm * MAXFORCE; fy = fy / fm * MAXFORCE; }
        return {
            n: n, cnt: cnt, attract: attract, weight: weight, bestIdx: bestIdx,
            aim: [cx, cy, vx, vy, lead], force: [fx, fy]
        };
    }

    var emaMax = {};
    function emaUpdate(key, value) {
        if (emaMax[key] === undefined) emaMax[key] = value;
        else emaMax[key] = emaMax[key] * 0.92 + value * 0.08;
        return Math.max(emaMax[key], 1e-9);
    }

    // Normalize an array to [0,1] by its EMA max magnitude.
    function normArr(key, arr) {
        var mx = 0;
        for (var i = 0; i < arr.length; i++) { var a = Math.abs(arr[i]); if (a > mx) mx = a; }
        var norm = emaUpdate(key, mx);
        var out = new Array(arr.length);
        for (var j = 0; j < arr.length; j++) out[j] = clamp(0, Math.abs(arr[j]) / norm, 1);
        return out;
    }

    function renderE3DBrain(ctx, sim, layout) {
        var boids = sim.boids;
        if (!boids || boids.length === 0) return;
        var pred = sim.predator;
        var pp = { x: pred.position.x, y: pred.position.y, vx: pred.velocity.x, vy: pred.velocity.y };
        var S = computeStages(pp, boids);
        var n = S.n;

        // Sort boid display order by attractiveness (densest cluster on top).
        var order = [];
        for (var i = 0; i < n; i++) order.push(i);
        order.sort(function (a, b) { return S.attract[b] - S.attract[a]; });

        // Normalized per-stage values (gray brightness).
        var cntN = normArr('cnt', S.cnt);
        var attN = normArr('att', S.attract);
        var wN = normArr('wgt', S.weight);
        // aim = [cx, cy, vx, vy, lead] — each component lives on its own scale,
        // so normalize per-natural-unit (not jointly) or the big world-coord
        // centroid swamps velocity + lead and three of five dots go dark.
        var aimN = [
            clamp(0, Math.abs(S.aim[0]) / 1680, 1),
            clamp(0, Math.abs(S.aim[1]) / 1680, 1),
            clamp(0, Math.abs(S.aim[2]) / 6, 1),
            clamp(0, Math.abs(S.aim[3]) / 6, 1),
            clamp(0, S.aim[4] / 230.6, 1)
        ];
        var frcN = normArr('frc', S.force);

        var x0 = layout.x, y0 = layout.y, W = layout.w, H = layout.h;
        var dotR = layout.dotR || 3.0;
        var edgeW = layout.edgeW || 0.7;
        var GRAY = '85, 85, 85', RED = '140, 60, 60';

        // 5 columns: boids, attract, select, aim, steer.
        var sizes = [n, n, n, 5, 2];
        var labels = ['boids', 'attract', 'select', 'aim', 'steer'];
        var nCols = 5;
        var colSpacing = W / (nCols - 1);

        function colPositions(col, count) {
            var colX = x0 + col * colSpacing;
            var spacing = Math.min(layout.maxSpacing || 7, (H - 8) / Math.max(1, count - 1));
            var totalH = (count - 1) * spacing;
            var startY = y0 + (H - totalH) / 2;
            var pos = new Array(count);
            for (var k = 0; k < count; k++) pos[k] = { x: colX, y: startY + k * spacing };
            return pos;
        }
        // For the boid columns, the i-th visual row is order[i]; map boidIdx->row.
        var boidPos = colPositions(0, n);
        var attractPos = colPositions(1, n);
        var selectPos = colPositions(2, n);
        var aimPos = colPositions(3, 5);
        var steerPos = colPositions(4, 2);
        var rowOf = new Array(n);
        for (var r = 0; r < n; r++) rowOf[order[r]] = r;

        // --- Edges (drawn first, behind dots) ---------------------------------
        // boids->attract and attract->select are per-boid horizontal "carries";
        // brightness follows the downstream value (attract, then weight). We draw
        // the top-K by value to stay sparse like production.
        function horizEdges(prevPos, nextPos, vals, cap, topK) {
            var idx = [];
            for (var q = 0; q < n; q++) idx.push(q);
            idx.sort(function (a, b) { return vals[b] - vals[a]; });
            var lim = Math.min(topK, n), mx = vals[idx[0]] || 1e-9;
            for (var e = 0; e < lim; e++) {
                var bi = idx[e], rr = rowOf[bi];
                var al = clamp(0, vals[bi] / mx, 1) * cap;
                ctx.strokeStyle = 'rgba(' + GRAY + ', ' + al.toFixed(3) + ')';
                ctx.lineWidth = edgeW;
                ctx.beginPath();
                ctx.moveTo(prevPos[rr].x, prevPos[rr].y);
                ctx.lineTo(nextPos[rr].x, nextPos[rr].y);
                ctx.stroke();
            }
        }
        horizEdges(boidPos, attractPos, attN, 0.30, 22);
        horizEdges(attractPos, selectPos, wN, 0.42, 16);

        // select->aim: the surviving (high-weight) boids fan into the 5 aim nodes.
        var widx = [];
        for (var s = 0; s < n; s++) widx.push(s);
        widx.sort(function (a, b) { return S.weight[b] - S.weight[a]; });
        var wmx = S.weight[widx[0]] || 1e-9;
        var fanK = Math.min(18, n);
        for (var f = 0; f < fanK; f++) {
            var fb = widx[f], frow = rowOf[fb];
            var fal = clamp(0, S.weight[fb] / wmx, 1) * 0.5;
            for (var t = 0; t < 5; t++) {
                ctx.strokeStyle = 'rgba(' + GRAY + ', ' + (fal * (t < 4 ? 1 : 0.5)).toFixed(3) + ')';
                ctx.lineWidth = edgeW;
                ctx.beginPath();
                ctx.moveTo(selectPos[frow].x, selectPos[frow].y);
                ctx.lineTo(aimPos[t].x, aimPos[t].y);
                ctx.stroke();
            }
        }
        // aim->steer: all 5 into both outputs, brightness by aim magnitude.
        for (var ai = 0; ai < 5; ai++) {
            for (var so = 0; so < 2; so++) {
                var aal = (0.12 + 0.5 * aimN[ai]) * 0.6;
                ctx.strokeStyle = 'rgba(' + RED + ', ' + aal.toFixed(3) + ')';
                ctx.lineWidth = edgeW;
                ctx.beginPath();
                ctx.moveTo(aimPos[ai].x, aimPos[ai].y);
                ctx.lineTo(steerPos[so].x, steerPos[so].y);
                ctx.stroke();
            }
        }

        // --- Dots -------------------------------------------------------------
        function drawCol(pos, vals, rgb, rOverride) {
            var base = 0.10;
            for (var k = 0; k < pos.length; k++) {
                var v = vals[k];
                var alpha = base + (1 - base) * v * 0.82;
                ctx.fillStyle = 'rgba(' + rgb + ', ' + alpha.toFixed(3) + ')';
                ctx.beginPath();
                ctx.arc(pos[k].x, pos[k].y, rOverride || dotR, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        // boid columns: index by visual row = order[r].
        var cntRow = new Array(n), attRow = new Array(n), wRow = new Array(n);
        for (var rr2 = 0; rr2 < n; rr2++) {
            cntRow[rr2] = cntN[order[rr2]];
            attRow[rr2] = attN[order[rr2]];
            wRow[rr2] = wN[order[rr2]];
        }
        var bR = layout.boidDotR || 1.6;
        drawCol(boidPos, cntRow, GRAY, bR);
        drawCol(attractPos, attRow, GRAY, bR);
        drawCol(selectPos, wRow, GRAY, bR);
        drawCol(aimPos, aimN, GRAY, dotR + 0.8);
        drawCol(steerPos, frcN, RED, dotR + 1.4);

        // --- Labels -----------------------------------------------------------
        ctx.font = (layout.labelFont || '13px') + ' "Source Code Pro", ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(85, 85, 85, 0.62)';
        for (var c2 = 0; c2 < nCols; c2++) {
            ctx.fillText(labels[c2], x0 + c2 * colSpacing, y0 + H + 10);
        }
        // Title (matches production's "predator's brain" voice).
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(85, 85, 85, 0.78)';
        ctx.font = (layout.titleFont || '15px') + ' "Source Code Pro", ui-monospace, monospace';
        ctx.fillText("predator's brain", x0, y0 - 30);
        ctx.fillStyle = 'rgba(85, 85, 85, 0.42)';
        ctx.font = '11px "Source Code Pro", ui-monospace, monospace';
        ctx.fillText('E3D patrol policy · 7 learnable constants · 100% steer-match',
            x0, y0 - 12);

        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
    }

    window.renderE3DBrain = renderE3DBrain;
})();
