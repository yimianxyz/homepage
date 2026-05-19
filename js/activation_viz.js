// Real-time neural activation diagram, rendered onto the same canvas as
// the boids/predator simulation. Anchored to the bottom-right of the viewport
// so it lives in the empty gutter on every device — desktop has wide side
// margins, mobile portrait pins content to the top with the bottom half
// empty, and landscape phones get a tighter override below. Strictly
// ambient: gray-ladder palette for input/hidden, predator's muted red for
// the output column, no backing frame.
//
// Reads activations from window.__predatorModel, which is guaranteed to
// be loaded by the time the simulation starts rendering (see boids.js).

(function () {
    'use strict';

    var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function clamp(min, v, max) {
        return v < min ? min : v > max ? max : v;
    }

    // Per-layer EMA of the max activation magnitude, for stable normalization.
    var emaMax = [];
    function emaUpdate(layerIdx, value) {
        if (emaMax[layerIdx] === undefined) emaMax[layerIdx] = value;
        else emaMax[layerIdx] = emaMax[layerIdx] * 0.92 + value * 0.08;
        return Math.max(emaMax[layerIdx], 1e-3);
    }

    function renderActivationViz(ctx, sim) {
        var model = window.__predatorModel;
        var cw = sim.canvasWidth;
        var ch = sim.canvasHeight;

        // Genuine micro-viewports only — anything bigger gets the adaptive layout.
        if (cw < 260 || ch < 280) return;

        // Compact mode covers both narrow portrait phones and short landscape
        // phones: smaller H cap, tighter margins, single-line caption. Text
        // remains readable even if the widget overlaps the content card
        // because the canvas sits at z-index -1 (the page sits on top).
        var compact = cw < 480 || ch < 500;

        // Adaptive footprint, clamp(min, fluid, max) — mirrors the styles.css idiom.
        var W       = clamp(78, cw * 0.15, 158);
        var H       = clamp(54, ch * 0.13, compact ? 72 : 128);
        var marginX = clamp(compact ? 10 : 14, cw * 0.022, compact ? 20 : 32);
        var marginY = clamp(compact ? 10 : 14, ch * 0.022, compact ? 18 : 32);
        var dotR    = clamp(1.1, H / 70, 1.6);

        // Caption tier: two-line on roomy desktop, one-line on tablet / landscape,
        // none on narrow phones. Architecture columns are the real signal — the
        // caption is decoration.
        var captionLines = cw < 380 ? 0 : (cw < 520 || compact) ? 1 : 2;
        var captionReserve = captionLines === 2 ? 24 : captionLines === 1 ? 12 : 0;

        var x0 = cw - marginX - W;
        var y0 = ch - marginY - H - captionReserve;

        var layerSizes = [model.featureDim].concat(model.layers.map(function (L) { return L.outDim; }));
        var nLayers = layerSizes.length;

        // Activations per "row" (row 0 = normalized inputs).
        var activations = [model.lastNormalizedInput];
        for (var i = 0; i < model.layers.length; i++) {
            activations.push(model.layers[i].lastA);
        }

        // Per-column dot positions, vertically centered within H.
        var positions = [];
        var colSpacing = W / (nLayers - 1);
        for (var l = 0; l < nLayers; l++) {
            var n = layerSizes[l];
            var colX = x0 + l * colSpacing;
            var spacing = Math.min(4.4, (H - 6) / (n - 1));
            var totalH = (n - 1) * spacing;
            var startY = y0 + (H - totalH) / 2;
            var col = new Array(n);
            for (var k = 0; k < n; k++) col[k] = { x: colX, y: startY + k * spacing };
            positions.push(col);
        }

        // --- Edges -------------------------------------------------------
        // Draw only the top |signal| edges per layer to keep it visually
        // sparse; signal = |weight * prev_activation|. This is a "wired
        // synapse firing" effect that's responsive to current state.
        if (!prefersReducedMotion) {
            for (var L_idx = 0; L_idx < model.layers.length; L_idx++) {
                var Lr = model.layers[L_idx];
                var prev = activations[L_idx];
                var posPrev = positions[L_idx];
                var posNext = positions[L_idx + 1];
                var edges = [];
                for (var j = 0; j < Lr.outDim; j++) {
                    for (var iI = 0; iI < Lr.inDim; iI++) {
                        var sig = Math.abs(Lr.W[iI * Lr.outDim + j] * prev[iI]);
                        if (sig > 0.02) edges.push([iI, j, sig]);
                    }
                }
                edges.sort(function (a, b) { return b[2] - a[2]; });
                var maxEdges = Math.min(48, edges.length);
                var sigCap = edges.length ? edges[0][2] : 1;
                for (var e = 0; e < maxEdges; e++) {
                    var ed = edges[e];
                    var a = Math.min(0.42, (ed[2] / sigCap) * 0.42);
                    ctx.strokeStyle = 'rgba(85, 85, 85, ' + a.toFixed(3) + ')';
                    ctx.lineWidth = 0.6;
                    ctx.beginPath();
                    ctx.moveTo(posPrev[ed[0]].x, posPrev[ed[0]].y);
                    ctx.lineTo(posNext[ed[1]].x, posNext[ed[1]].y);
                    ctx.stroke();
                }
            }
        }

        // --- Dots --------------------------------------------------------
        for (var li = 0; li < nLayers; li++) {
            var nn = layerSizes[li];
            var pos = positions[li];
            var act = activations[li];
            var isOutput = (li === nLayers - 1);

            // Per-layer EMA normalization so the visual is stable.
            var localMax = 0;
            for (var kk = 0; kk < nn; kk++) {
                var a2 = Math.abs(act[kk]);
                if (a2 > localMax) localMax = a2;
            }
            var norm = emaUpdate(li, localMax);

            for (var k2 = 0; k2 < nn; k2++) {
                var v = Math.abs(act[k2]) / norm;
                if (v > 1) v = 1;
                var baseAlpha = 0.10;
                var alpha = baseAlpha + (1 - baseAlpha) * v * 0.78;
                var rgb = isOutput ? '140, 60, 60' : '85, 85, 85';
                ctx.fillStyle = 'rgba(' + rgb + ', ' + alpha.toFixed(3) + ')';
                ctx.beginPath();
                ctx.arc(pos[k2].x, pos[k2].y, dotR, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // --- Tiny caption -----------------------------------------------
        // Tiered: two lines on wide, one on tablet, none on narrow.
        if (captionLines >= 1) {
            ctx.font = '9px "Source Code Pro", ui-monospace, monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            ctx.fillStyle = 'rgba(85, 85, 85, 0.42)';
            ctx.fillText(layerSizes.join(' · '), x0 + W, y0 + H + 8);
            if (captionLines >= 2) {
                ctx.fillStyle = 'rgba(85, 85, 85, 0.28)';
                ctx.fillText('predator policy', x0 + W, y0 + H + 22);
            }
            ctx.textAlign = 'left';
            ctx.textBaseline = 'alphabetic';
        }
    }

    window.renderActivationViz = renderActivationViz;
})();
