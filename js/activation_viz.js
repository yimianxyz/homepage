// Real-time neural activation diagram, rendered onto the same canvas as
// the boids/predator simulation. Tucked into the top-right whitespace of
// the page so it never competes with the centered content. The dot/edge
// style matches the page's gray-ladder palette; the output column borrows
// the predator's muted red so the viz is visually tied to the body it
// controls.
//
// Reads activations from window.__predatorModel, which is guaranteed to
// be loaded by the time the simulation starts rendering (see boids.js).

(function () {
    'use strict';

    var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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

        // Don't crowd small screens.
        if (cw < 720 || ch < 480) return;

        // Top-right corner, fixed footprint that scales gently on narrow viewports.
        var W = Math.min(160, cw * 0.14);
        var H = Math.min(150, ch * 0.22);
        var marginX = 36;
        var marginY = 36;
        var x0 = cw - marginX - W;
        var y0 = marginY;

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
                ctx.arc(pos[k2].x, pos[k2].y, 1.6, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // --- Tiny caption -----------------------------------------------
        ctx.fillStyle = 'rgba(85, 85, 85, 0.42)';
        ctx.font = '9px "Source Code Pro", ui-monospace, monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(layerSizes.join(' · '), x0 + W, y0 + H + 8);
        ctx.fillStyle = 'rgba(85, 85, 85, 0.28)';
        ctx.fillText('predator policy', x0 + W, y0 + H + 22);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
    }

    window.renderActivationViz = renderActivationViz;
})();
