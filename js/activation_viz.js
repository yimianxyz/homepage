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

    // Input features cluster into contiguous "always-bright" and "always-dim"
    // blocks (unit vectors at 15-22, distances at 23-26, padding zeros at
    // 27-28, etc.), which makes the input column look stripey instead of a
    // live firing constellation. We scramble the *visual y-position* of each
    // input feature with a stride permutation; activation values and edge
    // endpoints are untouched, so the rendering stays correct. Stride 8 on
    // a length-35 column gives a permutation (gcd(8,35)=1), and 8's inverse
    // mod 35 is 22 ≈ 35/φ, so adjacent original features land ~22 rows apart
    // visually — the golden-ratio quasi-random spread.
    var inputDisplayInverse = null;
    function ensureInputPermutation(n) {
        if (inputDisplayInverse && inputDisplayInverse.length === n) return;
        inputDisplayInverse = new Array(n);
        var stride = 8;
        for (var visualPos = 0; visualPos < n; visualPos++) {
            var featureIdx = (visualPos * stride) % n;
            inputDisplayInverse[featureIdx] = visualPos;
        }
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

        // Three regimes:
        //   landscapePhone — short + wider than tall (≤480 tall). Content
        //     overflows the viewport, so bottom-right would land on top of
        //     text. Park a tiny version at TOP-right, above the first
        //     paragraph, in the strip next to the H1.
        //   compact (portrait phone) — bottom-right, weight bumped (bigger
        //     dots, bolder edges, higher alpha) so the viz reads on retina
        //     against the boid backdrop.
        //   default (desktop / tablet) — original bottom-right ambient look.
        var landscapePhone = ch < 480 && cw > ch;
        var compact = !landscapePhone && (cw < 480 || ch < 500);

        var W, H, marginX, marginY;
        if (landscapePhone) {
            W = clamp(86, cw * 0.16, 124);
            H = clamp(38, ch * 0.10, 52);
            marginX = clamp(10, cw * 0.018, 18);
            marginY = clamp(10, ch * 0.030, 18);
        } else if (compact) {
            W = clamp(104, cw * 0.18, 134);
            H = clamp(80,  ch * 0.13, 100);
            marginX = clamp(12, cw * 0.024, 22);
            marginY = clamp(12, ch * 0.022, 20);
        } else {
            W = clamp(78, cw * 0.15, 158);
            H = clamp(54, ch * 0.13, 128);
            marginX = clamp(14, cw * 0.022, 32);
            marginY = clamp(14, ch * 0.022, 32);
        }
        var bumped = landscapePhone || compact;
        var dotR         = bumped ? 2.0 : 1.6;
        var edgeWidth    = bumped ? 0.9 : 0.6;
        var edgeAlphaCap = bumped ? 0.58 : 0.42;
        var dotBaseAlpha = bumped ? 0.18 : 0.10;

        // Caption tier: hidden on landscape phones (no vertical room above
        // first paragraph), single line on narrow / compact, two on desktop.
        var captionLines;
        if (landscapePhone || cw < 380) captionLines = 0;
        else if (cw < 520 || compact) captionLines = 1;
        else captionLines = 2;
        var captionReserve = captionLines === 2 ? 24 : captionLines === 1 ? 12 : 0;

        var x0 = cw - marginX - W;
        // Landscape anchors to the TOP of the viewport (next to H1); other
        // regimes anchor to the bottom-right.
        var y0 = landscapePhone ? marginY : ch - marginY - H - captionReserve;

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
            // For the input column only, place feature k at the permuted
            // y-row; hidden/output columns keep natural order.
            if (l === 0 && n > 4) {
                ensureInputPermutation(n);
                for (var k = 0; k < n; k++) {
                    col[k] = { x: colX, y: startY + inputDisplayInverse[k] * spacing };
                }
            } else {
                for (var k = 0; k < n; k++) col[k] = { x: colX, y: startY + k * spacing };
            }
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
                    var a = Math.min(edgeAlphaCap, (ed[2] / sigCap) * edgeAlphaCap);
                    ctx.strokeStyle = 'rgba(85, 85, 85, ' + a.toFixed(3) + ')';
                    ctx.lineWidth = edgeWidth;
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
                var baseAlpha = dotBaseAlpha;
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
