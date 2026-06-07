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
    var infoStart = 0;   // time the "?" badge first appeared, for its breathing hint

    function clamp(min, v, max) {
        return v < min ? min : v > max ? max : v;
    }

    // Wide layers (the rho head is 128- and 64-neuron) would pack into a solid
    // bar at this widget size, so each column draws at most MAX_DOTS neurons,
    // evenly subsampled across the layer. Purely cosmetic: the subset's
    // activations are still their true values, and edges are drawn only between
    // displayed neurons.
    var MAX_DOTS = 22;
    function displayIndices(n) {
        if (n <= MAX_DOTS) {
            var all = new Array(n);
            for (var i = 0; i < n; i++) all[i] = i;
            return all;
        }
        var idx = new Array(MAX_DOTS);
        for (var m = 0; m < MAX_DOTS; m++) {
            idx[m] = Math.round(m * (n - 1) / (MAX_DOTS - 1));
        }
        return idx;
    }

    // Per-layer EMA of the max signal magnitude, for stable normalization.
    var emaMax = [];
    // Per-neuron EMA of the raw activation. The display shows the deviation
    // from this (how much each value is *changing*), not the magnitude, so
    // steady neurons fade and shifting ones flare — see the change block below.
    var actEma = [];
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

        // The "predator" header sits INSIDE the widget at top-right, in
        // the empty space above the 2-dot output column — that space is
        // there at every viewport size (the hidden + output columns are
        // vertically centered, the input column is dense). Below the
        // widget we reserve one line for the live "N left · M eaten"
        // numbers (skipped only on landscape, where the widget anchors
        // to the top and there's no room below it).
        var captionReserve = landscapePhone ? 0 : 12;

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

        // --- Show CHANGE, not magnitude ----------------------------------
        // Drive brightness by how much each activation is *shifting*, not its
        // raw value: keep a per-neuron EMA and display the deviation from it.
        // A neuron (or input) that holds steady fades to dark; one that moves
        // when the predator commits a new decision flares and eases back.
        // The policy only re-plans every D frames (predator_cheap.js), so the
        // raw activations step in place — but the EMA keeps chasing the held
        // value each render frame, so the deviation decays smoothly between
        // steps and the diagram breathes continuously instead of freezing on
        // an always-on path. One EMA per neuron; nothing downstream changes.
        var change = [];
        for (var ci = 0; ci < activations.length; ci++) {
            var av = activations[ci], em = actEma[ci];
            if (!em || em.length !== av.length) {
                em = actEma[ci] = new Float64Array(av.length);
                for (var s0 = 0; s0 < av.length; s0++) em[s0] = av[s0];   // start settled — no flash on load
            }
            var cv = new Float64Array(av.length);
            for (var s1 = 0; s1 < av.length; s1++) {
                cv[s1] = av[s1] - em[s1];
                em[s1] += 0.09 * cv[s1];                                  // ~16-frame decay ≈ the plan cadence
            }
            change.push(cv);
        }
        // Reduced-motion users get the calm raw-magnitude view instead.
        var dotSrc = prefersReducedMotion ? activations : change;

        // Per-column displayed-neuron indices + their screen positions, vertically
        // centered within H. posMap[l] is sparse (length = layer size); entries
        // exist only for displayed neurons, so edges to undrawn neurons are skipped.
        var disp = [];
        var posMap = [];
        var colSpacing = W / (nLayers - 1);
        for (var l = 0; l < nLayers; l++) {
            var idxs = displayIndices(layerSizes[l]);
            var colX = x0 + l * colSpacing;
            var spacing = Math.min(4.4, (H - 6) / Math.max(1, idxs.length - 1));
            var totalH = (idxs.length - 1) * spacing;
            var startY = y0 + (H - totalH) / 2;
            var pm = new Array(layerSizes[l]);
            for (var k = 0; k < idxs.length; k++) {
                pm[idxs[k]] = { x: colX, y: startY + k * spacing };
            }
            disp.push(idxs);
            posMap.push(pm);
        }

        // --- Edges -------------------------------------------------------
        // Draw only the top |signal| edges per layer to keep it visually
        // sparse; signal = |weight * prev_activation|. This is a "wired
        // synapse firing" effect that's responsive to current state. Only
        // edges between two displayed neurons are eligible.
        if (!prefersReducedMotion) {
            for (var L_idx = 0; L_idx < model.layers.length; L_idx++) {
                var Lr = model.layers[L_idx];
                var prev = change[L_idx];                 // edges fire on changing inputs, not steady ones
                var posPrev = posMap[L_idx];
                var posNext = posMap[L_idx + 1];
                var dispPrev = disp[L_idx];
                var dispNext = disp[L_idx + 1];
                var edges = [];
                for (var dj = 0; dj < dispNext.length; dj++) {
                    var j = dispNext[dj];
                    for (var di = 0; di < dispPrev.length; di++) {
                        var iI = dispPrev[di];
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
            var pos = posMap[li];
            var dispCol = disp[li];
            var act = dotSrc[li];
            var isOutput = (li === nLayers - 1);

            // Per-layer EMA normalization (over displayed neurons) so the
            // visual is stable.
            var localMax = 0;
            for (var kk = 0; kk < dispCol.length; kk++) {
                var a2 = Math.abs(act[dispCol[kk]]);
                if (a2 > localMax) localMax = a2;
            }
            var norm = emaUpdate(li, localMax);

            for (var k2 = 0; k2 < dispCol.length; k2++) {
                var ni = dispCol[k2];
                var v = Math.abs(act[ni]) / norm;
                if (v > 1) v = 1;
                var baseAlpha = dotBaseAlpha;
                var alpha = baseAlpha + (1 - baseAlpha) * v * 0.78;
                var rgb = isOutput ? '140, 60, 60' : '85, 85, 85';
                ctx.fillStyle = 'rgba(' + rgb + ', ' + alpha.toFixed(3) + ')';
                ctx.beginPath();
                ctx.arc(pos[ni].x, pos[ni].y, dotR, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // --- Labels ------------------------------------------------------
        ctx.font = '9px "Source Code Pro", ui-monospace, monospace';
        ctx.textBaseline = 'middle';

        // Header "predator's brain" plus a small "?" info badge to its right,
        // the pair centered above the widget on the graph's axis (x0 + W/2).
        // The columns span x0 (inputs) to x0+W (output), so the box midpoint
        // is the graph's visual center; a title centered over its figure (with
        // the live caption centered below) is the standard, balanced layout.
        // Above the widget is clear canvas at every viewport (y0 >= marginY >=
        // 10px, top dots start at >= y0). Possessive matches the page's voice
        // ("my master's degree from Cornell"). Tapping the badge opens a DOM
        // panel (wired in boids.js / index.html) that explains the boid +
        // predator rules and the tap-to-hatch interaction; its hit-region is
        // published on window.__vizInfo in canvas units, which equal CSS px
        // here (canvas_init.js sizes the canvas to the visual viewport), so a
        // pointer's clientX/clientY compares against it directly.
        var title = "predator's brain";
        ctx.textAlign = 'left';
        var titleW = ctx.measureText(title).width;
        var iconR = 6, iconGap = 6;
        var groupLeft = x0 + W / 2 - (titleW + iconGap + iconR * 2) / 2;
        // Title, badge ring, and "?" all sit on one centerline (textBaseline
        // 'middle'), so they line up exactly — no per-glyph nudge needed.
        var headerY = y0 - 8;
        ctx.fillStyle = 'rgba(85, 85, 85, 0.28)';
        ctx.fillText(title, groupLeft, headerY);

        // "?" badge — slightly more present than the title (it's interactive),
        // brighter still on hover (window.__vizInfoHover, set on mousemove).
        var iconX = groupLeft + titleW + iconGap + iconR;
        // "i'm interactive" hint: after an ~8s freeze (let the reader take in the
        // page first — the viz sits off the reading path, where the boids already
        // move) the badge gently, continuously breathes between its resting and
        // hover brightness. Honors prefers-reduced-motion. glow: 0 = rest, 1 = hover.
        var hovered = !!window.__vizInfoHover, glow = hovered ? 1 : 0;
        if (!hovered && !prefersReducedMotion) {
            if (!infoStart) infoStart = Date.now();
            var e = (Date.now() - infoStart) / 1000;                         // seconds since first shown
            if (e > 8) glow = (1 - Math.cos((e - 8) * (Math.PI / 2.5))) / 2; // freeze 8s, then a slow, smooth breath (~5s/cycle)
        }
        ctx.strokeStyle = 'rgba(85, 85, 85, ' + (0.30 + glow * 0.25) + ')';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(iconX, headerY, iconR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = 'rgba(85, 85, 85, ' + (0.46 + glow * 0.26) + ')';
        ctx.textAlign = 'center';
        ctx.fillText('?', iconX, headerY);

        // The whole widget is one hit-target, not just the badge: hovering or
        // tapping anywhere over it behaves exactly like the "?" (boids.js drives
        // __vizInfoHover and the panel toggle off this rect), so the icon and the
        // area stay one consistent affordance. The box wraps the header, the
        // graph (x0..x0+W), the badge, and the caption, with a little slack.
        var pad = 4;
        var boxL = Math.min(x0, groupLeft) - pad;
        var boxR = Math.max(x0 + W, iconX + iconR) + pad;
        var boxT = headerY - iconR - pad;
        var boxB = (landscapePhone ? y0 + H : y0 + H + 17) + pad;
        window.__vizInfo = { x: boxL, y: boxT, w: boxR - boxL, h: boxB - boxT };

        // Live numbers caption below the widget, centered on the same axis as
        // the header. Always shown except on landscape phones (where the
        // widget anchors to the top of the viewport, no room below it).
        ctx.textBaseline = 'top';
        if (!landscapePhone) {
            // On hover the caption becomes the affordance hint; otherwise it's
            // the live tally. Hovering signals intent to act, not to read the
            // counter — so the line that's already there does double duty as a
            // frameless, in-place tooltip (no floating box to clash with the
            // boxless aesthetic, nothing new to clip near the viewport edge).
            var hint = !!window.__vizInfoHover;
            ctx.fillStyle = 'rgba(85, 85, 85, ' + (hint ? 0.6 : 0.42) + ')';
            var caption = hint ? "what's this?"
                : (sim.boids ? sim.boids.length : 0) + ' left · ' + (sim.boidsEaten || 0) + ' eaten';
            ctx.fillText(caption, x0 + W / 2, y0 + H + 8);
        }

        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
    }

    window.renderActivationViz = renderActivationViz;
})();
