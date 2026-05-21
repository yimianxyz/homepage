// Single source of truth for predator feature encoding. Used by the live
// page (loaded as a <script>) and by dev/ Node scripts (via require).
// The feature vector layout is frozen; changing it requires retraining.

(function (root, factory) {
    'use strict';
    var mod = factory();
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = mod;
    }
    var globals = [];
    if (typeof globalThis !== 'undefined') globals.push(globalThis);
    if (typeof window !== 'undefined' && window !== globalThis) globals.push(window);
    for (var i = 0; i < globals.length; i++) {
        var g = globals[i];
        g.POLICY_K = mod.POLICY_K;
        g.POLICY_R = mod.POLICY_R;
        g.POLICY_PAD = mod.POLICY_PAD;
        g.FEATURE_DIM = mod.FEATURE_DIM;
        g.buildPredatorFeatures = mod.buildPredatorFeatures;
    }
})(this, function () {
    'use strict';

    var POLICY_R = 80;
    var POLICY_K = 4;
    var POLICY_PAD = 2000;
    var PREDATOR_MAX_SPEED = 2.5;   // matches js/predator.js
    var PREDATOR_MAX_FORCE = 0.05;
    // Feature layout (append-only; old models read only the first N via
    // model.featureDim, so adding features here is safe for shipped weights):
    //   [0..1]   velocity (vx, vy)
    //   [2..3]   autoTarget offset (dxA, dyA)
    //   [4..5]   unit direction to autoTarget (ux_A, uy_A)
    //   [6]      dA = ||autoTarget offset||
    //   [7..14]  K=4 nearest boid offsets (dx1, dy1, ..., dx4, dy4)
    //   [15..22] unit directions to nearest K boids (ux1, uy1, ..., ux4, uy4)
    //   [23..26] d1..d4 = ||nearest boid offset||
    //   [27..28] reserved (legacy padding; always 0 for v1 compat)
    //   --- v2 additions (new models use these; v1 models ignore via featureDim) ---
    //   [29..30] seek_boid_xy = exact rule steering for the "hunt" branch
    //                           = fastLimit(fastSetMag((dx1,dy1), MAX_SPEED) - vel, MAX_FORCE)
    //   [31..32] seek_auto_xy = exact rule steering for the "patrol" branch
    //   [33]     inRange_smooth = clamp((R - d1)/10 + 0.5, 0, 1)
    //                             0 well outside R, 1 well inside, 10px transition.
    //   --- v3 addition (binary indicator so the network can fit the rule's
    //   --- exact branch discontinuity, not just smooth-mix near R-edge) ---
    //   [34]     inRange_binary = (d1 < R && dx1 != PAD) ? 1 : 0
    //   --- v4 additions: per-K-nearest boid velocities + relative velocity
    //   --- so the network has the raw signal needed to anticipate motion ---
    //   [35..36] vx1, vy1  = velocity of nearest boid (raw, not relative)
    //   [37..38] vx2, vy2  = velocity of 2nd nearest
    //   [39..40] vx3, vy3  = velocity of 3rd nearest
    //   [41..42] vx4, vy4  = velocity of 4th nearest
    //   --- v4 padding for absent boids: 0 (zero velocity is a fine sentinel
    //   --- because the position is already PAD-coded; the NN can read the
    //   --- inRange flag and the d_k distance to know if slot k is real) ---
    //   --- v5 addition: precomputed velocity-aware hunt steering. The NN
    //   --- distilled from the velocity-aware rule needs this slot as the
    //   --- "ready-made answer" the same way seek_boid_xy (29,30) gave it
    //   --- the answer for the original rule. Without it, H=4 ran out of
    //   --- capacity trying to reconstruct rule_v2's steering from raw inputs.
    //   [43..44] seek_boid_v2_xy = fastLimit(fastSetMag((dx1 + α·(bvx1-vx),
    //                                                    dy1 + α·(bvy1-vy)),
    //                                                   MAX_SPEED) - vel,
    //                                       MAX_FORCE)
    //            with α = PREDICT_ALPHA below.
    // Total: 45 features.
    var FEATURE_DIM = 45;
    var PREDICT_ALPHA = 8;   // matches the rule_v2 α used to generate v5 dataset

    function fastMag(x, y) {
        var ax = Math.abs(x), ay = Math.abs(y);
        return Math.max(ax, ay) * 0.96 + Math.min(ax, ay) * 0.398;
    }

    // Compute the exact rule steering for a single seek target. Matches
    // policy_spec.rulePolicy / Vector.iFastLimit / Vector.iFastSetMagnitude.
    function seekStep(dx, dy, vx, vy, out, outIdx) {
        var m = fastMag(dx, dy);
        var dx0, dy0;
        if (m === 0) { dx0 = 0; dy0 = 0; }
        else { dx0 = dx * PREDATOR_MAX_SPEED / m; dy0 = dy * PREDATOR_MAX_SPEED / m; }
        var sx = dx0 - vx, sy = dy0 - vy;
        var sm = fastMag(sx, sy);
        if (sm > PREDATOR_MAX_FORCE) {
            var f = PREDATOR_MAX_FORCE / sm;
            out[outIdx] = sx * f; out[outIdx + 1] = sy * f;
        } else {
            out[outIdx] = sx; out[outIdx + 1] = sy;
        }
    }

    function buildPredatorFeatures(predPos, predVel, boids, autoTarget) {
        var out = new Float32Array(FEATURE_DIM);
        var vx = predVel.x, vy = predVel.y;
        out[0] = vx;
        out[1] = vy;
        var dxA = autoTarget.x - predPos.x;
        var dyA = autoTarget.y - predPos.y;
        out[2] = dxA;
        out[3] = dyA;
        var dA = Math.sqrt(dxA * dxA + dyA * dyA);
        if (dA > 1e-9) { out[4] = dxA / dA; out[5] = dyA / dA; }
        else            { out[4] = 0;       out[5] = 0;       }
        out[6] = dA;

        var n = boids.length;
        // pair = [d^2, dx, dy, vx, vy] so we can retrieve the K-nearest boids'
        // velocities (slots 35..42) without a second pass.
        var pairs = new Array(n);
        for (var i = 0; i < n; i++) {
            var dx = boids[i].position.x - predPos.x;
            var dy = boids[i].position.y - predPos.y;
            var bv = boids[i].velocity;
            var bvx = bv ? bv.x : 0;
            var bvy = bv ? bv.y : 0;
            pairs[i] = [dx * dx + dy * dy, dx, dy, bvx, bvy];
        }
        pairs.sort(function (a, b) { return a[0] - b[0]; });
        var dx1 = POLICY_PAD, dy1 = POLICY_PAD, d1 = POLICY_PAD;
        for (var k = 0; k < POLICY_K; k++) {
            if (k < n) {
                var dx2 = pairs[k][1], dy2 = pairs[k][2];
                var d = Math.sqrt(pairs[k][0]);
                out[7 + 2 * k] = dx2;
                out[8 + 2 * k] = dy2;
                if (d > 1e-9) {
                    out[15 + 2 * k] = dx2 / d;
                    out[16 + 2 * k] = dy2 / d;
                } else {
                    out[15 + 2 * k] = 0; out[16 + 2 * k] = 0;
                }
                out[23 + k] = d;
                // v4: per-boid velocities (slots 35,36 / 37,38 / 39,40 / 41,42)
                out[35 + 2 * k] = pairs[k][3];
                out[36 + 2 * k] = pairs[k][4];
                if (k === 0) { dx1 = dx2; dy1 = dy2; d1 = d; }
            } else {
                out[7 + 2 * k] = POLICY_PAD;
                out[8 + 2 * k] = POLICY_PAD;
                out[15 + 2 * k] = 1;       // sentinel unit dir
                out[16 + 2 * k] = 0;
                out[23 + k] = POLICY_PAD;
                // velocity defaults to 0 (Float32Array initial value)
            }
        }
        // Legacy padding slots [27], [28] left as 0 (Float32Array default).
        // v2 additions: precomputed seek vectors + smooth in-range indicator.
        seekStep(dx1, dy1, vx, vy, out, 29);
        seekStep(dxA, dyA, vx, vy, out, 31);
        var t = (POLICY_R - d1) / 10 + 0.5;
        out[33] = t < 0 ? 0 : (t > 1 ? 1 : t);
        // v3 addition: binary in-range indicator. Matches the rule's branch
        // exactly (d1<R AND a real boid in slot 0, not PAD). Lets the network
        // model the rule's discontinuity at d1=R precisely, instead of having
        // to smear it across a finite transition region.
        out[34] = (d1 < POLICY_R && dx1 !== POLICY_PAD) ? 1 : 0;
        // v5 addition: precomputed velocity-aware hunt steering. Uses the
        // nearest boid's velocity (slots 35,36 above) to compute rule_v2's
        // exact hunt-branch output. The distilled NN can blend this slot
        // with seek_auto_xy via the inRange_binary feature just like the
        // v3 model did.
        if (d1 < POLICY_R && dx1 !== POLICY_PAD) {
            var bvx1 = out[35], bvy1 = out[36];
            seekStep(dx1 + PREDICT_ALPHA * (bvx1 - vx),
                     dy1 + PREDICT_ALPHA * (bvy1 - vy),
                     vx, vy, out, 43);
        } else {
            out[43] = 0; out[44] = 0;
        }

        return out;
    }

    return {
        POLICY_R: POLICY_R,
        POLICY_K: POLICY_K,
        POLICY_PAD: POLICY_PAD,
        FEATURE_DIM: FEATURE_DIM,
        buildPredatorFeatures: buildPredatorFeatures,
    };
});
