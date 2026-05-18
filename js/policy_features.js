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
    // Feature layout (frozen; changing requires regen + retrain):
    //   [0..1]   velocity (vx, vy)
    //   [2..3]   autoTarget offset (dxA, dyA)
    //   [4..5]   unit direction to autoTarget (ux_A, uy_A)
    //   [6]      dA = ||autoTarget offset||
    //   [7..14]  K=4 nearest boid offsets (dx1, dy1, ..., dx4, dy4)
    //   [15..22] unit directions to nearest K boids (ux1, uy1, ..., ux4, uy4)
    //   [23..26] d1..d4 = ||nearest boid offset||
    // Total: 27 features.
    //
    // Unit directions and magnitudes are pure geometry, not policy. They
    // short-circuit the 1/r normalization that small ReLU MLPs approximate
    // poorly. With these features the seek() transform becomes a near-linear
    // combination, fittable by a very small network.
    var FEATURE_DIM = 4 + 2 + 2 + 1 + 2 * POLICY_K + 2 * POLICY_K + POLICY_K;

    function buildPredatorFeatures(predPos, predVel, boids, autoTarget) {
        var out = new Float32Array(FEATURE_DIM);
        out[0] = predVel.x;
        out[1] = predVel.y;
        var dxA = autoTarget.x - predPos.x;
        var dyA = autoTarget.y - predPos.y;
        out[2] = dxA;
        out[3] = dyA;
        var dA = Math.sqrt(dxA * dxA + dyA * dyA);
        if (dA > 1e-9) { out[4] = dxA / dA; out[5] = dyA / dA; }
        else            { out[4] = 0;       out[5] = 0;       }
        out[6] = dA;

        var n = boids.length;
        var pairs = new Array(n);
        for (var i = 0; i < n; i++) {
            var dx = boids[i].position.x - predPos.x;
            var dy = boids[i].position.y - predPos.y;
            pairs[i] = [dx * dx + dy * dy, dx, dy];
        }
        pairs.sort(function (a, b) { return a[0] - b[0]; });
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
            } else {
                out[7 + 2 * k] = POLICY_PAD;
                out[8 + 2 * k] = POLICY_PAD;
                out[15 + 2 * k] = 1;       // sentinel unit dir
                out[16 + 2 * k] = 0;
                out[23 + k] = POLICY_PAD;
            }
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
