// Library-free forward pass for the radial (RadialPool) predator PATROL policy.
// One source of truth: loaded as a <script> by the live page AND require()d by
// the Node eval/parity scripts, so page runtime and offline eval agree exactly.
//
// This is the 10,352-param minimal net (patrol cos_med 0.9878, == the 129k
// transformer ceiling at 1/12.5 the size). It maps the RAW boid set directly to
// the predator's patrol steering force — it does internally what the old
// computeEvolvedTarget (density-weighted cluster select) + the 35-feat seek-net
// did in two stages. Chase (a boid within POLICY_R) stays analytic seek-nearest
// in predator.js; this net is only invoked in the patrol regime it was trained on.
//
// Architecture (mirrors dev/distill_e2e/rel_net.py RadialPool + rho, mode='radial'):
//   per boid i, feat = [dx/HALF, dy/HALF, (bvx-pvx)/BMAX, (bvy-pvy)/BMAX, dist/HALF]
//   xn          = (feat - fmean) / fstd                       (standardized)
//   counts[k]   = Σ_j sigmoid((R2k - |p_i-p_j|^2) * tk/(R2k+1))  soft radial counts
//   nbpos/nbvel = primary-radius(g0)-gated neighbourhood centroid / mean velocity
//   score(i)    = MLP([counts(K), xn(5), nboff/HALF(2), nbvel/BMAX(2)]) -> scalar
//   w           = softmax(score / tau)                        sharp E3D-style select
//   pooled(8)   = Σ_i w_i [pos/HALF, vel/BMAX, nbpos/HALF, nbvel/BMAX]
//   out(2)      = rho([pooled(8), pvel(2)])
//   force       = clipMag(clipMag(out, MAX_SPEED) - predVel, MAX_FORCE)
//
// Weight file (predator_radial_weights.json): logR/logt/log_tau, score & rho MLPs
// (W stored inDim-major: W[i*outDim+j]), fmean/fstd, and the sim constants.

(function (root, factory) {
    'use strict';
    var mod = factory();
    if (typeof module !== 'undefined' && module.exports) module.exports = mod;
    if (typeof globalThis !== 'undefined') globalThis.PredatorRadial = mod;
    if (typeof window !== 'undefined' && window !== globalThis) window.PredatorRadial = mod;
})(this, function () {
    'use strict';

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    // Dense layer: cur (inDim) -> z (outDim), W stored inDim-major (W[i*outDim+j]).
    function denseForward(layer, cur) {
        var W = layer.W, b = layer.b, inDim = layer.inDim, outDim = layer.outDim;
        var z = layer.lastZ, a = layer.lastA;
        for (var j = 0; j < outDim; j++) {
            var s = b[j];
            for (var i = 0; i < inDim; i++) s += W[i * outDim + j] * cur[i];
            z[j] = s;
            a[j] = layer.relu ? (s > 0 ? s : 0) : s;
        }
        return a;
    }

    function runMLP(layers, input) {
        var cur = input;
        for (var l = 0; l < layers.length; l++) cur = denseForward(layers[l], cur);
        return cur;
    }

    function parseLayers(arr) {
        return arr.map(function (L) {
            return {
                inDim: L.inDim, outDim: L.outDim,
                relu: L.activation === 'relu',
                W: new Float32Array(L.W), b: new Float32Array(L.b),
                lastZ: new Float32Array(L.outDim), lastA: new Float32Array(L.outDim),
            };
        });
    }

    function clipMag(x, y, m) {
        var n = Math.sqrt(x * x + y * y) + 1e-12;
        var s = m / n; if (s > 1) s = 1;
        return [x * s, y * s];
    }

    function loadModel(json) {
        var K = json.K;
        var model = {
            arch: 'radial',
            params: json.params,
            K: K,
            HALF: json.HALF, BOID_MAX: json.BOID_MAX,
            MAX_SPEED: json.PRED_MAX_SPEED, MAX_FORCE: json.PRED_MAX_FORCE,
            POLICY_R: json.POLICY_R,
            fmean: new Float32Array(json.fmean),
            fstd: new Float32Array(json.fstd),
            R2: new Float32Array(K),
            coef: new Float32Array(K),         // tk / (R2k + 1)
            tau: Math.min(1e2, Math.max(1e-2, Math.exp(json.log_tau))),
            score: parseLayers(json.score),
            rho: parseLayers(json.rho),
        };
        for (var k = 0; k < K; k++) {
            var R = Math.exp(json.logR[k]);
            var t = Math.min(1e3, Math.max(1e-3, Math.exp(json.logt[k])));
            model.R2[k] = R * R;
            model.coef[k] = t / (R * R + 1);
        }

        // activation_viz.js contract: visualize the rho decision head as a
        // feed-forward stack (10 -> 128 -> 64 -> 2). featureDim = rho input dim.
        model.featureDim = json.rho[0].inDim;
        model.layers = model.rho;
        model.lastNormalizedInput = new Float32Array(model.featureDim);

        var scoreIn = new Float32Array(K + 5 + 4);
        var rhoIn = new Float32Array(model.featureDim);

        // predPos/predVel: {x,y}; boids: [{position:{x,y}, velocity:{x,y}}]
        model.forward = function (predPos, predVel, boids) {
            var n = boids.length;
            if (n === 0) return [0, 0];
            var HALF = model.HALF, BMAX = model.BOID_MAX;
            var px = predPos.x, py = predPos.y, pvx = predVel.x, pvy = predVel.y;
            var fmean = model.fmean, fstd = model.fstd, R2 = model.R2, coef = model.coef;

            var posX = new Float32Array(n), posY = new Float32Array(n);
            var velX = new Float32Array(n), velY = new Float32Array(n);
            var xn = new Float32Array(n * 5);
            for (var i = 0; i < n; i++) {
                var b = boids[i];
                var dx = b.position.x - px, dy = b.position.y - py;
                posX[i] = dx; posY[i] = dy;
                velX[i] = b.velocity.x; velY[i] = b.velocity.y;
                var dist = Math.sqrt(dx * dx + dy * dy);
                var f0 = dx / HALF, f1 = dy / HALF;
                var f2 = (b.velocity.x - pvx) / BMAX, f3 = (b.velocity.y - pvy) / BMAX;
                var f4 = dist / HALF;
                var o = i * 5;
                xn[o]     = (f0 - fmean[0]) / fstd[0];
                xn[o + 1] = (f1 - fmean[1]) / fstd[1];
                xn[o + 2] = (f2 - fmean[2]) / fstd[2];
                xn[o + 3] = (f3 - fmean[3]) / fstd[3];
                xn[o + 4] = (f4 - fmean[4]) / fstd[4];
            }

            var score = new Float32Array(n);
            var nbpX = new Float32Array(n), nbpY = new Float32Array(n);
            var nbvX = new Float32Array(n), nbvY = new Float32Array(n);
            for (i = 0; i < n; i++) {
                for (var k = 0; k < K; k++) scoreIn[k] = 0;          // counts
                var g0sum = 0, cpx = 0, cpy = 0, cvx = 0, cvy = 0;
                var pix = posX[i], piy = posY[i];
                for (var j = 0; j < n; j++) {
                    var ex = pix - posX[j], ey = piy - posY[j];
                    var d2 = ex * ex + ey * ey;
                    for (k = 0; k < K; k++) scoreIn[k] += sigmoid((R2[k] - d2) * coef[k]);
                    var g0 = sigmoid((R2[0] - d2) * coef[0]);
                    g0sum += g0;
                    cpx += g0 * posX[j]; cpy += g0 * posY[j];
                    cvx += g0 * velX[j]; cvy += g0 * velY[j];
                }
                var gs = g0sum < 1e-6 ? 1e-6 : g0sum;
                var npx = cpx / gs, npy = cpy / gs, nvx = cvx / gs, nvy = cvy / gs;
                nbpX[i] = npx; nbpY[i] = npy; nbvX[i] = nvx; nbvY[i] = nvy;
                var oo = i * 5;
                scoreIn[K]     = xn[oo];     scoreIn[K + 1] = xn[oo + 1];
                scoreIn[K + 2] = xn[oo + 2]; scoreIn[K + 3] = xn[oo + 3];
                scoreIn[K + 4] = xn[oo + 4];
                scoreIn[K + 5] = (npx - pix) / HALF;
                scoreIn[K + 6] = (npy - piy) / HALF;
                scoreIn[K + 7] = nvx / BMAX;
                scoreIn[K + 8] = nvy / BMAX;
                score[i] = runMLP(model.score, scoreIn)[0];
            }

            // sharp masked softmax selection over all (alive) boids
            var tau = model.tau, smax = -Infinity;
            for (i = 0; i < n; i++) { score[i] /= tau; if (score[i] > smax) smax = score[i]; }
            var wsum = 0;
            for (i = 0; i < n; i++) { score[i] = Math.exp(score[i] - smax); wsum += score[i]; }
            var inv = 1 / wsum;

            // pooled(8) = Σ w_i [pos/HALF, vel/BMAX, nbpos/HALF, nbvel/BMAX]
            var p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;
            for (i = 0; i < n; i++) {
                var w = score[i] * inv;
                p0 += w * posX[i] / HALF; p1 += w * posY[i] / HALF;
                p2 += w * velX[i] / BMAX; p3 += w * velY[i] / BMAX;
                p4 += w * nbpX[i] / HALF; p5 += w * nbpY[i] / HALF;
                p6 += w * nbvX[i] / BMAX; p7 += w * nbvY[i] / BMAX;
            }
            rhoIn[0] = p0; rhoIn[1] = p1; rhoIn[2] = p2; rhoIn[3] = p3;
            rhoIn[4] = p4; rhoIn[5] = p5; rhoIn[6] = p6; rhoIn[7] = p7;
            rhoIn[8] = pvx / model.MAX_SPEED; rhoIn[9] = pvy / model.MAX_SPEED;
            model.lastNormalizedInput.set(rhoIn);

            var out = runMLP(model.rho, rhoIn);
            var desired = clipMag(out[0], out[1], model.MAX_SPEED);
            return clipMag(desired[0] - pvx, desired[1] - pvy, model.MAX_FORCE);
        };

        return model;
    }

    return { loadModel: loadModel };
});
