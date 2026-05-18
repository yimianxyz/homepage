// Tiny library-free NN forward pass for the predator policy.
// Same file is used by the live page (loaded as a <script>) and by the
// Node-side eval/diagnose/verify scripts (via require). One source of
// truth, so eval results match the page's runtime exactly.
//
// Weight file format (predator_weights.json):
//   {
//     "version": 1,
//     "K": 4,                 // matches POLICY_K in dev/policy_spec.js
//     "featureDim": 12,       // matches FEATURE_DIM
//     "inputScale": 1/100,    // optional: input is multiplied by this before fwd
//     "outputScale": 0.05,    // optional: output is multiplied by this (so the
//                             //           net can emit O(1) values then we
//                             //           rescale to the steering magnitude)
//     "layers": [
//       { "inDim": 12, "outDim": 8, "activation": "relu",
//         "W": [96 floats, row-major shape (inDim, outDim)],
//         "b": [8 floats] },
//       { "inDim": 8,  "outDim": 2, "activation": "linear",
//         "W": [16 floats], "b": [2 floats] }
//     ]
//   }

(function (root, factory) {
    'use strict';
    var mod = factory();
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = mod;
    }
    if (typeof globalThis !== 'undefined') globalThis.PredatorNN = mod;
    if (typeof window !== 'undefined' && window !== globalThis) window.PredatorNN = mod;
})(this, function () {
    'use strict';

    function applyActivation(z, kind, out) {
        var n = z.length;
        if (kind === 'relu') {
            for (var i = 0; i < n; i++) out[i] = z[i] > 0 ? z[i] : 0;
        } else if (kind === 'tanh') {
            for (var i = 0; i < n; i++) out[i] = Math.tanh(z[i]);
        } else if (kind === 'linear' || !kind) {
            for (var i = 0; i < n; i++) out[i] = z[i];
        } else if (kind === 'sigmoid') {
            for (var i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-z[i]));
        } else {
            throw new Error('Unknown activation: ' + kind);
        }
    }

    // Build a Model object from a parsed weights JSON. The model exposes:
    //   model.forward(features) -> Float32Array of length 2
    //   model.layers[i].lastZ, model.layers[i].lastA  (cached for future viz)
    //   model.lastInput, model.lastOutput
    //
    // Preprocessing applied to features in this order (each step optional):
    //   1. If json.inputMean / json.inputStd are present, standardize:
    //        x' = (x - mean[i]) / std[i]
    //   2. Else if json.inputScale is present, scale uniformly: x' = x * inputScale.
    //   3. Output is multiplied by json.outputScale.
    function loadModel(json) {
        var inMean = json.inputMean ? new Float32Array(json.inputMean) : null;
        var inStd = json.inputStd ? new Float32Array(json.inputStd) : null;
        var model = {
            version: json.version || 1,
            K: json.K,
            featureDim: json.featureDim,
            inputScale: typeof json.inputScale === 'number' ? json.inputScale : 1,
            inputMean: inMean,
            inputStd: inStd,
            outputScale: typeof json.outputScale === 'number' ? json.outputScale : 1,
            clipMagnitude: typeof json.clipMagnitude === 'number' ? json.clipMagnitude : null,
            layers: [],
            lastInput: null,
            lastOutput: null,
        };
        for (var i = 0; i < json.layers.length; i++) {
            var L = json.layers[i];
            model.layers.push({
                inDim: L.inDim,
                outDim: L.outDim,
                activation: L.activation || 'linear',
                W: new Float32Array(L.W),
                b: new Float32Array(L.b),
                lastZ: new Float32Array(L.outDim),
                lastA: new Float32Array(L.outDim),
            });
        }
        var scratch = []; // reusable buffers per layer for input
        for (var i = 0; i < model.layers.length; i++) {
            scratch.push(new Float32Array(model.layers[i].inDim));
        }

        model.lastNormalizedInput = new Float32Array(model.featureDim);
        model.forward = function (features) {
            model.lastInput = features;
            var cur = scratch[0];
            if (model.inputMean && model.inputStd) {
                for (var i = 0; i < features.length; i++) {
                    cur[i] = (features[i] - model.inputMean[i]) / model.inputStd[i];
                }
            } else {
                for (var i = 0; i < features.length; i++) cur[i] = features[i] * model.inputScale;
            }
            for (var i = 0; i < features.length; i++) model.lastNormalizedInput[i] = cur[i];

            for (var li = 0; li < model.layers.length; li++) {
                var L = model.layers[li];
                var z = L.lastZ;
                var W = L.W;
                var b = L.b;
                var inDim = L.inDim;
                var outDim = L.outDim;
                for (var j = 0; j < outDim; j++) {
                    var s = b[j];
                    for (var i2 = 0; i2 < inDim; i2++) {
                        s += W[i2 * outDim + j] * cur[i2];
                    }
                    z[j] = s;
                }
                applyActivation(z, L.activation, L.lastA);
                // Prepare next layer's input buffer.
                if (li + 1 < model.layers.length) {
                    cur = scratch[li + 1];
                    for (var k = 0; k < outDim; k++) cur[k] = L.lastA[k];
                } else {
                    cur = L.lastA;
                }
            }
            // Final output: rescale.
            var out = new Float32Array(cur.length);
            for (var i3 = 0; i3 < cur.length; i3++) out[i3] = cur[i3] * model.outputScale;
            // Optional alpha-max-beta-min clip so the runtime cannot exceed
            // the rule's fastLimit cap. Matches Vector.iFastLimit exactly.
            if (typeof model.clipMagnitude === 'number' && out.length === 2) {
                var ax = Math.abs(out[0]);
                var ay = Math.abs(out[1]);
                var mag = Math.max(ax, ay) * 0.96 + Math.min(ax, ay) * 0.398;
                if (mag > model.clipMagnitude) {
                    var s = model.clipMagnitude / mag;
                    out[0] *= s;
                    out[1] *= s;
                }
            }
            model.lastOutput = out;
            return out;
        };

        return model;
    }

    return { loadModel: loadModel };
});
