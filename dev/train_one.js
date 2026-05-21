// Single-architecture trainer. Hand-rolled Adam + MSE on a sequence of
// Dense+activation layers. No external deps -- keeps repo lean and gives the
// agent full visibility into every training internal.
//
//   node dev/train_one.js --arch '{"id":"K4_H8_relu","layers":[{"units":8,"activation":"relu"},{"units":2,"activation":"linear"}]}' \
//                          --dataset dev/dataset.bin --epochs 80 --batch 256 --lr 3e-3 \
//                          --out dev/weights/K4_H8_relu.json --report dev/reports/K4_H8_relu.train.json
//
// Output files:
//   weights file  -- consumable by js/predator_nn.js (browser + node)
//   training report -- loss curve + per-epoch val loss + final stats

'use strict';

const fs = require('fs');
const path = require('path');
const spec = require('./policy_spec');

// ---- args ----
function parseArgs(argv) {
    const args = {
        arch: null,
        dataset: 'dev/dataset.bin',
        epochs: 80,
        batch: 256,
        lr: 3e-3,
        valFrac: 0.1,
        seed: 1234,
        inputScale: 1 / 200,
        outputScale: 0.05,
        clip: 10,
        edgeOversample: 3,    // how many copies of R-edge samples (d1 in [R-15, R+15])
        out: null,
        report: null,
        logEvery: 1,
        // New (opt-in) knobs. Defaults preserve old behavior bit-for-bit:
        cosine: 0,            // 1 = cosine anneal lr -> cosineMinLr over full epoch budget
        cosineMinLr: 1e-5,
        warmupEpochs: 0,
        loss: 'mse',          // 'mse' | 'huber'
        huberDelta: 0.05,
        edgeWeight: 1,        // per-sample loss weight on R-edge (d1 in [R-15, R+15]). 1 = no extra weight.
        ema: 0,               // 0 = off. >0 = exponential moving average decay for shadow weights.
        bestBy: 'val',        // 'val' | 'emaVal'
        maxLossLambda: 0,     // add lambda * max-batch-error to loss (push down worst-case)
    };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--arch') args.arch = JSON.parse(argv[++i]);
        else if (a === '--archFile') args.arch = JSON.parse(fs.readFileSync(argv[++i], 'utf8'));
        else if (a === '--dataset') args.dataset = argv[++i];
        else if (a === '--epochs') args.epochs = +argv[++i];
        else if (a === '--batch') args.batch = +argv[++i];
        else if (a === '--lr') args.lr = +argv[++i];
        else if (a === '--valFrac') args.valFrac = +argv[++i];
        else if (a === '--seed') args.seed = +argv[++i];
        else if (a === '--inputScale') args.inputScale = +argv[++i];
        else if (a === '--outputScale') args.outputScale = +argv[++i];
        else if (a === '--clip') args.clip = +argv[++i];
        else if (a === '--edgeOversample') args.edgeOversample = +argv[++i];
        else if (a === '--out') args.out = argv[++i];
        else if (a === '--report') args.report = argv[++i];
        else if (a === '--logEvery') args.logEvery = +argv[++i];
        else if (a === '--cosine') args.cosine = +argv[++i];
        else if (a === '--cosineMinLr') args.cosineMinLr = +argv[++i];
        else if (a === '--warmupEpochs') args.warmupEpochs = +argv[++i];
        else if (a === '--loss') args.loss = argv[++i];
        else if (a === '--huberDelta') args.huberDelta = +argv[++i];
        else if (a === '--edgeWeight') args.edgeWeight = +argv[++i];
        else if (a === '--ema') args.ema = +argv[++i];
        else if (a === '--bestBy') args.bestBy = argv[++i];
        else if (a === '--maxLossLambda') args.maxLossLambda = +argv[++i];
    }
    if (!args.arch) throw new Error('--arch or --archFile required');
    if (!args.out) args.out = `dev/weights/${args.arch.id}.json`;
    if (!args.report) args.report = `dev/reports/${args.arch.id}.train.json`;
    return args;
}

// ---- mulberry32 for deterministic shuffles + init ----
function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function gaussian(rand) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---- Model ----
function buildModel(arch, inputDim, rand) {
    const layers = [];
    let prevDim = inputDim;
    for (const L of arch.layers) {
        const inDim = prevDim;
        const outDim = L.units;
        const W = new Float32Array(inDim * outDim);
        const b = new Float32Array(outDim);
        // Init: He for ReLU, Xavier otherwise.
        const fanIn = inDim;
        const std = (L.activation === 'relu')
            ? Math.sqrt(2 / fanIn)
            : Math.sqrt(1 / fanIn);
        for (let k = 0; k < W.length; k++) W[k] = gaussian(rand) * std;
        // b stays 0
        layers.push({
            inDim, outDim, activation: L.activation,
            W, b,
            // Gradient buffers
            gW: new Float32Array(W.length),
            gb: new Float32Array(b.length),
            // Adam state
            mW: new Float32Array(W.length), vW: new Float32Array(W.length),
            mb: new Float32Array(b.length), vb: new Float32Array(b.length),
            // Forward caches (per-sample within a batch)
            // For batched compute we store per-batch in scratch space below.
        });
        prevDim = outDim;
    }
    return { layers, inputDim, outputDim: prevDim };
}

// ---- Forward / backward on a mini-batch ----
function applyAct(z, n, act, out) {
    if (act === 'relu') for (let i = 0; i < n; i++) out[i] = z[i] > 0 ? z[i] : 0;
    else if (act === 'tanh') for (let i = 0; i < n; i++) out[i] = Math.tanh(z[i]);
    else if (act === 'sigmoid') for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-z[i]));
    else for (let i = 0; i < n; i++) out[i] = z[i];
}

function actGrad(z, a, n, act, dA, dZ) {
    if (act === 'relu') for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : 0;
    else if (act === 'tanh') for (let i = 0; i < n; i++) dZ[i] = dA[i] * (1 - a[i] * a[i]);
    else if (act === 'sigmoid') for (let i = 0; i < n; i++) dZ[i] = dA[i] * a[i] * (1 - a[i]);
    else for (let i = 0; i < n; i++) dZ[i] = dA[i];
}

// Returns the per-batch forward caches; for batchsize B and layer with outDim D the
// activations are stored as [B*D] Float32Array (row-major: sample i, unit j at i*D+j).
function forwardBatch(model, Xbatch, B) {
    const inputDim = model.inputDim;
    const caches = [];
    let cur = Xbatch;
    let curDim = inputDim;
    let curB = B;
    for (let li = 0; li < model.layers.length; li++) {
        const L = model.layers[li];
        const z = new Float32Array(B * L.outDim);
        for (let i = 0; i < B; i++) {
            const rowOff = i * L.outDim;
            const inOff = i * curDim;
            for (let j = 0; j < L.outDim; j++) {
                let s = L.b[j];
                for (let k = 0; k < curDim; k++) {
                    s += L.W[k * L.outDim + j] * cur[inOff + k];
                }
                z[rowOff + j] = s;
            }
        }
        const a = new Float32Array(B * L.outDim);
        applyAct(z, B * L.outDim, L.activation, a);
        caches.push({ z, a, inDim: curDim, outDim: L.outDim });
        cur = a;
        curDim = L.outDim;
    }
    return { output: cur, caches };
}

function backwardBatch(model, Xbatch, Ybatch, caches, B, opts) {
    // Build dL/dA for the final layer. Supports:
    //   loss: 'mse' (default) | 'huber'
    //   per-sample weights (default: all 1)
    //   maxLossLambda: extra gradient through max_{i,j} |err|
    // When opts is omitted, computes the exact same gradient as the original
    // unweighted MSE path so prior runs are bit-for-bit reproducible.
    const last = caches[caches.length - 1];
    const finalDim = last.outDim;
    let dA = new Float32Array(B * finalDim);

    if (!opts || (opts.loss === 'mse' && !opts.weights && !opts.maxLossLambda)) {
        const scale = 2 / (B * finalDim);
        for (let i = 0; i < B; i++) {
            for (let j = 0; j < finalDim; j++) {
                const idx = i * finalDim + j;
                dA[idx] = scale * (last.a[idx] - Ybatch[idx]);
            }
        }
    } else {
        const loss = opts.loss || 'mse';
        const huberDelta = opts.huberDelta || 0.05;
        const weights = opts.weights; // length B, may be null
        const maxLossLambda = opts.maxLossLambda || 0;
        // Normalize so the loss matches a mean of |err|^2/2 (mse) or huber.
        // We want gradient magnitudes comparable to the plain MSE path so
        // the same lr works.
        let totalW = 0;
        if (weights) {
            for (let i = 0; i < B; i++) totalW += weights[i];
        } else {
            totalW = B;
        }
        const denom = totalW * finalDim;
        // Track max-error position for max-loss gradient.
        let maxErr = 0, maxIdx = -1, maxSign = 0;
        for (let i = 0; i < B; i++) {
            const w = weights ? weights[i] : 1;
            for (let j = 0; j < finalDim; j++) {
                const idx = i * finalDim + j;
                const err = last.a[idx] - Ybatch[idx];
                let g;
                if (loss === 'huber') {
                    if (Math.abs(err) < huberDelta) g = err;
                    else g = huberDelta * (err > 0 ? 1 : -1);
                } else {
                    g = err;
                }
                dA[idx] = (2 * w * g) / denom;
                const ae = Math.abs(err);
                if (ae > maxErr) { maxErr = ae; maxIdx = idx; maxSign = err > 0 ? 1 : -1; }
            }
        }
        if (maxLossLambda > 0 && maxIdx >= 0) {
            // d/d(a[maxIdx]) of (lambda * max_abs_err) = lambda * sign(err) (subgrad).
            dA[maxIdx] += maxLossLambda * maxSign;
        }
    }
    // Backprop through layers in reverse.
    for (let li = model.layers.length - 1; li >= 0; li--) {
        const L = model.layers[li];
        const cache = caches[li];
        const dZ = new Float32Array(B * L.outDim);
        actGrad(cache.z, cache.a, B * L.outDim, L.activation, dA, dZ);

        // gW = prev_a^T @ dZ  (prev_a shape [B, inDim], dZ shape [B, outDim])
        const prev_a = (li === 0) ? Xbatch : caches[li - 1].a;
        L.gW.fill(0);
        L.gb.fill(0);
        for (let i = 0; i < B; i++) {
            const inOff = i * L.inDim;
            const outOff = i * L.outDim;
            for (let j = 0; j < L.outDim; j++) {
                L.gb[j] += dZ[outOff + j];
                const dzj = dZ[outOff + j];
                for (let k = 0; k < L.inDim; k++) {
                    L.gW[k * L.outDim + j] += prev_a[inOff + k] * dzj;
                }
            }
        }

        if (li > 0) {
            const dA_prev = new Float32Array(B * L.inDim);
            for (let i = 0; i < B; i++) {
                const inOff = i * L.inDim;
                const outOff = i * L.outDim;
                for (let k = 0; k < L.inDim; k++) {
                    let s = 0;
                    for (let j = 0; j < L.outDim; j++) {
                        s += L.W[k * L.outDim + j] * dZ[outOff + j];
                    }
                    dA_prev[inOff + k] = s;
                }
            }
            dA = dA_prev;
        }
    }
}

function adamStep(model, lr, beta1, beta2, eps, step) {
    const b1c = 1 - Math.pow(beta1, step);
    const b2c = 1 - Math.pow(beta2, step);
    for (const L of model.layers) {
        for (let i = 0; i < L.W.length; i++) {
            const g = L.gW[i];
            L.mW[i] = beta1 * L.mW[i] + (1 - beta1) * g;
            L.vW[i] = beta2 * L.vW[i] + (1 - beta2) * g * g;
            const mhat = L.mW[i] / b1c;
            const vhat = L.vW[i] / b2c;
            L.W[i] -= lr * mhat / (Math.sqrt(vhat) + eps);
        }
        for (let j = 0; j < L.b.length; j++) {
            const g = L.gb[j];
            L.mb[j] = beta1 * L.mb[j] + (1 - beta1) * g;
            L.vb[j] = beta2 * L.vb[j] + (1 - beta2) * g * g;
            const mhat = L.mb[j] / b1c;
            const vhat = L.vb[j] / b2c;
            L.b[j] -= lr * mhat / (Math.sqrt(vhat) + eps);
        }
    }
}

// ---- IO / preprocessing ----
function loadDataset(binPath) {
    const buf = fs.readFileSync(binPath);
    const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const metaPath = binPath.replace(/\.bin$/, '.meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const ROW = meta.rowFloats;
    const n = arr.length / ROW;
    return { arr, meta, n, ROW };
}

// Per-feature standardization: compute mean and std across the (raw) feature
// columns of the dataset, then apply x' = (x - mean)/std. Robust to widely
// different feature scales (velocity O(2.5) vs offsets O(500) vs sentinel
// 2000) which a single scalar inputScale cannot bridge.
function computeFeatureStats(arr, n, ROW, inputDim) {
    const sum = new Float64Array(inputDim);
    const sumSq = new Float64Array(inputDim);
    for (let i = 0; i < n; i++) {
        const off = i * ROW;
        for (let k = 0; k < inputDim; k++) {
            const v = arr[off + k];
            sum[k] += v;
            sumSq[k] += v * v;
        }
    }
    const mean = new Float32Array(inputDim);
    const std = new Float32Array(inputDim);
    for (let k = 0; k < inputDim; k++) {
        mean[k] = sum[k] / n;
        const v = sumSq[k] / n - mean[k] * mean[k];
        std[k] = Math.max(Math.sqrt(Math.max(0, v)), 1e-3);
    }
    return { mean, std };
}

function preprocess(arr, n, ROW, inputDim, mean, std, outputScale) {
    // X: [n, inputDim], Y: [n, 2]
    const X = new Float32Array(n * inputDim);
    const Y = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
        const off = i * ROW;
        for (let k = 0; k < inputDim; k++) {
            X[i * inputDim + k] = (arr[off + k] - mean[k]) / std[k];
        }
        // Targets divided by outputScale so the network can emit O(1).
        Y[i * 2] = arr[off + inputDim] / outputScale;
        Y[i * 2 + 1] = arr[off + inputDim + 1] / outputScale;
    }
    return { X, Y };
}

function rEdgeMask(arr, n, ROW, inputDim) {
    const R = spec.POLICY_R;
    const mask = new Uint8Array(n);
    const D1 = spec.F.D1;
    for (let i = 0; i < n; i++) {
        const d = arr[i * ROW + D1];
        if (d > R - 15 && d < R + 15) mask[i] = 1;
    }
    return mask;
}

function buildIndex(n, valFrac, edgeMask, oversample, rand) {
    const all = [];
    for (let i = 0; i < n; i++) all.push(i);
    // Shuffle
    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const t = all[i]; all[i] = all[j]; all[j] = t;
    }
    const nVal = Math.floor(n * valFrac);
    const valIdx = all.slice(0, nVal);
    const trainIdx = all.slice(nVal);
    // Oversample R-edge in train.
    if (oversample > 1 && edgeMask) {
        const extra = [];
        for (const i of trainIdx) {
            if (edgeMask[i]) {
                for (let k = 1; k < oversample; k++) extra.push(i);
            }
        }
        for (const i of extra) trainIdx.push(i);
        for (let i = trainIdx.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            const t = trainIdx[i]; trainIdx[i] = trainIdx[j]; trainIdx[j] = t;
        }
    }
    return { trainIdx, valIdx };
}

function gatherBatch(X, Y, indices, start, B, inputDim, weightsAll) {
    const xb = new Float32Array(B * inputDim);
    const yb = new Float32Array(B * 2);
    const wb = weightsAll ? new Float32Array(B) : null;
    for (let i = 0; i < B; i++) {
        const idx = indices[start + i];
        for (let k = 0; k < inputDim; k++) xb[i * inputDim + k] = X[idx * inputDim + k];
        yb[i * 2] = Y[idx * 2];
        yb[i * 2 + 1] = Y[idx * 2 + 1];
        if (wb) wb[i] = weightsAll[idx];
    }
    return { xb, yb, wb };
}

function evalLoss(model, X, Y, indices, inputDim) {
    let total = 0;
    const B = Math.min(1024, indices.length);
    for (let off = 0; off < indices.length; off += B) {
        const cur = Math.min(B, indices.length - off);
        const { xb, yb } = gatherBatch(X, Y, indices, off, cur, inputDim);
        const { output } = forwardBatch(model, xb, cur);
        for (let i = 0; i < cur * 2; i++) {
            const d = output[i] - yb[i];
            total += d * d;
        }
    }
    return total / (indices.length * 2);
}

function ensureDir(p) { fs.mkdirSync(path.dirname(p), { recursive: true }); }

function saveModel(model, arch, args, mean, std, outPath) {
    const layersJson = model.layers.map(L => ({
        inDim: L.inDim,
        outDim: L.outDim,
        activation: L.activation,
        W: Array.from(L.W),
        b: Array.from(L.b),
    }));
    const out = {
        version: 1,
        id: arch.id,
        K: spec.POLICY_K,
        // Use the actual trained input dim (matches the dataset's featureDim
        // and the layer-0 W shape). spec.FEATURE_DIM may have drifted higher
        // since the dataset was generated.
        featureDim: mean.length,
        inputMean: Array.from(mean),
        inputStd: Array.from(std),
        outputScale: args.outputScale,
        clipMagnitude: 0.05,
        layers: layersJson,
    };
    ensureDir(outPath);
    fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
}

function main() {
    const args = parseArgs(process.argv);
    const tStart = Date.now();
    process.stdout.write(JSON.stringify({ phase: 'start', arch: args.arch, dataset: args.dataset, lr: args.lr, epochs: args.epochs, batch: args.batch }) + '\n');

    const rand = mulberry32(args.seed);
    const { arr, meta, n, ROW } = loadDataset(args.dataset);
    process.stdout.write(JSON.stringify({ phase: 'loaded', n, featureDim: meta.featureDim }) + '\n');

    const inputDim = meta.featureDim;
    const { mean, std } = computeFeatureStats(arr, n, ROW, inputDim);
    process.stdout.write(JSON.stringify({ phase: 'normalize', mean: Array.from(mean), std: Array.from(std) }) + '\n');
    const { X, Y } = preprocess(arr, n, ROW, inputDim, mean, std, args.outputScale);
    const edgeMask = rEdgeMask(arr, n, ROW, inputDim);
    const { trainIdx, valIdx } = buildIndex(n, args.valFrac, edgeMask, args.edgeOversample, rand);
    process.stdout.write(JSON.stringify({ phase: 'split', train: trainIdx.length, val: valIdx.length, edgeRaw: edgeMask.reduce((s,v)=>s+v,0) }) + '\n');

    // Build per-sample weights (over the whole dataset, indexed by original i).
    // When --edgeWeight > 1, R-edge samples get that multiplier in the loss.
    let perSampleWeights = null;
    if (args.edgeWeight !== 1) {
        perSampleWeights = new Float32Array(n);
        for (let i = 0; i < n; i++) perSampleWeights[i] = edgeMask[i] ? args.edgeWeight : 1;
    }

    const model = buildModel(args.arch, inputDim, rand);
    const totalParams = model.layers.reduce((s, L) => s + L.W.length + L.b.length, 0);
    process.stdout.write(JSON.stringify({ phase: 'model', layers: model.layers.map(L => ({inDim:L.inDim, outDim:L.outDim, activation:L.activation})), totalParams }) + '\n');

    // EMA shadow weights (used for late-stage smoothing).
    let ema = null;
    if (args.ema > 0) {
        ema = model.layers.map(L => ({ W: Float32Array.from(L.W), b: Float32Array.from(L.b) }));
    }

    const initialValLoss = evalLoss(model, X, Y, valIdx, inputDim);
    process.stdout.write(JSON.stringify({ phase: 'pretrain', initialValLoss, cosine: args.cosine, loss: args.loss, edgeWeight: args.edgeWeight, ema: args.ema, maxLossLambda: args.maxLossLambda }) + '\n');

    function curLr(epoch) {
        if (epoch < args.warmupEpochs) return args.lr * (epoch + 1) / args.warmupEpochs;
        if (!args.cosine) return args.lr;
        // cosine from args.lr -> args.cosineMinLr over [warmupEpochs, epochs)
        const t = (epoch - args.warmupEpochs) / Math.max(1, args.epochs - args.warmupEpochs);
        const cos = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, Math.max(0, t))));
        return args.cosineMinLr + (args.lr - args.cosineMinLr) * cos;
    }

    function copyModelWeights(target) {
        for (let li = 0; li < model.layers.length; li++) {
            const L = model.layers[li];
            target[li].W = Float32Array.from(L.W);
            target[li].b = Float32Array.from(L.b);
        }
    }

    function emaUpdate(decay) {
        for (let li = 0; li < model.layers.length; li++) {
            const L = model.layers[li];
            const sh = ema[li];
            for (let i = 0; i < L.W.length; i++) sh.W[i] = decay * sh.W[i] + (1 - decay) * L.W[i];
            for (let i = 0; i < L.b.length; i++) sh.b[i] = decay * sh.b[i] + (1 - decay) * L.b[i];
        }
    }

    function emaValLoss() {
        if (!ema) return Infinity;
        // Swap shadow into model temporarily.
        const tmp = model.layers.map(L => ({ W: L.W, b: L.b }));
        for (let li = 0; li < model.layers.length; li++) {
            model.layers[li].W = ema[li].W;
            model.layers[li].b = ema[li].b;
        }
        const v = evalLoss(model, X, Y, valIdx, inputDim);
        for (let li = 0; li < model.layers.length; li++) {
            model.layers[li].W = tmp[li].W;
            model.layers[li].b = tmp[li].b;
        }
        return v;
    }

    const lossCurve = [];
    let step = 0;
    let bestVal = Infinity, bestEpoch = -1;
    let bestWeights = null;
    let bestSource = null; // 'live' | 'ema'
    const backwardOpts = (args.loss === 'huber' || perSampleWeights || args.maxLossLambda > 0)
        ? { loss: args.loss, huberDelta: args.huberDelta, maxLossLambda: args.maxLossLambda }
        : null;
    for (let epoch = 0; epoch < args.epochs; epoch++) {
        const epochLr = curLr(epoch);
        // Shuffle train indices.
        for (let i = trainIdx.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            const t = trainIdx[i]; trainIdx[i] = trainIdx[j]; trainIdx[j] = t;
        }
        let epochLossSum = 0, epochSamples = 0;
        for (let off = 0; off < trainIdx.length; off += args.batch) {
            const B = Math.min(args.batch, trainIdx.length - off);
            const { xb, yb, wb } = gatherBatch(X, Y, trainIdx, off, B, inputDim, perSampleWeights);
            const { output, caches } = forwardBatch(model, xb, B);
            // batch loss for reporting (always plain MSE so curves are comparable across configs)
            for (let i = 0; i < B * 2; i++) {
                const d = output[i] - yb[i];
                epochLossSum += d * d;
            }
            epochSamples += B * 2;
            if (backwardOpts) {
                backwardBatch(model, xb, yb, caches, B, Object.assign({ weights: wb }, backwardOpts));
            } else {
                backwardBatch(model, xb, yb, caches, B);
            }
            step++;
            adamStep(model, epochLr, 0.9, 0.999, 1e-8, step);
            if (ema) emaUpdate(args.ema);
        }
        const trainLoss = epochLossSum / epochSamples;
        const valLoss = evalLoss(model, X, Y, valIdx, inputDim);
        let emaVal = null;
        if (ema) emaVal = emaValLoss();
        lossCurve.push({ epoch, trainLoss, valLoss, lr: epochLr, emaVal });

        const candidateVal = (args.bestBy === 'emaVal' && emaVal != null) ? emaVal : valLoss;
        if (candidateVal < bestVal) {
            bestVal = candidateVal; bestEpoch = epoch;
            if (args.bestBy === 'emaVal' && ema) {
                bestWeights = ema.map(L => ({ W: Array.from(L.W), b: Array.from(L.b) }));
                bestSource = 'ema';
            } else {
                bestWeights = model.layers.map(L => ({ W: Array.from(L.W), b: Array.from(L.b) }));
                bestSource = 'live';
            }
        }
        if (epoch % args.logEvery === 0 || epoch === args.epochs - 1) {
            const out = { phase: 'epoch', epoch, trainLoss, valLoss, bestVal, bestEpoch, lr: epochLr };
            if (emaVal != null) out.emaVal = emaVal;
            process.stdout.write(JSON.stringify(out) + '\n');
        }
    }

    // Restore best weights.
    for (let li = 0; li < model.layers.length; li++) {
        const L = model.layers[li];
        L.W = Float32Array.from(bestWeights[li].W);
        L.b = Float32Array.from(bestWeights[li].b);
    }

    saveModel(model, args.arch, args, mean, std, args.out);

    const report = {
        startedAt: new Date(tStart).toISOString(),
        elapsedMs: Date.now() - tStart,
        args,
        n, totalParams,
        finalTrainLoss: lossCurve[lossCurve.length - 1].trainLoss,
        finalValLoss: lossCurve[lossCurve.length - 1].valLoss,
        bestValLoss: bestVal,
        bestEpoch,
        bestSource,
        lossCurve,
    };
    ensureDir(args.report);
    fs.writeFileSync(args.report, JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify({ phase: 'done', bestVal, bestEpoch, bestSource, weightsPath: args.out, reportPath: args.report, elapsedMs: report.elapsedMs }) + '\n');
}

if (require.main === module) main();

module.exports = { buildModel, forwardBatch, backwardBatch, adamStep, mulberry32 };
