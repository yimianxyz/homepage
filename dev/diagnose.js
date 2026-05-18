// Deeper introspection on a candidate weights file. Designed for the agent
// (me) to read after eval.js, when judging WHY a model fails and what the
// next architectural move should be.
//
//   node dev/diagnose.js --weights dev/weights/<id>.json --report dev/reports/<id>.diag.json
//
// Probes the model's behaviour on a sampled state distribution:
//   - Error vs distance-to-nearest-boid (binned).
//   - Error vs |autoTarget offset|.
//   - Error sharply localised around the in-range boundary (R-edge).
//   - Output magnitude distribution vs the 0.05 cap (saturation analysis).
//   - Dead / duplicate hidden units (per-layer).
//   - Weight + activation histograms (per-layer).
//   - Top-50 failure exemplars dumped in full.

'use strict';

const fs = require('fs');
const path = require('path');

const { Oracle } = require('./oracle');
const spec = require('./policy_spec');
const { loadModel } = require('../js/predator_nn');

function parseArgs(argv) {
    const args = { testSeeds: [201, 203, 211, 223, 227, 229], framesPerSeed: 5000 };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--weights') args.weights = argv[++i];
        else if (a === '--report') args.report = argv[++i];
        else if (a === '--framesPerSeed') args.framesPerSeed = +argv[++i];
        else if (a === '--seeds') args.testSeeds = argv[++i].split(',').map(s => +s);
    }
    if (!args.weights) throw new Error('--weights required');
    if (!args.report) throw new Error('--report required');
    return args;
}

function ensureDir(p) { fs.mkdirSync(path.dirname(p), { recursive: true }); }

function quantile(sorted, q) {
    if (sorted.length === 0) return null;
    const i = Math.floor(q * (sorted.length - 1));
    return sorted[i];
}

function summarize(arr) {
    if (arr.length === 0) return { n: 0 };
    const s = arr.slice().sort((a, b) => a - b);
    const mean = s.reduce((a, b) => a + b, 0) / s.length;
    return {
        n: s.length, min: s[0], max: s[s.length - 1], mean,
        p25: quantile(s, 0.25), p50: quantile(s, 0.5), p75: quantile(s, 0.75),
        p95: quantile(s, 0.95), p99: quantile(s, 0.99),
    };
}

function collectStates(seeds, framesPerSeed) {
    const X = [], Y = [], meta = [];
    const F = spec.F;
    for (const seed of seeds) {
        const o = new Oracle({ seed, numBoids: 120 });
        for (let i = 0; i < framesPerSeed; i++) {
            o.step(false);
            const f = o.sim.predator._lastFeatures;
            const y = o.sim.predator._lastOutput;
            if (!f || !y) continue;
            X.push(Array.from(f));
            Y.push([y[0], y[1]]);
            const d1 = f[F.D1];
            const dAuto = f[F.DA];
            const sp = Math.sqrt(f[F.VX] * f[F.VX] + f[F.VY] * f[F.VY]);
            const branch = (d1 < spec.POLICY_R && f[F.DX1] !== spec.POLICY_PAD) ? 'hunt' : 'patrol';
            meta.push({ d1, dAuto, sp, branch, seed, frame: i });
        }
    }
    return { X, Y, meta };
}

function binErrors(X, Y, meta, model, key, bins) {
    // bins: array of upper-edges (last is Infinity). Returns per-bin error stats.
    const bucketSq = bins.map(_ => []);
    for (let i = 0; i < X.length; i++) {
        const o = model.forward(new Float32Array(X[i]));
        const dx = o[0] - Y[i][0];
        const dy = o[1] - Y[i][1];
        const sq = dx * dx + dy * dy;
        const k = meta[i][key];
        for (let b = 0; b < bins.length; b++) {
            if (k < bins[b]) { bucketSq[b].push(sq); break; }
        }
    }
    return bins.map((upper, b) => ({
        upper,
        n: bucketSq[b].length,
        mse: bucketSq[b].length ? bucketSq[b].reduce((s, v) => s + v, 0) / bucketSq[b].length : null,
        rmse: bucketSq[b].length ? Math.sqrt(bucketSq[b].reduce((s, v) => s + v, 0) / bucketSq[b].length) : null,
    }));
}

function unitDiagnostics(model, X) {
    // For each hidden layer, gather activations across all X. Report:
    //   - per-unit zero-rate (fraction of inputs where lastA == 0)
    //   - per-unit mean/std absolute activation
    //   - pairwise correlation matrix (compressed: top duplicate pairs)
    const layers = model.layers.map((L, idx) => ({
        idx, inDim: L.inDim, outDim: L.outDim, activation: L.activation,
        sums: new Float64Array(L.outDim),
        sumsSq: new Float64Array(L.outDim),
        absSums: new Float64Array(L.outDim),
        zeros: new Int32Array(L.outDim),
        co: Array.from({ length: L.outDim }, () => new Float64Array(L.outDim)),
        count: 0,
    }));

    for (let i = 0; i < X.length; i++) {
        model.forward(new Float32Array(X[i]));
        for (let li = 0; li < model.layers.length; li++) {
            const L = model.layers[li];
            const a = L.lastA;
            const stats = layers[li];
            stats.count++;
            for (let j = 0; j < a.length; j++) {
                stats.sums[j] += a[j];
                stats.sumsSq[j] += a[j] * a[j];
                stats.absSums[j] += Math.abs(a[j]);
                if (Math.abs(a[j]) < 1e-12) stats.zeros[j]++;
            }
            // Cross-products for correlation.
            for (let j = 0; j < a.length; j++) {
                for (let k = j; k < a.length; k++) {
                    stats.co[j][k] += a[j] * a[k];
                }
            }
        }
    }

    return layers.map(stats => {
        const N = stats.count;
        const means = Array.from(stats.sums).map(v => v / N);
        const vars_ = Array.from(stats.sumsSq).map((v, j) => v / N - means[j] * means[j]);
        const stds = vars_.map(v => Math.sqrt(Math.max(0, v)));
        const absMeans = Array.from(stats.absSums).map(v => v / N);
        const zeroRate = Array.from(stats.zeros).map(v => v / N);

        // Top duplicate pairs (correlation > 0.99).
        const dups = [];
        for (let j = 0; j < stats.outDim; j++) {
            for (let k = j + 1; k < stats.outDim; k++) {
                const cov = stats.co[j][k] / N - means[j] * means[k];
                const sd = stds[j] * stds[k];
                if (sd > 1e-12) {
                    const corr = cov / sd;
                    if (Math.abs(corr) > 0.99) dups.push({ a: j, b: k, corr });
                }
            }
        }

        return {
            layer: stats.idx,
            activation: stats.activation,
            outDim: stats.outDim,
            unitMean: means,
            unitStd: stds,
            unitAbsMean: absMeans,
            unitZeroRate: zeroRate,
            deadUnits: zeroRate.map((z, i) => z > 0.99 ? i : -1).filter(i => i >= 0),
            duplicatePairs: dups,
        };
    });
}

function weightHistograms(model) {
    return model.layers.map((L, idx) => {
        const w = Array.from(L.W);
        const b = Array.from(L.b);
        return {
            layer: idx,
            wStats: summarize(w.map(Math.abs)),
            bStats: summarize(b.map(Math.abs)),
            wRaw: summarize(w),
            bRaw: summarize(b),
        };
    });
}

function saturationStats(model, X) {
    const mags = [];
    for (let i = 0; i < X.length; i++) {
        const o = model.forward(new Float32Array(X[i]));
        mags.push(Math.sqrt(o[0] * o[0] + o[1] * o[1]));
    }
    return {
        outputMagnitude: summarize(mags),
        fractionAtCap: mags.filter(m => m > 0.049).length / mags.length,
        fractionBelowEps: mags.filter(m => m < 1e-6).length / mags.length,
    };
}

function worstExemplars(model, X, Y, meta, topN) {
    const all = [];
    for (let i = 0; i < X.length; i++) {
        const o = model.forward(new Float32Array(X[i]));
        const dx = o[0] - Y[i][0];
        const dy = o[1] - Y[i][1];
        const errMag = Math.sqrt(dx * dx + dy * dy);
        all.push({ i, errMag });
    }
    all.sort((a, b) => b.errMag - a.errMag);
    return all.slice(0, topN).map(e => ({
        errMag: e.errMag,
        input: X[e.i],
        oracleOut: Y[e.i],
        nnOut: Array.from(model.forward(new Float32Array(X[e.i]))),
        meta: meta[e.i],
    }));
}

function main() {
    const args = parseArgs(process.argv);
    const tStart = Date.now();
    process.stdout.write(JSON.stringify({ phase: 'start', args }) + '\n');

    const json = JSON.parse(fs.readFileSync(args.weights, 'utf8'));
    const model = loadModel(json);

    process.stdout.write(JSON.stringify({ phase: 'collect' }) + '\n');
    const { X, Y, meta } = collectStates(args.testSeeds, args.framesPerSeed);

    process.stdout.write(JSON.stringify({ phase: 'binning' }) + '\n');
    const dNearestBins = [10, 30, 60, 75, 78, 80, 82, 85, 100, 150, 300, 1e6, Infinity];
    const dAutoBins = [50, 200, 500, 1000, 2000, Infinity];
    const errByDNearest = binErrors(X, Y, meta, model, 'd1', dNearestBins);
    const errByDAuto    = binErrors(X, Y, meta, model, 'dAuto', dAutoBins);

    process.stdout.write(JSON.stringify({ phase: 'rEdge' }) + '\n');
    const rEdgeBins = [70, 73, 76, 78, 79, 80, 81, 82, 84, 87, 90];
    const errByREdge = binErrors(X, Y, meta, model, 'd1', rEdgeBins);

    process.stdout.write(JSON.stringify({ phase: 'units' }) + '\n');
    const units = unitDiagnostics(model, X);

    process.stdout.write(JSON.stringify({ phase: 'weights' }) + '\n');
    const weights = weightHistograms(model);

    process.stdout.write(JSON.stringify({ phase: 'saturation' }) + '\n');
    const saturation = saturationStats(model, X);

    process.stdout.write(JSON.stringify({ phase: 'worst' }) + '\n');
    const worst = worstExemplars(model, X, Y, meta, 50);

    const report = {
        startedAt: new Date(tStart).toISOString(),
        elapsedMs: Date.now() - tStart,
        args,
        nStates: X.length,
        errByDNearest, errByDAuto, errByREdge,
        units, weights, saturation,
        worst,
    };

    ensureDir(args.report);
    fs.writeFileSync(args.report, JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify({ phase: 'done', elapsedMs: report.elapsedMs, reportPath: args.report }) + '\n');
}

if (require.main === module) main();

module.exports = { collectStates, binErrors, unitDiagnostics, weightHistograms, saturationStats, worstExemplars };
