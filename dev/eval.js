// Rich machine-readable eval for a candidate NN predator policy.
//
//   node dev/eval.js --weights dev/weights/<id>.json --report dev/reports/<id>.eval.json
//
// All numbers are dumped to JSON; this script is an INSTRUMENT. The agent
// is the judge. Default thresholds shown in the "gates" field of the output
// are advisory.

'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const { Oracle, runTrace } = require('./oracle');
const spec = require('./policy_spec');
const { loadModel } = require('../js/predator_nn');

// ---- CLI ----
function parseArgs(argv) {
    const args = { seeds: null, frames: 2000, testStates: 50000, divSeeds: 8, behaviorSeeds: 4 };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--weights') args.weights = argv[++i];
        else if (a === '--report') args.report = argv[++i];
        else if (a === '--frames') args.frames = +argv[++i];
        else if (a === '--testStates') args.testStates = +argv[++i];
        else if (a === '--divSeeds') args.divSeeds = +argv[++i];
        else if (a === '--behaviorSeeds') args.behaviorSeeds = +argv[++i];
        else if (a === '--seeds') args.seeds = argv[++i].split(',').map(s => +s);
    }
    if (!args.weights) throw new Error('--weights required');
    if (!args.report) throw new Error('--report required');
    // Held-out seeds for eval, disjoint from training/dataset seeds (0..49).
    args.testSeeds = args.seeds || [101, 103, 107, 109, 113, 127, 131, 137];
    return args;
}

function ensureDir(p) { fs.mkdirSync(path.dirname(p), { recursive: true }); }
function hashFile(p) {
    return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex').slice(0, 16);
}

// ---- 1. Build the NN forward callable. ----
function loadNN(weightsPath) {
    const json = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
    const model = loadModel(json);
    const totalParams = model.layers.reduce((s, L) => s + L.W.length + L.b.length, 0);
    const info = {
        weightsPath,
        weightHash: hashFile(weightsPath),
        K: model.K,
        featureDim: model.featureDim,
        inputScale: model.inputScale,
        outputScale: model.outputScale,
        layers: model.layers.map(L => ({
            inDim: L.inDim, outDim: L.outDim, activation: L.activation,
            params: L.W.length + L.b.length,
        })),
        totalParams,
    };
    const nnFn = (features) => model.forward(features);
    return { model, info, nnFn };
}

// ---- 2. Collect held-out (features, target_output) pairs. ----
function collectHeldOutStates(testSeeds, framesPerSeed) {
    const X = [];
    const Y = [];
    const meta = [];
    const F = spec.F;
    for (const seed of testSeeds) {
        const o = new Oracle({ seed, numBoids: 120 });
        for (let i = 0; i < framesPerSeed; i++) {
            o.step(false);
            const f = o.sim.predator._lastFeatures;
            const y = o.sim.predator._lastOutput;
            if (!f || !y) continue;
            X.push(Array.from(f));
            Y.push([y[0], y[1]]);
            const d1 = f[F.D1];
            const branch = (d1 < spec.POLICY_R && f[F.DX1] !== spec.POLICY_PAD) ? 'hunt' : 'patrol';
            meta.push({ d1, branch, seed, frame: i });
        }
    }
    return { X, Y, meta };
}

// ---- 3. Regression metrics. ----
function regressionMetrics(model, X, Y, meta) {
    const n = X.length;
    let sumSq = 0, sumAbs = 0, maxAbs = 0;
    let sumSqHunt = 0, nHunt = 0;
    let sumSqPatrol = 0, nPatrol = 0;
    let sumSqEdge = 0, nEdge = 0;
    const perDimSq = [0, 0];
    const errMags = [];
    const worst = []; // top-K worst by abs error magnitude

    for (let i = 0; i < n; i++) {
        const xArr = X[i];
        const y = Y[i];
        const xf = new Float32Array(xArr);
        const o = model.forward(xf);
        const dx = o[0] - y[0];
        const dy = o[1] - y[1];
        const sq = dx * dx + dy * dy;
        const ax = Math.abs(dx), ay = Math.abs(dy);
        sumSq += sq;
        sumAbs += ax + ay;
        perDimSq[0] += dx * dx;
        perDimSq[1] += dy * dy;
        const m = Math.max(ax, ay);
        if (m > maxAbs) maxAbs = m;
        errMags.push(Math.sqrt(sq));

        const md = meta[i];
        if (md.branch === 'hunt') { sumSqHunt += sq; nHunt += 1; }
        else { sumSqPatrol += sq; nPatrol += 1; }
        if (md.d1 >= spec.POLICY_R - 5 && md.d1 <= spec.POLICY_R + 5) {
            sumSqEdge += sq; nEdge += 1;
        }

        worst.push({ idx: i, errMag: Math.sqrt(sq), branch: md.branch, d1: md.d1 });
    }

    worst.sort((a, b) => b.errMag - a.errMag);
    const top10 = worst.slice(0, 10).map(w => ({
        idx: w.idx, errMag: w.errMag, branch: w.branch, d1: w.d1,
        input: X[w.idx], oracle: Y[w.idx], nn: Array.from(model.forward(new Float32Array(X[w.idx]))),
    }));

    // Histogram of error magnitudes (log bins).
    const bins = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, Infinity];
    const hist = bins.map(_ => 0);
    for (const m of errMags) {
        for (let b = 0; b < bins.length; b++) if (m < bins[b]) { hist[b]++; break; }
    }

    return {
        n,
        mse: sumSq / n,
        mae: sumAbs / (2 * n),
        maxAbs,
        perDimMse: [perDimSq[0] / n, perDimSq[1] / n],
        byBranch: {
            hunt: { n: nHunt, mse: nHunt ? sumSqHunt / nHunt : null },
            patrol: { n: nPatrol, mse: nPatrol ? sumSqPatrol / nPatrol : null },
        },
        rEdge: { n: nEdge, mse: nEdge ? sumSqEdge / nEdge : null },
        errorMagHistogram: { bins, counts: hist },
        top10Worst: top10,
    };
}

// ---- 4. Decision-branch agreement. ----
// For each state, the rule's branch is in meta. For the NN's implied branch:
// we say it's "hunt" if cos(NN_out, seek_to_nearest_boid) > 0.9, else "patrol".
// More robust: choose the branch whose canonical seek output is closer to the
// NN output by L2.
function decisionBranchMetrics(model, X, meta) {
    let hh = 0, hp = 0, ph = 0, pp = 0;
    let edgeRight = 0, edgeTotal = 0;
    const F = spec.F;
    for (let i = 0; i < X.length; i++) {
        const f = new Float32Array(X[i]);
        const nnOut = model.forward(f);
        const seekToBoid = spec.fastLimit(
            spec.fastSetMagnitude(f[F.DX1], f[F.DY1], spec.PREDATOR_MAX_SPEED)[0] - f[F.VX],
            spec.fastSetMagnitude(f[F.DX1], f[F.DY1], spec.PREDATOR_MAX_SPEED)[1] - f[F.VY],
            spec.PREDATOR_MAX_FORCE
        );
        const seekToAuto = spec.fastLimit(
            spec.fastSetMagnitude(f[F.DXA], f[F.DYA], spec.PREDATOR_MAX_SPEED)[0] - f[F.VX],
            spec.fastSetMagnitude(f[F.DXA], f[F.DYA], spec.PREDATOR_MAX_SPEED)[1] - f[F.VY],
            spec.PREDATOR_MAX_FORCE
        );
        const dh = (nnOut[0] - seekToBoid[0]) ** 2 + (nnOut[1] - seekToBoid[1]) ** 2;
        const da = (nnOut[0] - seekToAuto[0]) ** 2 + (nnOut[1] - seekToAuto[1]) ** 2;
        const nnBranch = dh < da ? 'hunt' : 'patrol';

        const ruleBranch = meta[i].branch;
        if (ruleBranch === 'hunt' && nnBranch === 'hunt') hh++;
        else if (ruleBranch === 'hunt' && nnBranch === 'patrol') hp++;
        else if (ruleBranch === 'patrol' && nnBranch === 'hunt') ph++;
        else pp++;

        if (meta[i].d1 >= spec.POLICY_R - 5 && meta[i].d1 <= spec.POLICY_R + 5) {
            edgeTotal++;
            if (ruleBranch === nnBranch) edgeRight++;
        }
    }
    const total = hh + hp + ph + pp;
    return {
        agreement: (hh + pp) / total,
        confusion: { hh, hp, ph, pp, total },
        rEdgeAgreement: edgeTotal ? edgeRight / edgeTotal : null,
        rEdgeTotal: edgeTotal,
    };
}

// ---- 5. Per-frame divergence (parallel NN-vs-rule oracles, shared seed). ----
function divergenceMetrics(model, seeds, frames) {
    const out = [];
    for (const seed of seeds) {
        const oR = new Oracle({ seed, numBoids: 120 });
        const oN = new Oracle({ seed, numBoids: 120, nnFn: (f) => model.forward(f) });
        const linfCurve = []; // sampled
        let firstBigDiverge = -1; // first frame >0.5 px
        let firstChaoticBreak = -1; // first frame where catch count differs
        let ruleCatches = [], nnCatches = [];
        let prevR = oR.sim.boids.length, prevN = oN.sim.boids.length;
        let finalLinf = 0;
        for (let i = 0; i < frames; i++) {
            const sR = oR.step(true);
            const sN = oN.step(true);
            const d = Math.max(Math.abs(sR.px - sN.px), Math.abs(sR.py - sN.py));
            if (d > finalLinf) finalLinf = d;
            if (i % 50 === 0) linfCurve.push([i, d]);
            if (firstBigDiverge < 0 && d > 0.5) firstBigDiverge = i;
            if (sR.boidCount < prevR) { ruleCatches.push(i); prevR = sR.boidCount; }
            if (sN.boidCount < prevN) { nnCatches.push(i); prevN = sN.boidCount; }
            if (firstChaoticBreak < 0 && (sR.boidCount !== sN.boidCount)) firstChaoticBreak = i;
        }
        // Longest common prefix of catch frame sequences.
        let lcp = 0;
        while (lcp < ruleCatches.length && lcp < nnCatches.length && ruleCatches[lcp] === nnCatches[lcp]) lcp++;
        out.push({
            seed, frames,
            firstBigDivergeFrame: firstBigDiverge,
            firstChaoticBreakFrame: firstChaoticBreak,
            finalLinf,
            linfCurve,
            ruleCatchCount: ruleCatches.length,
            nnCatchCount: nnCatches.length,
            catchLCP: lcp,
            firstCatchDiff: lcp < Math.min(ruleCatches.length, nnCatches.length)
                ? { idx: lcp, rule: ruleCatches[lcp], nn: nnCatches[lcp] }
                : null,
        });
    }
    return out;
}

// ---- 6. Aggregate behavioural stats over the same divergence runs. ----
function summarizeBehavior(model, seeds, frames) {
    const ruleStats = [];
    const nnStats = [];
    for (const seed of seeds) {
        const oR = new Oracle({ seed, numBoids: 120 });
        const oN = new Oracle({ seed, numBoids: 120, nnFn: (f) => model.forward(f) });
        const rR = { speeds: [], huntCount: 0, total: 0, finalSize: 0, catches: 0 };
        const rN = { speeds: [], huntCount: 0, total: 0, finalSize: 0, catches: 0 };
        let prevR = oR.sim.boids.length, prevN = oN.sim.boids.length;
        for (let i = 0; i < frames; i++) {
            const sR = oR.step(true);
            const sN = oN.step(true);
            const sp = Math.sqrt(sR.vx * sR.vx + sR.vy * sR.vy);
            const sp2 = Math.sqrt(sN.vx * sN.vx + sN.vy * sN.vy);
            rR.speeds.push(sp); rN.speeds.push(sp2);
            const f = oR.sim.predator._lastFeatures;
            if (f) {
                if (f[spec.F.D1] < spec.POLICY_R) rR.huntCount++;
                rR.total++;
            }
            const fn = oN.sim.predator._lastFeatures;
            if (fn) {
                if (fn[spec.F.D1] < spec.POLICY_R) rN.huntCount++;
                rN.total++;
            }
            if (sR.boidCount < prevR) { rR.catches++; prevR = sR.boidCount; }
            if (sN.boidCount < prevN) { rN.catches++; prevN = sN.boidCount; }
            rR.finalSize = sR.size; rN.finalSize = sN.size;
        }
        const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
        const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
        ruleStats.push({
            seed, catches: rR.catches, meanSpeed: mean(rR.speeds), stdSpeed: std(rR.speeds),
            huntFrac: rR.total ? rR.huntCount / rR.total : null, finalSize: rR.finalSize,
        });
        nnStats.push({
            seed, catches: rN.catches, meanSpeed: mean(rN.speeds), stdSpeed: std(rN.speeds),
            huntFrac: rN.total ? rN.huntCount / rN.total : null, finalSize: rN.finalSize,
        });
    }
    function summarize(arr, key) {
        const vs = arr.map(o => o[key]).filter(v => v != null);
        return { mean: vs.reduce((s, v) => s + v, 0) / vs.length, n: vs.length };
    }
    return {
        perSeed: { rule: ruleStats, nn: nnStats },
        summary: {
            catches:   { rule: summarize(ruleStats, 'catches'),   nn: summarize(nnStats, 'catches') },
            meanSpeed: { rule: summarize(ruleStats, 'meanSpeed'), nn: summarize(nnStats, 'meanSpeed') },
            huntFrac:  { rule: summarize(ruleStats, 'huntFrac'),  nn: summarize(nnStats, 'huntFrac') },
            finalSize: { rule: summarize(ruleStats, 'finalSize'), nn: summarize(nnStats, 'finalSize') },
        },
    };
}

// ---- main ----
function main() {
    const args = parseArgs(process.argv);
    const tStart = Date.now();
    process.stdout.write(JSON.stringify({ phase: 'start', args }) + '\n');

    const { model, info, nnFn } = loadNN(args.weights);

    process.stdout.write(JSON.stringify({ phase: 'collect_states', testSeeds: args.testSeeds }) + '\n');
    // Use a fraction of seeds to hit the requested testStates count.
    const framesPerSeed = Math.ceil(args.testStates / args.testSeeds.length);
    const { X, Y, meta } = collectHeldOutStates(args.testSeeds, framesPerSeed);
    process.stdout.write(JSON.stringify({ phase: 'collected', n: X.length }) + '\n');

    process.stdout.write(JSON.stringify({ phase: 'regression' }) + '\n');
    const regression = regressionMetrics(model, X, Y, meta);

    process.stdout.write(JSON.stringify({ phase: 'decision' }) + '\n');
    const decision = decisionBranchMetrics(model, X, meta);

    process.stdout.write(JSON.stringify({ phase: 'divergence' }) + '\n');
    const divSeeds = args.testSeeds.slice(0, args.divSeeds);
    const divergence = divergenceMetrics(model, divSeeds, args.frames);

    let behavior = null;
    if (args.behaviorSeeds > 0) {
        process.stdout.write(JSON.stringify({ phase: 'behavior' }) + '\n');
        const behaviorSeeds = args.testSeeds.slice(0, Math.min(args.behaviorSeeds, args.testSeeds.length));
        behavior = summarizeBehavior(model, behaviorSeeds, args.frames);
    }

    // Advisory gates -- I read these but my judgement is final.
    const gates = {
        mse:                  regression.mse < 1e-6,
        maxAbs:               regression.maxAbs < 1e-3,
        decisionAgreement:    decision.agreement > 0.9999,
        rEdgeAgreement:       decision.rEdgeAgreement == null || decision.rEdgeAgreement > 0.999,
        catchLCPAllSeeds:     divergence.every(d => d.catchLCP === d.ruleCatchCount),
    };
    if (behavior) {
        gates.meanSpeedClose = Math.abs(behavior.summary.meanSpeed.rule.mean - behavior.summary.meanSpeed.nn.mean) / behavior.summary.meanSpeed.rule.mean < 0.02;
        gates.huntFracClose = Math.abs((behavior.summary.huntFrac.rule.mean || 0) - (behavior.summary.huntFrac.nn.mean || 0)) < 0.02;
    }
    gates.allPassed = Object.values(gates).every(v => v === true);

    const report = {
        startedAt: new Date(tStart).toISOString(),
        elapsedMs: Date.now() - tStart,
        args,
        model: info,
        gates,
        metrics: { regression, decision, divergence, behavior },
    };

    ensureDir(args.report);
    fs.writeFileSync(args.report, JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify({ phase: 'done', elapsedMs: report.elapsedMs, allPassed: gates.allPassed, reportPath: args.report }) + '\n');
}

if (require.main === module) main();

module.exports = { collectHeldOutStates, regressionMetrics, decisionBranchMetrics, divergenceMetrics, summarizeBehavior, loadNN };
