// Population-based Evolution Strategies for the predator policy.
//
// Each generation samples K candidate directions, evaluates all K on the
// same fixed seed set, and accepts the BEST candidate if it beats the
// current best by z>zThresh (paired delta vs current best). This is
// strictly more efficient than hill climb when the true improvement
// signal is rare (most candidates worse, a few much better), because
// hill climb only takes 1 sample per try and frequently misses the
// rare wins.
//
// With antithetic pairing (--antithetic), each direction is evaluated
// at θ+σε AND θ-σε, doubling samples per direction at no extra random
// perturbation cost. Useful for symmetry-breaking searches.
//
//   node dev/rl_pop_es.js --base js/predator_weights.json --sigma 0.10 \
//       --K 4 --gens 5 --seeds 16 --zThresh 2.0 \
//       --out dev/weights/pop_v1.json --log dev/reports/pop_v1.log

'use strict';

const fs = require('fs');
const path = require('path');
const { evalPolicy } = require('./eval_tte');

function parseArgs(argv) {
    const a = {
        base: 'js/predator_weights.json',
        sigma: 0.10,
        K: 4,
        gens: 5,
        seeds: 16,
        seedStart: 100,
        maxFrames: 5000,
        numBoids: 120,
        workers: 4,
        out: 'dev/weights/pop_candidate.json',
        log: null,
        rngSeed: 1234,
        zThresh: 2.0,
        antithetic: false,
    };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--base') a.base = argv[++i];
        else if (k === '--sigma') a.sigma = +argv[++i];
        else if (k === '--K') a.K = +argv[++i];
        else if (k === '--gens') a.gens = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--numBoids') a.numBoids = +argv[++i];
        else if (k === '--workers') a.workers = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--log') a.log = argv[++i];
        else if (k === '--rngSeed') a.rngSeed = +argv[++i];
        else if (k === '--zThresh') a.zThresh = +argv[++i];
        else if (k === '--antithetic') a.antithetic = true;
    }
    return a;
}

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
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function cloneWeights(w) {
    return JSON.parse(JSON.stringify(w));
}

// Generate a random perturbation direction (one Gaussian sample per param).
// Returned as a flat array; applied via applyDirection.
function sampleDirection(weights, rand) {
    const dir = [];
    for (const L of weights.layers) {
        const dW = new Array(L.W.length);
        for (let i = 0; i < L.W.length; i++) dW[i] = gaussian(rand);
        const db = new Array(L.b.length);
        for (let i = 0; i < L.b.length; i++) db[i] = gaussian(rand);
        dir.push({ dW, db });
    }
    return dir;
}

// Apply σ·direction to weights (in-place); pass sign=-1 for antithetic.
function applyDirection(weights, direction, sigma, sign) {
    for (let li = 0; li < weights.layers.length; li++) {
        const L = weights.layers[li];
        const D = direction[li];
        for (let i = 0; i < L.W.length; i++) L.W[i] += sign * sigma * D.dW[i];
        for (let i = 0; i < L.b.length; i++) L.b[i] += sign * sigma * D.db[i];
    }
}

async function evalCandidate(weightsObj, seeds, maxFrames, numBoids, workers, tmpPath) {
    fs.writeFileSync(tmpPath, JSON.stringify(weightsObj));
    return evalPolicy(
        { kind: 'weights', path: path.resolve(tmpPath) },
        { seeds, maxFrames, numBoids, workers }
    );
}

function pairedStats(candPerSeed, bestPerSeed) {
    const n = candPerSeed.length;
    const deltas = candPerSeed.map((c, i) => c - bestPerSeed[i]);
    const dMean = deltas.reduce((a, b) => a + b, 0) / n;
    const dVar = deltas.reduce((a, b) => a + (b - dMean) ** 2, 0) / n;
    const dSE = Math.sqrt(dVar / n);
    const z = dSE > 0 ? dMean / dSE : 0;
    return { deltas, dMean, dSE, z };
}

async function main() {
    const args = parseArgs(process.argv);
    const seeds = Array.from({ length: args.seeds }, (_, i) => args.seedStart + i);
    const rand = mulberry32(args.rngSeed);

    const tmpPath = '/tmp/rl_pop_candidate_weights.json';
    const baseWeights = JSON.parse(fs.readFileSync(args.base, 'utf8'));
    const totalParams = baseWeights.layers.reduce((s, L) => s + L.W.length + L.b.length, 0);

    const logLines = [];
    const log = (obj) => {
        const line = JSON.stringify(obj);
        console.log(line);
        logLines.push(line);
        if (args.log) fs.writeFileSync(args.log, logLines.join('\n') + '\n');
    };

    log({
        phase: 'start', base: args.base, sigma: args.sigma, K: args.K,
        gens: args.gens, seeds, maxFrames: args.maxFrames, numBoids: args.numBoids,
        totalParams, zThresh: args.zThresh, antithetic: args.antithetic,
    });

    // Score the baseline once.
    const baseSummary = await evalCandidate(baseWeights, seeds, args.maxFrames, args.numBoids, args.workers, tmpPath);
    const basePerSeed = baseSummary.perSeed.map(p => p.catches);
    const baseMean = baseSummary.meanCatches;
    log({
        phase: 'baseline',
        meanCatches: +baseMean.toFixed(3),
        perSeed: basePerSeed,
        elapsedSec: +(baseSummary.elapsedMs / 1000).toFixed(1),
    });

    let bestWeights = baseWeights;
    let bestMean = baseMean;
    let bestPerSeed = basePerSeed;

    for (let g = 1; g <= args.gens; g++) {
        // Sample K directions; evaluate each candidate.
        const candidates = [];
        for (let k = 0; k < args.K; k++) {
            const dir = sampleDirection(bestWeights, rand);
            // Positive candidate.
            const candPos = cloneWeights(bestWeights);
            applyDirection(candPos, dir, args.sigma, +1);
            const sPos = await evalCandidate(candPos, seeds, args.maxFrames, args.numBoids, args.workers, tmpPath);
            const posPerSeed = sPos.perSeed.map(p => p.catches);
            const posStats = pairedStats(posPerSeed, bestPerSeed);
            candidates.push({
                gen: g, k, sign: '+', weights: candPos, perSeed: posPerSeed,
                meanCatches: sPos.meanCatches, ...posStats,
                elapsedSec: +(sPos.elapsedMs / 1000).toFixed(1),
            });
            log({
                phase: 'cand', gen: g, k, sign: '+',
                meanCatches: +sPos.meanCatches.toFixed(3),
                dMean: +posStats.dMean.toFixed(3),
                dSE: +posStats.dSE.toFixed(3),
                z: +posStats.z.toFixed(3),
                elapsedSec: +(sPos.elapsedMs / 1000).toFixed(1),
            });
            if (args.antithetic) {
                // Negative candidate (same direction, opposite sign).
                const candNeg = cloneWeights(bestWeights);
                applyDirection(candNeg, dir, args.sigma, -1);
                const sNeg = await evalCandidate(candNeg, seeds, args.maxFrames, args.numBoids, args.workers, tmpPath);
                const negPerSeed = sNeg.perSeed.map(p => p.catches);
                const negStats = pairedStats(negPerSeed, bestPerSeed);
                candidates.push({
                    gen: g, k, sign: '-', weights: candNeg, perSeed: negPerSeed,
                    meanCatches: sNeg.meanCatches, ...negStats,
                    elapsedSec: +(sNeg.elapsedMs / 1000).toFixed(1),
                });
                log({
                    phase: 'cand', gen: g, k, sign: '-',
                    meanCatches: +sNeg.meanCatches.toFixed(3),
                    dMean: +negStats.dMean.toFixed(3),
                    dSE: +negStats.dSE.toFixed(3),
                    z: +negStats.z.toFixed(3),
                    elapsedSec: +(sNeg.elapsedMs / 1000).toFixed(1),
                });
            }
        }

        // Pick the best candidate (highest z) and accept if z > zThresh.
        candidates.sort((a, b) => b.z - a.z);
        const winner = candidates[0];
        const accept = winner.z > args.zThresh && winner.dMean > 0;
        const entry = {
            phase: 'gen', gen: g,
            winnerMean: +winner.meanCatches.toFixed(3),
            winnerDMean: +winner.dMean.toFixed(3),
            winnerSE: +winner.dSE.toFixed(3),
            winnerZ: +winner.z.toFixed(3),
            accept,
            bestMean: +bestMean.toFixed(3),
        };
        if (accept) {
            bestWeights = winner.weights;
            bestMean = winner.meanCatches;
            bestPerSeed = winner.perSeed;
            entry.bestMean = +bestMean.toFixed(3);
            fs.mkdirSync(path.dirname(args.out), { recursive: true });
            fs.writeFileSync(args.out, JSON.stringify(bestWeights, null, 2));
        }
        log(entry);
    }

    if (bestWeights === baseWeights) {
        fs.mkdirSync(path.dirname(args.out), { recursive: true });
        fs.writeFileSync(args.out, JSON.stringify(bestWeights, null, 2));
    }

    log({
        phase: 'done',
        bestMean: +bestMean.toFixed(3),
        deltaVsBaseline: +(bestMean - baseMean).toFixed(3),
        deltaVsBaselinePct: +(100 * (bestMean - baseMean) / baseMean).toFixed(2),
        outWeights: args.out,
    });
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
