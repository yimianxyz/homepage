// Evolution Strategies / hill-climbing search for a better predator policy.
// Starts from a baseline weights file, perturbs the layer weights/biases
// with Gaussian noise, evaluates each candidate via dev/eval_tte.js's
// catches-per-frame metric on a fixed seed set, and keeps the best.
//
//   node dev/rl_es.js --base js/predator_weights.json --sigma 0.05 \
//                     --tries 30 --seeds 8 --maxFrames 5000 \
//                     --out dev/weights/rl_v1.json \
//                     --log dev/reports/rl_v1.log
//
// Algorithm choice: simple random-search hill climb. ES with K candidates
// per generation needs K evals to estimate the gradient; hill climb needs
// 1 eval per try and is the Occam baseline. We can graduate to NES later
// if hill climb converges too slowly.

'use strict';

const fs = require('fs');
const path = require('path');
const { evalPolicy } = require('./eval_tte');

function parseArgs(argv) {
    const a = {
        base: 'js/predator_weights.json',
        sigma: 0.05,
        tries: 30,
        seeds: 8,
        seedStart: 100,
        maxFrames: 5000,
        numBoids: 120,
        workers: 4,
        out: 'dev/weights/rl_candidate.json',
        log: null,
        rngSeed: 1234,
        zThresh: 1.0,
    };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--base') a.base = argv[++i];
        else if (k === '--sigma') a.sigma = +argv[++i];
        else if (k === '--tries') a.tries = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--numBoids') a.numBoids = +argv[++i];
        else if (k === '--workers') a.workers = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--log') a.log = argv[++i];
        else if (k === '--rngSeed') a.rngSeed = +argv[++i];
        else if (k === '--zThresh') a.zThresh = +argv[++i];
    }
    return a;
}

// mulberry32 — same PRNG the sim uses, so the RL trainer is fully
// deterministic given --rngSeed.
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

// Box-Muller — turn two uniforms into one standard normal.
function gaussian(rand) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Deep-copy a weights JSON so each candidate has its own arrays.
function cloneWeights(w) {
    const out = JSON.parse(JSON.stringify(w));
    return out;
}

// Perturb a candidate in-place: add σ·𝒩(0,1) to every W and b entry.
// inputMean/inputStd/outputScale/clipMagnitude are FIXED — those are
// fingerprints of the feature-encoding pipeline, not learnable.
function perturbInPlace(weights, sigma, rand) {
    for (const L of weights.layers) {
        for (let i = 0; i < L.W.length; i++) L.W[i] += sigma * gaussian(rand);
        for (let i = 0; i < L.b.length; i++) L.b[i] += sigma * gaussian(rand);
    }
}

// One eval cycle: write weights to a temp file, run evalPolicy, return its
// meanCatches summary. We persist to disk so the worker threads can read
// the policy spec by path (worker_threads can't easily share Function refs).
async function evalCandidate(weightsObj, seeds, maxFrames, numBoids, workers, tmpPath) {
    fs.writeFileSync(tmpPath, JSON.stringify(weightsObj));
    const summary = await evalPolicy(
        { kind: 'weights', path: path.resolve(tmpPath) },
        { seeds, maxFrames, numBoids, workers }
    );
    return summary;
}

function median(arr) {
    const s = arr.slice().sort((a, b) => a - b);
    const n = s.length;
    return n % 2 === 0 ? (s[n / 2 - 1] + s[n / 2]) / 2 : s[(n - 1) / 2];
}

async function main() {
    const args = parseArgs(process.argv);
    const seeds = Array.from({ length: args.seeds }, (_, i) => args.seedStart + i);
    const rand = mulberry32(args.rngSeed);

    const tmpPath = '/tmp/rl_candidate_weights.json';
    const baseWeights = JSON.parse(fs.readFileSync(args.base, 'utf8'));
    const totalParams = baseWeights.layers.reduce((s, L) => s + L.W.length + L.b.length, 0);

    const logLines = [];
    const log = (obj) => {
        const line = JSON.stringify(obj);
        console.log(line);
        logLines.push(line);
        if (args.log) fs.writeFileSync(args.log, logLines.join('\n') + '\n');
    };

    log({ phase: 'start', base: args.base, sigma: args.sigma, tries: args.tries, seeds, maxFrames: args.maxFrames, numBoids: args.numBoids, totalParams, zThresh: args.zThresh });

    // 1. Score the baseline (so all subsequent perturbations are comparable).
    const baseSummary = await evalCandidate(baseWeights, seeds, args.maxFrames, args.numBoids, args.workers, tmpPath);
    const basePerSeed = baseSummary.perSeed.map(p => p.catches);
    const baseMean = baseSummary.meanCatches;
    log({ phase: 'baseline', meanCatches: +baseMean.toFixed(3), perSeed: basePerSeed, elapsedSec: +(baseSummary.elapsedMs / 1000).toFixed(1) });

    // 2. Hill-climb. Each try: perturb baseline by σ𝒩(0,1), eval, accept if
    //    the *mean of paired deltas* > 0 AND the improvement is statistically
    //    meaningful (delta_mean - delta_stderr > 0).
    let bestWeights = baseWeights;
    let bestMean = baseMean;
    let bestPerSeed = basePerSeed;

    for (let t = 1; t <= args.tries; t++) {
        const cand = cloneWeights(bestWeights);
        perturbInPlace(cand, args.sigma, rand);
        const s = await evalCandidate(cand, seeds, args.maxFrames, args.numBoids, args.workers, tmpPath);
        const candPerSeed = s.perSeed.map(p => p.catches);
        const candMean = s.meanCatches;
        // Paired delta (candidate − current best), same seeds.
        const deltas = candPerSeed.map((c, i) => c - bestPerSeed[i]);
        const dMean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
        const dVar = deltas.reduce((a, b) => a + (b - dMean) ** 2, 0) / deltas.length;
        const dSE = Math.sqrt(dVar / deltas.length);
        const z = dSE > 0 ? dMean / dSE : 0;
        // Accept if delta is positive AND statistically meaningful at the
        // configured z-threshold. Tighter z reduces false positives at the
        // cost of missing weak-but-real improvements.
        const accept = dMean > 0 && z > args.zThresh;
        const entry = {
            phase: 'try',
            t,
            candMean: +candMean.toFixed(3),
            dMean: +dMean.toFixed(3),
            dSE: +dSE.toFixed(3),
            z: +z.toFixed(3),
            accept,
            elapsedSec: +(s.elapsedMs / 1000).toFixed(1),
            bestMean: +bestMean.toFixed(3),
        };
        if (accept) {
            bestWeights = cand;
            bestMean = candMean;
            bestPerSeed = candPerSeed;
            entry.bestMean = +bestMean.toFixed(3);
            // Persist current best at every accept so we can pick it up if
            // the run is interrupted.
            fs.mkdirSync(path.dirname(args.out), { recursive: true });
            fs.writeFileSync(args.out, JSON.stringify(bestWeights, null, 2));
        }
        log(entry);
    }

    // Final write (in case nothing was accepted, we still emit the baseline).
    if (bestWeights === baseWeights) {
        // Nothing accepted — write the unchanged baseline to make pipelines simple.
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
