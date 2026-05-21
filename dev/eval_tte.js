// End-to-end time-to-extinction (TTE) evaluation for any predator policy.
// This is the SINGLE SOURCE OF TRUTH for the RL search: every candidate
// policy gets scored by its mean TTE across a fixed set of held-out seeds.
// Lower TTE = predator catches all boids faster = better.
//
//   node dev/eval_tte.js [weights.json] [--seeds N] [--maxFrames F]
//                        [--numBoids B] [--workers W] [--policy null|random|weights]
//
// Defaults: js/predator_weights.json, 16 seeds (100..115), 12000 maxFrames,
// 120 boids, 4 workers.

'use strict';

const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// Lazy require so workers (which inherit __filename) don't try to spin up an
// Oracle until needed. The main thread requires these eagerly anyway.
let Oracle, loadModel;

// -----------------------------------------------------------------------
// Single-seed run. Constructs an Oracle with the given nnFn and steps until
// either extinction (all boids eaten) or maxFrames. Returns a struct with
// TTE, the list of frame numbers at which catches happened, and a flag for
// whether extinction was actually reached.
function evalSeed(nnFn, seed, maxFrames, numBoids, autoTargetMode) {
    if (!Oracle) Oracle = require('./oracle').Oracle;
    const oracle = new Oracle({ seed, numBoids, nnFn, autoTargetMode: autoTargetMode || 'random' });
    const initialBoids = oracle.sim.boids.length;
    let prevCount = initialBoids;
    const catchFrames = [];

    for (let f = 0; f < maxFrames; f++) {
        oracle.step(false);
        const count = oracle.sim.boids.length;
        while (count < prevCount) {
            catchFrames.push(f);
            prevCount--;
        }
        if (count === 0) {
            return {
                seed,
                tte: f + 1,
                catches: catchFrames.length,
                catchFrames,
                extinct: true,
                remaining: 0,
            };
        }
    }
    return {
        seed,
        tte: maxFrames,
        catches: catchFrames.length,
        catchFrames,
        extinct: false,
        remaining: oracle.sim.boids.length,
    };
}

// -----------------------------------------------------------------------
// Build an nnFn from a "policy spec":
//   { kind: 'weights', path: 'js/predator_weights.json' }   -- shipped/learned NN
//   { kind: 'null' }                                          -- always [0, 0]
//   { kind: 'random' }                                        -- random direction at MAX_FORCE
//
// In worker context this is called inside the worker; in main context only
// for single-threaded runs.
function buildNNFn(spec) {
    if (spec.kind === 'rule') {
        // Pure analytical rule: head toward nearest boid in range, else
        // toward autoTarget. Useful for distinguishing distillation loss
        // from rule-level limits when we change autoTargetMode.
        const rulePolicy = require('./policy_spec').rulePolicy;
        return (features) => rulePolicy(features);
    }
    if (spec.kind === 'weights') {
        if (!loadModel) loadModel = require('../js/predator_nn').loadModel;
        const json = JSON.parse(fs.readFileSync(spec.path, 'utf8'));
        const model = loadModel(json);
        return (features) => model.forward(features);
    }
    if (spec.kind === 'null') {
        const out = new Float32Array([0, 0]);
        return (_features) => out;
    }
    if (spec.kind === 'random') {
        // Match the predator's force cap so the sanity check stresses *direction*,
        // not magnitude.
        const PREDATOR_MAX_FORCE = 0.05;
        // Deterministic "random" using mulberry32 seeded once per process.
        let state = 0x9E3779B9 >>> 0;
        const rand = function () {
            state = (state + 0x6D2B79F5) >>> 0;
            let t = state;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        const out = new Float32Array(2);
        return (_features) => {
            const angle = rand() * 2 * Math.PI;
            out[0] = Math.cos(angle) * PREDATOR_MAX_FORCE;
            out[1] = Math.sin(angle) * PREDATOR_MAX_FORCE;
            return out;
        };
    }
    throw new Error('unknown policy spec: ' + JSON.stringify(spec));
}

// -----------------------------------------------------------------------
// Worker entry. Receives a list of seeds + the policy spec + run params,
// returns a list of evalSeed results.
if (!isMainThread) {
    const { policySpec, seeds, maxFrames, numBoids, autoTargetMode } = workerData;
    const nnFn = buildNNFn(policySpec);
    const out = seeds.map(s => evalSeed(nnFn, s, maxFrames, numBoids, autoTargetMode));
    parentPort.postMessage(out);
    return;  // unreachable; node exits after postMessage on worker end
}

// -----------------------------------------------------------------------
// Top-level eval. Distributes seeds across workers, gathers results,
// computes aggregate stats. Lower-is-better metric: meanTTE.
async function evalPolicy(policySpec, opts) {
    opts = opts || {};
    const seeds = opts.seeds || Array.from({ length: 16 }, (_, i) => 100 + i);
    const maxFrames = opts.maxFrames || 12000;
    const numBoids = opts.numBoids || 120;
    const workers = Math.min(opts.workers || 4, seeds.length);
    const autoTargetMode = opts.autoTargetMode || 'random';

    // Chunk seeds across workers.
    const chunkSize = Math.ceil(seeds.length / workers);
    const chunks = [];
    for (let i = 0; i < seeds.length; i += chunkSize) {
        chunks.push(seeds.slice(i, i + chunkSize));
    }

    const t0 = Date.now();
    const allResults = await Promise.all(chunks.map(chunkSeeds =>
        new Promise((resolve, reject) => {
            const w = new Worker(__filename, {
                workerData: { policySpec, seeds: chunkSeeds, maxFrames, numBoids, autoTargetMode },
            });
            w.on('message', resolve);
            w.on('error', reject);
            w.on('exit', code => { if (code !== 0) reject(new Error('worker exit ' + code)); });
        })
    ));
    const elapsedMs = Date.now() - t0;

    const results = [].concat(...allResults);
    const ttes = results.map(r => r.tte);
    const catches = results.map(r => r.catches);
    const extinctCount = results.filter(r => r.extinct).length;
    const mean = ttes.reduce((a, b) => a + b, 0) / ttes.length;
    const variance = ttes.reduce((a, b) => a + (b - mean) ** 2, 0) / ttes.length;
    const sorted = ttes.slice().sort((a, b) => a - b);
    const median = sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[(sorted.length - 1) / 2];

    return {
        policySpec,
        seedsRun: seeds.length,
        maxFrames,
        numBoids,
        elapsedMs,
        meanTTE: mean,
        stdTTE: Math.sqrt(variance),
        medianTTE: median,
        minTTE: sorted[0],
        maxTTE: sorted[sorted.length - 1],
        meanCatches: catches.reduce((a, b) => a + b, 0) / catches.length,
        extinctRate: extinctCount / results.length,
        perSeed: results.map(r => ({
            seed: r.seed, tte: r.tte, catches: r.catches, extinct: r.extinct, remaining: r.remaining,
        })),
    };
}

// -----------------------------------------------------------------------
// CLI.
function parseArgs(argv) {
    const args = {
        weights: 'js/predator_weights.json',
        numSeeds: 16,
        seedStart: 100,
        maxFrames: 12000,
        numBoids: 120,
        workers: 4,
        policy: null,            // 'null' | 'random' — overrides weights when set
        report: null,            // optional path to dump full JSON
        autoTarget: 'random',    // 'random' | 'nearest_boid' | 'flock_centroid' | 'farthest_in_K'
    };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (!a.startsWith('--') && i === 2) args.weights = a;
        else if (a === '--seeds') args.numSeeds = +argv[++i];
        else if (a === '--seedStart') args.seedStart = +argv[++i];
        else if (a === '--maxFrames') args.maxFrames = +argv[++i];
        else if (a === '--numBoids') args.numBoids = +argv[++i];
        else if (a === '--workers') args.workers = +argv[++i];
        else if (a === '--policy') args.policy = argv[++i];
        else if (a === '--report') args.report = argv[++i];
        else if (a === '--autoTarget') args.autoTarget = argv[++i];
    }
    return args;
}

if (require.main === module) {
    const args = parseArgs(process.argv);
    const seeds = Array.from({ length: args.numSeeds }, (_, i) => args.seedStart + i);
    const policySpec = args.policy === 'null' ? { kind: 'null' }
                     : args.policy === 'random' ? { kind: 'random' }
                     : args.policy === 'rule' ? { kind: 'rule' }
                     : { kind: 'weights', path: path.resolve(args.weights) };
    evalPolicy(policySpec, {
        seeds, maxFrames: args.maxFrames, numBoids: args.numBoids, workers: args.workers,
        autoTargetMode: args.autoTarget,
    }).then(summary => {
        const compact = {
            policy: policySpec.kind === 'weights' ? path.basename(policySpec.path) : policySpec.kind,
            autoTarget: args.autoTarget,
            seedsRun: summary.seedsRun,
            maxFrames: summary.maxFrames,
            numBoids: summary.numBoids,
            meanTTE: +summary.meanTTE.toFixed(1),
            stdTTE: +summary.stdTTE.toFixed(1),
            medianTTE: summary.medianTTE,
            minTTE: summary.minTTE,
            maxTTE: summary.maxTTE,
            extinctRate: +summary.extinctRate.toFixed(3),
            meanCatches: +summary.meanCatches.toFixed(1),
            elapsedSec: +(summary.elapsedMs / 1000).toFixed(1),
        };
        console.log(JSON.stringify(compact, null, 2));
        if (args.report) {
            fs.mkdirSync(path.dirname(args.report), { recursive: true });
            fs.writeFileSync(args.report, JSON.stringify(summary, null, 2));
        }
    }).catch(e => { console.error(e); process.exit(1); });
}

module.exports = { evalSeed, evalPolicy, buildNNFn };
