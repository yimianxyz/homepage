// Generate the (features, target_steering) training dataset by running the
// rule-based oracle for a list of seeds and capturing the JS rule's actual
// output each frame.
//
//   node dev/gen_dataset.js --seeds 0-19 --framesPerSeed 5000 --out dev/dataset.bin
//
// Output:
//   dev/dataset.bin     : float32 array, N rows of FEATURE_DIM+2 = 14 floats.
//   dev/dataset.meta.json : metadata for the trainer.

'use strict';

const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

const spec = require('./policy_spec');

function parseRange(s) {
    if (!s) return null;
    const out = [];
    for (const part of s.split(',')) {
        const m = part.match(/^(\d+)-(\d+)$/);
        if (m) {
            const a = +m[1], b = +m[2];
            for (let i = a; i <= b; i++) out.push(i);
        } else {
            out.push(+part);
        }
    }
    return out;
}

function parseArgs(argv) {
    const args = {
        seeds: parseRange('0-19'),
        framesPerSeed: 5000,
        out: 'dev/dataset.bin',
        workers: 4,
        numBoids: 120,
    };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--seeds') args.seeds = parseRange(argv[++i]);
        else if (a === '--framesPerSeed') args.framesPerSeed = +argv[++i];
        else if (a === '--out') args.out = argv[++i];
        else if (a === '--workers') args.workers = +argv[++i];
        else if (a === '--numBoids') args.numBoids = +argv[++i];
    }
    return args;
}

// ---- Worker: process a list of seeds and return the buffer. ----
function workerMain() {
    const { seeds, framesPerSeed, numBoids } = workerData;
    const { Oracle } = require('./oracle');
    const FD = spec.FEATURE_DIM;
    const ROW = FD + 2;
    const buf = new Float32Array(seeds.length * framesPerSeed * ROW);
    let wIdx = 0;
    let validRows = 0;
    for (const seed of seeds) {
        const o = new Oracle({ seed, numBoids });
        for (let i = 0; i < framesPerSeed; i++) {
            o.step(false);
            const f = o.sim.predator._lastFeatures;
            const y = o.sim.predator._lastOutput;
            if (!f || !y) { wIdx += ROW; continue; }
            for (let k = 0; k < FD; k++) buf[wIdx + k] = f[k];
            buf[wIdx + FD] = y[0];
            buf[wIdx + FD + 1] = y[1];
            wIdx += ROW;
            validRows++;
        }
        parentPort.postMessage({ kind: 'progress', seed, validRows });
    }
    // Trim invalid rows if any (rare).
    parentPort.postMessage({ kind: 'done', buffer: buf.buffer, validRows, rowsWritten: wIdx / ROW }, [buf.buffer]);
}

if (!isMainThread) {
    workerMain();
} else {
    runMain();
}

function chunk(arr, n) {
    const out = []; const k = Math.ceil(arr.length / n);
    for (let i = 0; i < arr.length; i += k) out.push(arr.slice(i, i + k));
    return out;
}

async function runMain() {
    const args = parseArgs(process.argv);
    const FD = spec.FEATURE_DIM;
    const ROW = FD + 2;

    const totalRows = args.seeds.length * args.framesPerSeed;
    process.stdout.write(JSON.stringify({ phase: 'start', seeds: args.seeds.length, framesPerSeed: args.framesPerSeed, totalRows, workers: args.workers }) + '\n');
    const tStart = Date.now();

    const seedChunks = chunk(args.seeds, args.workers);
    const promises = seedChunks.map(seedSubset => new Promise((resolve, reject) => {
        const w = new Worker(__filename, {
            workerData: { seeds: seedSubset, framesPerSeed: args.framesPerSeed, numBoids: args.numBoids },
        });
        w.on('message', msg => {
            if (msg.kind === 'progress') {
                process.stdout.write(JSON.stringify({ phase: 'progress', seed: msg.seed, validRows: msg.validRows }) + '\n');
            } else if (msg.kind === 'done') {
                resolve({ buffer: msg.buffer, validRows: msg.validRows, rowsWritten: msg.rowsWritten });
            }
        });
        w.on('error', reject);
        w.on('exit', code => { if (code !== 0) reject(new Error('worker exit ' + code)); });
    }));

    const chunks = await Promise.all(promises);
    // Concatenate buffers into one big Float32Array, then write to disk.
    const totalRowsActual = chunks.reduce((s, c) => s + c.rowsWritten, 0);
    const merged = new Float32Array(totalRowsActual * ROW);
    let off = 0;
    for (const c of chunks) {
        const part = new Float32Array(c.buffer);
        merged.set(part.subarray(0, c.rowsWritten * ROW), off);
        off += c.rowsWritten * ROW;
    }
    fs.mkdirSync(path.dirname(args.out), { recursive: true });
    fs.writeFileSync(args.out, Buffer.from(merged.buffer, merged.byteOffset, merged.byteLength));
    const metaPath = args.out.replace(/\.bin$/, '.meta.json');
    const meta = {
        n: totalRowsActual,
        featureDim: FD,
        outputDim: 2,
        rowFloats: ROW,
        seeds: args.seeds,
        framesPerSeed: args.framesPerSeed,
        numBoids: args.numBoids,
        generatedAt: new Date().toISOString(),
        elapsedMs: Date.now() - tStart,
    };
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
    process.stdout.write(JSON.stringify({ phase: 'done', n: totalRowsActual, out: args.out, metaPath, elapsedMs: meta.elapsedMs }) + '\n');
}
