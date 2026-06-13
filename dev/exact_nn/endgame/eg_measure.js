// eg_measure.js — L1e NN-share preview: run egboidPick on endgame commit records,
// compare its argmin egBoid to prod's logged egIdx, emit {margin, agree, n, cell}
// (predicted scan-t margin in FRAMES) for verifier/tau_calibrate.js, and report the
// egBoid agreement + a coverage table. The D4 twin of validate_student.js.
//
//   node eg_measure.js --weights eg_weights.json --data data_eg [--max N]
//   WRITE_CALIB=calib.json node eg_measure.js ...     # also dump the calib record
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { loadEgStudent } = require('./egboidPick.js');
const { egPick } = require('./eg_scan.js');

function parseArgs() {
    const a = { weights: path.join(__dirname, 'eg_weights.json'),
        data: path.join(__dirname, 'data_eg'), max: Infinity };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--weights') a.weights = process.argv[++i];
        else if (k === '--data') a.data = process.argv[++i];
        else if (k === '--max') a.max = +process.argv[++i];
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

function main() {
    const a = parseArgs();
    const pick = loadEgStudent(a.weights);
    let n = 0, agree = 0, sole = 0, contested = 0, contestedAgree = 0, egDerivedMis = 0;
    const calib = [];
    for (const f of fs.readdirSync(a.data).filter(x => x.endsWith('.commits.jsonl.gz')).sort()) {
        for (const line of zlib.gunzipSync(fs.readFileSync(path.join(a.data, f))).toString().split('\n')) {
            if (!line || n >= a.max) continue;
            const r = JSON.parse(line); n++;
            // independent prod egBoid (cross-check vs logged)
            const p = egPick(r.px, r.py, r.boids, r.W, r.Hc);
            if (p.egIdx !== r.egIdx) egDerivedMis++;
            const snap = { px: r.px, py: r.py, bx: r.boids.map(b => b.x), by: r.boids.map(b => b.y),
                bvx: r.boids.map(b => b.vx), bvy: r.boids.map(b => b.vy) };
            const res = pick(snap, { W: r.W, Hc: r.Hc });
            const ok = res.egIdx === r.egIdx;
            if (ok) agree++;
            const reachable = r.boids.filter(b => b.t != null).length;
            if (reachable >= 2) { contested++; if (ok) contestedAgree++; } else sole++;
            calib.push({ margin: Number.isFinite(res.margin) ? res.margin : null, agree: ok, n: r.n || r.boids.length, cell: r.cell });
        }
    }
    const out = { commits: n, egBoid_agree: +(agree / n).toFixed(4),
        sole_reachable: sole, contested, contested_agree: +(contestedAgree / Math.max(contested, 1)).toFixed(4),
        egDerivedMismatch: egDerivedMis };
    console.log(JSON.stringify(out, null, 1));
    if (process.env.WRITE_CALIB) { fs.writeFileSync(process.env.WRITE_CALIB, JSON.stringify(calib)); process.stderr.write('wrote calib ' + process.env.WRITE_CALIB + '\n'); }
}
main();
