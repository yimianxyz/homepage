// eg_pack.js — pack endgame commit records into per-commit feature/label arrays
// for the L1e scan-t regressor. Features computed in JS (eg_features.js) so they
// are bit-identical to the deploy egboidPick (the D1 parity lesson). Emits a
// gzipped JSON the python trainer reads verbatim — NO python feature math.
//
// Per commit: boids padded to MAXEG=5; feat[5][NFEAT], label[5] (scan-t/100, null
// -> TMAX/100=14), mask[5], egIdx (prod's committed boid), margin (true 2nd-min -
// min scan-t over reachable, null if <2 reachable), n, cell, seed.
//
//   node eg_pack.js --data data_eg --out packed_eg.json.gz
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { egBoidFeatures, EG_NFEAT, TMAX } = require('./eg_features.js');
const { egPick } = require('./eg_scan.js');

const MAXEG = 5;

function parseArgs() {
    const a = { data: path.join(__dirname, 'data_eg'), out: path.join(__dirname, 'packed_eg.json.gz') };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--data') a.data = process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i];
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

function main() {
    const a = parseArgs();
    const dirs = a.data.split(',');   // multi-dir: tag each commit with its source index
    const feat = [], label = [], mask = [], egIdx = [], margin = [], ncol = [], cell = [], seed = [], src = [];
    let skipped = 0, egDerivedMismatch = 0;
    for (let di = 0; di < dirs.length; di++) {
      const dir = dirs[di];
      const files = fs.readdirSync(dir).filter(f => f.endsWith('.commits.jsonl.gz')).sort();
      for (const f of files) {
        for (const line of zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString().split('\n')) {
            if (!line) continue;
            const r = JSON.parse(line);
            const n = r.boids.length;
            if (n < 1 || n > MAXEG) { skipped++; continue; }
            // cross-check: derived egBoid must equal prod's logged egIdx (sound label)
            const p = egPick(r.px, r.py, r.boids, r.W, r.Hc);
            if (p.egIdx !== r.egIdx) egDerivedMismatch++;
            const fr = [], lb = [], mk = [];
            for (let i = 0; i < MAXEG; i++) {
                if (i < n) {
                    const b = r.boids[i];
                    fr.push(egBoidFeatures(r.px, r.py, b.x, b.y, b.vx, b.vy, r.W, r.Hc));
                    lb.push((b.t == null ? TMAX : b.t) / 100.0);
                    mk.push(1);
                } else { fr.push(new Array(EG_NFEAT).fill(0)); lb.push(TMAX / 100.0); mk.push(0); }
            }
            // true margin over reachable scan-t
            const reach = r.boids.map(b => b.t).filter(t => t != null).sort((x, y) => x - y);
            const mg = reach.length >= 2 ? reach[1] - reach[0] : null;
            feat.push(fr); label.push(lb); mask.push(mk); egIdx.push(r.egIdx);
            margin.push(mg); ncol.push(n); cell.push(r.cell); seed.push(r.seed); src.push(di);
        }
      }
    }
    const out = { nfeat: EG_NFEAT, maxeg: MAXEG, count: feat.length,
        feat, label, mask, egIdx, margin, n: ncol, cell, seed, src, dirs,
        egDerivedMismatch, skipped };
    fs.writeFileSync(a.out, zlib.gzipSync(JSON.stringify(out)));
    process.stderr.write(`packed ${feat.length} commits -> ${a.out}; egDerivedMismatch=${egDerivedMismatch} skipped(N>5 or 0)=${skipped}\n`);
}

main();
