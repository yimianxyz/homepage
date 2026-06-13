// independent_margin_check.js — ADVERSARIAL cross-check of the logged dmargin.
//
// oracle_logger.recheckPlan re-runs labelAndDedup() inline, but that's the SAME
// function that produced the value — a bug in labelAndDedup would pass its own
// recheck (circular). This tool recomputes the coordinate-dedup top1−top2 margin
// and the canonical label with a DELIBERATELY DIFFERENT implementation
// (string-keyed grouping, separate winner-group resolution, independent argmax)
// straight from the raw record fields (cands/score/bi/pidx), and diffs against
// the logged lab/dmargin/nDistinct/margin/bi over EVERY record in the data dir.
//
// Known fidelity caveat it also MEASURES: cands are logged as plain JSON numbers,
// so a -0 candidate coordinate round-trips to +0 — grouping here (from JSON)
// could differ from the log-time f64hex grouping ONLY for -0 coords. The tool
// counts cand coords that are exactly 0 to bound this; if any mismatch is found
// it is reported with the record so we can tell a real bug from the -0 artifact.
//
//   node dev/exact_nn/independent_margin_check.js [--data dir] [--max N]
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

function parseArgs(argv) {
    const a = { data: path.join(__dirname, 'data'), max: Infinity };
    for (let i = 2; i < argv.length; i++) {
        if (argv[i] === '--data') a.data = argv[++i];
        else if (argv[i] === '--max') a.max = +argv[++i];
        else throw new Error('unknown arg ' + argv[i]);
    }
    return a;
}

// INDEPENDENT recompute (string keys, not f64hex; explicit winner-group logic).
function independent(rec) {
    const cands = rec.cands;                         // [[x,y]...]
    const score = rec.score.map(v => v === null ? -Infinity : v);
    const bi = rec.bi;
    // 1) independent argmax (first strict max) — must equal logged bi
    let am = 0, best = -Infinity;
    for (let k = 0; k < score.length; k++) { if (score[k] > best) { best = score[k]; am = k; } }
    // 2) group by exact coord string; canonical index = lowest in group
    const groups = new Map();                          // key -> {minIdx, max}
    for (let k = 0; k < cands.length; k++) {
        const key = cands[k][0] + '|' + cands[k][1];   // string key (independent of f64hex)
        const g = groups.get(key);
        if (!g) groups.set(key, { minIdx: k, max: score[k] });
        else { if (k < g.minIdx) g.minIdx = k; if (score[k] > g.max) g.max = score[k]; }
    }
    const winKey = cands[bi][0] + '|' + cands[bi][1];
    const ti = groups.get(winKey).minIdx;              // canonical label index
    // 3) dedup margin: winner group max − best OTHER group max
    let other = -Infinity;
    for (const [key, g] of groups) if (key !== winKey && g.max > other) other = g.max;
    const winMax = groups.get(winKey).max;
    const dmargin = other === -Infinity ? Infinity : winMax - other;
    // 4) slot-level runner-up margin (independent of dedup)
    let ru = -Infinity;
    for (let k = 0; k < score.length; k++) if (k !== bi && score[k] > ru) ru = score[k];
    const margin = score[bi] - ru;
    return { am, ti, dmargin, nDistinct: groups.size, margin };
}

function* iterDecisions(dir) {
    for (const f of fs.readdirSync(dir).sort()) {
        if (!f.endsWith('.decisions.jsonl.gz')) continue;
        const buf = zlib.gunzipSync(fs.readFileSync(path.join(dir, f)));
        for (const line of buf.toString('utf8').split('\n')) { if (line) yield JSON.parse(line); }
    }
}

function main() {
    const opt = parseArgs(process.argv);
    let n = 0, mismatches = 0, zeroCoord = 0;
    const samples = [];
    for (const rec of iterDecisions(opt.data)) {
        if (n >= opt.max) break;
        n++;
        for (const c of rec.cands) { if (c[0] === 0) zeroCoord++; if (c[1] === 0) zeroCoord++; }
        const ind = independent(rec);
        const logDm = rec.dmargin === null ? Infinity : rec.dmargin;
        const probs = [];
        if (ind.am !== rec.bi) probs.push(`argmax ${ind.am}!=bi ${rec.bi}`);
        if (ind.ti !== rec.lab.ti) probs.push(`ti ${ind.ti}!=${rec.lab.ti}`);
        if (ind.dmargin !== logDm) probs.push(`dmargin ${ind.dmargin}!=${logDm}`);
        if (ind.nDistinct !== rec.nDistinct) probs.push(`nDistinct ${ind.nDistinct}!=${rec.nDistinct}`);
        if (ind.margin !== rec.margin) probs.push(`margin ${ind.margin}!=${rec.margin}`);
        // label coords: independent ti must point at the SAME coords the logger hashed
        const _dv = new DataView(new ArrayBuffer(8));
        const hex = x => { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); };
        if (hex(rec.cands[ind.ti][0]) !== rec.lab.tx || hex(rec.cands[ind.ti][1]) !== rec.lab.ty)
            probs.push('label coords differ');
        if (probs.length) {
            mismatches++;
            if (samples.length < 8) samples.push({ seed: rec.seed, cell: rec.cell, f: rec.f, N: rec.N, probs });
        }
    }
    console.log(JSON.stringify({ records: n, mismatches, zeroCandCoords: zeroCoord, samples }, null, 1));
    if (mismatches) { console.error(`FAIL: ${mismatches}/${n} records disagree with the logged margin/label`); process.exit(2); }
    console.log(`PASS: all ${n} records' dedup margin + label independently reproduced (zero-coord cands seen: ${zeroCoord}).`);
}
main();
