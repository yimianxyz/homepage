// predict_sdec.js — turn a measured student score-error σ into a predicted
// per-cell S_dec, by integrating σ against the ACTUAL logged margins (the audit's
// ask: "S_dec(σ) is an integral over the margin density, not a τ-lookup").
//
// Two models, both first-order (binding-competitor; independent N(0,σ²) per
// rolled score; gap noise = σ·√(#movable sides)):
//   general (L1s/L1p): every score carries error → gap noise = √2·σ; flip prob
//     for a competitor at gap g is Φ(−g/(√2σ)); exact ties (g=0) flip w.p. ½.
//   l1r: ONLY the 4 rolled scores carry error. A competitor gap is movable iff
//     the winner's max-slot is rolled (winner can fall) OR the competitor's
//     max-slot is rolled (it can rise); non-movable gaps never flip. gap noise
//     = σ·√(#rolled sides ∈ {0,1,2}).
// S_dec = mean over plans of (1 − P(any competitor overtakes)); binding
// approximation uses the single largest per-competitor flip prob (upper-bounds
// S_dec slightly; a union bound would lower it). Reported per cell × N bucket.
//
//   node dev/exact_nn/predict_sdec.js --sigma 0.01 [--data dir]
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

function erf(x) {                       // Abramowitz-Stegun 7.1.26, ~1e-7
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return x >= 0 ? y : -y;
}
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));

function parseArgs(argv) {
    const a = { data: path.join(__dirname, 'data'), sigmas: null };
    for (let i = 2; i < argv.length; i++) {
        if (argv[i] === '--data') a.data = argv[++i];
        else if (argv[i] === '--sigma') a.sigmas = (a.sigmas || []).concat(+argv[++i]);
        else throw new Error('unknown arg ' + argv[i]);
    }
    if (!a.sigmas) a.sigmas = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1];   // default scan
    return a;
}

function* iter(dir) {
    for (const f of fs.readdirSync(dir).sort()) {
        if (!f.endsWith('.decisions.jsonl.gz')) continue;
        for (const line of zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString().split('\n'))
            if (line) yield JSON.parse(line);
    }
}

// per-plan: list of competitor gaps with their movable-side counts for L1r,
// plus the general (always-movable) gaps.
function planGaps(rec) {
    const score = rec.score.map(v => v === null ? -Infinity : v);
    const rolled = new Set(rec.pidx.slice(0, 4));
    const key = k => rec.cands[k][0] + '|' + rec.cands[k][1];
    const groups = new Map();
    for (let k = 0; k < rec.cands.length; k++) {
        const kk = key(k);
        let g = groups.get(kk);
        if (!g) { g = { max: -Infinity, maxSlot: -1 }; groups.set(kk, g); }
        if (score[k] > g.max) { g.max = score[k]; g.maxSlot = k; }
    }
    const winKey = key(rec.bi);
    const wg = groups.get(winKey);
    const winnerRolled = rolled.has(wg.maxSlot);
    const out = [];
    for (const [k, g] of groups) {
        if (k === winKey) continue;
        const gap = wg.max - g.max;                          // ≥ 0
        const compRolled = rolled.has(g.maxSlot);
        const l1rSides = (winnerRolled ? 1 : 0) + (compRolled ? 1 : 0);
        out.push({ gap, l1rSides });
    }
    return out;
}

function flip(gap, noise) { return noise > 0 ? Phi(-gap / noise) : (gap === 0 ? 0.5 : 0); }

function main() {
    const opt = parseArgs(process.argv);
    // cellId -> bucket -> {n, matchGeneral:[σ]->sum, matchL1r:[σ]->sum}
    const cells = {};
    const bucket = N => N <= 14 ? '6-14' : '15+';
    for (const rec of iter(opt.data)) {
        const cid = rec.cell, b = bucket(rec.N);
        cells[cid] = cells[cid] || {};
        const st = cells[cid][b] = cells[cid][b] || { n: 0, gen: opt.sigmas.map(() => 0), l1r: opt.sigmas.map(() => 0) };
        st.n++;
        const gaps = planGaps(rec);
        opt.sigmas.forEach((s, si) => {
            // binding competitor = max flip prob
            let fGen = 0, fL1r = 0;
            for (const { gap, l1rSides } of gaps) {
                const pg = flip(gap, Math.SQRT2 * s);
                if (pg > fGen) fGen = pg;
                const pl = flip(gap, s * Math.sqrt(l1rSides));
                if (pl > fL1r) fL1r = pl;
            }
            st.gen[si] += 1 - fGen;
            st.l1r[si] += 1 - fL1r;
        });
    }
    const pf = x => (x * 100).toFixed(2) + '%';
    console.log('Predicted S_dec vs student score-error σ (binding-competitor, first-order).');
    console.log('Two models: general (L1s/L1p, all scores noisy) | l1r (only 4 rolled scores noisy).\n');
    for (const cid of Object.keys(cells).sort()) {
        for (const b of ['6-14', '15+']) {
            const st = cells[cid][b]; if (!st) continue;
            console.log(`${cid} N${b} (n=${st.n}):`);
            opt.sigmas.forEach((s, si) => {
                console.log(`   σ=${String(s).padEnd(7)}  S_dec general ${pf(st.gen[si] / st.n).padStart(7)}   l1r ${pf(st.l1r[si] / st.n).padStart(7)}`);
            });
        }
    }
    // overall
    console.log('\nOVERALL:');
    const tot = { n: 0, gen: opt.sigmas.map(() => 0), l1r: opt.sigmas.map(() => 0) };
    for (const cid of Object.keys(cells)) for (const b of ['6-14', '15+']) {
        const st = cells[cid][b]; if (!st) continue;
        tot.n += st.n; opt.sigmas.forEach((s, si) => { tot.gen[si] += st.gen[si]; tot.l1r[si] += st.l1r[si]; });
    }
    opt.sigmas.forEach((s, si) =>
        console.log(`   σ=${String(s).padEnd(7)}  S_dec general ${pf(tot.gen[si] / tot.n).padStart(7)}   l1r ${pf(tot.l1r[si] / tot.n).padStart(7)}`));
}
main();
