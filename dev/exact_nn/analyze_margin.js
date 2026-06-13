// analyze_margin.js — DELIVERABLE ZERO (SPEC §6.3): the dedup-by-coordinates
// top1−top2 score-margin CDF over oracle-logged plan decisions, stratified by
// N (6–14 vs 15+) and device cell. This number sizes the entire GPU program:
// the left tail of the margin distribution is the L1 near-tie risk, i.e. the
// ceiling on any NN-alone system's decision-agreement (S_dec) before a single
// net is trained.
//
// Each decision record already carries `dmargin` (top1−top2 over coordinate-
// deduped candidate groups; group score = max over its bitwise-equal slots) and
// was replay-verified at log time (oracle_logger.recheckPlan). This tool only
// aggregates — it does not recompute the policy.
//
// Two lenses, both reported:
//  (A) general margin CDF (dmargin): the L1s/L1p / L1h-gate-relevant number —
//      a student whose reconstructed-score error has scale σ flips the committed
//      COORDINATES on (roughly) the plans with dmargin < σ. P(dmargin≈0) is the
//      irreducible floor even for a near-perfect student.
//  (B) L1r lens: L1r keeps cp_features+vprior EXACT and learns only the 4 rolled
//      scores, so a plan can mismatch ONLY when the winning group's score
//      depends on a rolled candidate. We report the share of plans whose winner
//      is a rolled candidate (bi ∈ pidx) and, among those, the margin to the
//      best COMPETING group — the L1r-specific risk surface.
//
//   node dev/exact_nn/analyze_margin.js [--data dir] [--out report.json] [--md report.md]
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

function parseArgs(argv) {
    const a = { data: path.join(__dirname, 'data'), out: null, md: null, glob: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--data') a.data = argv[++i];
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--md') a.md = argv[++i];
        else if (k === '--glob') a.glob = argv[++i];   // substring filter on filenames
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}

function* iterDecisions(dataDir, globSub) {
    for (const f of fs.readdirSync(dataDir).sort()) {
        if (!f.endsWith('.decisions.jsonl.gz')) continue;
        if (globSub && !f.includes(globSub)) continue;
        const buf = zlib.gunzipSync(fs.readFileSync(path.join(dataDir, f)));
        for (const line of buf.toString('utf8').split('\n')) {
            if (!line) continue;
            yield JSON.parse(line);
        }
    }
}

// margin thresholds for the CDF (P(dmargin <= τ)); spans exact-tie .. wide.
const TAUS = [0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1, 2, 5];
const PCTL = [0, 0.1, 0.5, 1, 2, 5, 10, 25, 50];   // percent points to report

function newStratum() {
    return { n: 0, margins: [], // dmargin values
        winnerRolled: 0,        // bi ∈ pidx
        rolledRiskMargins: [],  // dmargin where winner is rolled (L1r risk surface)
        exactTie: 0,            // dmargin === 0 (distinct coord groups tie exactly)
        singleGroup: 0 };       // nDistinct === 1 (all 16 cands one coordinate)
}

function add(strat, rec) {
    strat.n++;
    // dmargin null ⟺ +Infinity (coordinate forced: no competing distinct group).
    // Decoding to +Inf — NOT 0 — is essential: a forced coordinate is maximally
    // SAFE, the opposite of a near-tie. Treating null as 0 would corrupt the
    // left tail (the whole point of this analysis).
    const dm = rec.dmargin === null ? Infinity : rec.dmargin;
    strat.margins.push(dm);
    const winnerRolled = rec.pidx.includes(rec.bi);
    if (winnerRolled) { strat.winnerRolled++; strat.rolledRiskMargins.push(dm); }
    if (dm === 0) strat.exactTie++;
    if (rec.nDistinct === 1) strat.singleGroup++;
}

function quantiles(sorted, ps) {
    const out = {};
    for (const p of ps) {
        if (!sorted.length) { out[p] = null; continue; }
        const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
        out[p] = sorted[idx];
    }
    return out;
}

function cdf(sorted, taus) {
    // P(dmargin <= τ) via binary search on the sorted ascending array.
    const out = {};
    for (const t of taus) {
        let lo = 0, hi = sorted.length;
        while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= t) lo = m + 1; else hi = m; }
        out[t] = sorted.length ? lo / sorted.length : null;
    }
    return out;
}

function summarize(strat) {
    const sorted = strat.margins.slice().sort((a, b) => a - b);
    const rolledSorted = strat.rolledRiskMargins.slice().sort((a, b) => a - b);
    return {
        n: strat.n,
        exactTieFrac: strat.n ? strat.exactTie / strat.n : null,
        singleGroupFrac: strat.n ? strat.singleGroup / strat.n : null,
        winnerRolledFrac: strat.n ? strat.winnerRolled / strat.n : null,
        cdf: cdf(sorted, TAUS),
        pctl: quantiles(sorted, PCTL),
        rolled: { n: strat.winnerRolled, cdf: cdf(rolledSorted, TAUS),
                  pctl: quantiles(rolledSorted, PCTL) },
    };
}

function nbucket(N) { return N <= 14 ? '6-14' : '15+'; }

function asciiCdf(sorted, label) {
    // compact log-scale CDF sketch for the issue comment
    const lines = [`  ${label}  (n=${sorted.length})`];
    for (const t of [1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1, 2]) {
        let lo = 0, hi = sorted.length;
        while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= t) lo = m + 1; else hi = m; }
        const frac = sorted.length ? lo / sorted.length : 0;
        const bar = '█'.repeat(Math.round(frac * 40));
        lines.push(`    ≤${String(t).padEnd(6)} ${(frac * 100).toFixed(2).padStart(6)}%  ${bar}`);
    }
    return lines.join('\n');
}

function main() {
    const opt = parseArgs(process.argv);
    const cells = {};            // cellId -> { '6-14':strat, '15+':strat, all:strat }
    const overall = newStratum();
    let total = 0;
    for (const rec of iterDecisions(opt.data, opt.glob)) {
        total++;
        const cid = rec.cell;
        if (!cells[cid]) cells[cid] = { '6-14': newStratum(), '15+': newStratum(), all: newStratum() };
        add(cells[cid][nbucket(rec.N)], rec);
        add(cells[cid].all, rec);
        add(overall, rec);
    }
    if (!total) { console.error('no decision records found in ' + opt.data); process.exit(1); }

    const report = { total, generatedFrom: path.resolve(opt.data), taus: TAUS, pctlPoints: PCTL,
        overall: summarize(overall), byCell: {} };
    for (const [cid, s] of Object.entries(cells)) {
        report.byCell[cid] = { '6-14': summarize(s['6-14']), '15+': summarize(s['15+']), all: summarize(s.all) };
    }

    // ---- console summary ----
    const o = report.overall;
    console.log(`\n=== DELIVERABLE ZERO: deduped top1−top2 margin CDF — ${total} plan decisions ===\n`);
    console.log('OVERALL:');
    console.log(asciiCdf(overall.margins.slice().sort((a, b) => a - b), 'all cells, all N'));
    console.log(`    exact-tie (dmargin=0): ${(o.exactTieFrac * 100).toFixed(3)}%   ` +
                `single-coord-group: ${(o.singleGroupFrac * 100).toFixed(2)}%   ` +
                `winner-is-rolled: ${(o.winnerRolledFrac * 100).toFixed(2)}%`);
    console.log('');
    for (const bucket of ['6-14', '15+']) {
        console.log(`N ${bucket}:`);
        for (const cid of Object.keys(cells).sort()) {
            const st = cells[cid][bucket];
            if (!st.n) continue;
            console.log(asciiCdf(st.margins.slice().sort((a, b) => a - b), `${cid}`));
        }
        console.log('');
    }
    // key sizing numbers
    console.log('SIZING (trusted fraction = P(dmargin > τ); S_dec ceiling if student perfect above τ):');
    for (const t of [1e-6, 1e-3, 1e-2, 0.1]) {
        const trusted = 1 - o.cdf[t];
        console.log(`  τ=${t}:  trusted ${(trusted * 100).toFixed(3)}%   (≈ S_dec ceiling for a τ-gated NN-alone system)`);
    }

    if (opt.out) { fs.writeFileSync(opt.out, JSON.stringify(report, null, 1)); console.log('\nwrote ' + opt.out); }
    if (opt.md) { fs.writeFileSync(opt.md, renderMd(report, cells)); console.log('wrote ' + opt.md); }
}

function pct(x) { return x == null ? '—' : (x * 100).toFixed(2) + '%'; }
function num(x) { return x == null ? '—' : (Math.abs(x) >= 1e-4 || x === 0 ? x.toFixed(5) : x.toExponential(2)); }

function renderMd(report, cells) {
    const m = [];
    m.push('# Deliverable Zero — deduped top1−top2 margin CDF');
    m.push('');
    m.push(`Plan decisions analyzed: **${report.total}**.  Margin = top1−top2 over `);
    m.push('coordinate-deduped candidate groups (SPEC §3); each value replay-verified at log time.');
    m.push('');
    const o = report.overall;
    m.push('## Overall');
    m.push(`- exact ties (dmargin=0, distinct coord groups): **${pct(o.exactTieFrac)}**`);
    m.push(`- single-coordinate-group plans (all 16 cands one point): ${pct(o.singleGroupFrac)}`);
    m.push(`- winner is a rolled candidate (bi ∈ pidx): ${pct(o.winnerRolledFrac)}  ← L1r risk surface`);
    m.push('');
    m.push('CDF P(dmargin ≤ τ):');
    m.push('');
    m.push('| τ | ' + Object.keys(o.cdf).map(t => String(t)).join(' | ') + ' |');
    m.push('|' + '---|'.repeat(Object.keys(o.cdf).length + 1));
    m.push('| P(≤τ) | ' + Object.values(o.cdf).map(pct).join(' | ') + ' |');
    m.push('');
    m.push('Trusted fraction = P(dmargin > τ) — the S_dec ceiling for a τ-gated NN-alone policy:');
    m.push('');
    for (const t of [1e-6, 1e-3, 1e-2, 0.1]) m.push(`- τ=${t}: **${pct(1 - o.cdf[t])}**`);
    m.push('');
    m.push('## By device cell × N bucket');
    m.push('');
    m.push('| cell | N | n | exact-tie | P(≤1e-3) | P(≤1e-2) | P(≤0.1) | median dmargin | winner-rolled |');
    m.push('|---|---|---|---|---|---|---|---|---|');
    for (const cid of Object.keys(cells).sort()) {
        for (const b of ['6-14', '15+']) {
            const s = report.byCell[cid][b];
            if (!s.n) continue;
            m.push(`| ${cid} | ${b} | ${s.n} | ${pct(s.exactTieFrac)} | ${pct(s.cdf[1e-3])} | ${pct(s.cdf[1e-2])} | ${pct(s.cdf[0.1])} | ${num(s.pctl[50])} | ${pct(s.winnerRolledFrac)} |`);
        }
    }
    m.push('');
    m.push('## L1r lens (winner-is-rolled plans only) — margin to best competing group');
    m.push('');
    m.push('| cell | N | rolled-winner plans | P(≤1e-3) | P(≤1e-2) | P(≤0.1) | median |');
    m.push('|---|---|---|---|---|---|---|');
    for (const cid of Object.keys(cells).sort()) {
        for (const b of ['6-14', '15+']) {
            const s = report.byCell[cid][b];
            if (!s.rolled.n) continue;
            m.push(`| ${cid} | ${b} | ${s.rolled.n} | ${pct(s.rolled.cdf[1e-3])} | ${pct(s.rolled.cdf[1e-2])} | ${pct(s.rolled.cdf[0.1])} | ${num(s.rolled.pctl[50])} |`);
        }
    }
    m.push('');
    return m.join('\n') + '\n';
}

main();
