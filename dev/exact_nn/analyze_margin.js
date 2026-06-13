// analyze_margin.js — DELIVERABLE ZERO (SPEC §6.3): the dedup-by-coordinates
// top1−top2 score-margin CDF over oracle-logged plan decisions, stratified by
// N (6–14 vs 15+) and device cell. It sizes the D1 (planCheap argmax) half of
// the program for the planner-modal regime; what it is and is NOT (per the
// adversarial audit) is stated explicitly in DZERO.md and reproduced here:
//
//   * P(dmargin > τ) = max NN-alone COVERAGE of a prod-margin-gated hybrid (the
//     fraction the NN may decide alone); its complement is the FALLBACK LOAD. It
//     is NOT an S_dec accuracy bound — dmargin is PROD's margin, but L1h gates on
//     the STUDENT's margin (SPEC §4c), so this CDF cannot see L1h's only failure
//     mode (student-confident-and-wrong; needs the §4c adversarial search).
//   * The genuine σ-independent S_dec floor is the EXACT-TIE rate (bitwise score
//     tie across distinct coords → index tiebreak, unlearnable by a continuous
//     net; a score/pointer student misses ≥~½).
//   * A real student with score error σ has S_dec ≈ 1 − ∫φ(m)Φ(−m/(√2σ))dm — an
//     integral over the margin density, not a τ-lookup.
//
// Each record carries `dmargin` (top1−top2 over coordinate-deduped groups; group
// score = max over bitwise-equal slots), replay-verified at log time AND
// independently re-derived (independent_margin_check.js). This tool aggregates.
//
// L1r lens (CORRECTED): the rolled set is pidx[0:4] (NOT `pidx.includes` — pidx
// is the full 16-perm, so that test was a tautology). L1r perturbs ONLY rolled
// scores; l1rLens() computes the flip margin over movable competitors and the
// exact-for-free share. Scope caveats (none-profile, D1-only, autocorrelation)
// in DZERO.md.
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

// The L1r lens (corrected). The ROLLED set is pidx[0:4] (prod rolls exactly the
// top-K_roll=4 by ballistic pscore — `pidx` is the FULL 16-perm, so the old
// `pidx.includes(bi)` was a tautology). L1r keeps cp_features+vprior EXACT and
// learns ONLY the 4 rolled scores, so under L1r ONLY rolled candidates' scores
// can move. A plan's committed coordinate can flip under L1r iff a competing
// coordinate group can be reordered past the winner by perturbing rolled scores:
//   - a competing group can RISE iff it contains any rolled slot (it could be
//     pushed up), and/or
//   - the winner can FALL iff the winner group's max-scoring slot is rolled.
// l1rMargin = winner_score − best competing group score AMONG movable competitors
// (conservative: uses current group maxes, so it under-states the perturbation a
// sub-max rolled slot needs — i.e. it OVER-states L1r risk, the safe direction).
// exactForFree = no movable competitor AND winner not movable ⇒ the coordinate is
// immune to ANY rolled perturbation, L1r reproduces it bitwise regardless of σ.
function l1rLens(rec) {
    const score = rec.score.map(v => v === null ? -Infinity : v);
    const rolled = new Set(rec.pidx.slice(0, 4));
    const bi = rec.bi;
    const key = k => rec.cands[k][0] + '|' + rec.cands[k][1];
    const groups = new Map();
    for (let k = 0; k < rec.cands.length; k++) {
        const kk = key(k);
        let g = groups.get(kk);
        if (!g) { g = { max: -Infinity, maxSlot: -1, anyRolled: false }; groups.set(kk, g); }
        if (score[k] > g.max) { g.max = score[k]; g.maxSlot = k; }
        if (rolled.has(k)) g.anyRolled = true;
    }
    const winKey = key(bi);
    const wg = groups.get(winKey);
    const winnerMaxRolled = rolled.has(wg.maxSlot);
    let l1rMargin = Infinity, movable = false;
    for (const [k, g] of groups) {
        if (k === winKey) continue;
        if (g.anyRolled || winnerMaxRolled) {
            movable = true;
            const gap = wg.max - g.max;
            if (gap < l1rMargin) l1rMargin = gap;
        }
    }
    return { winnerRolled: rolled.has(bi), l1rMargin, exactForFree: !movable };
}

function newStratum() {
    return { n: 0, margins: [],   // dmargin values (SPEC §6.3 primary)
        l1rMargins: [],           // L1r-flip margins over MOVABLE plans
        winnerRolled: 0,          // committed slot bi ∈ rolled set (pidx[0:4])
        exactForFree: 0,          // L1r reproduces the coordinate regardless of σ
        exactTie: 0,              // dmargin === 0 (distinct coord groups tie exactly)
        singleGroup: 0,           // nDistinct === 1 (all 16 cands one coordinate)
        seeds: new Set() };       // distinct games (for plans-per-game)
}

function add(strat, rec) {
    strat.n++;
    strat.seeds.add(rec.seed);
    // dmargin null ⟺ +Infinity (coordinate forced: no competing distinct group).
    // Decoding to +Inf — NOT 0 — is essential: a forced coordinate is maximally
    // SAFE, the opposite of a near-tie. Treating null as 0 would corrupt the
    // left tail (the whole point of this analysis).
    const dm = rec.dmargin === null ? Infinity : rec.dmargin;
    strat.margins.push(dm);
    if (dm === 0) strat.exactTie++;
    if (rec.nDistinct === 1) strat.singleGroup++;
    const L = l1rLens(rec);
    if (L.winnerRolled) strat.winnerRolled++;
    if (L.exactForFree) strat.exactForFree++; else strat.l1rMargins.push(L.l1rMargin);
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

// rule-of-three: 0 observed in n trials ⇒ true rate ≤ 3/n at ~95%.
function ruleOfThree(count, n) { return count === 0 && n > 0 ? 3 / n : null; }

function summarize(strat) {
    const sorted = strat.margins.slice().sort((a, b) => a - b);
    const l1rSorted = strat.l1rMargins.slice().sort((a, b) => a - b);
    return {
        n: strat.n,
        games: strat.seeds.size,
        plansPerGame: strat.seeds.size ? +(strat.n / strat.seeds.size).toFixed(1) : null,
        exactTieFrac: strat.n ? strat.exactTie / strat.n : null,
        exactTieCount: strat.exactTie,
        exactTieRuleOf3: ruleOfThree(strat.exactTie, strat.n),  // ceiling when 0 observed
        // genuine NN-alone S_dec ceiling FROM EXACT TIES alone (a continuous net
        // can't reproduce a bitwise score tie + index tiebreak): 1 − exact-tie.
        sDecFloorFromTies: strat.n ? 1 - strat.exactTie / strat.n : null,
        singleGroupFrac: strat.n ? strat.singleGroup / strat.n : null,
        winnerRolledFrac: strat.n ? strat.winnerRolled / strat.n : null,
        exactForFreeFrac: strat.n ? strat.exactForFree / strat.n : null,
        cdf: cdf(sorted, TAUS),
        pctl: quantiles(sorted, PCTL),
        // L1r-flip-margin CDF over the movable plans (exact-for-free excluded —
        // those never flip). The fraction movable = 1 − exactForFreeFrac.
        l1r: { movable: l1rSorted.length, cdf: cdf(l1rSorted, TAUS), pctl: quantiles(l1rSorted, PCTL) },
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
    // key sizing numbers — P(dmargin>τ) is a COVERAGE/fallback number, NOT S_dec.
    console.log('SIZING — P(dmargin > τ) = max NN-alone COVERAGE of a prod-margin-gated system');
    console.log('(= fraction the NN may decide alone; the complement is the fallback load).');
    console.log('It is NOT an S_dec accuracy bound: dmargin is PROD\'s margin, L1h gates on the');
    console.log('STUDENT\'s margin; real S_dec(σ) is an integral over the margin density.');
    for (const t of [1e-6, 1e-3, 1e-2, 0.1]) {
        console.log(`  τ=${t}:  P(dmargin>τ) ${((1 - o.cdf[t]) * 100).toFixed(3)}%   fallback load ${(o.cdf[t] * 100).toFixed(3)}%`);
    }
    console.log(`\nGENUINE NN-alone S_dec floor (from exact ties alone, σ-independent): ${(o.sDecFloorFromTies * 100).toFixed(3)}% overall`);
    console.log(`  worst cell×bucket: ${worstTieCell(report).label} → ${(worstTieCell(report).floor * 100).toFixed(2)}% ceiling`);
    console.log(`winner-is-rolled (corrected, pidx[0:4]): ${(o.winnerRolledFrac * 100).toFixed(2)}%   ` +
                `L1r exact-for-free: ${(o.exactForFreeFrac * 100).toFixed(2)}%`);

    if (opt.out) { fs.writeFileSync(opt.out, JSON.stringify(report, null, 1)); console.log('\nwrote ' + opt.out); }
    if (opt.md) { fs.writeFileSync(opt.md, renderMd(report, cells)); console.log('wrote ' + opt.md); }
}

function worstTieCell(report) {
    let worst = { floor: 1, label: '—' };
    for (const [cid, s] of Object.entries(report.byCell)) {
        for (const b of ['6-14', '15+']) {
            if (s[b].n && s[b].sDecFloorFromTies < worst.floor)
                worst = { floor: s[b].sDecFloorFromTies, label: `${cid} N${b}` };
        }
    }
    return worst;
}

function pct(x) { return x == null ? '—' : (x * 100).toFixed(2) + '%'; }
function num(x) { return x == null ? '—' : (x === Infinity ? '∞' : (Math.abs(x) >= 1e-4 || x === 0 ? x.toFixed(5) : x.toExponential(2))); }
function tieCell(s) {  // exact-tie with rule-of-three ceiling when 0 observed
    return s.exactTieCount === 0 ? `0 (≤${pct(s.exactTieRuleOf3)})` : `${pct(s.exactTieFrac)} (${s.exactTieCount})`;
}

function renderMd(report, cells) {
    const m = [];
    const o = report.overall;
    m.push('# Deliverable Zero — coordinate-dedup top1−top2 margin CDF');
    m.push('');
    m.push(`**${report.total}** plan decisions (${o.games} games), 6 device cells, **none-profile (no spawns), planner regime (N≥6 / D1) only.** ` +
           'Margin = top1−top2 over coordinate-deduped candidate groups (SPEC §3); every value ' +
           'replay-verified at log time AND independently re-derived (independent_margin_check.js, 0/' +
           `${report.total} mismatches).`);
    m.push('');
    m.push('## What this number is — and is NOT');
    m.push('- **P(dmargin > τ)** = the maximum NN-alone **COVERAGE** of a prod-margin-gated hybrid ' +
           '(fraction the NN may decide alone); its complement is the **fallback load**. It is **NOT** an ' +
           'S_dec accuracy bound: `dmargin` is *prod\'s* margin, but L1h gates on the *student\'s* margin ' +
           '(SPEC §4c), so this CDF structurally cannot see L1h\'s only failure mode (student-confident-and-wrong).');
    m.push('- The **genuine σ-independent S_dec floor** is the **exact-tie rate**: when prod\'s own top groups ' +
           'tie bitwise (distinct coordinates, identical score), the committed target is set by the index ' +
           'tiebreak — unlearnable by a continuous net. A score/pointer student (L1s/L1p) misses ≥~½ of these.');
    m.push('- A real student with score-reconstruction error σ has S_dec ≈ 1 − ∫ φ(m)·Φ(−m/(√2·σ)) dm over the ' +
           'margin density φ — an **integral**, not a τ-lookup. The decisive unknown is the student\'s achievable σ; ' +
           'measure it on a small L1r run, then integrate against this CDF.');
    m.push('');
    m.push('## Overall');
    m.push(`- exact ties (dmargin=0, distinct coords) → **NN-alone S_dec floor ${pct(o.sDecFloorFromTies)}**: exact-tie ${pct(o.exactTieFrac)} (${o.exactTieCount})`);
    m.push(`- winner is a ROLLED candidate (pidx[0:4]): **${pct(o.winnerRolledFrac)}** — so ~${pct(1 - o.winnerRolledFrac)} of decisions commit a non-rolled vprior candidate (L1r scores those exactly)`);
    m.push(`- **L1r exact-for-free** (committed coord immune to ANY rolled-score perturbation): ${pct(o.exactForFreeFrac)}`);
    m.push(`- single-coordinate-group plans: ${pct(o.singleGroupFrac)}; plans/game ${o.plansPerGame}`);
    m.push('');
    m.push('CDF P(dmargin ≤ τ):');
    m.push('');
    m.push('| τ | ' + Object.keys(o.cdf).map(t => String(t)).join(' | ') + ' |');
    m.push('|' + '---|'.repeat(Object.keys(o.cdf).length + 1));
    m.push('| P(≤τ) | ' + Object.values(o.cdf).map(pct).join(' | ') + ' |');
    m.push('');
    m.push('NN-alone coverage ceiling = P(dmargin > τ) (NOT S_dec — see above); fallback load = P(≤τ):');
    m.push('');
    for (const t of [1e-6, 1e-3, 1e-2, 0.1]) m.push(`- τ=${t}: coverage **${pct(1 - o.cdf[t])}**, fallback load ${pct(o.cdf[t])}`);
    m.push('');
    m.push('## By device cell × N bucket (SPEC §6.3 stratification)');
    m.push('');
    m.push('exact-tie shown as rate (count), or `0 (≤rule-of-three ceiling)` when none observed.');
    m.push('');
    m.push('| cell | N | n | games | exact-tie → S_dec floor | P(≤1e-3) | P(≤1e-2) | P(≤0.1) | median dmargin | winner-rolled |');
    m.push('|---|---|---|---|---|---|---|---|---|---|');
    for (const cid of Object.keys(cells).sort()) {
        for (const b of ['6-14', '15+']) {
            const s = report.byCell[cid][b];
            if (!s.n) continue;
            m.push(`| ${cid} | ${b} | ${s.n} | ${s.games} | ${tieCell(s)} → ${pct(s.sDecFloorFromTies)} | ${pct(s.cdf[1e-3])} | ${pct(s.cdf[1e-2])} | ${pct(s.cdf[0.1])} | ${num(s.pctl[50])} | ${pct(s.winnerRolledFrac)} |`);
        }
    }
    m.push('');
    m.push('## L1r risk surface — flip margin (only the 4 rolled scores can move the decision)');
    m.push('');
    m.push('Excludes exact-for-free plans (no rolled candidate can flip the committed coordinate). ' +
           'l1rMargin is a CONSERVATIVE proxy (uses current group maxes → over-states risk).');
    m.push('');
    m.push('| cell | N | exact-for-free | movable plans | P(≤1e-3) | P(≤1e-2) | P(≤0.1) | median |');
    m.push('|---|---|---|---|---|---|---|---|');
    for (const cid of Object.keys(cells).sort()) {
        for (const b of ['6-14', '15+']) {
            const s = report.byCell[cid][b];
            if (!s.n) continue;
            m.push(`| ${cid} | ${b} | ${pct(s.exactForFreeFrac)} | ${s.l1r.movable} | ${pct(s.l1r.cdf[1e-3])} | ${pct(s.l1r.cdf[1e-2])} | ${pct(s.l1r.cdf[0.1])} | ${num(s.l1r.pctl[50])} |`);
        }
    }
    m.push('');
    m.push('## Caveats (scope of this estimate)');
    m.push('- **Optimistic lower bound on tie/near-tie density:** none-profile only. Spawn / 5→6→5 ' +
           're-crossing / same-coord double-tap states (SPEC §5, §7 top risk) contribute ZERO plans here ' +
           'and plausibly carry a heavier tail; measured next via `shard_runner --spawnFrac`.');
    m.push('- **D1 only:** the N≤5 intercept/egBoid-commit decision (D4 / L1e) has its own, still-unmeasured margin distribution.');
    m.push('- **Autocorrelation:** plans within a game-to-extinction are correlated; effective n < n, especially the thin 6-14 cells. Per-cell tail percentiles below ~3/n are counting-noise dominated.');
    m.push('- **Train seeds only** (100000–160000, all < 270000): the sealed verification set (≥270000) is untouched.');
    m.push('');
    return m.join('\n') + '\n';
}

main();
