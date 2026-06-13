// tau_calibrate.js — freeze the L1h margin threshold τ from the CALIBRATION set
// (SPEC §4c). This is the pivotal number: side-a's finding (#5) shows NN-alone
// D1 plateaus ~37% (rollout-dominated), so the exact path is L1h — NN fast-path
// where the student's deduped top-2 margin ≥ τ, else exact rollout fallback. The
// L1h NN-share = the trusted fraction at τ, and τ is set by the student's
// CONFIDENCE CALIBRATION, not its raw accuracy. This tool owns that.
//
// Anti-circularity (SPEC §4c): τ is chosen ONLY on the calibration range
// ([270000,280000)); it is frozen (written + hashed) BEFORE the one-shot sealed
// verdict (verifier/verdict.js reads the frozen τ). Never tuned on sealed seeds.
//
// INPUT — the student's calibration record (side-a delivers this with the JS
// export). JSON array (or JSONL via --jsonl), one entry per plan decision:
//   { margin: <student deduped top-2 margin>, agree: <bool student-committed-
//     coord == prod-committed-coord>, n?: <boid count>, cell?: <WxH> }
// (agree is computed against prod's committed TARGET COORDINATES, deduped — §3.)
//
// OUTPUT — frozen_tau.json: chosen τ, trusted fraction (= L1h NN-share), the
// full risk-vs-τ tail curve, a reliability (calibration) curve, and the
// rule-of-three residual projection for the sealed run. Also flags if margin is
// NOT a usable confidence signal (non-monotone risk) — in which case τ-gating
// can't deliver and the report must say so.
//
//   node tau_calibrate.js --in calib.json [--target-trusted-disagree 0] [--out frozen_tau.json]
//   node tau_calibrate.js --mock 20000 --mock-sigma 0.02   # self-test the logic
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function parseArgs() {
    const a = { in: null, jsonl: false, out: path.join(__dirname, 'frozen_tau.json'),
        targetTrustedDisagree: 0, mock: 0, mockSigma: 0.02, mockSeed: 1, label: null };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--in') a.in = process.argv[++i];
        else if (k === '--jsonl') a.jsonl = true;
        else if (k === '--out') a.out = process.argv[++i];
        else if (k === '--target-trusted-disagree') a.targetTrustedDisagree = +process.argv[++i];
        else if (k === '--mock') a.mock = +process.argv[++i];
        else if (k === '--mock-sigma') a.mockSigma = +process.argv[++i];
        else if (k === '--label') a.label = process.argv[++i];
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

// ---- mock calibration data (self-test only) -------------------------------
// Models a student whose committed coord disagrees with prod with probability
// Φ(−margin/(√2 σ)) (side-a's integral kernel): bigger margin ⇒ student is
// confident ⇒ less likely wrong. Margins drawn ~ prod's observed left-tail-heavy
// shape (exponential-ish). Validates that τ-selection finds the threshold above
// which trusted disagreements vanish, and that the curve is monotone.
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function normCdf(z) { // Abramowitz-Stegun 7.1.26 erf
    const s = z < 0 ? -1 : 1; const x = Math.abs(z) / Math.SQRT2;
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return 0.5 * (1 + s * y);
}
function genMock(n, sigma, seed) {
    const r = mulberry32(seed >>> 0), out = [];
    for (let i = 0; i < n; i++) {
        const margin = -Math.log(1 - r() * 0.999) * 0.08;        // heavy left tail, median ~0.055
        const pDisagree = normCdf(-margin / (Math.SQRT2 * sigma));
        out.push({ margin, agree: r() >= pDisagree });
    }
    return out;
}

// ---- curves + τ selection --------------------------------------------------
function analyze(recs, targetTrustedDisagree) {
    const data = recs.filter(d => typeof d.margin === 'number' && Number.isFinite(d.margin))
        .map(d => ({ m: d.margin, ok: !!d.agree }))
        .sort((x, y) => x.m - y.m);
    const N = data.length;
    // suffix counts: for a τ, trusted = {m >= τ}; trusted disagreements = #{m>=τ & !ok}
    // walk from high margin down; track trusted set growing as τ decreases.
    // reliability curve: bin by margin, empirical disagreement rate per bin.
    const BINS = [0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, Infinity];
    const bin = BINS.slice(0, -1).map((lo, i) => ({ lo, hi: BINS[i + 1], n: 0, dis: 0 }));
    for (const d of data) { for (const b of bin) if (d.m >= b.lo && d.m < b.hi) { b.n++; if (!d.ok) b.dis++; break; } }
    const reliability = bin.filter(b => b.n > 0).map(b => ({ marginRange: [b.lo, b.hi === Infinity ? null : b.hi], n: b.n, disagreeRate: +(b.dis / b.n).toFixed(6) }));
    // monotonicity check: disagreeRate should fall (weakly) as margin rises.
    // Only compare bins with enough samples, and allow a binomial-SE tolerance
    // so sparse-high-margin sampling noise doesn't false-flag (real data has few
    // high-margin plans). A genuine non-monotonicity (margin not a confidence
    // signal) would persist across well-populated bins.
    let monotone = true;
    const big = reliability.filter(b => b.n >= 50);
    for (let i = 1; i < big.length; i++) {
        const prev = big[i - 1], cur = big[i];
        const se = Math.sqrt(Math.max(prev.disagreeRate * (1 - prev.disagreeRate), 1e-9) / prev.n)
                 + Math.sqrt(Math.max(cur.disagreeRate * (1 - cur.disagreeRate), 1e-9) / cur.n);
        if (cur.disagreeRate > prev.disagreeRate + 3 * se + 1e-6) monotone = false;   // higher-margin bin materially worse
    }

    // τ candidates = the distinct margins (+0); for each, trusted fraction + trusted disagreements
    // efficient: iterate ascending; maintain trustedDisagree = total !ok among m>=τ.
    const totalBad = data.filter(d => !d.ok).length;
    // build τ -> (trustedCount, trustedBad) by scanning ascending and subtracting.
    let seenBad = 0, seen = 0; const curve = [];
    // sample the curve at the BIN edges + at the minimal-safe τ
    let tauSafe = Infinity, trustedAtSafe = 0;
    // scan from smallest margin up; at threshold τ=data[i].m, trusted={j>=i}
    for (let i = 0; i < N; i++) {
        const trusted = N - i, trustedBad = totalBad - seenBad;
        if (trustedBad <= targetTrustedDisagree && data[i].m < tauSafe) { tauSafe = data[i].m; trustedAtSafe = trusted; }
        if (!data[i].ok) seenBad++; seen++;
    }
    // curve at representative τ
    const taus = [0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3];
    for (const tau of taus) {
        let trusted = 0, trustedBad = 0;
        for (const d of data) if (d.m >= tau) { trusted++; if (!d.ok) trustedBad++; }
        curve.push({ tau, trustedFrac: +(trusted / N).toFixed(6), trustedDisagreements: trustedBad,
            trustedDisagreeRate: trusted ? +(trustedBad / trusted).toFixed(8) : null });
    }
    // rule-of-three projection: if the sealed run yields 0 trusted disagreements
    // in n trusted decisions, per-decision risk <= 3/n at 95%.
    return { N, totalDisagreements: totalBad, overallAgree: +(1 - totalBad / N).toFixed(6),
        marginIsUsableConfidenceSignal: monotone,
        reliability, riskVsTau: curve,
        chosenTau: tauSafe === Infinity ? null : tauSafe,
        trustedFracAtChosenTau: tauSafe === Infinity ? 0 : +(trustedAtSafe / N).toFixed(6),
        note: tauSafe === Infinity
            ? 'NO τ achieves the trusted-disagreement target on calibration — even the highest-margin plans disagree. τ-gating cannot deliver; report L1h NN-share ≈ 0 and rely on L0.'
            : 'τ = smallest calibration margin with ≤ target trusted disagreements; trustedFrac = L1h NN-share estimate. Frozen for the one-shot sealed verdict; rule-of-three on the sealed trusted decisions bounds residual risk.' };
}

function main() {
    const a = parseArgs();
    let recs, source;
    if (a.mock) { recs = genMock(a.mock, a.mockSigma, a.mockSeed); source = `MOCK(n=${a.mock},sigma=${a.mockSigma})`; }
    else if (a.in) {
        const raw = fs.readFileSync(a.in, 'utf8');
        recs = a.jsonl ? raw.trim().split('\n').map(l => JSON.parse(l)) : JSON.parse(raw);
        source = path.basename(a.in);
    } else throw new Error('need --in <calib.json> or --mock <n>');

    const res = analyze(recs, a.targetTrustedDisagree);
    const frozen = {
        spec: 'SPEC §4c — frozen L1h τ from the calibration set',
        source, label: a.label, calibPlans: res.N,
        targetTrustedDisagree: a.targetTrustedDisagree,
        chosenTau: res.chosenTau, L1h_NNshare_estimate: res.trustedFracAtChosenTau,
        marginIsUsableConfidenceSignal: res.marginIsUsableConfidenceSignal,
        overallStudentAgree: res.overallAgree, totalDisagreements: res.totalDisagreements,
        riskVsTau: res.riskVsTau, reliability: res.reliability,
        note: res.note,
        inputSha256: crypto.createHash('sha256').update(JSON.stringify(recs)).digest('hex'),
        FROZEN: true,
    };
    console.log(JSON.stringify(frozen, null, 1));
    if (!a.mock) { fs.writeFileSync(a.out, JSON.stringify(frozen, null, 1)); process.stderr.write('froze τ -> ' + a.out + '\n'); }
}

if (require.main === module) main();
module.exports = { analyze, genMock };
