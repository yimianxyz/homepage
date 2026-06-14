// tau_calibrate_eg.js — freeze the L1e endgame margin threshold τ from the
// CALIBRATION set (the scan-t-margin analogue of tau_calibrate.js). The L1e gate
// commits the NN's egBoid iff (a) the eg_bound certificate fires [ZERO-RISK, no
// τ] OR (b) the NN's deduped scan-t margin ≥ τ [trusted]. So τ governs ONLY the
// NON-certified commits — certified commits are exact by the certificate's
// soundness regardless of margin (and we ASSERT 0 disagreements among them here:
// a false cert on the calibration distribution would be a soundness violation).
//
// τ = smallest non-certified margin (in FRAMES) with ≤target trusted disagreements.
// NN-share at τ = certShare + (non-cert trusted fraction). Frozen ONE-SHOT, then
// the sealed verdict (verdict_l1e.js, sealOffset 40) checks it generalizes.
//
//   node tau_calibrate_eg.js --in calib_eg.json [--target-trusted-disagree 0]
//                            [--out frozen_tau_eg.json]
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function parseArgs() {
    const a = { in: path.join(__dirname, 'calib_eg.json'), target: 0,
        out: path.join(__dirname, 'frozen_tau_eg.json'), label: null };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--in') a.in = process.argv[++i];
        else if (k === '--target-trusted-disagree') a.target = +process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i];
        else if (k === '--label') a.label = process.argv[++i];
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

// reliability bins in FRAMES (scan-t margin). Infinity = sole-reachable (n=1 or
// only one boid reachable) → margin is +Inf, always trusted, trivially correct.
const FBINS = [0, 1, 2, 4, 8, 16, 32, 64, Infinity];

function analyze(records, target) {
    const all = records.filter(d => typeof d.margin === 'number' || d.margin === null);
    const total = all.length;
    const certified = all.filter(d => d.cert);
    const nonCert = all.filter(d => !d.cert);
    // SOUNDNESS cross-check: certified commits must ALL agree with prod.
    const certifiedDisagreements = certified.filter(d => !d.agree).length;

    // reliability on NON-certified commits (where margin is the only signal).
    const finite = nonCert.map(d => ({ m: Number.isFinite(d.margin) ? d.margin : Infinity, ok: !!d.agree }));
    const bin = FBINS.slice(0, -1).map((lo, i) => ({ lo, hi: FBINS[i + 1], n: 0, dis: 0 }));
    for (const d of finite) for (const b of bin) if (d.m >= b.lo && d.m < b.hi) { b.n++; if (!d.ok) b.dis++; break; }
    const reliability = bin.filter(b => b.n > 0).map(b => ({
        marginFramesRange: [b.lo, b.hi === Infinity ? null : b.hi], n: b.n,
        disagreeRate: +(b.dis / b.n).toFixed(6) }));
    // monotonicity (binomial-SE tolerance, ≥50-sample bins)
    let monotone = true;
    const big = reliability.filter(b => b.n >= 50);
    for (let i = 1; i < big.length; i++) {
        const p = big[i - 1], c = big[i];
        const se = Math.sqrt(Math.max(p.disagreeRate * (1 - p.disagreeRate), 1e-9) / p.n)
                 + Math.sqrt(Math.max(c.disagreeRate * (1 - c.disagreeRate), 1e-9) / c.n);
        if (c.disagreeRate > p.disagreeRate + 3 * se + 1e-6) monotone = false;
    }

    // τ-sweep on NON-certified: trusted = {m >= τ}; find smallest τ with ≤target disagreements.
    const sorted = finite.slice().sort((a, b) => a.m - b.m);
    const M = sorted.length, totalBad = sorted.filter(d => !d.ok).length;
    let seenBad = 0, tauSafe = Infinity, trustedAtSafe = 0;
    for (let i = 0; i < M; i++) {
        const trusted = M - i, trustedBad = totalBad - seenBad;
        if (trustedBad <= target && sorted[i].m < tauSafe) { tauSafe = sorted[i].m; trustedAtSafe = trusted; }
        if (!sorted[i].ok) seenBad++;
    }
    // risk-vs-τ curve at representative frame thresholds
    const taus = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64];
    const curve = taus.map(tau => {
        let tr = 0, bad = 0;
        for (const d of finite) if (d.m >= tau) { tr++; if (!d.ok) bad++; }
        return { tauFrames: tau, nonCertTrustedFrac: +(tr / Math.max(1, total)).toFixed(6),
            trustedDisagreements: bad, trustedDisagreeRate: tr ? +(bad / tr).toFixed(8) : null };
    });

    const certShare = +(certified.length / total).toFixed(6);
    const nonCertTrustedFrac = tauSafe === Infinity ? 0 : +(trustedAtSafe / total).toFixed(6);
    const nnShare = +(certShare + nonCertTrustedFrac).toFixed(6);
    return {
        total, certified: certified.length, certShare, certifiedDisagreements,
        nonCertified: nonCert.length,
        overallAgree: +(1 - all.filter(d => !d.agree).length / total).toFixed(6),
        marginIsUsableConfidenceSignal: monotone,
        chosenTauFrames: tauSafe === Infinity ? null : tauSafe,
        nonCertTrustedFracAtTau: nonCertTrustedFrac,
        NNshare_estimate: tauSafe === Infinity ? certShare : nnShare,
        reliability, riskVsTau: curve,
        note: tauSafe === Infinity
            ? 'No finite τ achieves the trusted-disagreement target on non-certified commits — NN-share = cert-share only (zero-risk path).'
            : 'τ = smallest non-certified scan-t margin (frames) with ≤target trusted disagreements. NN-share = certShare + non-cert trusted fraction. Frozen ONE-SHOT; the sealed verdict checks generalization.',
    };
}

function main() {
    const a = parseArgs();
    const raw = JSON.parse(fs.readFileSync(a.in, 'utf8'));
    const records = Array.isArray(raw) ? raw : raw.records;
    const res = analyze(records, a.target);
    const frozen = {
        spec: 'frozen L1e τ (scan-t margin, frames) from the calibration set',
        source: path.basename(a.in), label: a.label,
        distribution: raw.summary ? raw.summary.distribution : null,
        seedRange: raw.summary ? raw.summary.seedRange : null,
        calibCommits: res.total, targetTrustedDisagree: a.target,
        chosenTau: res.chosenTauFrames,                 // FRAMES — verdict reads this
        certShare: res.certShare, certifiedDisagreements: res.certifiedDisagreements,
        nonCertTrustedFracAtTau: res.nonCertTrustedFracAtTau,
        L1e_NNshare_estimate: res.NNshare_estimate,
        marginIsUsableConfidenceSignal: res.marginIsUsableConfidenceSignal,
        overallNNagree: res.overallAgree,
        riskVsTau: res.riskVsTau, reliability: res.reliability,
        note: res.note,
        inputSha256: crypto.createHash('sha256').update(JSON.stringify(records)).digest('hex'),
        FROZEN: true,
    };
    if (res.certifiedDisagreements > 0)
        frozen.SOUNDNESS_ALERT = 'CERTIFIED commits disagree with prod (' + res.certifiedDisagreements
            + ') — eg_bound certificate is UNSOUND on the calibration distribution. INVESTIGATE before any verdict.';
    console.log(JSON.stringify(frozen, null, 1));
    fs.writeFileSync(a.out, JSON.stringify(frozen, null, 1));
    process.stderr.write('froze L1e τ -> ' + a.out + (frozen.SOUNDNESS_ALERT ? '  [SOUNDNESS_ALERT]' : '') + '\n');
    process.exit(res.certifiedDisagreements > 0 ? 3 : 0);
}
if (require.main === module) main();
module.exports = { analyze, FBINS };
