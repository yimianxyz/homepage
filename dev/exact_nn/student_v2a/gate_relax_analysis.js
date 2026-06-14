#!/usr/bin/env node
// Gate-relaxation analysis for the L1h exact-NN barrier.
//
// Question: SPEC §4c allows a "rule-of-three" RESIDUAL. If we relax the
// "0 trusted disagreements" gate to allow a trusted-disagreement RATE r,
// does the NN fast-path share (trusted fraction) become non-trivial while
// remaining (approximately) exact?
//
// For each allowed rate r, find the SMALLEST tau such that among plans with
// margin >= tau, the disagreement fraction <= r. Report the resulting
// trustedFrac = (#plans with margin>=tau) / total.
//
// We also report the disagreement rate in the very top margin band, to see
// whether it ever approaches 0 (the "monotone but floors" claim).

const fs = require('fs');
const path = require('path');

const FILE = path.join(__dirname, 'calib_margins_indep.json');
const data = JSON.parse(fs.readFileSync(FILE, 'utf8'));
const N = data.length;

// Sort by margin DESCENDING. We sweep a threshold tau from high to low.
// For a given tau, the trusted set is { margin >= tau }.
// As tau decreases, we accumulate plans in margin order.
const sorted = data.slice().sort((a, b) => b.margin - a.margin);

// Precompute suffix-from-top cumulative stats:
// cum[i] = stats for the top (i+1) plans by margin = {trusted count, disagree count}
// disagree := !agree (a "trusted disagreement" = we committed the student's
// target but it disagreed with the committed prod target).
const cumDisagree = new Array(N);
const cumCount = new Array(N);
let dis = 0;
for (let i = 0; i < N; i++) {
  if (!sorted[i].agree) dis++;
  cumDisagree[i] = dis;
  cumCount[i] = i + 1;
}

// For a given allowed rate r, we want the LARGEST trusted set (smallest tau,
// i.e. largest index i) such that cumDisagree[i] / cumCount[i] <= r.
// Because disagreement rate is NOT necessarily monotone in i, we scan all i
// and take the max count satisfying the constraint. This is the most generous
// reading of "find the smallest tau such that disagreement fraction <= r".
function bestForRate(r) {
  let bestIdx = -1;
  for (let i = 0; i < N; i++) {
    const rate = cumDisagree[i] / cumCount[i];
    if (rate <= r) bestIdx = i; // keep extending; take the deepest feasible
  }
  if (bestIdx < 0) {
    return { r, feasible: false, trustedFrac: 0, count: 0, tau: null,
             disagreements: 0, rate: null };
  }
  return {
    r,
    feasible: true,
    count: bestIdx + 1,
    trustedFrac: (bestIdx + 1) / N,
    tau: sorted[bestIdx].margin,         // smallest margin admitted
    disagreements: cumDisagree[bestIdx],
    rate: cumDisagree[bestIdx] / (bestIdx + 1),
  };
}

// ALSO report the STRICT prefix reading: the smallest tau such that the
// prefix {top-k} satisfies rate <= r for the LARGEST k that is still a
// contiguous-from-top feasible prefix (first violation stops us). This is the
// stricter, arguably more honest reading because a real gate is a threshold:
// everything above tau is trusted, so the relevant rate is the rate of the
// whole admitted set, and you cannot "skip" a bad band.
function bestPrefixForRate(r) {
  // Largest k such that for the prefix of size k, rate <= r.
  // (Same as bestForRate since the admitted set IS the prefix. Kept for
  //  clarity / documentation.)
  return bestForRate(r);
}

console.log('=== Gate-relaxation analysis (side-b indep calibration) ===');
console.log('total plans N =', N);
const totalDisagree = cumDisagree[N - 1];
console.log('overall disagreement (untrusted-all) =',
  totalDisagree, '/', N, '=', (totalDisagree / N).toFixed(6));
console.log('overall AGREE rate =', (1 - totalDisagree / N).toFixed(6));
console.log('');

// Baseline: the strict r=0 gate (the committed L1h gate).
console.log('--- per-rate results (trusted set = {margin >= tau}) ---');
const rates = [0, 1e-4, 1e-3, 1e-2];
for (const r of rates) {
  const res = bestForRate(r);
  if (!res.feasible) {
    console.log(`r=${r}: INFEASIBLE (even top-1 plan exceeds rate)`);
    continue;
  }
  console.log(
    `r=${r.toExponential(0).padStart(7)}: ` +
    `tau=${res.tau.toFixed(6).padStart(11)}  ` +
    `count=${String(res.count).padStart(6)}  ` +
    `trustedFrac=${res.trustedFrac.toFixed(6)}  ` +
    `(disagreements=${res.disagreements}, realRate=${res.rate.toExponential(3)})`
  );
}
console.log('');

// Top-of-distribution behaviour: bin the top plans and show disagreement rate
// as we descend. Does the top band ever approach 0 disagreement?
console.log('--- top-of-distribution disagreement (cumulative-from-top) ---');
const checkpoints = [10, 30, 100, 300, 1000, 3000, 10000, 30000, N];
for (const k of checkpoints) {
  if (k > N) continue;
  const idx = k - 1;
  console.log(
    `top ${String(k).padStart(6)} plans: ` +
    `tau>=${sorted[idx].margin.toFixed(6).padStart(10)}  ` +
    `disagreeRate=${(cumDisagree[idx] / cumCount[idx]).toFixed(6)}  ` +
    `(${cumDisagree[idx]}/${cumCount[idx]})`
  );
}
console.log('');

// The absolute best (lowest) cumulative disagreement rate achievable, and at
// what depth. Also the single-band minimum: among NON-overlapping margin bands,
// the lowest local disagreement rate.
let minCumRate = Infinity, minCumIdx = -1;
for (let i = 0; i < N; i++) {
  // require a minimum sample to avoid 1-plan flukes
  if (i + 1 < 30) continue;
  const rate = cumDisagree[i] / cumCount[i];
  if (rate < minCumRate) { minCumRate = rate; minCumIdx = i; }
}
console.log('best cumulative-from-top disagreement rate (>=30 plans):',
  minCumRate.toFixed(6), 'at depth', minCumIdx + 1,
  'tau>=', sorted[minCumIdx].margin.toFixed(6));

// Local band minimum: split into deciles of margin and report disagreement rate
console.log('');
console.log('--- local margin bands (high->low margin), disagreement rate ---');
const BANDS = 20;
for (let b = 0; b < BANDS; b++) {
  const lo = Math.floor((b) * N / BANDS);
  const hi = Math.floor((b + 1) * N / BANDS);
  let d = 0;
  for (let i = lo; i < hi; i++) if (!sorted[i].agree) d++;
  const cnt = hi - lo;
  console.log(
    `band ${String(b).padStart(2)} [margin ${sorted[hi-1].margin.toFixed(4)}..${sorted[lo].margin.toFixed(4)}]: ` +
    `disagreeRate=${(d / cnt).toFixed(4)}  (${d}/${cnt})`
  );
}
console.log('');

// Margin>1 band (the "high-conf floor" claim ~17-20%).
let m1d = 0, m1c = 0;
for (const p of data) { if (p.margin > 1) { m1c++; if (!p.agree) m1d++; } }
console.log('margin>1 band: count=', m1c,
  'disagreeRate=', m1c ? (m1d / m1c).toFixed(4) : 'n/a', `(${m1d}/${m1c})`);

// Even more extreme: top 1% by margin
const top1pct = Math.floor(N * 0.01);
let t1d = 0;
for (let i = 0; i < top1pct; i++) if (!sorted[i].agree) t1d++;
console.log('top 1% by margin: count=', top1pct,
  'disagreeRate=', (t1d / top1pct).toFixed(4), `tau>=${sorted[top1pct-1].margin.toFixed(4)}`);

// The very highest margins (top 50, individually-ish)
console.log('');
console.log('--- top 50 plans by margin (raw) ---');
let agreeTop50 = 0;
for (let i = 0; i < 50 && i < N; i++) if (sorted[i].agree) agreeTop50++;
console.log('agree among top 50 highest-margin plans:', agreeTop50, '/ 50');

// SUMMARY for structured output
console.log('');
console.log('=== SUMMARY ===');
const r0 = bestForRate(0);
const r1e4 = bestForRate(1e-4);
const r1e3 = bestForRate(1e-3);
const r1e2 = bestForRate(1e-2);
console.log(JSON.stringify({
  N,
  overallAgree: 1 - totalDisagree / N,
  r0_trustedFrac: r0.trustedFrac, r0_tau: r0.tau, r0_count: r0.count,
  r1e4_trustedFrac: r1e4.trustedFrac, r1e4_tau: r1e4.tau, r1e4_count: r1e4.count,
  r1e3_trustedFrac: r1e3.trustedFrac, r1e3_tau: r1e3.tau, r1e3_count: r1e3.count,
  r1e2_trustedFrac: r1e2.trustedFrac, r1e2_tau: r1e2.tau, r1e2_count: r1e2.count,
  bestCumDisagreeRate: minCumRate,
  margin_gt1_disagreeRate: m1c ? m1d / m1c : null,
  top1pct_disagreeRate: t1d / top1pct,
}, null, 2));
