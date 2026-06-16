// tstar_fit.js — fit candidate T(screen,N0) formulas to the throughput SURFACE
// (not the noisy argmax) and pick the best by PREVALENCE-WEIGHTED throughput. Also
// runs the confound regression (T* ~ logArea + N0) on the cross cells to separate the
// area effect from the boid-count effect, and applies the COMPLEXITY BAR (if the best
// continuous formula beats fixed-8 by <2% on 1920×1080, prefer a device-class step).
//
//   node tstar_fit.js fit_surface.json [eval_surface.json]
// If eval_surface is given, formulas are FIT on the first and SCORED on the second
// (held-out → sealed). Deployment cells = real-N0 cells (forcedN0===0). Cross cells
// (forcedN0>0) feed the confound regression only.
'use strict';
const fs = require('fs');

// Approx global screen prevalence (StatCounter 2025: 360×800 then 390×844 top mobile;
// 1920×1080 ~23% top desktop then 1366×768, 1536×864). Mobile ≈55% of global traffic.
// Within-class shares are renormalised over the tested cells. Documented assumption;
// a uniform-within-class sensitivity check is printed alongside.
const PREV = {
    // mobile (logical CSS px) — sum ≈ 0.55
    '360x800': 0.20, '390x844': 0.15, '393x852': 0.07, '412x915': 0.06, '414x896': 0.07,
    // laptop / desktop — sum ≈ 0.45 (1920×1080 dominant)
    '1280x720': 0.02, '1366x768': 0.07, '1440x900': 0.035, '1536x864': 0.07, '1512x982': 0.03,
    '1600x900': 0.025, '1680x1050': 0.015, '1920x1080': 0.17, '2560x1440': 0.035,
};
const clampT = (t, Ts) => Math.max(Ts[0], Math.min(Ts[Ts.length - 1], Math.round(t)));
const meanAtT = (cell, T) => { const p = cell.perT.find(x => x.T === T); return p ? p.mean : null; };

function loadSurface(fp) { const s = JSON.parse(fs.readFileSync(fp, 'utf8'));
    s.deploy = s.cells.filter(c => !c.forcedN0); s.cross = s.cells.filter(c => c.forcedN0); return s; }
function wkey(c) { return `${c.W}x${c.H}`; }
function weights(cells, mode) {
    const w = {};
    if (mode === 'uniformClass') { const mob = cells.filter(c => c.N0 === 60), des = cells.filter(c => c.N0 !== 60);
        for (const c of mob) w[c.key] = 0.55 / mob.length; for (const c of des) w[c.key] = 0.45 / des.length; }
    else { let tot = 0; for (const c of cells) { w[c.key] = PREV[wkey(c)] != null ? PREV[wkey(c)] : 0.01; tot += w[c.key]; }
        for (const c of cells) w[c.key] /= tot; }
    return w;
}

// weighted throughput of a cell→T policy over deployment cells (uses the surface curve)
function wThru(cells, w, Ts, policy) {
    let s = 0; for (const c of cells) { const T = clampT(policy(c), Ts); const m = meanAtT(c, T); if (m != null) s += w[c.key] * m; } return s;
}

// ---- candidate formula family (each returns {name, T(cell), params}) ----
function fitFormulas(fitCells, Ts) {
    const w = weights(fitCells, 'prev');
    const cand = [];
    // fixed baselines + best fixed
    for (const T of Ts) cand.push({ name: 'fixed:T=' + T, kind: 'fixed', T: () => T, params: { T } });
    // theta*N0  (≈ device step since N0∈{60,120})
    let best = null;
    for (let th = 0.02; th <= 0.22; th += 0.001) { const f = c => th * c.N0;
        const v = wThru(fitCells, w, Ts, f); if (!best || v > best.v) best = { v, th }; }
    cand.push({ name: 'theta*N0', kind: 'theta', T: c => best.th * c.N0, params: { theta: +best.th.toFixed(3) } });
    // lindim: a + b*sqrtArea
    best = null;
    for (let a = -12; a <= 12; a += 0.5) for (let b = 0; b <= 0.02; b += 0.0002) { const f = c => a + b * c.sqrtArea;
        const v = wThru(fitCells, w, Ts, f); if (!best || v > best.v) best = { v, a, b }; }
    cand.push({ name: 'a+b*sqrtA', kind: 'lindim', T: c => best.a + best.b * c.sqrtArea, params: { a: +best.a.toFixed(2), b: +best.b.toFixed(5) } });
    // power: a*area^p
    best = null;
    for (let p = 0.1; p <= 1.0; p += 0.02) for (let a = 0.0005; a <= 0.5; a *= 1.1) { const f = c => a * Math.pow(c.area, p);
        const v = wThru(fitCells, w, Ts, f); if (!best || v > best.v) best = { v, a, p }; }
    cand.push({ name: 'a*A^p', kind: 'power', T: c => best.a * Math.pow(c.area, best.p), params: { a: +best.a.toPrecision(3), p: +best.p.toFixed(2) } });
    // density: a + b*(N0/area*1e6)
    best = null;
    for (let a = -4; a <= 18; a += 0.5) for (let b = -200; b <= 200; b += 2) { const f = c => a + b * (c.N0 / c.area * 1e6);
        const v = wThru(fitCells, w, Ts, f); if (!best || v > best.v) best = { v, a, b }; }
    cand.push({ name: 'a+b*dens', kind: 'density', T: c => best.a + best.b * (c.N0 / c.area * 1e6), params: { a: +best.a.toFixed(2), b: +best.b.toFixed(2) } });
    // mindim: a + b*min(W,H)
    best = null;
    for (let a = -8; a <= 12; a += 0.5) for (let b = 0; b <= 0.02; b += 0.0002) { const f = c => a + b * c.minDim;
        const v = wThru(fitCells, w, Ts, f); if (!best || v > best.v) best = { v, a, b }; }
    cand.push({ name: 'a+b*minDim', kind: 'mindim', T: c => best.a + best.b * c.minDim, params: { a: +best.a.toFixed(2), b: +best.b.toFixed(5) } });
    // device-class step (explicit lookup): per-class throughput-weighted-optimal T
    const cls = { mob: fitCells.filter(c => c.N0 === 60), des: fitCells.filter(c => c.N0 !== 60) };
    const bestClassT = grp => { const wl = weights(grp, 'prev'); let bb = null;
        for (const T of Ts) { let s = 0; for (const c of grp) s += wl[c.key] * (meanAtT(c, T) || 0); if (!bb || s > bb.s) bb = { s, T }; } return bb.T; };
    const Tm = cls.mob.length ? bestClassT(cls.mob) : 6, Td = cls.des.length ? bestClassT(cls.des) : 8;
    cand.push({ name: 'device-step', kind: 'step', T: c => (c.N0 === 60 ? Tm : Td), params: { mobile: Tm, desktop: Td } });
    return cand;
}

// ---- confound regression: T* ~ a + b*log(area) + c*N0 over cross cells (OLS) ----
function regress(cells, yOf) {
    // design: [1, logArea_centered, N0_centered]
    const n = cells.length; if (n < 4) return null;
    const la = cells.map(c => Math.log(c.area)), n0 = cells.map(c => c.N0), y = cells.map(yOf);
    const mLa = avg(la), mN0 = avg(n0), mY = avg(y);
    const X = cells.map((c, i) => [1, la[i] - mLa, n0[i] - mN0]);
    // normal equations (3x3)
    const XtX = mat3(); const Xty = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) { Xty[a] += X[i][a] * y[i];
        for (let b = 0; b < 3; b++) XtX[a][b] += X[i][a] * X[i][b]; }
    const beta = solve3(XtX, Xty); if (!beta) return null;
    // residual std + coef se (approx)
    let rss = 0; for (let i = 0; i < n; i++) { const yh = beta[0] * 1 + beta[1] * X[i][1] + beta[2] * X[i][2]; rss += (y[i] - yh) ** 2; }
    const s2 = rss / Math.max(1, n - 3); const inv = inv3(XtX);
    const se = inv ? [0, 1, 2].map(k => Math.sqrt(s2 * inv[k][k])) : [null, null, null];
    return { intercept: beta[0] + mY * 0 + beta[0] * 0 || beta[0], bLogArea: beta[1], bN0: beta[2],
             seLogArea: se[1], seN0: se[2], tLogArea: se[1] ? beta[1] / se[1] : null, tN0: se[2] ? beta[2] / se[2] : null, n };
}
function avg(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
function mat3() { return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; }
function solve3(A, b) { const M = A.map((r, i) => r.concat(b[i]));
    for (let col = 0; col < 3; col++) { let piv = col; for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
        if (Math.abs(M[piv][col]) < 1e-12) return null; [M[col], M[piv]] = [M[piv], M[col]];
        for (let r = 0; r < 3; r++) if (r !== col) { const f = M[r][col] / M[col][col]; for (let k = col; k <= 3; k++) M[r][k] -= f * M[col][k]; } }
    return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]]; }
function inv3(A) { const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(r => r.slice()); const M = A.map(r => r.slice());
    for (let col = 0; col < 3; col++) { let piv = col; for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
        if (Math.abs(M[piv][col]) < 1e-12) return null; [M[col], M[piv]] = [M[piv], M[col]]; [I[col], I[piv]] = [I[piv], I[col]];
        const d = M[col][col]; for (let k = 0; k < 3; k++) { M[col][k] /= d; I[col][k] /= d; }
        for (let r = 0; r < 3; r++) if (r !== col) { const f = M[r][col]; for (let k = 0; k < 3; k++) { M[r][k] -= f * M[col][k]; I[r][k] -= f * I[col][k]; } } }
    return I; }
// plateau center = mean of plateau T's (robust optimal-T proxy for regression)
function plateauCenter(c) { return c.plateau && c.plateau.length ? avg(c.plateau) : c.Tstar; }

function main() {
    const fit = loadSurface(process.argv[2]);
    const ev = process.argv[3] ? loadSurface(process.argv[3]) : fit;
    const Ts = fit.Ts;
    const cands = fitFormulas(fit.deploy, Ts);
    // score on eval deployment cells (prevalence + uniform-class)
    const wPrev = weights(ev.deploy, 'prev'), wUni = weights(ev.deploy, 'uniformClass');
    const base8 = wThru(ev.deploy, wPrev, Ts, () => 8), base5 = wThru(ev.deploy, wPrev, Ts, () => 5);
    console.log(`# FORMULA FIT — fit=${fit.seedSet}  eval=${ev.seedSet}  (deploy cells=${ev.deploy.length})`);
    console.log(`# baselines weighted-thru (e-4): fixed8=${(base8 * 1e4).toFixed(3)}  fixed5=${(base5 * 1e4).toFixed(3)}\n`);
    const rows = [];
    for (const cand of cands) {
        const vP = wThru(ev.deploy, wPrev, Ts, cand.T), vU = wThru(ev.deploy, wUni, Ts, cand.T);
        const gain8 = (vP - base8) / base8 * 100, gain5 = (vP - base5) / base5 * 100;
        const assign = ev.deploy.map(c => `${wkey(c)}@${c.N0}→${clampT(cand.T(c), Ts)}`);
        rows.push({ name: cand.name, params: cand.params, wThruPrev: vP, wThruUni: vU, gain8, gain5, assign });
    }
    rows.sort((a, b) => b.wThruPrev - a.wThruPrev);
    for (const r of rows) console.log(`${pad(r.name, 13)} thru=${(r.wThruPrev * 1e4).toFixed(3)} (uni ${(r.wThruUni * 1e4).toFixed(3)})  vs8=${sgn(r.gain8)}%  vs5=${sgn(r.gain5)}%  ${JSON.stringify(r.params)}`);
    // 1920x1080 specific gain (complexity bar) — best continuous vs fixed-8
    const c1080 = ev.deploy.find(c => c.W === 1920 && c.H === 1080 && c.N0 !== 60);
    console.log('\n# Per-deployment-screen T assignment (winner): ' + (rows[0] ? rows[0].assign.join('  ') : ''));
    if (c1080) { const m8 = meanAtT(c1080, 8);
        console.log('# 1920×1080 complexity-bar check (vs fixed-8):');
        for (const r of rows.slice(0, 6)) { const T = clampT(cands.find(c => c.name === r.name).T(c1080), Ts);
            const m = meanAtT(c1080, T); console.log(`   ${pad(r.name, 13)} →T=${T}  thru=${(m * 1e4).toFixed(3)} vs T8 ${(m8 * 1e4).toFixed(3)}  = ${sgn((m - m8) / m8 * 100)}%`); } }
    // confound regression on cross cells (+ all cells)
    console.log('\n# CONFOUND REGRESSION  optimalT ~ a + b·ln(area) + c·N0');
    for (const [lab, cells] of [['cross-only', ev.cross], ['all-cells', ev.cells]]) {
        if (cells.length < 4) { console.log(`  ${lab}: n=${cells.length} (too few)`); continue; }
        const reg = regress(cells, plateauCenter);
        if (!reg) { console.log(`  ${lab}: singular`); continue; }
        console.log(`  ${lab} (n=${reg.n}): b·ln(area)=${reg.bLogArea.toFixed(3)} (t=${fmt(reg.tLogArea)})   c·N0=${reg.bN0.toFixed(4)} (t=${fmt(reg.tN0)})`);
        console.log(`     → ΔT per ×2 area = ${(reg.bLogArea * Math.log(2)).toFixed(2)};  ΔT per +60 boids = ${(reg.bN0 * 60).toFixed(2)}`);
    }
    // dump the cross cells T* for inspection
    if (ev.cross.length) { console.log('\n# Cross cells (confound grid)  key → T* [plateau]');
        for (const c of ev.cross.sort((a, b) => a.area - b.area || a.N0 - b.N0)) console.log(`   ${pad(c.key, 18)} area=${(c.area/1e6).toFixed(2)}M N0=${c.N0}  T*=${c.Tstar} [${c.plateau.join(',')}]`); }
}
function pad(s, n) { return (s + ' '.repeat(n)).slice(0, n); }
function sgn(x) { return (x >= 0 ? '+' : '') + x.toFixed(2); }
function fmt(x) { return x == null ? 'n/a' : x.toFixed(2); }
if (require.main === module) main();
