'use strict';
// (1) wall-time cost of short-K (sim only) vs full, to test "cost win shrinks as
//     you re-add flocking".
// (2) Sharper regime test: among boot-decided plans, fraction where the BEST
//     cheap-enough surrogate (short30) AND the best-accuracy surrogate
//     (nopredavoid) get the rolled-argmax right, stratified by local density and
//     by true decisive margin (the boot gap between top-2 rolled). If there is a
//     regime where boot is geometrically determined to ~0.01, the rolled-argmax
//     should be ~perfect there with a cheap surrogate.
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const IP = require('./intermediate_probe.js');
const DATA_DIR = path.join(__dirname, '..', 'data_1e6');
function* records(file) { const txt = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8'); for (const line of txt.split('\n')) { if (!line) continue; yield JSON.parse(line); } }

(async () => {
    await IP.init();
    const F = (c, s, a, p) => ({ incCoh: c, incSep: s, incAlign: a, incPredAvoid: p });
    const MODES = {
        short30: { kind: 'short', K: 30, fl: F(1, 1, 1, 1) },
        nopredavoid: { kind: 'flags', fl: F(1, 1, 1, 0) },
        full: { kind: 'full' },
    };
    const R = IP.makeRollout({});
    const N_REC = +(process.argv[2] || 700);
    const STRIDE = +(process.argv[3] || 37);
    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (STRIDE > 1) files = files.filter((_, i) => i % STRIDE === 0);

    const ns = {}; const cnt = {}; for (const m of Object.keys(MODES)) { ns[m] = 0; cnt[m] = 0; }
    // boot-decided argmax accuracy by (density bucket) x (true decisive margin bucket)
    const ND = 60;
    const buckets = {}; // key: mode|dens|marg -> {ok,tot}
    function bump(m, dens, marg, ok) { const k = m + '|' + dens + '|' + marg; if (!buckets[k]) buckets[k] = { ok: 0, tot: 0 }; buckets[k].tot++; if (ok) buckets[k].ok++; }

    let nrec = 0;
    outer:
    for (const file of files) {
        for (const r of records(path.join(DATA_DIR, file))) {
            const s = r.s, cands = r.cands;
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            let near = 0; for (let i = 0; i < s.bx.length; i++) { const dx = s.bx[i] - s.px, dy = s.by[i] - s.py; if (dx * dx + dy * dy < (2 * ND) ** 2) near++; }
            const dens = near <= 3 ? 'sparse' : (near <= 10 ? 'mid' : 'dense');
            const trueCatch = r.rolled.map(x => x[1]); const trueBoot = r.rolled.map(x => x[2] === null ? -Infinity : x[2]);
            const allEq = trueCatch.every(c => c === trueCatch[0]);
            // true decisive margin = top1-top2 boot among rolled (boot-decided only)
            let mb = trueBoot.slice().sort((a, b) => b - a); const decMarg = (mb.length >= 2 && isFinite(mb[0]) && isFinite(mb[1])) ? mb[0] - mb[1] : Infinity;
            const margB = decMarg < 0.01 ? '<0.01' : (decMarg < 0.05 ? '0.01-0.05' : (decMarg < 0.2 ? '0.05-0.2' : '>=0.2'));

            for (const m of Object.keys(MODES)) {
                const cheapBoot = [];
                for (let rk = 0; rk < r.rolled.length; rk++) {
                    const ci = r.rolled[rk][0]; const cand = cands[ci];
                    const t0 = process.hrtime.bigint();
                    const out = R.rollScore(s, cand[0], cand[1], MODES[m]);
                    const t1 = process.hrtime.bigint();
                    ns[m] += Number(t1 - t0); cnt[m]++;
                    cheapBoot.push(out.boot);
                }
                if (allEq) {
                    let tArg = 0, tB = -Infinity; for (let k = 0; k < trueBoot.length; k++) if (trueBoot[k] > tB) { tB = trueBoot[k]; tArg = k; }
                    let cArg = 0, cB = -Infinity; for (let k = 0; k < cheapBoot.length; k++) if (cheapBoot[k] > cB) { cB = cheapBoot[k]; cArg = k; }
                    bump(m, dens, margB, tArg === cArg);
                }
            }
            if (++nrec >= N_REC) break outer;
        }
    }
    const rep = { nrec, usPerRollout: {}, regimeArgmaxAgree: {} };
    for (const m of Object.keys(MODES)) rep.usPerRollout[m] = (ns[m] / cnt[m]) / 1000;
    for (const k of Object.keys(buckets).sort()) rep.regimeArgmaxAgree[k] = { agree: buckets[k].ok / buckets[k].tot, n: buckets[k].tot };
    console.log(JSON.stringify(rep, null, 2));
    fs.writeFileSync(path.join(__dirname, 'cost_regime_result.json'), JSON.stringify(rep, null, 2));
})();
