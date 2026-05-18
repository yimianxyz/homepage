// Side-by-side gate + metric table for multiple eval reports.
//   node dev/compare.js dev/reports/*.eval.json
'use strict';
const fs = require('fs');
const path = require('path');

function fmt(v, w, prec) {
    if (v == null || v === undefined) return 'n/a'.padStart(w);
    if (typeof v === 'boolean') return (v ? 'PASS' : 'FAIL').padStart(w);
    if (typeof v !== 'number') return String(v).padStart(w);
    if (Math.abs(v) < 1 && Math.abs(v) > 1e-15) {
        if (v < 1e-3) return v.toExponential(prec).padStart(w);
        return v.toFixed(prec).padStart(w);
    }
    return v.toFixed(prec).padStart(w);
}

function main() {
    const files = process.argv.slice(2);
    if (!files.length) {
        console.error('usage: node dev/compare.js <eval.json> [...]');
        process.exit(1);
    }
    const reports = files.map(f => {
        const r = JSON.parse(fs.readFileSync(f, 'utf8'));
        return { f, r };
    });
    // Header
    const name = (f) => path.basename(f).replace(/\.eval\.json$/, '');
    const W = 14;
    const cols = reports.map(x => name(x.f).padStart(W).slice(-W));
    console.log('metric'.padEnd(22) + cols.join(' '));
    console.log('-'.repeat(22 + cols.length * (W + 1)));
    const rows = [
        ['params', x => x.r.model.totalParams, 0],
        ['mse', x => x.r.metrics.regression.mse, 3],
        ['maxAbs', x => x.r.metrics.regression.maxAbs, 4],
        ['mse(hunt)', x => x.r.metrics.regression.byBranch.hunt && x.r.metrics.regression.byBranch.hunt.mse, 3],
        ['mse(patrol)', x => x.r.metrics.regression.byBranch.patrol && x.r.metrics.regression.byBranch.patrol.mse, 3],
        ['mse(rEdge)', x => x.r.metrics.regression.rEdge && x.r.metrics.regression.rEdge.mse, 3],
        ['decisionAgr', x => x.r.metrics.decision.agreement, 4],
        ['rEdgeAgr', x => x.r.metrics.decision.rEdgeAgreement, 4],
        ['gates.allPassed', x => x.r.gates.allPassed],
        ['gates.mse', x => x.r.gates.mse],
        ['gates.maxAbs', x => x.r.gates.maxAbs],
        ['gates.decisionAgr', x => x.r.gates.decisionAgreement],
        ['gates.rEdgeAgr', x => x.r.gates.rEdgeAgreement],
        ['gates.catchLCPAll', x => x.r.gates.catchLCPAllSeeds],
        ['gates.meanSpeedClose', x => x.r.gates && x.r.gates.meanSpeedClose],
        ['gates.huntFracClose', x => x.r.gates && x.r.gates.huntFracClose],
    ];
    for (const [label, fn, prec] of rows) {
        const cells = reports.map(x => fmt(fn(x), W, prec));
        console.log(label.padEnd(22) + cells.join(' '));
    }
    console.log();
    // Divergence summary
    console.log('Divergence (catchLCP / ruleCatches per seed):');
    for (let i = 0; i < reports.length; i++) {
        const r = reports[i].r;
        const divs = (r.metrics.divergence || []).map(d =>
            `${d.seed}:${d.catchLCP}/${d.ruleCatchCount}`).join(' ');
        console.log('  ' + name(reports[i].f).padEnd(W) + ' ' + divs);
    }
    console.log();
    console.log('Behavior mean catches (rule | nn) per arch:');
    for (let i = 0; i < reports.length; i++) {
        const r = reports[i].r;
        const b = r.metrics.behavior && r.metrics.behavior.summary;
        if (!b) { console.log('  ' + name(reports[i].f).padEnd(W) + ' (no behavior)'); continue; }
        console.log('  ' + name(reports[i].f).padEnd(W) + ` ${fmt(b.catches.rule.mean, 8, 2)} | ${fmt(b.catches.nn.mean, 8, 2)}   meanSpd ${fmt(b.meanSpeed.rule.mean, 6, 3)}|${fmt(b.meanSpeed.nn.mean, 6, 3)}   huntFrac ${fmt(b.huntFrac.rule.mean, 6, 3)}|${fmt(b.huntFrac.nn.mean, 6, 3)}`);
    }
}

if (require.main === module) main();
