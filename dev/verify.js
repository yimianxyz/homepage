// Final gate. Loads the shipped js/predator_weights.json and runs the full
// eval suite. Writes a single JSON report at dev/reports/final.json. I read
// this report and decide pass/fail.
//
//   node dev/verify.js

'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const WEIGHTS = path.join(__dirname, '..', 'js', 'predator_weights.json');
const REPORT = path.join(__dirname, 'reports', 'final.json');

function main() {
    if (!fs.existsSync(WEIGHTS)) {
        const out = { status: 'fail', reason: 'no shipped weights at ' + WEIGHTS };
        fs.mkdirSync(path.dirname(REPORT), { recursive: true });
        fs.writeFileSync(REPORT, JSON.stringify(out, null, 2));
        process.stdout.write(JSON.stringify(out) + '\n');
        process.exit(2);
    }
    const r = spawnSync('node', [
        path.join(__dirname, 'eval.js'),
        '--weights', WEIGHTS,
        '--report', REPORT,
        '--frames', '5000',
        '--testStates', '50000',
        '--divSeeds', '8',
    ], { stdio: 'inherit' });
    if (r.status !== 0) {
        process.stdout.write(JSON.stringify({ status: 'fail', reason: 'eval exited ' + r.status }) + '\n');
        process.exit(r.status || 1);
    }
    // Summarize.
    const report = JSON.parse(fs.readFileSync(REPORT, 'utf8'));
    const g = report.gates;
    process.stdout.write(JSON.stringify({
        status: g.allPassed ? 'pass' : 'inspect',
        gates: g,
        regressionMse: report.metrics.regression.mse,
        regressionMaxAbs: report.metrics.regression.maxAbs,
        decisionAgreement: report.metrics.decision.agreement,
        rEdgeAgreement: report.metrics.decision.rEdgeAgreement,
        divergenceLCP: report.metrics.divergence.map(d => `${d.seed}:${d.catchLCP}/${d.ruleCatchCount}`),
        finalLinf: report.metrics.divergence.map(d => `${d.seed}:${d.finalLinf.toFixed(3)}`),
    }, null, 2) + '\n');
}

if (require.main === module) main();
