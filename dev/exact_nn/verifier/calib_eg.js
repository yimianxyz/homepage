// calib_eg.js — collect the L1e endgame CALIBRATION record (the scan-t-margin
// analogue of the L1h calib). Drives prod endgame games on the PUBLISHED
// calibration seed range [270000,280000) and, at every egBoid commit decision,
// logs the NN student's would-be pick: {margin, cert, agree, n, cell}. The gate
// runs in SHADOW mode (candidates/l1e.js with EXACTNN_EG_CALIB=1) — it always
// abstains, so the game follows prod's EXACT trajectory and the logged commits
// are prod's real endgame states (no distribution drift from the candidate).
//
// `agree` = (NN argmin == eg_scan.egPick argmin), where eg_scan is the
// independently-verified bit-identical reimpl of prod's intercept() scan. A
// verdict_l1e --tau 0 --no-cert run cross-checks egPick≡prod's REAL egBoid (its
// harness egDisagree must equal the !agree count here) — so this calib is sound,
// not reimpl-trusting.
//
// Output: a JSON array consumable by tau_calibrate_eg.js (freezes τ ONE-SHOT on
// THIS set, never on sealed). Two distributions (report both — side-a's audit #3):
//   --scatter (default) : startBoids cycled 2..5 + scatter (cheap, high-volume,
//                         matches side-a's train/calib methodology → comparable)
//   --natural           : startBoids=0 full games → the endgame the planner
//                         actually produces (the deployable distribution)
//
//   node calib_eg.js --seeds 800 [--scatter|--natural] --out calib_eg.json
'use strict';
const path = require('path');
const fs = require('fs');
const { runGame } = require('../diff_harness.js');

const DEVICE_MATRIX = [
    { W: 390, H: 844 }, { W: 820, H: 1180 }, { W: 1024, H: 768 },
    { W: 1512, H: 982 }, { W: 1680, H: 1050 }, { W: 2560, H: 1440 },
];

function parseArgs(argv) {
    const a = { seeds: 600, natural: false, cells: null, maxFrames: null,
        out: path.join(__dirname, 'calib_eg.json'), base: 270000 };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--natural') a.natural = true;
        else if (k === '--scatter') a.natural = false;
        else if (k === '--cells') a.cells = argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--base') a.base = +argv[++i];          // seed range base (calib=270000)
        else if (k === '--out') a.out = argv[++i];
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}
function cells(opt) {
    if (!opt.cells) return DEVICE_MATRIX;
    return opt.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });
}

async function main() {
    const opt = parseArgs(process.argv);
    if (opt.base >= 280000 && opt.base < 290000) throw new Error('refusing dev/buffer range');
    if (opt.base >= 290000) throw new Error('REFUSING to calibrate on the SEALED range (≥290000)');
    const candidate = path.join(__dirname, '..', 'candidates', 'l1e.js');
    process.env.EXACTNN_EG_CALIB = '1';
    global.__l1eCalibLog = [];

    const cs = cells(opt);
    const maxFrames = opt.maxFrames || (opt.natural ? 30000 : 4000);
    let games = 0;
    for (const c of cs) {
        for (let i = 0; i < opt.seeds; i++) {
            const seed = opt.base + i;
            const startBoids = opt.natural ? 0 : (2 + (i % 4));     // 2..5
            await runGame({
                policyDir: path.join(__dirname, '..', '..', '..', 'js'),
                W: c.W, H: c.H, mode: 'lockstep', resync: true, decisions: true,
                fastRender: true, mismatchLimit: 1, postExtinct: 0,
                startBoids, scatter: !opt.natural, uaMobile: false, maxFrames,
            }, seed, candidate);
            games++;
        }
        const n = global.__l1eCalibLog.length;
        console.error(`[cell ${c.W}x${c.H}] commits so far: ${n}`);
    }
    const log = global.__l1eCalibLog;
    // summary
    const fin = log.filter(d => Number.isFinite(d.margin));
    const dis = log.filter(d => !d.agree).length;
    const certN = log.filter(d => d.cert).length;
    const soleReach = log.filter(d => d.n === 1).length;
    const summary = {
        distribution: opt.natural ? 'natural(full-game)' : 'scatter(startBoids 2..5)',
        seedRange: [opt.base, opt.base + opt.seeds], cells: cs.map(c => c.W + 'x' + c.H),
        games, commits: log.length, finiteMargin: fin.length,
        disagreements: dis, overallAgree: +(1 - dis / log.length).toFixed(6),
        certShare: +(certN / log.length).toFixed(6),
        soleReachable_n1: soleReach,
    };
    console.error(JSON.stringify(summary, null, 1));
    fs.writeFileSync(opt.out, JSON.stringify({ summary, records: log }));
    console.error('wrote ' + log.length + ' commit records -> ' + opt.out);
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { DEVICE_MATRIX };
