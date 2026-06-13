// shard_runner.js — runs the EXACT-NN L0 verification matrix across local cores.
//
// Builds the cell list (device × regime × spawn × UA × pristine slices), fans
// cells across CONC child processes (one diff_harness.js invocation each),
// streams results to runs/matrix_results.jsonl, and prints an aggregate.
// Exit 0 only if EVERY cell reports 0 force mismatches and 0 decision
// disagreements. Seeds: held-out discipline (>=270000), disjoint range per
// cell (base 272000 + cellIdx*1000).
//
//   node dev/exact_nn/shard_runner.js [--conc N] [--out dir] [--quick]
//
// --quick cuts seeds ~10x for a smoke pass of the whole matrix shape.
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const HARNESS = path.join(__dirname, 'diff_harness.js');
const L0 = path.join(__dirname, 'candidates', 'l0.js');

function buildCells(quick) {
    const q = (n) => quick ? Math.max(1, Math.ceil(n / 10)) : n;
    // device cells: [W, H, uaMobile, tag]
    const DEVICES = [
        [390, 844, false, 'mob390'],       // width<=768 -> NUM_BOIDS=60 like prod
        [820, 1180, false, 'desk820'],     // desktop window at iPad size -> N=120
        [820, 1180, true, 'ipad820'],      // REAL iPad UA -> N=60 (regex path)
        [1024, 768, false, 'desk1024'],
        [1512, 982, false, 'desk1512'],
        [1680, 1050, false, 'desk1680'],
        [2560, 1440, false, 'desk2560'],
    ];
    const cells = [];
    let idx = 0;
    const push = (tag, extra) => {
        cells.push(Object.assign({ tag, seedStart: 272000 + idx * 1000 }, extra));
        idx++;
    };
    for (const [W, H, ua, dtag] of DEVICES) {
        const base = { W, H, ua };
        // full games: spawn-point flock -> planner phase -> natural 6->5 gate -> endgame -> extinction
        push(dtag + ':full', Object.assign({}, base, { seeds: q(W >= 2000 ? 12 : W >= 1400 ? 20 : W <= 800 ? 50 : 30), maxFrames: 30000 }));
        // endgame-only: scattered singletons at every N<=5
        for (const n of [1, 2, 3, 4, 5]) {
            push(dtag + ':end' + n, Object.assign({}, base, { startBoids: n, scatter: true, seeds: q(100), maxFrames: 6000 }));
        }
        // gate-crossing: start just above the gate, scattered
        for (const n of [6, 7, 8]) {
            push(dtag + ':gate' + n, Object.assign({}, base, { startBoids: n, scatter: true, seeds: q(60), maxFrames: 12000 }));
        }
        // spawn-schedule games (tap-to-spawn): endgame commit then upward
        // re-crossings; two same-coordinate spawns -> duplicate boids/candidates
        const spawnA = JSON.stringify([
            { frame: 120, x: Math.floor(W / 2), y: Math.floor(H / 2) },
            { frame: 120, x: Math.floor(W / 2), y: Math.floor(H / 2) },
            { frame: 300, x: 100, y: 100 }]);
        const spawnB = JSON.stringify([
            { frame: 200, x: 5, y: 5 }, { frame: 201, x: W - 5, y: H - 5 },
            { frame: 500, x: Math.floor(W / 2), y: 5 }, { frame: 501, x: Math.floor(W / 2), y: 5 }]);
        push(dtag + ':spawnA', Object.assign({}, base, { startBoids: 5, scatter: true, seeds: q(60), maxFrames: 12000, spawnScript: spawnA }));
        push(dtag + ':spawnB', Object.assign({}, base, { startBoids: 3, scatter: true, seeds: q(60), maxFrames: 12000, spawnScript: spawnB }));
        // pristine slice: the COMMITTED artifact, no debug transform anywhere
        push(dtag + ':pristine-full', Object.assign({}, base, { seeds: q(4), maxFrames: 30000, noDecisions: true }));
    }
    return cells;
}

function cellArgs(c) {
    const a = ['--candidate', L0, '--W', c.W, '--H', c.H,
        '--seedStart', c.seedStart, '--seeds', c.seeds,
        '--maxFrames', c.maxFrames, '--json'];
    if (c.startBoids) a.push('--startBoids', c.startBoids, );
    if (c.scatter) a.push('--scatter');
    if (c.ua) a.push('--uaMobile');
    if (c.spawnScript) a.push('--spawnScript', c.spawnScript);
    if (c.noDecisions) a.push('--noDecisions');
    return a.map(String);
}

function runCell(c, outDir) {
    return new Promise((resolve) => {
        const args = [HARNESS, ...cellArgs(c),
            '--out', path.join(outDir, 'mismatches.jsonl')];
        const t0 = Date.now();
        const ch = spawn(process.execPath, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let out = '', err = '';
        ch.stdout.on('data', d => out += d);
        ch.stderr.on('data', d => err += d);
        ch.on('close', (code) => {
            let parsed = null;
            try { parsed = JSON.parse(out.trim().split('\n').pop()); } catch (e) { /* leave null */ }
            resolve({ tag: c.tag, code, result: parsed, err: err.slice(0, 2000),
                wallSec: +((Date.now() - t0) / 1000).toFixed(1) });
        });
    });
}

async function main() {
    const argv = process.argv;
    let conc = Math.max(1, os.cpus().length - 0), outDir = path.join(__dirname, 'runs'), quick = false;
    for (let i = 2; i < argv.length; i++) {
        if (argv[i] === '--conc') conc = +argv[++i];
        else if (argv[i] === '--out') outDir = argv[++i];
        else if (argv[i] === '--quick') quick = true;
    }
    fs.mkdirSync(outDir, { recursive: true });
    const cells = buildCells(quick);
    console.log('matrix: ' + cells.length + ' cells, conc=' + conc + (quick ? ' (QUICK)' : ''));
    const results = [];
    const resStream = fs.createWriteStream(path.join(outDir, 'matrix_results.jsonl'), { flags: 'a' });
    let next = 0, running = 0, done = 0, t0 = Date.now();
    await new Promise((resolveAll) => {
        const pump = () => {
            while (running < conc && next < cells.length) {
                const c = cells[next++];
                running++;
                runCell(c, outDir).then((r) => {
                    running--; done++;
                    results.push(r);
                    resStream.write(JSON.stringify(r) + '\n');
                    const res = r.result || {};
                    console.log('[' + done + '/' + cells.length + '] ' + r.tag +
                        '  frames=' + (res.frames != null ? res.frames : '?') +
                        '  mm=' + (res.mismatches != null ? res.mismatches : '?') +
                        '  dec=' + (res.decisions ? (res.decisions.planDisagree + '/' + res.decisions.egDisagree) : '-') +
                        '  fpm/core=' + (res.framesPerMinPerCore || '?') +
                        '  wall=' + r.wallSec + 's' + (r.code !== 0 ? '  EXIT=' + r.code : ''));
                    pump();
                });
            }
            if (running === 0 && next >= cells.length) resolveAll();
        };
        pump();
    });
    resStream.end();

    // aggregate
    let frames = 0, mm = 0, planD = 0, egD = 0, plans = 0, egc = 0, games = 0, cleared = 0;
    const reg = { planner: 0, intercept: 0, zero: 0 };
    let bad = [];
    for (const r of results) {
        const res = r.result;
        if (!res) { bad.push(r.tag + ' (no result, exit ' + r.code + ')'); continue; }
        frames += res.frames; games += res.games; cleared += res.cleared;
        mm += res.mismatches;
        reg.planner += res.framesByRegime.planner;
        reg.intercept += res.framesByRegime.intercept;
        reg.zero += res.framesByRegime.zero;
        if (res.decisions) {
            plans += res.decisions.plans; planD += res.decisions.planDisagree;
            egc += res.decisions.egCommits; egD += res.decisions.egDisagree;
        }
        if (r.code !== 0) bad.push(r.tag + ' (exit ' + r.code + ', mm=' + res.mismatches + ')');
    }
    const wallMin = (Date.now() - t0) / 60000;
    const summary = { cells: cells.length, games, cleared, frames,
        framesByRegime: reg, forceMismatches: mm,
        decisions: { plans, planDisagree: planD, egCommits: egc, egDisagree: egD },
        wallMin: +wallMin.toFixed(1),
        framesPerMin: Math.round(frames / wallMin),
        bad };
    fs.writeFileSync(path.join(outDir, 'matrix_summary.json'), JSON.stringify(summary, null, 2));
    console.log(JSON.stringify(summary, null, 2));
    process.exit(bad.length || mm || planD || egD ? 2 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
