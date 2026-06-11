// certify_oracle.js — the FARM GATE: proves the instrumented fork
// (oracle_policy.js, hooks ACTIVE) is bitwise-identical to pristine prod
// before any dataset shard may be produced.
//
// Per device cell (dev/exact_nn/device_matrix.js):
//   full+gate   2 seeds, lockstep, full games to extinction (gate crossed
//               naturally; both regimes asserted present)
//   fork-traj   1 seed, fork mode (sim DRIVEN by the fork — the farm
//               configuration exactly)
//   spawn-mid   1 seed, lockstep, scripted taps during the planner phase
//               incl. a same-coordinate double-tap
//   spawn-recross  probe the frame where N first reaches 5, then re-run with
//               taps scheduled there: forces 5->6->7->...->5 gate re-crossing
//               with the egBoid/frame-counter state alive across it
//   spawn-spam  lockstep, 48 taps in 48 consecutive frames (mobile cell goes
//               past 100, desktop past 160 live boids)
//
// Hook coverage is asserted per run (planStart==planEnd>0, roll==4*planEnd,
// frameEnd==frames) — a certificate that never exercised the logging paths
// would be vacuous.
//
// Output: dev/exact_nn/CERT.json  { certRunId, oracleSha, srcSha, runs, totals }
//         dev/exact_nn/CERT.md    human-readable summary
// oracle_logger.js refuses to farm unless CERT.json matches the current
// oracle_policy.js sha. Any mismatch anywhere => no certificate (exit 2).
//
//   node dev/exact_nn/certify_oracle.js [--concurrency 4]
//   node dev/exact_nn/certify_oracle.js --cell ipad_820x1180   (one cell, JSON to stdout)
//
// Cert seeds live in the held-out range (>=270000): certification is
// verification, not training; train seeds stay untouched for the dataset.
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');
const { CELLS } = require('./device_matrix.js');

const CAND = path.join(__dirname, 'oracle_candidate.js');
const CERT_SEED_BASE = 271200;   // disjoint from diff_harness selftest seeds (2710xx)

function sha256File(f) { return crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex'); }

function baseOpt(cell, extra) {
    return Object.assign({
        policyDir: path.join(__dirname, '..', '..', 'js'),
        W: cell.W, H: cell.H, startBoids: cell.startBoids, scatter: false,
        frameMs: cell.frameMs, maxFrames: cell.maxFrames,
        mode: 'lockstep', mismatchLimit: 50, ulpEvery: 997,
        out: null, fastRender: true, spawnScript: null,
    }, extra || {});
}

function hookCheck(r) {
    const c = global.__oracleCertCounts || {};
    const probs = [];
    if (!(c.planEnd > 0)) probs.push('planEnd==0');
    if (c.planStart !== c.planEnd) probs.push(`planStart ${c.planStart} != planEnd ${c.planEnd}`);
    if (c.roll !== 4 * c.planEnd) probs.push(`roll ${c.roll} != 4*planEnd`);
    if (c.frameEnd !== r.frames) probs.push(`frameEnd ${c.frameEnd} != frames ${r.frames}`);
    return { counts: Object.assign({}, c), problems: probs };
}

// step an uninstrumented game until N first <= 5; return that frame index
async function probeGateFrame(cell, seed) {
    const { createGame } = require('./stepper.js');
    const g = await createGame({
        policyDir: path.join(__dirname, '..', '..', 'js'),
        W: cell.W, H: cell.H, seed, startBoids: cell.startBoids,
        frameMs: cell.frameMs, fastRender: true,
    });
    while (g.boidCount() > 5 && g.frame() < cell.maxFrames) g.step();
    return g.boidCount() <= 5 ? g.frame() : -1;
}

async function runCell(cell) {
    const { runGame } = require('./diff_harness.js');
    const runs = [];
    async function rec(tag, opt, seed, expectHooks) {
        const r = await runGame(opt, seed, CAND);
        const h = expectHooks === false ? { counts: null, problems: [] } : hookCheck(r);
        runs.push({
            cell: cell.id, tag, seed, mode: opt.mode,
            spawns: opt.spawnScript ? opt.spawnScript.length : 0,
            frames: r.frames, cleared: r.cleared,
            framesByRegime: r.framesByRegime,
            mismatches: r.mismatchCount,
            firstMismatchFrame: r.firstMismatchFrame,
            mismatchSamples: r.mismatches.slice(0, 3),
            hookCounts: h.counts, hookProblems: h.problems,
        });
        return r;
    }

    // 1) full games incl. natural gate crossing
    for (let i = 0; i < 2; i++) {
        const r = await rec('full+gate', baseOpt(cell), CERT_SEED_BASE + 0 + i);
        const reg = r.framesByRegime;
        if (!(reg.planner > 0 && reg.intercept > 0)) {
            runs[runs.length - 1].hookProblems.push('gate not crossed (planner=' + reg.planner + ' intercept=' + reg.intercept + ')');
        }
    }
    // 2) fork-driven trajectory (the farm configuration)
    await rec('fork-traj', baseOpt(cell, { mode: 'fork' }), CERT_SEED_BASE + 2);
    // 3) planner-phase spawns incl. same-coordinate double-tap
    const mid = [
        { frame: 400, x: cell.W * 0.25, y: cell.H * 0.25 },
        { frame: 400, x: cell.W * 0.25, y: cell.H * 0.25 },   // double-tap, same coord, same frame
        { frame: 401, x: cell.W * 0.25, y: cell.H * 0.25 },   // and next frame
        { frame: 1200, x: cell.W * 0.7, y: cell.H * 0.6 },
        { frame: 2000, x: cell.W * 0.1, y: cell.H * 0.9 },
    ];
    await rec('spawn-mid', baseOpt(cell, { spawnScript: mid }), CERT_SEED_BASE + 3);
    // 4) endgame gate re-crossing: spawn while N<=5 with egBoid state alive
    const probeSeed = CERT_SEED_BASE + 4;
    const f5 = await probeGateFrame(cell, probeSeed);
    if (f5 >= 0) {
        const rc = [
            { frame: f5 + 40, x: cell.W * 0.5, y: cell.H * 0.2 },
            { frame: f5 + 40, x: cell.W * 0.5, y: cell.H * 0.2 },  // same-coord double
            { frame: f5 + 44, x: cell.W * 0.8, y: cell.H * 0.8 },
        ];
        const r = await rec('spawn-recross', baseOpt(cell, { spawnScript: rc }), probeSeed);
        if (r.frames <= f5 + 44) runs[runs.length - 1].hookProblems.push('game ended before recross spawns');
    } else {
        runs.push({ cell: cell.id, tag: 'spawn-recross', seed: probeSeed, skipped: 'N never reached 5 within maxFrames', mismatches: 0, hookProblems: [] });
    }
    // 5) spawn spam past the device boid cap
    const spam = [];
    for (let k = 0; k < 48; k++) {
        spam.push({ frame: 300 + k, x: cell.W * (0.2 + 0.6 * ((k % 5) / 4)), y: cell.H * (0.2 + 0.6 * (((k * 7) % 9) / 8)) });
    }
    await rec('spawn-spam', baseOpt(cell, { spawnScript: spam }), CERT_SEED_BASE + 5);

    return runs;
}

async function main() {
    const argv = process.argv;
    const cellArg = argv.includes('--cell') ? argv[argv.indexOf('--cell') + 1] : null;
    const conc = argv.includes('--concurrency') ? +argv[argv.indexOf('--concurrency') + 1]
        : Math.max(1, Math.min(4, os.cpus().length));

    if (cellArg) {   // child mode: one cell, JSON to stdout
        const cell = CELLS.find(c => c.id === cellArg);
        if (!cell) throw new Error('unknown cell ' + cellArg);
        console.log(JSON.stringify(await runCell(cell)));
        return;
    }

    // gate 0: the committed fork must be freshly derivable from prod source
    const { execFileSync } = require('child_process');
    execFileSync('node', [path.join(__dirname, 'gen_oracle_policy.js'), '--verify'], { stdio: 'inherit' });

    const t0 = Date.now();
    const queue = CELLS.slice();
    const allRuns = [];
    await Promise.all(Array.from({ length: conc }, async () => {
        while (queue.length) {
            const cell = queue.shift();
            console.error(`[cert] ${cell.id} starting...`);
            const out = await new Promise((res, rej) => {
                const p = spawn('node', [__filename, '--cell', cell.id], { stdio: ['ignore', 'pipe', 'inherit'] });
                let s = '';
                p.stdout.on('data', d => s += d);
                p.on('close', code => code === 0 ? res(s) : rej(new Error(cell.id + ' rc=' + code)));
            });
            const runs = JSON.parse(out.trim().split('\n').pop());
            allRuns.push(...runs);
            const mm = runs.reduce((t, r) => t + (r.mismatches || 0), 0);
            console.error(`[cert] ${cell.id} done: ${runs.length} runs, ${runs.reduce((t, r) => t + (r.frames || 0), 0)} frames, ${mm} mismatches (${((Date.now() - t0) / 60000).toFixed(1)}m)`);
        }
    }));

    const totals = {
        runs: allRuns.length,
        frames: allRuns.reduce((t, r) => t + (r.frames || 0), 0),
        plans: allRuns.reduce((t, r) => t + ((r.hookCounts && r.hookCounts.planEnd) || 0), 0),
        mismatches: allRuns.reduce((t, r) => t + (r.mismatches || 0), 0),
        hookProblems: allRuns.reduce((t, r) => t + ((r.hookProblems && r.hookProblems.length) || 0), 0),
    };
    const oracleSha = sha256File(path.join(__dirname, 'oracle_policy.js'));
    const srcSha = sha256File(path.join(__dirname, '..', '..', 'js', 'predator_cheap.js'));
    const ok = totals.mismatches === 0 && totals.hookProblems === 0;
    const cert = {
        ok, oracleSha, srcSha,
        certRunId: crypto.createHash('sha256')
            .update(oracleSha + JSON.stringify(allRuns)).digest('hex').slice(0, 16),
        date: new Date().toISOString(), node: process.version,
        certSeedBase: CERT_SEED_BASE, totals,
        runs: allRuns.map(r => { const c = Object.assign({}, r); delete c.mismatchSamples; return c; }),
        mismatchSamples: allRuns.flatMap(r => r.mismatchSamples || []).slice(0, 10),
    };
    fs.writeFileSync(path.join(__dirname, 'CERT.json'), JSON.stringify(cert, null, 1));

    const md = [];
    md.push('# oracle_policy.js certification ' + (ok ? '— PASS' : '— **FAIL**'));
    md.push('');
    md.push(`- certRunId: \`${cert.certRunId}\`  oracleSha: \`${oracleSha.slice(0, 16)}…\`  srcSha: \`${srcSha.slice(0, 16)}…\``);
    md.push(`- date: ${cert.date}  node ${process.version}`);
    md.push(`- totals: **${totals.frames}** frames lockstep/fork-compared bitwise, **${totals.plans}** plans hook-logged, **${totals.mismatches}** mismatches, ${totals.hookProblems} hook problems`);
    md.push('');
    md.push('| cell | tag | seed | frames | cleared | planner/intercept | plans | mismatches |');
    md.push('|---|---|---|---|---|---|---|---|');
    for (const r of allRuns) {
        if (r.skipped) { md.push(`| ${r.cell} | ${r.tag} | ${r.seed} | — | — | — | — | skipped: ${r.skipped} |`); continue; }
        md.push(`| ${r.cell} | ${r.tag} | ${r.seed} | ${r.frames} | ${r.cleared} | ${r.framesByRegime.planner}/${r.framesByRegime.intercept} | ${(r.hookCounts && r.hookCounts.planEnd) != null ? r.hookCounts.planEnd : '—'} | ${r.mismatches}${r.hookProblems.length ? ' ⚠ ' + r.hookProblems.join('; ') : ''} |`);
    }
    fs.writeFileSync(path.join(__dirname, 'CERT.md'), md.join('\n') + '\n');
    console.log(JSON.stringify({ ok, certRunId: cert.certRunId, totals }));
    process.exit(ok ? 0 : 2);
}

main().catch(e => { console.error(e); process.exit(1); });
