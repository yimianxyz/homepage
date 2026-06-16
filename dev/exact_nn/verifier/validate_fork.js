// validate_fork.js — PROVE endgame_fork.forkRun reproduces the trusted full-game
// diff_harness.runGame BITWISE for the same (screen, N0, seed, T). Compares
// frames / eaten / cleared / trajDigest. Any mismatch ⇒ the fork's snapshot/restore
// is incomplete ⇒ the fast runner is NOT used. 100% match ⇒ greenlit.
'use strict';
const path = require('path');
const { runGame } = require('../diff_harness.js');
const { forkRun } = require('./endgame_fork.js');

const POLICY_DIR = path.join(__dirname, '..', '..', '..', 'js');
const CAND = path.join(__dirname, '..', 'candidates', 'split.js');
const MF = 60000, FORK_N = 13;

// (W, H, uaMobile, N0)  — N0=0 ⇒ natural (uaMobile decides 60/120); else forced
const CASES = [
    [390, 844, true, 0],     // mobile natural (60)
    [414, 896, true, 0],     // mobile natural (60)
    [1366, 768, false, 0],   // laptop natural (120)
    [1920, 1080, false, 0],  // desktop natural (120)
    [390, 844, false, 120],  // CROSS: mobile screen forced 120
    [1920, 1080, false, 60], // CROSS: desktop screen forced 60
];
const SEEDS = [270000, 270001, 270002];
const TS = [3, 5, 8, 11, 12];

async function fullGame(W, H, uaMobile, N0, seed, T) {
    process.env.EXACTNN_SPLIT_RULE = 'count'; process.env.EXACTNN_SPLIT_T = String(T);
    const r = await runGame({ policyDir: POLICY_DIR, W, H, startBoids: N0, scatter: false,
        uaMobile, maxFrames: MF, postExtinct: 0, decisions: false, fastRender: true,
        mismatchLimit: 0, mode: 'fork', resync: false }, seed, CAND);
    delete process.env.EXACTNN_SPLIT_T; delete process.env.EXACTNN_SPLIT_RULE;
    return { frames: r.frames, eaten: r.eaten, cleared: r.cleared, trajDigest: r.trajDigest };
}

(async () => {
    let pass = 0, fail = 0; const fails = [];
    for (const [W, H, uaMobile, N0] of CASES) {
        for (const seed of SEEDS) {
            const fk = await forkRun({ W, H, uaMobile, N0, seed, Ts: TS, rule: 'count',
                forkN: FORK_N, maxFrames: MF, digest: true });
            const byT = {}; for (const r of fk.results) byT[r.T] = r;
            for (const T of TS) {
                const full = await fullGame(W, H, uaMobile, N0, seed, T);
                const f = byT[T];
                const ok = f.frames === full.frames && f.eaten === full.eaten &&
                           f.cleared === full.cleared && f.trajDigest === full.trajDigest;
                if (ok) pass++; else {
                    fail++; fails.push({ W, H, uaMobile, N0, seed, T,
                        fork: { frames: f.frames, eaten: f.eaten, cleared: f.cleared, d: f.trajDigest },
                        full: { frames: full.frames, eaten: full.eaten, cleared: full.cleared, d: full.trajDigest } });
                }
            }
            process.stderr.write(`[${W}x${H} ua=${uaMobile} N0=${N0} seed=${seed}] prefix=${fk.prefixFrames}f forkN=${fk.forkNcount} ✓pass=${pass} ✗fail=${fail}\n`);
        }
    }
    console.log('\n==== VALIDATION ' + (fail === 0 ? 'PASS' : 'FAIL') + ' ====  pass=' + pass + ' fail=' + fail);
    if (fail) console.log(JSON.stringify(fails.slice(0, 10), null, 2));
    process.exit(fail === 0 ? 0 : 1);
})();
