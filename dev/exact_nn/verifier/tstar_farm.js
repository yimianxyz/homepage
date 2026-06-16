// tstar_farm.js — paired-seed throughput farm for the T*(screen,N0) surface, built
// on the validated fork runner (endgame_fork.js). Per (cell, seed) it runs ONE
// planner prefix then forks the endgame for every T — so all T are paired per seed
// by construction (same prefix → same fork state). Cells carry their OWN N0 (real
// deployment 0→60/120 via uaMobile, or a forced count for the confound cross).
//
//   node tstar_farm.js --cells "390,844,1,0;1920,1080,0,0;390,844,0,120" \
//        --Ts 1,2,3,4,5,6,7,8,9,10,11,12 --seeds 200 --seedBase 270000 --out farm.json
//   sealed:  --sealed --sealOffset 0   (draws from the p2 salt block)
'use strict';
const path = require('path'), fs = require('fs');
const { forkRun } = require('./endgame_fork.js');

function parseArgs(argv) {
    const a = { cells: null, Ts: '1,2,3,4,5,6,7,8,9,10,11,12', rule: 'count',
        seeds: 200, seedBase: 270000, sealed: false, sealOffset: 0,
        forkN: 13, maxFrames: 60000, out: null };
    for (let i = 2; i < argv.length; i++) { const k = argv[i];
        if (k === '--cells') a.cells = argv[++i]; else if (k === '--Ts') a.Ts = argv[++i];
        else if (k === '--rule') a.rule = argv[++i]; else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--seedBase') a.seedBase = +argv[++i]; else if (k === '--sealed') a.sealed = true;
        else if (k === '--sealOffset') a.sealOffset = +argv[++i]; else if (k === '--forkN') a.forkN = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i]; else if (k === '--out') a.out = argv[++i];
        else throw new Error('unknown arg ' + k); }
    if (!a.cells) throw new Error('--cells required');
    return a;
}
function parseCells(spec) {   // "W,H,ua,N0;..." → [{W,H,uaMobile,N0,key}]
    return spec.split(';').filter(Boolean).map(c => {
        const [W, H, ua, N0] = c.split(',').map(Number);
        return { W, H, uaMobile: !!ua, N0: N0 || 0,
                 key: `${W}x${H}_ua${ua ? 1 : 0}_N${N0 || 0}` };
    });
}
function seedList(opt) {
    if (opt.sealed) { const seal = require('./seal_seeds.js');
        const all = seal.sealedSeeds(fs.readFileSync(seal.SALT_PATH), opt.sealOffset + opt.seeds + 1);
        return all.slice(opt.sealOffset, opt.sealOffset + opt.seeds); }
    const o = []; for (let i = 0; i < opt.seeds; i++) o.push(opt.seedBase + i); return o;
}

async function main() {
    const opt = parseArgs(process.argv);
    const cells = parseCells(opt.cells);
    const Ts = opt.Ts.split(',').map(Number);
    const seeds = seedList(opt);
    const R = {};
    const t0 = Date.now();
    for (const cell of cells) {
        R[cell.key] = { W: cell.W, H: cell.H, uaMobile: cell.uaMobile, N0eff: cell.N0 || (cell.uaMobile ? 60 : 120),
                        forcedN0: cell.N0 || 0, byT: {}, prefix: [] };
        for (const T of Ts) R[cell.key].byT[T] = [];
        const ct0 = Date.now();
        for (const seed of seeds) {
            const fk = await forkRun({ W: cell.W, H: cell.H, uaMobile: cell.uaMobile, N0: cell.N0,
                seed, Ts, rule: opt.rule, forkN: opt.forkN, maxFrames: opt.maxFrames, digest: false });
            R[cell.key].prefix.push({ seed, prefixFrames: fk.prefixFrames, forkNcount: fk.forkNcount, capped: fk.prefixCapped });
            for (const r of fk.results)
                R[cell.key].byT[r.T].push({ seed, thru: r.thru, frames: r.frames, eaten: r.eaten, cleared: r.cleared });
        }
        // quick per-cell summary to stderr
        const ms = Date.now() - ct0;
        const summ = Ts.map(T => { const rows = R[cell.key].byT[T];
            const mt = rows.reduce((s, x) => s + x.thru, 0) / rows.length;
            return `T${T}=${(mt * 1e4).toFixed(2)}`; }).join(' ');
        const clr = R[cell.key].byT[Ts[0]].length ? (R[cell.key].byT[8] || R[cell.key].byT[Ts[0]]).filter(x => x.cleared).length : 0;
        process.stderr.write(`[${cell.key}] n=${seeds.length} ${(ms / seeds.length).toFixed(0)}ms/seed clr@${(R[cell.key].byT[8] ? 8 : Ts[0])}=${clr}/${seeds.length} | ${summ} (e-4)\n`);
    }
    const report = { metric: 'throughput=eaten/frames (paired seeds, fork)', rule: opt.rule, Ts,
        cells: cells.map(c => c.key), seeds: seeds.length,
        seedSet: opt.sealed ? ('SEALED@off' + opt.sealOffset) : ('held-out@' + opt.seedBase),
        forkN: opt.forkN, maxFrames: opt.maxFrames, wallMs: Date.now() - t0, R };
    if (opt.out) { fs.mkdirSync(path.dirname(opt.out), { recursive: true }); fs.writeFileSync(opt.out, JSON.stringify(report)); }
    console.log('DONE ' + (opt.out || '') + ` (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
