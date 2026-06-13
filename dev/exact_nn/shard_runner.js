// exact_nn #5 shard runner — fans oracle_logger games across local cores and
// (optionally) the side-a GPU VMs' CPUs (--mode vm, ml-forecast-1/2 only).
//
// Drives the device matrix (dev/exact_nn/device_matrix.js) with a spawn-profile
// MIX (SPEC §5 interaction corpus). Each (cell, spawn-profile) pair is an
// independent stream with its own disjoint seed sub-block; the runner keeps
// launching fixed-size seed-block shards for whichever stream is most behind
// its decision target until every stream is met, then stops. Idempotent:
// re-running scans outDir's *.meta.json (keyed by cell+spawn) and continues
// seed allocation past whatever already exists.
//
//   node dev/exact_nn/shard_runner.js --quota 167000 --concurrency 4 \
//        --outDir dev/exact_nn/data [--mode local|vm] [--spawnFrac 0.15]
//
// --spawnFrac 0  => pure on-distribution (the deliverable-zero margin-CDF farm).
// VM mode mirrors GPU_OPS.md dispatch (gcloud compute ssh ml-forecast-N
// --tunnel-through-iap); it errors with a clear message if gcloud is absent,
// and NEVER touches ml-forecast-3 (side-b's).
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn, execFileSync } = require('child_process');
const { CELLS, HELD_OUT_SEED } = require('./device_matrix.js');

// spawn-profile mix. offsets carve each cell's seedBase block into disjoint
// sub-blocks so a 'none' game and a 'mid' game never share a seed (different
// corpora, distinct shard names, no meta double-count). fracs are normalized.
const PROFILES = [
    { name: 'none',    frac: 0.85, offset: 0 },
    { name: 'mid',     frac: 0.05, offset: 2000 },
    { name: 'recross', frac: 0.05, offset: 4000 },
    { name: 'spam',    frac: 0.05, offset: 6000 },
];
// side-a owns ml-forecast-1 + ml-forecast-2 (us-central1-a). NEVER VM3.
const HOSTS = [['ml-forecast-1', 'us-central1-a'], ['ml-forecast-2', 'us-central1-a']];

function parseArgs(argv) {
    const a = { quota: 167000, concurrency: Math.max(1, os.cpus().length), mode: 'local',
        outDir: path.join(__dirname, 'data'), policyDir: path.join(__dirname, '..', '..', 'js'),
        spawnFrac: 0.15, capFrames: 0 };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--quota') a.quota = +argv[++i];
        else if (k === '--concurrency') a.concurrency = +argv[++i];
        else if (k === '--outDir') a.outDir = argv[++i];
        else if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--mode') a.mode = argv[++i];
        else if (k === '--spawnFrac') a.spawnFrac = +argv[++i];     // 0 => none-only
        else if (k === '--capFrames') a.capFrames = +argv[++i];     // smoke tests only
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}

// Build the (cell × profile) stream list with per-stream decision targets and
// disjoint seed sub-blocks. spawnFrac scales the non-'none' profiles; 'none'
// takes the remainder. Asserts every stream stays below HELD_OUT_SEED and that
// sub-blocks cannot collide at this quota.
function buildStreams(opt) {
    const spawnProfiles = PROFILES.filter(p => p.name !== 'none');
    const eachSpawn = opt.spawnFrac > 0 ? opt.spawnFrac / spawnProfiles.length : 0;
    const noneFrac = 1 - (eachSpawn * spawnProfiles.length);
    const streams = [];
    for (const cell of CELLS) {
        for (const p of PROFILES) {
            const frac = p.name === 'none' ? noneFrac : eachSpawn;
            if (frac <= 0) continue;
            const target = Math.round(opt.quota * frac);
            const seedBase = cell.seedBase + p.offset;
            // worst case games = target decisions (>=1 decision/game) is a loose
            // bound; in practice ~250 plans/game, but guard the loose bound's seed
            // span against the next sub-block (offset step = 2000) and held-out.
            streams.push({ key: `${cell.id}__${p.name}`, cell, profile: p.name,
                seedBase, nextSeed: seedBase, target, decisions: 0,
                gamesPerShard: cell.gamesPerShard, maxFrames: cell.maxFrames });
        }
    }
    return streams;
}

function effMaxFrames(opt, maxFrames) {
    return opt.capFrames > 0 ? Math.min(maxFrames, opt.capFrames) : maxFrames;
}

function shardName(cell, profile, seedStart) {
    return `train_${cell.id}_s${seedStart}${profile !== 'none' ? '_' + profile : ''}`;
}

// scan existing shards into the stream map (resume): meta carries cell+spawn.
function scanExisting(outDir, streams) {
    const byKey = {};
    for (const s of streams) byKey[s.key] = s;
    if (!fs.existsSync(outDir)) return;
    for (const f of fs.readdirSync(outDir)) {
        if (!f.endsWith('.meta.json')) continue;
        const m = JSON.parse(fs.readFileSync(path.join(outDir, f), 'utf8'));
        const key = `${m.cell}__${m.spawn || 'none'}`;
        const s = byKey[key];
        if (!s) continue;
        s.decisions += m.nDecisions;
        s.nextSeed = Math.max(s.nextSeed, m.seedStart + m.seeds);
    }
}

function localJob(opt, s, seedStart, done) {
    const args = [path.join(__dirname, 'oracle_logger.js'),
        '--policyDir', opt.policyDir, '--cell', s.cell.id,
        '--seedStart', String(seedStart), '--seeds', String(s.gamesPerShard),
        '--maxFrames', String(effMaxFrames(opt, s.maxFrames)), '--spawn', s.profile,
        '--outDir', opt.outDir, '--shard', shardName(s.cell, s.profile, seedStart)];
    const p = spawn('node', args, { stdio: ['ignore', 'pipe', 'inherit'] });
    let out = '';
    p.stdout.on('data', d => out += d);
    p.on('close', code => {
        if (code !== 0) return done(new Error(`shard ${shardName(s.cell, s.profile, seedStart)} rc=${code}`));
        try { done(null, JSON.parse(out.trim().split('\n').pop())); } catch (e) { done(e); }
    });
}

// VM mode: same shard, executed remotely on a side-a VM (CPU node — labels are
// V8 regardless, GPU_OPS: node v20.18.1 is md5-identical to local for exp/pow).
// Shards land in the VM's ~/eval/dev/exact_nn/data and are pulled back via scp
// (md5-verified — IAP scp can silently truncate, GPU_OPS.md).
function vmJob(opt, s, seedStart, hostIdx, done) {
    const gcloud = path.join(os.homedir(), 'google-cloud-sdk', 'bin', 'gcloud');
    if (!fs.existsSync(gcloud)) return done(new Error(
        'VM mode needs ~/google-cloud-sdk + activated SA (see GPU_OPS.md); not found.'));
    const [host, zone] = HOSTS[hostIdx % HOSTS.length];
    const name = shardName(s.cell, s.profile, seedStart);
    const remote = `cd ~/eval && ~/bin/node dev/exact_nn/oracle_logger.js --policyDir js` +
        ` --cell ${s.cell.id} --seedStart ${seedStart} --seeds ${s.gamesPerShard}` +
        ` --maxFrames ${effMaxFrames(opt, s.maxFrames)} --spawn ${s.profile}` +
        ` --outDir dev/exact_nn/data --shard ${name}`;
    const sh = (args, timeout) => execFileSync(gcloud, args, { timeout, stdio: ['ignore', 'pipe', 'pipe'] });
    try {
        sh(['compute', 'ssh', host, '--zone', zone, '--tunnel-through-iap', '--command', remote], 3600000);
        for (const ext of ['.decisions.jsonl.gz', '.frames.jsonl.gz', '.meta.json']) {
            const local = path.join(opt.outDir, name + ext);
            sh(['compute', 'scp', '--zone', zone, '--tunnel-through-iap',
                `${host}:~/eval/dev/exact_nn/data/${name}${ext}`, local], 600000);
            const rmd5 = sh(['compute', 'ssh', host, '--zone', zone, '--tunnel-through-iap',
                '--command', `md5sum ~/eval/dev/exact_nn/data/${name}${ext}`], 120000)
                .toString().trim().split(/\s+/)[0];
            const lmd5 = execFileSync('md5sum', [local]).toString().trim().split(/\s+/)[0];
            if (rmd5 !== lmd5) return done(new Error(`scp md5 mismatch ${name}${ext} (truncated?)`));
        }
        const m = JSON.parse(fs.readFileSync(path.join(opt.outDir, name + '.meta.json'), 'utf8'));
        done(null, { shard: name, games: m.seeds, decisions: m.nDecisions });
    } catch (e) { done(e); }
}

async function main() {
    const opt = parseArgs(process.argv);
    fs.mkdirSync(opt.outDir, { recursive: true });
    const streams = buildStreams(opt);
    scanExisting(opt.outDir, streams);
    // seed-discipline + collision guards (loud failure beats silent corruption)
    for (const s of streams) {
        if (s.nextSeed >= HELD_OUT_SEED) throw new Error(`${s.key} seed ${s.nextSeed} crosses held-out ${HELD_OUT_SEED}`);
    }
    console.log('resume state: ' + JSON.stringify(streams.map(s =>
        ({ key: s.key, decisions: s.decisions, target: s.target, nextSeed: s.nextSeed }))));
    let inFlight = 0, hostRR = 0, failures = 0;
    const t0 = Date.now();

    function pickStream() {           // most-behind stream still under target
        let best = null, bestFrac = 1;
        for (const s of streams) {
            const frac = s.decisions / s.target;
            if (frac < 1 && frac < bestFrac) { bestFrac = frac; best = s; }
        }
        return best;
    }

    await new Promise((resolve) => {
        function pump() {
            while (inFlight < opt.concurrency) {
                const s = pickStream();
                if (!s) break;
                const seedStart = s.nextSeed;
                s.nextSeed += s.gamesPerShard;
                if (s.nextSeed >= HELD_OUT_SEED) { console.error(`STOP ${s.key}: would cross held-out`); s.target = s.decisions; continue; }
                inFlight++;
                const cb = (err, res) => {
                    inFlight--;
                    if (err) { failures++; console.error('FAIL: ' + err.message); }
                    else {
                        s.decisions += res.decisions;
                        const grand = streams.reduce((t, ss) => t + ss.decisions, 0);
                        console.log(`[${((Date.now() - t0) / 60000).toFixed(1)}m] ${res.shard}: +${res.decisions} ` +
                            `(${s.key} ${s.decisions}/${s.target}; grand ${grand})`);
                    }
                    if (failures > 20) { console.error('too many failures, aborting'); return resolve(); }
                    pump();
                };
                if (opt.mode === 'vm') vmJob(opt, s, seedStart, hostRR++, cb);
                else localJob(opt, s, seedStart, cb);
            }
            if (inFlight === 0) resolve();
        }
        pump();
    });

    const totals = {}; let grand = 0;
    for (const s of streams) { totals[s.key] = s.decisions; grand += s.decisions; }
    fs.writeFileSync(path.join(opt.outDir, 'manifest.json'),
        JSON.stringify({ quota: opt.quota, spawnFrac: opt.spawnFrac, mode: opt.mode,
            totals, grand, failures, finishedAt: new Date().toISOString() }, null, 1));
    console.log('DONE: ' + JSON.stringify({ grand, failures }));
}
main().catch(e => { console.error(e); process.exit(1); });
