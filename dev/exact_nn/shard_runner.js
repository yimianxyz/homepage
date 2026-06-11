// exact_nn #5 shard runner — fans oracle_logger games across local cores
// (and the 3 GPU VMs' CPUs when gcloud is reachable; see --mode vm).
//
// Adaptive: per §5 device geometry it keeps launching fixed-size seed-block
// shards (train seeds < 270000, disjoint blocks per geometry) until that
// geometry's decision quota is met, then stops. Idempotent: re-running scans
// outDir's *.meta.json and continues seed allocation past whatever exists.
//
//   node dev/exact_nn/shard_runner.js --quota 167000 --concurrency 4 \
//        --outDir dev/exact_nn/data [--mode local]
//
// VM mode mirrors dev/fleet_clear.py's dispatch (gcloud compute ssh
// ml-forecast-N --tunnel-through-iap), requires ~/google-cloud-sdk + repo at
// ~/eval on the VMs; it errors out with a clear message if gcloud is absent.
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn, execFileSync } = require('child_process');

// §5 verification-corpus device matrix (SPEC.md). seedBase blocks are
// disjoint per geometry; all allocation stays far below the 270000 held-out
// boundary (oracle_logger refuses to cross it regardless).
const GEOMS = [
    { W: 390,  H: 844,  seedBase: 100000, maxFrames: 24000, gamesPerShard: 8 },
    { W: 820,  H: 1180, seedBase: 110000, maxFrames: 90000, gamesPerShard: 4 },
    { W: 1024, H: 768,  seedBase: 120000, maxFrames: 90000, gamesPerShard: 4 },
    { W: 1512, H: 982,  seedBase: 130000, maxFrames: 90000, gamesPerShard: 4 },
    { W: 1680, H: 1050, seedBase: 140000, maxFrames: 90000, gamesPerShard: 4 },
    { W: 2560, H: 1440, seedBase: 150000, maxFrames: 90000, gamesPerShard: 2 },
];
const HOSTS = [['ml-forecast-1', 'us-central1-a'], ['ml-forecast-2', 'us-central1-a'],
               ['ml-forecast-3', 'us-central1-c']];

function parseArgs(argv) {
    const a = { quota: 167000, concurrency: Math.max(1, os.cpus().length), mode: 'local',
        outDir: path.join(__dirname, 'data'), policyDir: path.join(__dirname, '..', '..', 'js'),
        capFrames: 0 };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--quota') a.quota = +argv[++i];
        else if (k === '--concurrency') a.concurrency = +argv[++i];
        else if (k === '--outDir') a.outDir = argv[++i];
        else if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--mode') a.mode = argv[++i];
        else if (k === '--capFrames') a.capFrames = +argv[++i]; // smoke tests only
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}

function effMaxFrames(opt, g) {
    return opt.capFrames > 0 ? Math.min(g.maxFrames, opt.capFrames) : g.maxFrames;
}

// scan existing shards: per geometry -> {decisions, nextSeed}
function scanExisting(outDir) {
    const st = {};
    for (const g of GEOMS) st[`${g.W}x${g.H}`] = { decisions: 0, nextSeed: g.seedBase };
    if (!fs.existsSync(outDir)) return st;
    for (const f of fs.readdirSync(outDir)) {
        if (!f.endsWith('.meta.json')) continue;
        const m = JSON.parse(fs.readFileSync(path.join(outDir, f), 'utf8'));
        const key = `${m.W}x${m.H}`;
        if (!st[key]) continue;
        st[key].decisions += m.nDecisions;
        st[key].nextSeed = Math.max(st[key].nextSeed, m.seedStart + m.seeds);
    }
    return st;
}

function shardName(g, seedStart) { return `train_${g.W}x${g.H}_s${seedStart}`; }

function localJob(opt, g, seedStart, done) {
    const args = [path.join(__dirname, 'oracle_logger.js'),
        '--policyDir', opt.policyDir, '--W', String(g.W), '--H', String(g.H),
        '--seedStart', String(seedStart), '--seeds', String(g.gamesPerShard),
        '--maxFrames', String(effMaxFrames(opt, g)), '--outDir', opt.outDir,
        '--shard', shardName(g, seedStart)];
    const p = spawn('node', args, { stdio: ['ignore', 'pipe', 'inherit'] });
    let out = '';
    p.stdout.on('data', d => out += d);
    p.on('close', code => {
        if (code !== 0) return done(new Error(`shard ${shardName(g, seedStart)} rc=${code}`));
        try { done(null, JSON.parse(out.trim().split('\n').pop())); } catch (e) { done(e); }
    });
}

// VM mode: same shard, executed remotely; shard files land in the VM's
// ~/eval/dev/exact_nn/data and are pulled back via gcloud scp afterwards.
function vmJob(opt, g, seedStart, hostIdx, done) {
    const gcloud = path.join(os.homedir(), 'google-cloud-sdk', 'bin', 'gcloud');
    if (!fs.existsSync(gcloud)) return done(new Error(
        'VM mode needs ~/google-cloud-sdk + credentials (see dev/fleet_clear.py); ' +
        'not reachable from this container — run --mode local or run from the lead container.'));
    const [host, zone] = HOSTS[hostIdx % HOSTS.length];
    const name = shardName(g, seedStart);
    const remote = `cd ~/eval && ~/bin/node dev/exact_nn/oracle_logger.js --policyDir js` +
        ` --W ${g.W} --H ${g.H} --seedStart ${seedStart} --seeds ${g.gamesPerShard}` +
        ` --maxFrames ${effMaxFrames(opt, g)} --outDir dev/exact_nn/data --shard ${name}`;
    try {
        execFileSync(gcloud, ['compute', 'ssh', host, '--zone', zone, '--tunnel-through-iap',
            '--command', remote], { timeout: 3600000 });
        for (const ext of ['.decisions.jsonl.gz', '.frames.jsonl.gz', '.meta.json']) {
            execFileSync(gcloud, ['compute', 'scp', '--zone', zone, '--tunnel-through-iap',
                `${host}:~/eval/dev/exact_nn/data/${name}${ext}`, opt.outDir + '/'], { timeout: 600000 });
        }
        const m = JSON.parse(fs.readFileSync(path.join(opt.outDir, name + '.meta.json'), 'utf8'));
        done(null, { shard: name, games: m.seeds, decisions: m.nDecisions });
    } catch (e) { done(e); }
}

async function main() {
    const opt = parseArgs(process.argv);
    fs.mkdirSync(opt.outDir, { recursive: true });
    const st = scanExisting(opt.outDir);
    console.log('resume state: ' + JSON.stringify(st));
    let inFlight = 0, hostRR = 0, failures = 0;
    const t0 = Date.now();

    function pickGeom() {
        // most-behind geometry (by fraction of quota) that still needs work
        let best = null, bestFrac = 1;
        for (const g of GEOMS) {
            const s = st[`${g.W}x${g.H}`];
            const frac = s.decisions / opt.quota;
            if (frac < 1 && frac < bestFrac) { bestFrac = frac; best = g; }
        }
        return best;
    }

    await new Promise((resolve) => {
        function pump() {
            while (inFlight < opt.concurrency) {
                const g = pickGeom();
                if (!g) break;
                const s = st[`${g.W}x${g.H}`];
                const seedStart = s.nextSeed;
                s.nextSeed += g.gamesPerShard;
                inFlight++;
                const cb = (err, res) => {
                    inFlight--;
                    if (err) { failures++; console.error('FAIL: ' + err.message); }
                    else {
                        s.decisions += res.decisions;
                        const total = GEOMS.reduce((t, gg) => t + st[`${gg.W}x${gg.H}`].decisions, 0);
                        console.log(`[${((Date.now() - t0) / 60000).toFixed(1)}m] ${res.shard}: +${res.decisions} ` +
                            `(${g.W}x${g.H} ${s.decisions}/${opt.quota}; total ${total})`);
                    }
                    if (failures > 20) { console.error('too many failures, aborting'); return resolve(); }
                    pump();
                };
                if (opt.mode === 'vm') vmJob(opt, g, seedStart, hostRR++, cb);
                else localJob(opt, g, seedStart, cb);
            }
            if (inFlight === 0) resolve();
        }
        pump();
    });

    const totals = {};
    let grand = 0;
    for (const g of GEOMS) { const d = st[`${g.W}x${g.H}`].decisions; totals[`${g.W}x${g.H}`] = d; grand += d; }
    fs.writeFileSync(path.join(opt.outDir, 'manifest.json'),
        JSON.stringify({ quotaPerGeom: opt.quota, totals, grand, finishedAt: new Date().toISOString() }, null, 1));
    console.log('DONE: ' + JSON.stringify({ totals, grand }));
}
main().catch(e => { console.error(e); process.exit(1); });
