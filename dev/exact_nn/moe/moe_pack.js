// moe_pack.js — pack the Phase-1 oracle logs into unified Phase-2 MoE tensors.
//   planner: dev/exact_nn/data_1e6  (decisions.jsonl.gz, N>5)
//   endgame: dev/exact_nn/endgame/data_eg{,2,_nat} (commits.jsonl.gz, N<=5)
// Uses moe_features.js (the SAME featurizer the deploy uses → bit-identical).
// Writes raw float32/int32 binaries + a small JSON meta per regime, plus a
// train/val split tag (by game seed → whole games stay on one side; sealed seeds
// >=270000 are never present in these dirs / are excluded).
//
//   node moe_pack.js --plannerTarget 500000 --out pack
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const readline = require('readline');
const F = require('./moe_features.js');

const CELLS = ['iphone_390x844', 'ipad_820x1180', 'desk_1024x768', 'desk_1512x982',
    'desk_1680x1050', 'desk_2560x1440', '390x844', '820x1180', '1024x768',
    '1512x982', '1680x1050', '2560x1440'];
const cellId = (c) => { const i = CELLS.indexOf(c); return i < 0 ? 99 : i; };

function arg(name, def) { const i = process.argv.indexOf('--' + name); return i >= 0 ? process.argv[i + 1] : def; }
const OUT = arg('out', 'pack');
const PLANNER_TARGET = parseInt(arg('plannerTarget', '500000'), 10);
const SEALED_MIN = 270000;

// game-coherent ~10% val split from seed hash (deterministic)
function isVal(seed) { return (((seed >>> 0) * 2654435761) >>> 0) % 10 === 0; }

// bitwise coord-class: group the 16 candidates by exact (x,y) bytes; class id =
// lowest member index (matches ds.dedup_classes / SPEC §3, exact not approximate).
function coordClasses(cands) {
    const first = {}; const cls = new Array(16);
    const buf = Buffer.allocUnsafe(16);
    for (let k = 0; k < 16; k++) {
        buf.writeDoubleLE(cands[k][0], 0); buf.writeDoubleLE(cands[k][1], 8);
        const key = buf.toString('latin1');
        if (!(key in first)) first[key] = k;
        cls[k] = first[key];
    }
    return cls;
}

function gzLines(file, onLine, onDone) {
    const rl = readline.createInterface({ input: fs.createReadStream(file).pipe(zlib.createGunzip()) });
    rl.on('line', onLine);
    rl.on('close', onDone);
}

// Streaming typed-array writer: buffers `push`ed numbers and flushes to the file
// as raw bytes every CHUNK elements, so memory stays bounded for any dataset size
// (a plain JS Array > ~2^28 elements overflows V8 fast-elements -> OOM).
class Writer {
    constructor(p, kind) { this.fd = fs.openSync(p, 'w'); this.kind = kind; this.buf = []; this.total = 0; this.path = p; }
    push(v) { this.buf.push(v); if (this.buf.length >= 1 << 20) this.flush(); }
    flush() {
        if (!this.buf.length) return;
        const ta = this.kind === 'i32' ? Int32Array.from(this.buf) : Float32Array.from(this.buf);
        fs.writeSync(this.fd, Buffer.from(ta.buffer, ta.byteOffset, ta.byteLength));
        this.total += this.buf.length; this.buf = [];
    }
    close() { this.flush(); fs.closeSync(this.fd); console.error(`  wrote ${this.path} (${this.total} elems)`); }
}

// ---------------- PLANNER ----------------
async function packPlanner() {
    const dir = path.join(__dirname, '..', 'data_1e6');
    let shards = fs.readdirSync(dir).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    const maxPerCell = parseInt(arg('maxShardsPerCell', '0'), 10);
    if (maxPerCell > 0) {
        // cell = the token between 'train_' and the trailing _sNNNNNN
        const byCell = {};
        for (const sh of shards) {
            const cell = sh.replace(/^train_/, '').replace(/_s\d+\.decisions\.jsonl\.gz$/, '');
            (byCell[cell] = byCell[cell] || []).push(sh);
        }
        const picked = [];
        for (const c of Object.keys(byCell)) picked.push(...byCell[c].slice(0, maxPerCell));
        shards = picked.sort();
        console.error(`[planner] cell-balanced: ${Object.keys(byCell).length} cells x<=${maxPerCell} = ${shards.length} shards`);
    }
    // estimate total decisions for a balanced stride
    let est = 0; for (const sh of shards) { const m = sh.replace('.decisions.jsonl.gz', '.meta.json'); try { est += JSON.parse(fs.readFileSync(path.join(dir, m))).nDecisions || 0; } catch (e) { } }
    const frac = Math.min(1, PLANNER_TARGET / Math.max(est, 1));
    const keepMod = Math.max(1, Math.round(1 / frac));   // keep 1 of every keepMod records, per shard (cell-balanced)
    console.error(`[planner] ${shards.length} shards, est ${est} decisions, frac ${frac.toFixed(3)}, keepMod ${keepMod}`);

    const wSlots = new Writer(`${OUT}/planner_slots.f32`, 'f32');
    const wScore = new Writer(`${OUT}/planner_score.f32`, 'f32');
    const wCls = new Writer(`${OUT}/planner_cls.i32`, 'i32');
    const wBicls = new Writer(`${OUT}/planner_bicls.i32`, 'i32');
    const wGate = new Writer(`${OUT}/planner_gate.f32`, 'f32');
    const meta = { N: [], split: [], seed: [], cell: [], dmargin: [] };
    let kept = 0, seen = 0, skippedSealed = 0;
    for (const sh of shards) {
        await new Promise((resolve) => {
            let li = 0;
            gzLines(path.join(dir, sh), (line) => {
                const r = JSON.parse(line);
                if (r.seed >= SEALED_MIN) { skippedSealed++; return; }
                if ((li++ % keepMod) !== 0) return;   // cell-balanced stride sample
                seen++;
                const s = r.s, n = r.N;
                const ps = F.plannerSlots(r.cands, r.feat, r.vprior, s.px, s.py, n, r.pidx, r.rolled);
                for (let k = 0; k < 16; k++) for (let d = 0; d < F.PLANNER_DIM; d++) wSlots.push(ps[k][d]);
                for (let k = 0; k < 16; k++) { const v = r.score[k]; wScore.push(v === null ? -1e9 : v); }
                const cc = coordClasses(r.cands);
                for (let k = 0; k < 16; k++) wCls.push(cc[k]);
                wBicls.push(cc[r.bi]);
                const ga = F.gateFeat(n, n / 120.0, s.psize);
                for (let d = 0; d < F.GATE_DIM; d++) wGate.push(ga[d]);
                meta.N.push(n); meta.split.push(isVal(r.seed) ? 1 : 0); meta.seed.push(r.seed);
                meta.cell.push(cellId(r.cell)); meta.dmargin.push(r.dmargin === null ? 1e9 : r.dmargin);
                kept++;
            }, resolve);
        });
        if (kept && kept % 50000 < keepMod) console.error(`[planner] kept ${kept}...`);
    }
    console.error(`[planner] DONE seen ${seen} kept ${kept} (sealed skipped ${skippedSealed})`);
    wSlots.close(); wScore.close(); wCls.close(); wBicls.close(); wGate.close();
    fs.writeFileSync(`${OUT}/planner_meta.json`, JSON.stringify({ P: kept, dim: F.PLANNER_DIM, gateDim: F.GATE_DIM, ...meta }));
}

// ---------------- ENDGAME ----------------
async function packEndgame() {
    const dirs = ['data_eg', 'data_eg2', 'data_eg_nat'].map(d => path.join(__dirname, '..', 'endgame', d));
    const wSlots = new Writer(`${OUT}/endgame_slots.f32`, 'f32');
    const wScant = new Writer(`${OUT}/endgame_scant.f32`, 'f32');
    const wValid = new Writer(`${OUT}/endgame_valid.i32`, 'i32');
    const wEg = new Writer(`${OUT}/endgame_egidx.i32`, 'i32');
    const wGate = new Writer(`${OUT}/endgame_gate.f32`, 'f32');
    const meta = { N: [], split: [], seed: [], cell: [], egmargin: [], src: [] };
    let kept = 0, srcIdx = 0;
    for (const dir of dirs) {
        if (!fs.existsSync(dir)) { srcIdx++; continue; }
        const files = fs.readdirSync(dir).filter(f => f.endsWith('.commits.jsonl.gz')).sort();
        for (const fn of files) {
            await new Promise((resolve) => {
                gzLines(path.join(dir, fn), (line) => {
                    const r = JSON.parse(line);
                    if (r.seed >= SEALED_MIN) return;
                    const n = r.boids.length; if (n < 1 || n > 5) return;
                    const ts = r.boids.map(b => (b.t === undefined ? null : b.t));
                    const es = F.endgameSlots(r.px, r.py, r.boids, ts, r.W, r.Hc);   // [n][20]
                    for (let k = 0; k < 16; k++) {
                        for (let d = 0; d < F.ENDGAME_DIM; d++) wSlots.push(k < n ? es[k][d] : 0.0);
                        wScant.push(k < n ? (ts[k] === null ? F.TMAX : ts[k]) : 1e9);
                        wValid.push(k < n ? 1 : 0);
                    }
                    wEg.push(r.egIdx);
                    const ga = F.gateFeat(n, n / 120.0, r.psize);
                    for (let d = 0; d < F.GATE_DIM; d++) wGate.push(ga[d]);
                    // eg margin = 2nd-min - min scan-t over reachable (Infinity if <2 reachable)
                    const fin = ts.filter(t => t !== null).sort((a, b) => a - b);
                    const egm = fin.length >= 2 ? (fin[1] - fin[0]) : 1e9;
                    meta.N.push(n); meta.split.push(isVal(r.seed) ? 1 : 0); meta.seed.push(r.seed);
                    meta.cell.push(cellId(r.cell)); meta.egmargin.push(egm); meta.src.push(srcIdx);
                    kept++;
                }, resolve);
            });
        }
        srcIdx++;
    }
    console.error(`[endgame] DONE kept ${kept}`);
    wSlots.close(); wScant.close(); wValid.close(); wEg.close(); wGate.close();
    fs.writeFileSync(`${OUT}/endgame_meta.json`, JSON.stringify({ E: kept, dim: F.ENDGAME_DIM, gateDim: F.GATE_DIM, ...meta }));
}

// pull TMAX for the endgame unreachable sentinel
F.TMAX = require(path.join(__dirname, '..', 'endgame', 'eg_features.js')).TMAX;

(async () => {
    if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });
    const which = arg('only', 'both');
    if (which === 'both' || which === 'planner') await packPlanner();
    if (which === 'both' || which === 'endgame') await packEndgame();
    console.error('PACK COMPLETE ->', OUT);
})();
