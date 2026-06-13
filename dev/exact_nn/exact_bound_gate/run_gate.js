// run_gate.js — execute the exact-bound gate on calib_data + a large data_1e6
// sample; print all four deliverables. Usage:
//   node run_gate.js calib            # all calib_data
//   node run_gate.js data <N> <every> # first N data_1e6 files, sample 1/every
'use strict';
var fs = require('fs'), zlib = require('zlib'), readline = require('readline');
var gate = require('./gate.js');

function pct(a, b) { return b ? (100 * a / b).toFixed(4) + '%' : 'n/a'; }

// Pre-scan true global boot range over the given files (for the ceiling Bstar).
async function bootRange(paths, sampleEvery) {
    var lo = Infinity, hi = -Infinity, n = 0;
    for (var i = 0; i < paths.length; i++) {
        var rl = readline.createInterface({ input: fs.createReadStream(paths[i]).pipe(zlib.createGunzip()) });
        var ln = 0;
        for await (var line of rl) {
            if (!line) continue; ln++;
            if (sampleEvery > 1 && (ln % sampleEvery)) continue;
            var d = JSON.parse(line);
            for (var r = 0; r < d.rolled.length; r++) { var b = d.rolled[r][2]; if (b < lo) lo = b; if (b > hi) hi = b; }
            n++;
        }
    }
    return { lo: lo, hi: hi, n: n };
}

function report(label, s, br) {
    console.log('\n================= ' + label + ' =================');
    console.log('plans: ' + s.n + '   (true global boot range used for ceiling: [' + br.lo.toFixed(4) + ', ' + br.hi.toFixed(4) + '])');
    console.log('--- Deliverable 1: SOUND-deploy certified NN-share ---');
    console.log('  SOUND gate (full IBP boot bound)  : ' + pct(s.sound, s.n) + '  (' + s.sound + ')   FALSE certs = ' + s.falseCert_sound);
    console.log('  losRate-capped IBP variant (|<=110|): ' + pct(s.soundT, s.n) + '  (' + s.soundT + ')   FALSE certs = ' + s.falseCert_soundT);
    console.log('  SOUNDNESS SELF-TEST (perfect bounds): ' + pct(s.perfect, s.n) + '  FALSE certs = ' + s.falseCert_perfect + '  (must be 100% / 0)');
    console.log('--- Deliverable 2: ORACLE CEILING ---');
    console.log('  no rolled in deduped top-2 (TRUE)  : ' + pct(s.noRolledTop2, s.n) + '  (' + s.noRolledTop2 + ')');
    console.log('  robust to ANY rolled perturbation  : ' + pct(s.ceiling, s.n) + '  (' + s.ceiling + ')   FALSE certs = ' + s.falseCert_ceiling);
    console.log('    [ceiling = certify with rolled in [0, Cmax_true + Bstar]]');
    console.log('--- Deliverable 3: TIGHTNESS BREAKDOWN (of the ' + s.nFail + ' SOUND-gate failures) ---');
    console.log('  certified if BOOT were perfect      : ' + pct(s.certTrueBoot, s.nFail) + '  (' + s.certTrueBoot + ')');
    console.log('  certified if CATCH were perfect     : ' + pct(s.certTrueCatch, s.nFail) + '  (' + s.certTrueCatch + ')');
    console.log('  blocked SOLELY by boot bound        : ' + pct(s.failBoot, s.nFail) + '  (' + s.failBoot + ')');
    console.log('  blocked SOLELY by catch bound       : ' + pct(s.failCatch, s.nFail) + '  (' + s.failCatch + ')');
    console.log('  either fix alone suffices           : ' + pct(s.failBoth, s.nFail) + '  (' + s.failBoth + ')');
    console.log('  needs BOTH bounds tightened         : ' + pct(s.failNeither, s.nFail) + '  (' + s.failNeither + ')');
}

(async function () {
    var which = process.argv[2] || 'calib';
    var dir, paths, sampleEvery = 1, label;
    if (which === 'calib') {
        dir = __dirname + '/../calib_data';
        paths = fs.readdirSync(dir).filter(function (f) { return f.endsWith('.decisions.jsonl.gz'); }).map(function (f) { return dir + '/' + f; });
        label = 'calib_data (ALL ' + paths.length + ' files)';
    } else {
        dir = __dirname + '/../data_1e6';
        var N = parseInt(process.argv[3] || '40', 10);
        sampleEvery = parseInt(process.argv[4] || '1', 10);
        var all = fs.readdirSync(dir).filter(function (f) { return f.endsWith('.decisions.jsonl.gz'); }).sort();
        // stride across the sorted list so all device cells are represented
        var stride = Math.max(1, Math.floor(all.length / N));
        var picked = [];
        for (var ii = 0; ii < all.length && picked.length < N; ii += stride) picked.push(all[ii]);
        paths = picked.map(function (f) { return dir + '/' + f; });
        label = 'data_1e6 (' + paths.length + ' files strided over ' + all.length + ', sample 1/' + sampleEvery + ')';
    }
    var t0 = Date.now();
    var br = await bootRange(paths, sampleEvery);
    var s = await gate.run(paths, label, sampleEvery, 0.0, br.hi);   // ceiling Bstar = observed global max boot on this set
    report(label, s, br);
    console.log('\n(' + ((Date.now() - t0) / 1000).toFixed(1) + 's)');
})();
