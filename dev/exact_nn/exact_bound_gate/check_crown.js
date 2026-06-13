// check_crown.js — CROWN-lite vs IBP tightness + soundness (no empirical
// boot/vprior may fall outside the CROWN envelope).
'use strict';
var fs = require('fs'), zlib = require('zlib'), readline = require('readline');
var fb = require('./feat_bounds.js');
var ibp = require('./ibp.js');
var crown = require('./crown.js');
var net = JSON.parse(fs.readFileSync(__dirname + '/../../../js/value_net.json', 'utf8'));

// geluEnvelope self-test: sample random intervals, verify the lines bracket gelu
var cp = require('../../../js/cheap_planner.js');
var bad = 0;
for (var t = 0; t < 20000; t++) {
    var l = -8 + Math.random() * 8, u = l + Math.random() * 6;
    var e = crown.geluEnvelope(l, u);
    for (var k = 0; k <= 50; k++) {
        var x = l + (u - l) * k / 50, gv = cp.cp_gelu(x);
        if (gv < e.aL * x + e.bL - 1e-7 || gv > e.aU * x + e.bU + 1e-7) bad++;
    }
}
console.log('geluEnvelope soundness violations (20k intervals x 51 pts):', bad);

function envIBP(W, Hc) { var raw = fb.rawFeatureBounds(W, Hc); var sb = fb.standardizeBounds(net, raw); return ibp.ibpValue(net, sb.lo, sb.hi); }
function envCROWN(W, Hc) { var raw = fb.rawFeatureBounds(W, Hc); var sb = fb.standardizeBounds(net, raw); return crown.crownValue(net, sb.lo, sb.hi); }

var files = fs.readdirSync(__dirname + '/../calib_data').filter(function (f) { return f.endsWith('.decisions.jsonl.gz'); });
(async function () {
    for (var fi = 0; fi < files.length; fi++) {
        var f = files[fi];
        var rl = readline.createInterface({ input: fs.createReadStream(__dirname + '/../calib_data/' + f).pipe(zlib.createGunzip()) });
        var ei = null, ec = null, viol = 0, n = 0;
        for await (var line of rl) {
            if (!line) continue;
            var d = JSON.parse(line);
            if (!ec) { ei = envIBP(d.cfg.W, d.cfg.Hc); ec = envCROWN(d.cfg.W, d.cfg.Hc); }
            for (var r = 0; r < d.rolled.length; r++) { var bo = d.rolled[r][2]; if (bo < ec.lo - 1e-7 || bo > ec.hi + 1e-7) viol++; }
            for (var kk = 0; kk < d.vprior.length; kk++) { var v = d.vprior[kk]; if (v < ec.lo - 1e-7 || v > ec.hi + 1e-7) viol++; }
            n++;
        }
        console.log(f.replace('.decisions.jsonl.gz', '').padEnd(34),
            'IBP w=' + (ei.hi - ei.lo).toExponential(2),
            'CROWN[' + ec.lo.toFixed(2) + ',' + ec.hi.toFixed(2) + '] w=' + (ec.hi - ec.lo).toFixed(2),
            'VIOL=' + viol);
        if (fi >= 5) break;   // 6 cells cover all distinct cfgs
    }
})();
