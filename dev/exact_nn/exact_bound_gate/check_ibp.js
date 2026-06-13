// check_ibp.js — sanity: (1) GELU trough; (2) IBP [Bmin,Bmax] per cell must
// sound-contain every empirical boot AND every empirical vprior in the data
// (boot and vprior are both cp_value outputs, so both must lie in the envelope).
'use strict';
var fs = require('fs'), zlib = require('zlib'), readline = require('readline');
var cp = require('../../../js/cheap_planner.js');
var fb = require('./feat_bounds.js');
var ibp = require('./ibp.js');
var net = JSON.parse(fs.readFileSync(__dirname + '/../../../js/value_net.json', 'utf8'));

console.log('GELU trough: x*=' + ibp.GELU_XMIN.toFixed(6) + ' gelu(x*)=' + ibp.GELU_VMIN.toFixed(6));
// spot: gelu monotone check around trough
console.log('gelu(-2)=' + cp.cp_gelu(-2).toFixed(5), 'gelu(-0.75)=' + cp.cp_gelu(-0.75).toFixed(5),
    'gelu(0)=' + cp.cp_gelu(0).toFixed(5), 'gelu(3)=' + cp.cp_gelu(3).toFixed(5));

function envelope(W, Hc) {
    var raw = fb.rawFeatureBounds(W, Hc);
    var sb = fb.standardizeBounds(net, raw);
    return ibp.ibpValue(net, sb.lo, sb.hi);
}

var files = fs.readdirSync(__dirname + '/../calib_data').filter(function (f) { return f.endsWith('.decisions.jsonl.gz'); });
(async function () {
    for (var fi = 0; fi < files.length; fi++) {
        var f = files[fi];
        var rl = readline.createInterface({ input: fs.createReadStream(__dirname + '/../calib_data/' + f).pipe(zlib.createGunzip()) });
        var env = null, bmin = Infinity, bmax = -Infinity, vmin = Infinity, vmax = -Infinity, viol = 0, n = 0;
        for await (var line of rl) {
            if (!line) continue;
            var d = JSON.parse(line);
            if (!env) env = envelope(d.cfg.W, d.cfg.Hc);
            for (var r = 0; r < d.rolled.length; r++) { var bo = d.rolled[r][2]; if (bo < bmin) bmin = bo; if (bo > bmax) bmax = bo; if (bo < env.lo - 1e-9 || bo > env.hi + 1e-9) viol++; }
            for (var k = 0; k < d.vprior.length; k++) { var v = d.vprior[k]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; if (v < env.lo - 1e-9 || v > env.hi + 1e-9) viol++; }
            n++;
        }
        console.log(f.replace('.decisions.jsonl.gz', '').padEnd(34),
            'IBP[' + env.lo.toFixed(3) + ',' + env.hi.toFixed(3) + ']',
            'empBoot[' + bmin.toFixed(3) + ',' + bmax.toFixed(3) + ']',
            'empVprior[' + vmin.toFixed(3) + ',' + vmax.toFixed(3) + ']',
            'VIOL=' + viol, 'n=' + n);
    }
})();
