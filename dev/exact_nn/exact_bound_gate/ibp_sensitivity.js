// ibp_sensitivity.js — which input feature drives the IBP output radius blowup?
// For each feature i, collapse ITS interval to a point (its midpoint) and rerun
// IBP; the drop in output width attributes the blowup. Also report the per-feature
// standardized half-width (radius), the dominant driver of layer-0 spread.
'use strict';
var fs = require('fs');
var fb = require('./feat_bounds.js');
var ibp = require('./ibp.js');
var net = JSON.parse(fs.readFileSync(__dirname + '/../../../js/value_net.json', 'utf8'));

var labels = ["rx/PS","ry/PS","dist/PS","isE3d","tgo/60","tbrx/PS","tbry/PS","tbvx/VS","tbvy/VS","tbDistCand/PS","rangepb/PS","closing/VS","losRate*50","dens/20","fleeAlign/VS","(rangepb-dist)/PS","tCatchNorm","minDist/PS","caught","pvx/VS","pvy/VS","fracAlive","psize/20"];

var W = 1024, Hc = 768;
var raw = fb.rawFeatureBounds(W, Hc);
var sb = fb.standardizeBounds(net, raw);
var full = ibp.ibpValue(net, sb.lo, sb.hi);
var fullW = full.hi - full.lo;
console.log('cfg', W + 'x' + Hc, 'full IBP width =', fullW.toExponential(3), '[' + full.lo.toExponential(3) + ',' + full.hi.toExponential(3) + ']');
console.log('idx feature              std-radius      width-if-collapsed   width-drop');
var rows = [];
for (var i = 0; i < 23; i++) {
    var lo2 = sb.lo.slice(), hi2 = sb.hi.slice();
    var mid = (lo2[i] + hi2[i]) / 2; lo2[i] = mid; hi2[i] = mid;
    var r = ibp.ibpValue(net, lo2, hi2);
    var w = r.hi - r.lo;
    rows.push({ i: i, lab: labels[i], rad: (sb.hi[i] - sb.lo[i]) / 2, w: w, drop: fullW - w });
}
rows.sort(function (a, b) { return b.drop - a.drop; });
for (var k = 0; k < rows.length; k++) {
    var x = rows[k];
    console.log(String(x.i).padStart(3), x.lab.padEnd(20), x.rad.toExponential(3).padStart(12), x.w.toExponential(3).padStart(16), x.drop.toExponential(3).padStart(14));
}
