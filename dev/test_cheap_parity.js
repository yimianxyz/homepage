// Node parity test: JS cheap_planner vs the Python fixture (dev/js_fixture.json).
// Verifies the 19 features, 4 ctx, value-net V, and top-1 ballistic pick all
// match sim_torch to tight tolerance, using the SAME candidates + state.
'use strict';
var fs = require('fs');
var cp = require('../js/cheap_planner.js');
var fx = JSON.parse(fs.readFileSync(__dirname + '/js_fixture.json', 'utf8'));
var net = JSON.parse(fs.readFileSync(__dirname + '/../js/value_net.json', 'utf8'));

var pmaxS = fx.consts.PRED_MAX_SPEED, pmaxF = fx.consts.PRED_MAX_FORCE;
var maxFeatErr = 0, maxVErr = 0, top1ok = 0, ballErr = 0;
var decisions = fx.fixtures.length;

for (var di = 0; di < decisions; di++) {
    var F = fx.fixtures[di], st = F.state;
    var state = { px: st.px, py: st.py, pvx: st.pvx, pvy: st.pvy, psize: st.psize,
        bx: st.bx, by: st.by, bvx: st.bvx, bvy: st.bvy, nAlive: st.n_alive };
    var cands = F.cand.map(function (c) { return { x: c[0], y: c[1] }; });
    var r = cp.cp_features(state, cands, pmaxS, pmaxF);
    var V = cp.cp_value(net, r.feat, r.ctx);
    var top1 = cp.cp_top1(r.feat);
    // feature parity
    for (var k = 0; k < 16; k++)
        for (var j = 0; j < 19; j++) {
            var e = Math.abs(r.feat[k][j] - F.feat[k][j]);
            if (e > maxFeatErr) maxFeatErr = e;
        }
    // ballistic parity (feat cols 16,17,18 already inside feat; F.ballistic dump too)
    for (k = 0; k < 16; k++) {
        ballErr = Math.max(ballErr, Math.abs(r.feat[k][16] - F.ballistic[k][0]),
            Math.abs(r.feat[k][18] - F.ballistic[k][2]));
    }
    // V parity
    for (k = 0; k < 16; k++) { var ev = Math.abs(V[k] - F.v[k]); if (ev > maxVErr) maxVErr = ev; }
    if (top1 === F.top1_ballistic) top1ok++;
    console.log('decision', F.frame, 'top1 js=', top1, 'py=', F.top1_ballistic,
        'V[top1] js=', V[top1].toFixed(4), 'py=', F.v[top1].toFixed(4));
}
console.log('--- parity ---');
console.log('max feature err :', maxFeatErr.toExponential(3));
console.log('max ballistic err:', ballErr.toExponential(3));
console.log('max V err       :', maxVErr.toExponential(3));
console.log('top1 match       :', top1ok + '/' + decisions);
var ok = maxFeatErr < 1e-4 && maxVErr < 1e-3 && top1ok === decisions;
console.log(ok ? 'PARITY OK' : 'PARITY FAIL');
process.exit(ok ? 0 : 1);
