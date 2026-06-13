// gen_oracle_policy.js — generates dev/exact_nn/oracle_policy.js, the
// INSTRUMENTED FORK of js/predator_cheap.js used by the oracle logger (#5).
//
// Spec contract ("instrumented fork, logging lines only" — #5 revision):
//   * every inserted line carries the /*ORACLE*/ marker and is a pure logging
//     hook guarded by `window.__oracle` (a no-op sink when unset);
//   * NO prod statement is modified, reordered, or re-indented — insertions
//     only. Mechanical proof: stripping all marker lines reproduces
//     js/predator_cheap.js byte-for-byte (gen --verify checks this, and the
//     certification gate re-checks it before any farming);
//   * anchors must match the prod source EXACTLY ONCE or generation hard-fails
//     (so a prod drift can never silently produce a stale fork).
//
// The per-frame state hook is a pass-through wrapper APPENDED after the
// `window.__cheap = {...}` assignment, inside the IIFE so it sees the
// closure-local persistent state (target / frame counter / egBoid) that no
// external wrapper can reach. It wraps unconditionally but logs only when
// window.__oracle is set; cert runs prove the wrapped fork is bit-identical
// to pristine prod either way.
//
//   node dev/exact_nn/gen_oracle_policy.js            # (re)generate
//   node dev/exact_nn/gen_oracle_policy.js --verify   # committed file fresh?
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const SRC = path.join(__dirname, '..', '..', 'js', 'predator_cheap.js');
const OUT = path.join(__dirname, 'oracle_policy.js');
const MARK = '/*ORACLE*/';

// where: 'after' inserts the line(s) on a new line following the anchor;
// 'before' inserts on the line above it. Indentation mirrors the anchor's.
const INSERTS = [
    {   // full plan inputs: state snapshot, 16 candidates, feature matrix, NN prior
        find: '        var vprior = cp_value(NET, fr.feat, fr.ctx);',
        where: 'after',
        lines: ['        ' + MARK + ' if (window.__oracle) window.__oracle.planStart(s, cands, fr, vprior, cfg);'],
    },
    {   // each rolled candidate: rollout catches + terminal value bootstrap
        find: '            score[ci] = rr.catches + boot;',
        where: 'after',
        lines: ['            ' + MARK + ' if (window.__oracle) window.__oracle.roll(ci, rr.catches, boot);'],
    },
    {   // final ranking + argmax, before the viz call
        find: '        if (vizModel) cp_value_viz(NET, fr.feat[bi], fr.ctx, vizModel);',
        where: 'before',
        lines: ['        ' + MARK + ' if (window.__oracle) window.__oracle.planEnd(pidx, score, bi);'],
    },
    {   // per-frame persistent closure state (target coords / frame counter /
        // egBoid identity) + the returned force — appended pass-through wrapper
        find: '            return steer(pred, boids);\n        }\n    };',
        where: 'after',
        lines: [
            '    ' + MARK + ' (function () { var __pf = window.__cheap.force;',
            '    ' + MARK + '   window.__cheap.force = function (pred, boids) { var r = __pf(pred, boids);',
            '    ' + MARK + '     if (window.__oracle) window.__oracle.frameEnd(target, frame, egBoid, boids, r, pred);',
            '    ' + MARK + '     return r; }; })();',
        ],
    },
];

function sha256(s) { return crypto.createHash('sha256').update(s).digest('hex'); }

function generate() {
    const src = fs.readFileSync(SRC, 'utf8');
    if (src.includes('ORACLE')) throw new Error('prod source already contains the marker?!');
    let out = src;
    for (const ins of INSERTS) {
        const first = out.indexOf(ins.find);
        if (first < 0) throw new Error('anchor not found (prod drifted?): ' + JSON.stringify(ins.find.slice(0, 60)));
        if (out.indexOf(ins.find, first + 1) >= 0) throw new Error('anchor not unique: ' + JSON.stringify(ins.find.slice(0, 60)));
        const block = ins.lines.join('\n');
        out = ins.where === 'after'
            ? out.replace(ins.find, ins.find + '\n' + block)
            : out.replace(ins.find, block + '\n' + ins.find);
    }
    const header = [
        '// oracle_policy.js — GENERATED instrumented fork of js/predator_cheap.js.',
        '// DO NOT EDIT BY HAND: regenerate with `node dev/exact_nn/gen_oracle_policy.js`',
        '// and re-certify (dev/exact_nn/certify_oracle.js) before farming.',
        '// Logging lines only: every inserted line carries ' + MARK + '; stripping them',
        '// reproduces the prod source byte-for-byte (gen --verify proves it).',
        '// source: js/predator_cheap.js sha256=' + sha256(src),
    ].join('\n') + '\n\n';   // real blank line terminates the header (strip() relies on it)
    return { text: header + out, srcSha: sha256(src) };
}

// strip rule: drop the generated header (every line up to and including the
// first blank line) and every line containing the marker; must equal prod src.
function strip(text) {
    const lines = text.split('\n');
    let i = 0;
    while (i < lines.length && lines[i] !== '') i++;
    return lines.slice(i + 1).filter(l => !l.includes(MARK)).join('\n');
}

function main() {
    const verify = process.argv.includes('--verify');
    const g = generate();
    if (strip(g.text) !== fs.readFileSync(SRC, 'utf8')) {
        throw new Error('round-trip failed: stripping marker lines does not reproduce prod source');
    }
    if (verify) {
        const cur = fs.readFileSync(OUT, 'utf8');
        if (cur !== g.text) { console.error('STALE: committed oracle_policy.js != regenerated'); process.exit(2); }
        console.log('VERIFY OK  oracleSha=' + sha256(cur) + '  srcSha=' + g.srcSha);
        return;
    }
    fs.writeFileSync(OUT, g.text);
    console.log('wrote ' + OUT + '  oracleSha=' + sha256(g.text) + '  srcSha=' + g.srcSha);
}

if (require.main === module) main();
module.exports = { generate, strip, SRC, OUT, MARK, sha256 };
