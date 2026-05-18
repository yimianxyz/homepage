// Take an existing v1 dataset (29 features, 2 targets, 31 floats/row) and
// extend it to v2 (34 features, 2 targets, 36 floats/row) by computing the
// 5 new features analytically from the v1 fields. No simulation needed.
//
// New features (appended at indices 29..33):
//   [29] seek_boid_x = exact steering for the "hunt" branch with target=(dx1,dy1)
//   [30] seek_boid_y
//   [31] seek_auto_x = exact steering for the "patrol" branch with target=(dxA,dyA)
//   [32] seek_auto_y
//   [33] inRange    = clamp((R - d1) / 5 + 0.5, 0, 1)  -- smooth 5px transition at R
//
//   node dev/extend_dataset.js --in dev/dataset.bin --out dev/dataset_v2.bin
'use strict';
const fs = require('fs');
const path = require('path');
const spec = require('./policy_spec');

function parseArgs(argv) {
    const args = { in: 'dev/dataset.bin', out: 'dev/dataset_v2.bin' };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--in') args.in = argv[++i];
        else if (a === '--out') args.out = argv[++i];
    }
    return args;
}

const MAX_SPEED = spec.PREDATOR_MAX_SPEED; // 2.5
const MAX_FORCE = spec.PREDATOR_MAX_FORCE; // 0.05
const R = spec.POLICY_R;                   // 80
const F = spec.F;
const NEW_FEATURE_COUNT = 5;

function seekStep(dx, dy, vx, vy) {
    // Match Vector.iFastSetMagnitude exactly: it's a no-op when fastMag is 0,
    // so desired = (0,0) and steering_pre = -velocity. Then iFastLimit.
    const m = spec.fastMagnitude(dx, dy);
    let dx0 = 0, dy0 = 0;
    if (m !== 0) {
        dx0 = dx * MAX_SPEED / m;
        dy0 = dy * MAX_SPEED / m;
    }
    const sx = dx0 - vx, sy = dy0 - vy;
    const sm = spec.fastMagnitude(sx, sy);
    if (sm > MAX_FORCE) {
        const f = MAX_FORCE / sm;
        return [sx * f, sy * f];
    }
    return [sx, sy];
}

function main() {
    const args = parseArgs(process.argv);
    const metaPath = args.in.replace(/\.bin$/, '.meta.json');
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const arr = new Float32Array(fs.readFileSync(args.in).buffer);
    const oldFD = meta.featureDim;
    const oldROW = meta.rowFloats;
    const n = arr.length / oldROW;
    const newFD = oldFD + NEW_FEATURE_COUNT;
    const newROW = newFD + (oldROW - oldFD); // keep same # of targets

    console.log(`in: n=${n} featureDim=${oldFD} rowFloats=${oldROW}`);
    console.log(`out: featureDim=${newFD} rowFloats=${newROW}`);

    const out = new Float32Array(n * newROW);
    for (let i = 0; i < n; i++) {
        const oOff = i * oldROW;
        const nOff = i * newROW;
        // Copy v1 features.
        for (let k = 0; k < oldFD; k++) out[nOff + k] = arr[oOff + k];
        // Compute v2 features.
        const vx = arr[oOff + F.VX], vy = arr[oOff + F.VY];
        const dx1 = arr[oOff + F.DX1], dy1 = arr[oOff + F.DY1];
        const dxA = arr[oOff + F.DXA], dyA = arr[oOff + F.DYA];
        const d1 = arr[oOff + F.D1];
        const sb = seekStep(dx1, dy1, vx, vy);
        const sa = seekStep(dxA, dyA, vx, vy);
        out[nOff + oldFD + 0] = sb[0];
        out[nOff + oldFD + 1] = sb[1];
        out[nOff + oldFD + 2] = sa[0];
        out[nOff + oldFD + 3] = sa[1];
        // Smooth in-range indicator: 1 well inside R, 0 well outside R,
        // linear ramp in a 10px window centered at R.
        const t = (R - d1) / 10 + 0.5;
        out[nOff + oldFD + 4] = t < 0 ? 0 : (t > 1 ? 1 : t);
        // Copy targets after the new features.
        for (let k = 0; k < (oldROW - oldFD); k++) {
            out[nOff + newFD + k] = arr[oOff + oldFD + k];
        }
    }

    fs.writeFileSync(args.out, Buffer.from(out.buffer, out.byteOffset, out.byteLength));
    const newMeta = Object.assign({}, meta, {
        featureDim: newFD,
        rowFloats: newROW,
        extendedFrom: path.basename(args.in),
        newFeatureNames: ['seek_boid_x', 'seek_boid_y', 'seek_auto_x', 'seek_auto_y', 'inRange_smooth'],
        extendedAt: new Date().toISOString(),
    });
    const newMetaPath = args.out.replace(/\.bin$/, '.meta.json');
    fs.writeFileSync(newMetaPath, JSON.stringify(newMeta, null, 2));
    console.log('wrote', args.out, 'and', newMetaPath);
}

if (require.main === module) main();
