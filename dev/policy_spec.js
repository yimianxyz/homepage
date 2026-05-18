// Single source of truth for the predator policy: feature encoding, the
// analytical rule (used as the oracle / training target), and the constants
// that the trainer, eval, diagnose, and runtime all must agree on.
//
// The feature vector layout is fixed forever; changing it requires
// regenerating the dataset and re-training.

'use strict';

const sharedFeatures = require('../js/policy_features');

// Constants pulled from the shared browser/Node module so there's exactly
// one source of truth.
const POLICY_R = sharedFeatures.POLICY_R;
const POLICY_K = sharedFeatures.POLICY_K;
const POLICY_PAD = sharedFeatures.POLICY_PAD;
const FEATURE_DIM = sharedFeatures.FEATURE_DIM;
const buildFeaturesShared = sharedFeatures.buildPredatorFeatures;

// Feature index map (single source of truth for downstream code).
const F = {
    VX: 0, VY: 1,
    DXA: 2, DYA: 3,
    UXA: 4, UYA: 5,
    DA: 6,
    DX1: 7, DY1: 8,
    UX1: 15, UY1: 16,
    D1: 23,
};

// Predator kinematic constants -- must match js/predator.js exactly.
const PREDATOR_MAX_SPEED = 2.5;
const PREDATOR_MAX_FORCE = 0.05;

// Alpha-max-beta-min magnitude approximation (matches Vector.getFastMagnitude).
function fastMagnitude(x, y) {
    const ax = Math.abs(x);
    const ay = Math.abs(y);
    return Math.max(ax, ay) * 0.96 + Math.min(ax, ay) * 0.398;
}

function fastSetMagnitude(x, y, mag) {
    const m = fastMagnitude(x, y);
    if (m === 0) return [0, 0];
    const s = mag / m;
    return [x * s, y * s];
}

function fastLimit(x, y, max) {
    const m = fastMagnitude(x, y);
    if (m > max) {
        const s = max / m;
        return [x * s, y * s];
    }
    return [x, y];
}

// True Euclidean distance for nearest-neighbour comparison. The original
// predator uses Vector.getDistance which is sqrt-based -- match it exactly.
function trueDist(x, y) {
    return Math.sqrt(x * x + y * y);
}

// Delegated to the shared UMD module so browser and Node agree on every bit.
const buildFeatures = buildFeaturesShared;

// The analytical rule, expressed in feature space. Returns [ax, ay].
function rulePolicy(features) {
    const vx = features[F.VX];
    const vy = features[F.VY];
    const dx1 = features[F.DX1];
    const dy1 = features[F.DY1];
    const d1 = features[F.D1];

    let tx, ty;
    if (d1 < POLICY_R && dx1 !== POLICY_PAD) {
        tx = dx1;
        ty = dy1;
    } else {
        tx = features[F.DXA];
        ty = features[F.DYA];
    }

    const desired = fastSetMagnitude(tx, ty, PREDATOR_MAX_SPEED);
    const steering = fastLimit(desired[0] - vx, desired[1] - vy, PREDATOR_MAX_FORCE);
    return steering;
}

function ruleBranch(features) {
    const d1 = features[F.D1];
    return (d1 < POLICY_R && features[F.DX1] !== POLICY_PAD) ? 'hunt' : 'patrol';
}

module.exports = {
    POLICY_R,
    POLICY_K,
    POLICY_PAD,
    PREDATOR_MAX_SPEED,
    PREDATOR_MAX_FORCE,
    FEATURE_DIM,
    F,
    fastMagnitude,
    fastSetMagnitude,
    fastLimit,
    trueDist,
    buildFeatures,
    rulePolicy,
    ruleBranch,
};
