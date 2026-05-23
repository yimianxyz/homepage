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
    // v4: per-K-nearest boid velocities (slots 35..42). Slot k uses
    // [35 + 2*k] = vx_k, [36 + 2*k] = vy_k.
    VX1: 35, VY1: 36,
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

// rulePolicy v2: in hunt mode, aim at the *predicted* boid position
// α frames into the future. The predator's own predicted position is
// also extrapolated, so the relative offset to track is:
//
//   predicted_offset = (boid.pos - pred.pos) + α × (boid.vel - pred.vel)
//                    = dx1 + α × (bvx - vx)   (and same for y)
//
// α=0 collapses to the original rule. Tuning α trades aim-precision
// (small α) against aim-anticipation (large α); too large overshoots.
function rulePolicy_v2(features, alpha) {
    const vx = features[F.VX];
    const vy = features[F.VY];
    const dx1 = features[F.DX1];
    const dy1 = features[F.DY1];
    const d1 = features[F.D1];
    const bvx1 = features[F.VX1];
    const bvy1 = features[F.VY1];

    let tx, ty;
    if (d1 < POLICY_R && dx1 !== POLICY_PAD) {
        tx = dx1 + alpha * (bvx1 - vx);
        ty = dy1 + alpha * (bvy1 - vy);
    } else {
        tx = features[F.DXA];
        ty = features[F.DYA];
    }

    const desired = fastSetMagnitude(tx, ty, PREDATOR_MAX_SPEED);
    const steering = fastLimit(desired[0] - vx, desired[1] - vy, PREDATOR_MAX_FORCE);
    return steering;
}

// rulePolicy v3: smart target selection. The original rule picks the
// SINGLE NEAREST boid within range. v3 looks at all K=4 nearest and picks
// the one with the best "catch ease" — closing speed minus distance
// penalty — so we don't waste compute chasing a boid that's escaping
// faster than we can follow.
//
// This is structurally analogous to the flock_centroid patrol fix: a
// small change to the policy's TARGET SELECTION, no NN retraining. The
// +39% patrol fix changed WHERE to patrol; v3 changes WHICH boid to
// chase. If the same kind of structural win exists in the attack
// branch, it lives here.
//
// Variants (selected by `mode`):
//   'score_minus_dist': score_k = closing_k - DIST_W * d_k
//   'closing_only':     score_k = closing_k
//   'time_to_catch':    score_k = -d_k / max(closing_k, eps)  (min t_catch)
//   'closing_lookahead':as 'score_minus_dist' but score the boid's
//                       position α frames in the future (combines v2 +
//                       smart selection)
function rulePolicy_v3(features, opts) {
    opts = opts || {};
    const mode = opts.mode || 'score_minus_dist';
    const DIST_W = opts.distW != null ? opts.distW : 0.05;
    const alpha = opts.alpha != null ? opts.alpha : 0;
    const EPS = 0.05;

    const vx = features[F.VX];
    const vy = features[F.VY];
    const dxA = features[F.DXA];
    const dyA = features[F.DYA];

    let bestK = -1;
    let bestScore = -Infinity;
    for (let k = 0; k < POLICY_K; k++) {
        const dxk = features[7 + 2 * k];
        if (dxk === POLICY_PAD) continue;
        const dyk = features[8 + 2 * k];
        const dk = features[23 + k];
        if (dk >= POLICY_R) continue;
        const bvxk = features[35 + 2 * k];
        const bvyk = features[36 + 2 * k];

        // For closing_lookahead: shift target by α·(boid_vel - pred_vel)
        let tx_k = dxk, ty_k = dyk, d_k = dk;
        if (alpha !== 0) {
            tx_k = dxk + alpha * (bvxk - vx);
            ty_k = dyk + alpha * (bvyk - vy);
            d_k = Math.sqrt(tx_k * tx_k + ty_k * ty_k);
            if (d_k < 1e-9) d_k = 1e-9;
        }

        // Closing speed = component of (pred_vel - boid_vel) along (dx, dy)
        // = positive means predator approaching the (predicted) boid position
        const closing = ((vx - bvxk) * tx_k + (vy - bvyk) * ty_k) / d_k;

        let score;
        if (mode === 'closing_only') {
            score = closing;
        } else if (mode === 'time_to_catch') {
            // Minimise d / max(closing, eps); equivalent to maximising -that.
            score = -d_k / Math.max(closing, EPS);
        } else {
            // score_minus_dist (default) or closing_lookahead with default DIST_W.
            score = closing - DIST_W * d_k;
        }

        if (score > bestScore) {
            bestScore = score;
            bestK = k;
        }
    }

    let tx, ty;
    if (bestK >= 0) {
        // Use the chosen boid's offset (with optional lookahead) as the seek target.
        const bvxk = features[35 + 2 * bestK];
        const bvyk = features[36 + 2 * bestK];
        tx = features[7 + 2 * bestK];
        ty = features[8 + 2 * bestK];
        if (alpha !== 0) {
            tx = tx + alpha * (bvxk - vx);
            ty = ty + alpha * (bvyk - vy);
        }
    } else {
        tx = dxA;
        ty = dyA;
    }

    const desired = fastSetMagnitude(tx, ty, PREDATOR_MAX_SPEED);
    const steering = fastLimit(desired[0] - vx, desired[1] - vy, PREDATOR_MAX_FORCE);
    return steering;
}

// rulePolicy v4: classic pursuit-curve intercept. For each candidate boid
// within range, solve the quadratic for time-to-intercept assuming the
// boid moves at constant velocity and the predator moves at MAX_SPEED in
// a straight line toward the lead point. Pick the boid with smallest
// intercept time, head to ITS LEAD POINT (not its current position).
//
// Math: solve |b + v_b * t - p| = s_p * t for smallest positive t.
//   Let d = b - p, then  (d + v_b·t)·(d + v_b·t) = s_p² t²
//   ⇒ |v_b|² t² + 2 d·v_b t + |d|² - s_p² t² = 0
//   ⇒ (|v_b|² - s_p²) t² + 2(d·v_b) t + |d|² = 0
// If |v_b| > s_p the boid can outrun the predator (no real positive root
// when discriminant < 0). Fall back to seeking the boid's current
// position in that case.
//
// Returns the steering toward the lead point.
function rulePolicy_v4(features, opts) {
    opts = opts || {};
    const fallback_dist_w = opts.distW != null ? opts.distW : 0.0;  // optional distance tiebreaker

    const vx = features[F.VX];
    const vy = features[F.VY];
    const dxA = features[F.DXA];
    const dyA = features[F.DYA];
    const sp = PREDATOR_MAX_SPEED;
    const sp2 = sp * sp;

    let bestK = -1;
    let bestT = Infinity;
    let best_lead_x = 0, best_lead_y = 0;

    for (let k = 0; k < POLICY_K; k++) {
        const dxk = features[7 + 2 * k];
        if (dxk === POLICY_PAD) continue;
        const dyk = features[8 + 2 * k];
        const dk = features[23 + k];
        if (dk >= POLICY_R) continue;
        const bvxk = features[35 + 2 * k];
        const bvyk = features[36 + 2 * k];

        // Quadratic: (|v_b|² - s_p²) t² + 2(d·v_b) t + |d|² = 0
        const v2 = bvxk * bvxk + bvyk * bvyk;
        const a = v2 - sp2;
        const b = 2 * (dxk * bvxk + dyk * bvyk);
        const c = dxk * dxk + dyk * dyk;

        let t;
        if (Math.abs(a) < 1e-9) {
            // Degenerate (boid speed == predator speed): linear in t.
            t = -c / b;
        } else {
            const disc = b * b - 4 * a * c;
            if (disc < 0) {
                // Boid can outrun predator (or both stationary), no perfect
                // intercept. Skip this boid (or fall back to seeking current).
                continue;
            }
            const sqd = Math.sqrt(disc);
            // Two roots; want smallest positive
            const t1 = (-b - sqd) / (2 * a);
            const t2 = (-b + sqd) / (2 * a);
            if (t1 > 0 && t2 > 0) t = Math.min(t1, t2);
            else if (t1 > 0)      t = t1;
            else if (t2 > 0)      t = t2;
            else                  continue; // no positive root
        }
        if (t < 0 || !isFinite(t)) continue;

        // Optional distance tiebreaker so we don't always pick "shortest t"
        // when t differences are tiny (often happens at close range).
        const adjusted_t = t + fallback_dist_w * dk;

        if (adjusted_t < bestT) {
            bestT = adjusted_t;
            bestK = k;
            best_lead_x = dxk + bvxk * t;
            best_lead_y = dyk + bvyk * t;
        }
    }

    let tx, ty;
    if (bestK >= 0) {
        tx = best_lead_x;
        ty = best_lead_y;
    } else {
        tx = dxA;
        ty = dyA;
    }

    const desired = fastSetMagnitude(tx, ty, PREDATOR_MAX_SPEED);
    const steering = fastLimit(desired[0] - vx, desired[1] - vy, PREDATOR_MAX_FORCE);
    return steering;
}

// rulePolicy v5: avoidance-aware multi-step prediction. The boid the
// predator is approaching is actively turning away (predator-avoidance
// acceleration ≈ 0.15 / frame in the flee direction when close). Linear
// extrapolation (rule_v2 / v3 with α-lookahead) under-counts how far
// the boid will move because it ignores this acceleration.
//
// v5 predicts the boid's future position T frames ahead by iterating:
//   boid_pos(t+1) = boid_pos(t) + boid_vel(t)
//   boid_vel(t+1) = boid_vel(t) + a_avoid(t)
//   a_avoid(t)   ≈ unit(boid_pos - pred_pos) · (R-d)/R · TURN_FACTOR,
//                  with fastLimit cap.
// (We ignore boid-boid flocking forces — too expensive to recompute.)
//
// We also predict the predator's straight-line motion at MAX_SPEED in
// the seek direction so the relative offset stays meaningful. Then for
// each candidate boid, we pick the one with the smallest projected
// distance at T frames, and head to its predicted position.
const PREDATOR_RANGE_V5 = POLICY_R;            // matches PREDATOR_RANGE / POLICY_R = 80
const PREDATOR_TURN_FACTOR_V5 = 0.3;
const BOID_MAX_FORCE_AVOID = 0.15;             // MAX_FORCE * 1.5 from boid.js

function rulePolicy_v5(features, opts) {
    opts = opts || {};
    const T = opts.steps != null ? opts.steps : 5;
    const score_w_dist = opts.distW != null ? opts.distW : 0.0;

    const vx = features[F.VX];
    const vy = features[F.VY];
    const dxA = features[F.DXA];
    const dyA = features[F.DYA];

    let bestK = -1;
    let bestScore = Infinity;
    let best_lead_x = 0, best_lead_y = 0;

    for (let k = 0; k < POLICY_K; k++) {
        const dxk = features[7 + 2 * k];
        if (dxk === POLICY_PAD) continue;
        const dyk = features[8 + 2 * k];
        const dk = features[23 + k];
        if (dk >= PREDATOR_RANGE_V5) continue;
        const bvxk0 = features[35 + 2 * k];
        const bvyk0 = features[36 + 2 * k];

        // Roll forward T frames. Track boid position relative to predator's
        // expected position. Predator pursues by seeking toward the boid's
        // current relative offset (a strong simplification; could iterate).
        let rx = dxk, ry = dyk;      // relative offset (boid - predator)
        let bvx = bvxk0, bvy = bvyk0;
        for (let t = 0; t < T; t++) {
            const dnow = Math.sqrt(rx * rx + ry * ry);
            if (dnow < 1e-9) break;
            // Avoidance accel (boid moves away from predator)
            let ax = 0, ay = 0;
            if (dnow < PREDATOR_RANGE_V5) {
                const strength = (PREDATOR_RANGE_V5 - dnow) / PREDATOR_RANGE_V5 * PREDATOR_TURN_FACTOR_V5;
                const um = fastMagnitude(rx, ry);
                if (um > 0) {
                    ax = (rx / um) * strength;
                    ay = (ry / um) * strength;
                    const fm = fastMagnitude(ax, ay);
                    if (fm > BOID_MAX_FORCE_AVOID) {
                        const s = BOID_MAX_FORCE_AVOID / fm;
                        ax *= s; ay *= s;
                    }
                }
            }
            // Boid step
            bvx = bvx + ax;
            bvy = bvy + ay;
            // Clamp boid speed (boid.js uses iFastLimit(MAX_SPEED=6))
            const bm = fastMagnitude(bvx, bvy);
            if (bm > 6.0) { const s = 6.0 / bm; bvx *= s; bvy *= s; }
            // Predator step: seek the current relative offset at MAX_SPEED
            const sm = fastMagnitude(rx, ry);
            let pvx = 0, pvy = 0;
            if (sm > 0) { pvx = (rx / sm) * PREDATOR_MAX_SPEED; pvy = (ry / sm) * PREDATOR_MAX_SPEED; }
            // Relative offset update: boid moves by bvx; predator moves by pvx.
            rx = rx + bvx - pvx;
            ry = ry + bvy - pvy;
        }
        const d_pred = Math.sqrt(rx * rx + ry * ry);

        // Score: smaller projected distance is better, optional distance penalty.
        const score = d_pred + score_w_dist * dk;
        if (score < bestScore) {
            bestScore = score;
            bestK = k;
            // Lead point = boid's predicted offset relative to PREDATOR's
            // CURRENT position. Re-derive as: dx_k + (Δboid - Δpred we just
            // simulated). Use the relative offset rx, ry plus predator's
            // expected forward motion.
            best_lead_x = rx + T * (PREDATOR_MAX_SPEED * (dxk / Math.max(dk, 1e-9)));
            best_lead_y = ry + T * (PREDATOR_MAX_SPEED * (dyk / Math.max(dk, 1e-9)));
        }
    }

    let tx, ty;
    if (bestK >= 0) {
        tx = best_lead_x;
        ty = best_lead_y;
    } else {
        tx = dxA;
        ty = dyA;
    }

    const desired = fastSetMagnitude(tx, ty, PREDATOR_MAX_SPEED);
    const steering = fastLimit(desired[0] - vx, desired[1] - vy, PREDATOR_MAX_FORCE);
    return steering;
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
    rulePolicy_v2,
    rulePolicy_v3,
    rulePolicy_v4,
    rulePolicy_v5,
    ruleBranch,
};
