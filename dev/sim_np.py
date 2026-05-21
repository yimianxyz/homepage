"""Vectorized NumPy port of the boid + predator simulation.

Eliminates the per-seed JS-eval bottleneck so we can run 64-256 parallel
sims in seconds. Matches js/boid.js + js/predator.js + js/simulation.js
behavior bit-for-bit (modulo float32/float64 rounding) under the
flock_centroid patrol policy.

Usage:
    from sim_np import Sim, load_weights
    weights = load_weights('js/predator_weights.json')
    sim = Sim(seeds=range(100, 116), num_boids=120, weights=weights)
    summary = sim.run(max_frames=5000)
    print(summary['mean_catches'])
"""

import json
import numpy as np
from pathlib import Path


# Constants — must match js/boid.js, js/predator.js, js/simulation.js.
N_BOIDS = 120
CANVAS_W = 1680
CANVAS_H = 1680
FRAME_MS = 12

# Boid dynamics
MAX_SPEED = 6.0
MAX_FORCE = 0.1
DESIRED_SEPARATION = 40.0
NEIGHBOR_DISTANCE = 60.0
BORDER_OFFSET = 10.0
EPSILON = 1e-7
SEP_MULT = 2.0
COH_MULT = 1.0
ALI_MULT = 1.0

# Predator dynamics
PREDATOR_RANGE = 80.0          # = POLICY_R, the hunting threshold
PREDATOR_TURN_FACTOR = 0.3     # boid's avoidance reaction strength
PREDATOR_MAX_SPEED = 2.5
PREDATOR_MAX_FORCE = 0.05
PREDATOR_BASE_SIZE = 12.0
PREDATOR_MAX_SIZE = PREDATOR_BASE_SIZE * 1.8       # 21.6
PREDATOR_GROWTH = 1.2
PREDATOR_DECAY = 0.002
PREDATOR_FEED_COOLDOWN_MS = 100

# Feature pipeline (matches js/policy_features.js v3 layout, 35-dim).
# Slots 35..42 (v4 velocities) and 43..44 (v5 seek_boid_v2) are optional;
# we generate them up to FEATURE_DIM determined by the loaded weights.
POLICY_K = 4
POLICY_PAD = 2000.0
PREDICT_ALPHA = 8              # used when FEATURE_DIM >= 45


def mulberry32_seq(seed: int, n: int) -> np.ndarray:
    """Generate n consecutive mulberry32 floats in [0,1). Matches js/rng.js."""
    out = np.empty(n, dtype=np.float64)
    s = np.uint32(seed)
    for i in range(n):
        s = (s + np.uint32(0x6D2B79F5)) & np.uint32(0xFFFFFFFF)
        t = s
        t = np.uint32(t ^ (t >> np.uint32(15))) * (t | np.uint32(1)) & np.uint32(0xFFFFFFFF)
        t = (t ^ (t + np.uint32(np.uint32(t ^ (t >> np.uint32(7))) * (t | np.uint32(61)) & np.uint32(0xFFFFFFFF)))) & np.uint32(0xFFFFFFFF)
        out[i] = ((t ^ (t >> np.uint32(14))) & np.uint32(0xFFFFFFFF)) / 4294967296.0
    return out


def fast_mag(x, y):
    """Alpha-max-beta-min magnitude approximation (Cornell ECE 5730).
    Matches js/vector.js getFastMagnitude exactly. Works element-wise on arrays."""
    ax = np.abs(x)
    ay = np.abs(y)
    return np.maximum(ax, ay) * 0.96 + np.minimum(ax, ay) * 0.398


def fast_set_magnitude(x, y, mag):
    """Scale (x, y) vectors to have fast-magnitude == mag. Returns (x', y')."""
    m = fast_mag(x, y)
    safe = m > 0
    s = np.where(safe, mag / np.where(safe, m, 1.0), 0.0)
    return x * s, y * s


def fast_limit(x, y, max_mag):
    """Cap (x, y) fast-magnitude at max_mag. Returns (x', y')."""
    m = fast_mag(x, y)
    cap = m > max_mag
    s = np.where(cap, max_mag / np.where(m > 0, m, 1.0), 1.0)
    return x * s, y * s


def load_weights(path: str | Path) -> dict:
    """Load js/predator_weights.json into a dict of NumPy arrays.

    Required fields:
      featureDim   : int
      inputMean    : list[float]   (length featureDim)
      inputStd     : list[float]   (length featureDim)
      outputScale  : float
      layers       : [{W:[...], b:[...], activation:'relu'|'linear', inDim, outDim}]
    """
    with open(path) as f:
        j = json.load(f)
    layers = []
    for L in j['layers']:
        in_dim = L.get('inDim') or len(L['b']) and (len(L['W']) // len(L['b']))
        out_dim = L.get('outDim') or len(L['b'])
        W = np.array(L['W'], dtype=np.float32).reshape(in_dim, out_dim)
        b = np.array(L['b'], dtype=np.float32)
        layers.append({'W': W, 'b': b, 'activation': L['activation']})
    # Use inputMean length as the ground-truth featureDim (some weight files
    # incorrectly save spec.FEATURE_DIM instead of the dataset's featureDim).
    true_fd = len(j['inputMean'])
    return {
        'featureDim': true_fd,
        'inputMean': np.array(j['inputMean'], dtype=np.float32),
        'inputStd': np.array(j['inputStd'], dtype=np.float32),
        'outputScale': float(j['outputScale']),
        'clipMagnitude': float(j.get('clipMagnitude', 0.05)),
        'layers': layers,
    }


def nn_forward(features: np.ndarray, weights: dict) -> np.ndarray:
    """Batched NN forward. features: (B, FEATURE_DIM). Returns (B, 2) steering.

    Matches js/predator_nn.js's forward pass: (x - inputMean)/inputStd ->
    layers -> output * outputScale -> clipped to clipMagnitude.
    """
    fd = weights['featureDim']
    x = (features[:, :fd] - weights['inputMean']) / weights['inputStd']
    for L in weights['layers']:
        x = x @ L['W'] + L['b']
        if L['activation'] == 'relu':
            x = np.maximum(x, 0.0)
        # 'linear' is identity, nothing to do
    x = x * weights['outputScale']
    # Clip magnitude (matches js/predator_nn.js)
    cm = weights['clipMagnitude']
    if cm > 0:
        mag = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        s = np.where(mag > cm, cm / np.maximum(mag, 1e-12), 1.0)
        x = x * s[:, None]
    return x.astype(np.float32)


def build_features(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive,
                   auto_target, feature_dim):
    """Build batched feature vectors matching js/policy_features.js.

    Args:
        pred_pos: (B, 2) predator positions
        pred_vel: (B, 2) predator velocities
        boid_pos: (B, N, 2) boid positions
        boid_vel: (B, N, 2) boid velocities
        boid_alive: (B, N) bool mask
        auto_target: (B, 2) current autoTarget
        feature_dim: int — number of features to fill (35, 43, or 45)

    Returns: (B, feature_dim) features.
    """
    B = pred_pos.shape[0]
    out = np.zeros((B, feature_dim), dtype=np.float32)
    vx = pred_pos[:, 0] * 0 + pred_vel[:, 0]  # keep dtype
    vy = pred_vel[:, 1]
    out[:, 0] = pred_vel[:, 0]
    out[:, 1] = pred_vel[:, 1]
    dxA = auto_target[:, 0] - pred_pos[:, 0]
    dyA = auto_target[:, 1] - pred_pos[:, 1]
    out[:, 2] = dxA
    out[:, 3] = dyA
    dA = np.sqrt(dxA * dxA + dyA * dyA)
    safe = dA > 1e-9
    out[:, 4] = np.where(safe, dxA / np.where(safe, dA, 1.0), 0.0)
    out[:, 5] = np.where(safe, dyA / np.where(safe, dA, 1.0), 0.0)
    out[:, 6] = dA

    # K-nearest boids per batch element. Distance for dead boids = inf so
    # they sort to the end. If fewer than K alive, pad with POLICY_PAD.
    dx = boid_pos[:, :, 0] - pred_pos[:, None, 0]    # (B, N)
    dy = boid_pos[:, :, 1] - pred_pos[:, None, 1]
    d2 = dx * dx + dy * dy
    d2_masked = np.where(boid_alive, d2, np.inf)

    # Sort by d2 (alive boids first, then inf for dead).
    order = np.argsort(d2_masked, axis=1)  # (B, N)
    # Take the K nearest.
    K = POLICY_K
    near_idx = order[:, :K]                # (B, K)
    rows = np.arange(B)[:, None]
    near_dx = dx[rows, near_idx]            # (B, K)
    near_dy = dy[rows, near_idx]
    near_d2 = d2_masked[rows, near_idx]
    near_alive = boid_alive[rows, near_idx]
    near_d = np.sqrt(np.where(near_alive, near_d2, 0.0))
    near_bvx = boid_vel[rows, near_idx, 0]
    near_bvy = boid_vel[rows, near_idx, 1]

    for k in range(K):
        # offset slots
        is_real = near_alive[:, k]
        dx_k = np.where(is_real, near_dx[:, k], POLICY_PAD)
        dy_k = np.where(is_real, near_dy[:, k], POLICY_PAD)
        d_k = np.where(is_real, near_d[:, k], POLICY_PAD)
        out[:, 7 + 2 * k] = dx_k
        out[:, 8 + 2 * k] = dy_k
        safe_k = is_real & (near_d[:, k] > 1e-9)
        out[:, 15 + 2 * k] = np.where(safe_k, near_dx[:, k] / np.where(safe_k, near_d[:, k], 1.0),
                                      np.where(is_real, 0.0, 1.0))  # sentinel (1,0) when absent
        out[:, 16 + 2 * k] = np.where(safe_k, near_dy[:, k] / np.where(safe_k, near_d[:, k], 1.0), 0.0)
        out[:, 23 + k] = d_k

    # v2/v3 precomputed seek vectors + smooth/binary in-range indicators.
    dx1 = out[:, 7]; dy1 = out[:, 8]; d1 = out[:, 23]
    in_range_real = (d1 < PREDATOR_RANGE) & (dx1 != POLICY_PAD)

    # seek_boid_xy = fastLimit(fastSetMag((dx1,dy1), MAX_SPEED) - vel, MAX_FORCE)
    dx0, dy0 = fast_set_magnitude(dx1, dy1, PREDATOR_MAX_SPEED)
    sx = dx0 - pred_vel[:, 0]
    sy = dy0 - pred_vel[:, 1]
    sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
    out[:, 29] = sx
    out[:, 30] = sy

    # seek_auto_xy
    dx0, dy0 = fast_set_magnitude(dxA, dyA, PREDATOR_MAX_SPEED)
    sx = dx0 - pred_vel[:, 0]
    sy = dy0 - pred_vel[:, 1]
    sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
    out[:, 31] = sx
    out[:, 32] = sy

    # smooth in-range indicator
    t = (PREDATOR_RANGE - d1) / 10.0 + 0.5
    out[:, 33] = np.clip(t, 0.0, 1.0)
    # binary in-range
    out[:, 34] = in_range_real.astype(np.float32)

    if feature_dim >= 43:
        # v4: per-K boid velocities at slots 35..42
        for k in range(K):
            is_real = near_alive[:, k]
            out[:, 35 + 2 * k] = np.where(is_real, near_bvx[:, k], 0.0)
            out[:, 36 + 2 * k] = np.where(is_real, near_bvy[:, k], 0.0)

    if feature_dim >= 45:
        # v5: seek_boid_v2 at slots 43..44 — anticipation via boid velocity
        bvx1 = out[:, 35]
        bvy1 = out[:, 36]
        tx = dx1 + PREDICT_ALPHA * (bvx1 - pred_vel[:, 0])
        ty = dy1 + PREDICT_ALPHA * (bvy1 - pred_vel[:, 1])
        dx0, dy0 = fast_set_magnitude(tx, ty, PREDATOR_MAX_SPEED)
        sx = dx0 - pred_vel[:, 0]
        sy = dy0 - pred_vel[:, 1]
        sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
        out[:, 43] = np.where(in_range_real, sx, 0.0)
        out[:, 44] = np.where(in_range_real, sy, 0.0)

    return out


class Sim:
    """Batched NumPy simulation of the boid + predator world.

    Each row of the batch is an independent seed (different initial boid
    positions/velocities). The predator policy is shared across all rows.
    """

    def __init__(self, seeds, weights, num_boids=N_BOIDS, auto_target='flock_centroid'):
        self.seeds = list(seeds)
        self.B = len(self.seeds)
        self.N = num_boids
        self.weights = weights
        self.auto_target_mode = auto_target
        self._initialize()

    def _initialize(self):
        """Seed each batch element and lay out initial state. Matches
        simulation.initialize() exactly:
          - simRandom() × 2 → start_x = floor(... * W), start_y = floor(... * H)
          - all N boids start at (start_x, start_y)
          - for each boid, simRandom() → randomAngle; vel = (cos, sin)
          - predator at canvas center
        """
        self.boid_pos = np.zeros((self.B, self.N, 2), dtype=np.float64)
        self.boid_vel = np.zeros((self.B, self.N, 2), dtype=np.float64)
        self.boid_alive = np.ones((self.B, self.N), dtype=bool)
        self.pred_pos = np.full((self.B, 2), [CANVAS_W / 2, CANVAS_H / 2], dtype=np.float64)
        self.pred_vel = np.zeros((self.B, 2), dtype=np.float64)
        self.pred_size = np.full((self.B,), PREDATOR_BASE_SIZE, dtype=np.float64)
        self.pred_auto = np.copy(self.pred_pos)
        self.pred_last_feed_ms = np.zeros((self.B,), dtype=np.float64)
        self.catches = np.zeros((self.B,), dtype=np.int32)

        for bi, seed in enumerate(self.seeds):
            # RNG sequence per js/simulation.js + js/predator.js:
            #   r[0],r[1]      = start_x, start_y for boid cluster
            #   r[2..2+N-1]    = randomAngle per boid (velocity = cos, sin)
            #   r[2+N], r[2+N+1] = predator initial velocity (each * 2 - 1)
            r = mulberry32_seq(seed, 2 + self.N + 2)
            start_x = float(int(r[0] * CANVAS_W))
            start_y = float(int(r[1] * CANVAS_H))
            self.boid_pos[bi, :, 0] = start_x
            self.boid_pos[bi, :, 1] = start_y
            angles = r[2:2 + self.N] * 2 * np.pi
            self.boid_vel[bi, :, 0] = np.cos(angles)
            self.boid_vel[bi, :, 1] = np.sin(angles)
            self.pred_vel[bi, 0] = r[2 + self.N] * 2 - 1
            self.pred_vel[bi, 1] = r[2 + self.N + 1] * 2 - 1

        self.frame = 0

    def _compute_boid_acceleration(self):
        """Vectorised cohesion + separation + alignment + predator avoidance.

        All work happens in (B, N, N) tensors. Dead boids are excluded by
        setting their pairwise distance to infinity in the relevant masks.
        """
        B, N = self.B, self.N
        pos = self.boid_pos              # (B, N, 2)
        vel = self.boid_vel
        alive = self.boid_alive          # (B, N)

        # Pairwise: delta[b, i, j] = pos[b, j] - pos[b, i].
        delta = pos[:, None, :, :] - pos[:, :, None, :]      # (B, N, N, 2)
        dist = np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2) + EPSILON  # (B, N, N)

        # Mask: only count alive j boids (i != j is fine because self-distance
        # ≈ EPSILON which exceeds 0 — but we must NOT count self in neighbor
        # aggregations; JS uses if (this === boid) continue).
        eye = np.eye(N, dtype=bool)[None, :, :]              # (1, N, N)
        alive_j = alive[:, None, :] & ~eye                   # (B, N, N) — j is alive AND j != i

        # Cohesion: mean position of alive neighbors within NEIGHBOR_DISTANCE.
        coh_mask = alive_j & (dist <= NEIGHBOR_DISTANCE)
        coh_count = coh_mask.sum(axis=2)                     # (B, N)
        coh_pos_sum = (pos[:, None, :, :] * coh_mask[..., None]).sum(axis=2)  # (B, N, 2)
        coh_has = coh_count > 0
        coh_avg = np.where(coh_has[..., None],
                           coh_pos_sum / np.where(coh_has[..., None], coh_count[..., None], 1.0),
                           pos)  # if no neighbors, target self (steering will be 0)
        # seek(coh_avg) - returns steering toward coh_avg
        seek_x = coh_avg[..., 0] - pos[..., 0]
        seek_y = coh_avg[..., 1] - pos[..., 1]
        dx0, dy0 = fast_set_magnitude(seek_x, seek_y, MAX_SPEED)
        cx = dx0 - vel[..., 0]
        cy = dy0 - vel[..., 1]
        cx, cy = fast_limit(cx, cy, MAX_FORCE)
        # No neighbors → zero cohesion
        cx = np.where(coh_has, cx, 0.0)
        cy = np.where(coh_has, cy, 0.0)

        # Separation: sum of (self - other)/distance over alive neighbors
        # within DESIRED_SEPARATION; then average; then seek-style limit.
        sep_mask = alive_j & (dist < DESIRED_SEPARATION) & (dist > 0)
        sep_count = sep_mask.sum(axis=2)
        # delta_self_minus_other[b, i, j] = pos[b, i] - pos[b, j] = -delta[b, i, j]
        # JS uses iNormalize (true normalize), so |v|/|v| = unit; then divide by distance.
        delta_neg = -delta                                   # (B, N, N, 2)
        # true magnitude
        true_mag = np.sqrt(delta_neg[..., 0] ** 2 + delta_neg[..., 1] ** 2) + EPSILON
        unit_x = delta_neg[..., 0] / true_mag
        unit_y = delta_neg[..., 1] / true_mag
        # divide by distance (distance already includes EPSILON)
        scaled_x = unit_x / dist
        scaled_y = unit_y / dist
        sep_x = (scaled_x * sep_mask).sum(axis=2)            # (B, N)
        sep_y = (scaled_y * sep_mask).sum(axis=2)
        sep_has = sep_count > 0
        sep_x_avg = np.where(sep_has, sep_x / np.where(sep_has, sep_count, 1.0), 0.0)
        sep_y_avg = np.where(sep_has, sep_y / np.where(sep_has, sep_count, 1.0), 0.0)
        # If magnitude > 0, normalize to MAX_SPEED, subtract velocity, limit force.
        m = fast_mag(sep_x_avg, sep_y_avg)
        applied = m > 0
        sx, sy = fast_set_magnitude(sep_x_avg, sep_y_avg, MAX_SPEED)
        sx = sx - vel[..., 0]
        sy = sy - vel[..., 1]
        sx, sy = fast_limit(sx, sy, MAX_FORCE)
        sep_x_final = np.where(applied, sx, 0.0)
        sep_y_final = np.where(applied, sy, 0.0)

        # Alignment: mean velocity of alive neighbors within NEIGHBOR_DISTANCE.
        ali_mask = alive_j & (dist < NEIGHBOR_DISTANCE) & (dist > 0)
        ali_count = ali_mask.sum(axis=2)
        ali_vel_sum = (vel[:, None, :, :] * ali_mask[..., None]).sum(axis=2)  # (B, N, 2)
        ali_has = ali_count > 0
        ali_avg = np.where(ali_has[..., None],
                           ali_vel_sum / np.where(ali_has[..., None], ali_count[..., None], 1.0),
                           np.zeros_like(ali_vel_sum))
        ax_, ay_ = fast_set_magnitude(ali_avg[..., 0], ali_avg[..., 1], MAX_SPEED)
        ax_ = ax_ - vel[..., 0]
        ay_ = ay_ - vel[..., 1]
        ax_, ay_ = fast_limit(ax_, ay_, MAX_FORCE)
        ax_ = np.where(ali_has, ax_, 0.0)
        ay_ = np.where(ali_has, ay_, 0.0)

        # Predator avoidance: for each boid, distance to predator. If <
        # PREDATOR_RANGE, push away with scaled magnitude.
        pdx = pos[..., 0] - self.pred_pos[:, None, 0]        # (B, N)
        pdy = pos[..., 1] - self.pred_pos[:, None, 1]
        pdist = np.sqrt(pdx * pdx + pdy * pdy) + EPSILON
        in_pr = pdist < PREDATOR_RANGE
        # avoidance vector = (pos - pred); normalize (fast); scale by strength.
        avx, avy = pdx, pdy
        # iFastNormalize: divide by fast magnitude.
        fm = fast_mag(avx, avy)
        fm_safe = np.where(fm > 0, fm, 1.0)
        avx = avx / fm_safe
        avy = avy / fm_safe
        strength = (PREDATOR_RANGE - pdist) / PREDATOR_RANGE
        avx = avx * strength * PREDATOR_TURN_FACTOR
        avy = avy * strength * PREDATOR_TURN_FACTOR
        # fast limit at MAX_FORCE * 1.5
        avx, avy = fast_limit(avx, avy, MAX_FORCE * 1.5)
        # Apply only when in range
        avx = np.where(in_pr, avx, 0.0)
        avy = np.where(in_pr, avy, 0.0)

        # Sum with multipliers (cohesion 1, separation 2, alignment 1, predator avoidance 1)
        acc_x = (cx * COH_MULT) + (sep_x_final * SEP_MULT) + (ax_ * ALI_MULT) + avx
        acc_y = (cy * COH_MULT) + (sep_y_final * SEP_MULT) + (ay_ * ALI_MULT) + avy

        return acc_x, acc_y

    def _step_boids(self):
        """Apply acceleration → velocity → position → wrap."""
        ax, ay = self._compute_boid_acceleration()
        # vel += acc
        new_vx = self.boid_vel[..., 0] + ax
        new_vy = self.boid_vel[..., 1] + ay
        # Boid velocity is limited via fastLimit to MAX_SPEED inside flock?
        # Actually looking at js/boid.js's run() — boid velocity is limited
        # to MAX_SPEED via iFastLimit after acceleration is applied.
        new_vx, new_vy = fast_limit(new_vx, new_vy, MAX_SPEED)
        self.boid_vel[..., 0] = new_vx
        self.boid_vel[..., 1] = new_vy
        self.boid_pos[..., 0] += new_vx
        self.boid_pos[..., 1] += new_vy
        # Wrap (BORDER_OFFSET = 10)
        self.boid_pos[..., 0] = np.where(
            self.boid_pos[..., 0] > CANVAS_W + BORDER_OFFSET,
            -BORDER_OFFSET, self.boid_pos[..., 0])
        self.boid_pos[..., 0] = np.where(
            self.boid_pos[..., 0] < -BORDER_OFFSET,
            CANVAS_W + BORDER_OFFSET, self.boid_pos[..., 0])
        self.boid_pos[..., 1] = np.where(
            self.boid_pos[..., 1] > CANVAS_H + BORDER_OFFSET,
            -BORDER_OFFSET, self.boid_pos[..., 1])
        self.boid_pos[..., 1] = np.where(
            self.boid_pos[..., 1] < -BORDER_OFFSET,
            CANVAS_H + BORDER_OFFSET, self.boid_pos[..., 1])

    def _update_auto_target(self):
        """Apply autoTargetMode. flock_centroid: centroid of alive boids.
        Only updates when no boid is within PREDATOR_RANGE (matching the
        rule's branch logic in predator.js)."""
        if self.auto_target_mode == 'random':
            # Not implementing random mode here — flock_centroid is the
            # deployment target. (Could add later if needed.)
            return
        if self.auto_target_mode != 'flock_centroid':
            raise NotImplementedError(f"autoTargetMode={self.auto_target_mode}")

        # Per-batch: any boid in range?
        dx = self.boid_pos[..., 0] - self.pred_pos[:, None, 0]
        dy = self.boid_pos[..., 1] - self.pred_pos[:, None, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d_masked = np.where(self.boid_alive, d, np.inf)
        any_in_range = (d_masked < PREDATOR_RANGE).any(axis=1)   # (B,)
        any_alive = self.boid_alive.any(axis=1)                  # (B,)

        # Centroid of alive boids
        alive_f = self.boid_alive.astype(np.float64)
        n_alive = alive_f.sum(axis=1)
        n_alive_safe = np.where(n_alive > 0, n_alive, 1.0)
        cx = (self.boid_pos[..., 0] * alive_f).sum(axis=1) / n_alive_safe
        cy = (self.boid_pos[..., 1] * alive_f).sum(axis=1) / n_alive_safe
        new_target_x = np.where(~any_in_range & any_alive, cx, self.pred_auto[:, 0])
        new_target_y = np.where(~any_in_range & any_alive, cy, self.pred_auto[:, 1])
        self.pred_auto[:, 0] = new_target_x
        self.pred_auto[:, 1] = new_target_y

    def _step_predator(self):
        """Build features → NN forward → update velocity, position."""
        self._update_auto_target()
        feats = build_features(
            self.pred_pos, self.pred_vel,
            self.boid_pos, self.boid_vel, self.boid_alive,
            self.pred_auto, self.weights['featureDim'],
        )
        steering = nn_forward(feats, self.weights)
        # vel += steering, limit, position += vel, wrap.
        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = new_vx
        self.pred_vel[:, 1] = new_vy
        self.pred_pos[:, 0] += new_vx
        self.pred_pos[:, 1] += new_vy
        # wrap
        self.pred_pos[:, 0] = np.where(
            self.pred_pos[:, 0] > CANVAS_W + 20, -20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = np.where(
            self.pred_pos[:, 0] < -20, CANVAS_W + 20, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = np.where(
            self.pred_pos[:, 1] > CANVAS_H + 20, -20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = np.where(
            self.pred_pos[:, 1] < -20, CANVAS_H + 20, self.pred_pos[:, 1])

    def _check_catches(self):
        """One boid eaten per frame if within catch radius and cooldown elapsed.
        Matches predator.js's checkForPrey: iterates boids in order, takes
        the first within catchRadius; size grows after each catch.
        """
        # cooldown: only catch if (current_time - lastFeed) >= feedCooldown
        cur_ms = self.frame * FRAME_MS
        cooldown_done = (cur_ms - self.pred_last_feed_ms) >= PREDATOR_FEED_COOLDOWN_MS
        # distance predator-boid
        dx = self.boid_pos[..., 0] - self.pred_pos[:, None, 0]
        dy = self.boid_pos[..., 1] - self.pred_pos[:, None, 1]
        d = np.sqrt(dx * dx + dy * dy)
        catch_radius = (self.pred_size * 0.7)[:, None]   # (B, 1)
        in_catch = (d < catch_radius) & self.boid_alive
        # For each batch, find first eligible boid (by index order — matches
        # JS iteration). argmax of mask returns first True.
        catch_avail = in_catch & cooldown_done[:, None]
        any_catch = catch_avail.any(axis=1)
        # argmax returns 0 if all False; mask with any_catch.
        first_idx = catch_avail.argmax(axis=1)
        # Apply catches.
        rows = np.arange(self.B)
        new_alive = self.boid_alive.copy()
        new_alive[rows, first_idx] = np.where(any_catch, False, new_alive[rows, first_idx])
        self.boid_alive = new_alive
        # Feed
        self.pred_size = np.where(any_catch,
                                  np.minimum(self.pred_size + PREDATOR_GROWTH, PREDATOR_MAX_SIZE),
                                  self.pred_size)
        self.pred_last_feed_ms = np.where(any_catch, cur_ms, self.pred_last_feed_ms)
        self.catches += any_catch.astype(np.int32)

    def _decay_size(self):
        self.pred_size = np.where(
            self.pred_size > PREDATOR_BASE_SIZE,
            np.maximum(self.pred_size - PREDATOR_DECAY, PREDATOR_BASE_SIZE),
            self.pred_size)

    def step(self):
        self._step_boids()
        self._step_predator()
        self._check_catches()
        self._decay_size()
        self.frame += 1

    def run(self, max_frames):
        for _ in range(max_frames):
            self.step()
        return {
            'mean_catches': float(self.catches.mean()),
            'per_seed_catches': self.catches.tolist(),
            'seeds': self.seeds,
        }


if __name__ == '__main__':
    import sys
    import time
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'js/predator_weights.json'
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    seeds = list(range(100, 100 + n_seeds))
    weights = load_weights(weights_path)
    print(f"loaded weights featureDim={weights['featureDim']}")
    sim = Sim(seeds=seeds, weights=weights, auto_target='flock_centroid')
    t0 = time.time()
    out = sim.run(max_frames)
    elapsed = time.time() - t0
    print(f"seeds: {out['seeds']}")
    print(f"per_seed_catches: {out['per_seed_catches']}")
    print(f"mean_catches: {out['mean_catches']:.2f}")
    print(f"elapsed: {elapsed:.1f}s  ({n_seeds}×{max_frames} frames @ {max_frames * n_seeds / elapsed:.0f} frame/s)")
