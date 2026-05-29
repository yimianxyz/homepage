"""PyTorch port of the boid + predator simulation. Same approximation as
sim_np.py (parallel boid updates, not sequential). Runs on CPU or CUDA.

Usage:
    from sim_torch import Sim, load_weights
    weights = load_weights('js/predator_weights.json', device='cuda')
    sim = Sim(seeds=range(100, 228), weights=weights, device='cuda')
    summary = sim.run(max_frames=5000)
"""

import json
import time
import numpy as np
import torch
from pathlib import Path


# Constants (identical to sim_np.py)
N_BOIDS = 120
CANVAS_W = 1680
CANVAS_H = 1680
FRAME_MS = 12

MAX_SPEED = 6.0
MAX_FORCE = 0.1
DESIRED_SEPARATION = 40.0
NEIGHBOR_DISTANCE = 60.0
BORDER_OFFSET = 10.0
EPSILON = 1e-7
SEP_MULT = 2.0
COH_MULT = 1.0
ALI_MULT = 1.0

PREDATOR_RANGE = 80.0
PREDATOR_TURN_FACTOR = 0.3
PREDATOR_MAX_SPEED = 2.5
PREDATOR_MAX_FORCE = 0.05
PREDATOR_BASE_SIZE = 12.0
PREDATOR_MAX_SIZE = PREDATOR_BASE_SIZE * 1.8
PREDATOR_GROWTH = 1.2
PREDATOR_DECAY = 0.002
PREDATOR_FEED_COOLDOWN_MS = 100

POLICY_K = 4
POLICY_PAD = 2000.0
PREDICT_ALPHA = 8


def mulberry32_seq(seed: int, n: int) -> np.ndarray:
    """Generate n consecutive mulberry32 floats. NumPy (faster than torch
    for this small CPU-side init)."""
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
    ax = torch.abs(x)
    ay = torch.abs(y)
    return torch.maximum(ax, ay) * 0.96 + torch.minimum(ax, ay) * 0.398


def fast_set_magnitude(x, y, mag):
    m = fast_mag(x, y)
    safe = m > 0
    s = torch.where(safe, mag / torch.where(safe, m, torch.ones_like(m)), torch.zeros_like(m))
    return x * s, y * s


def fast_limit(x, y, max_mag):
    m = fast_mag(x, y)
    cap = m > max_mag
    s = torch.where(cap, max_mag / torch.where(m > 0, m, torch.ones_like(m)), torch.ones_like(m))
    return x * s, y * s


def load_weights(path: str | Path, device='cpu', dtype=torch.float32) -> dict:
    # JS predator_nn.js stores weights, inputMean/Std as Float32Array; we
    # match that to keep numerical drift between sim_torch and JS minimal.
    """Load weights JSON, return tensors on `device`."""
    with open(path) as f:
        j = json.load(f)
    layers = []
    for L in j['layers']:
        in_dim = L.get('inDim') or len(L['b']) and (len(L['W']) // len(L['b']))
        out_dim = L.get('outDim') or len(L['b'])
        W = torch.tensor(L['W'], dtype=dtype, device=device).view(in_dim, out_dim)
        b = torch.tensor(L['b'], dtype=dtype, device=device)
        layers.append({'W': W, 'b': b, 'activation': L['activation']})
    true_fd = len(j['inputMean'])
    return {
        'featureDim': true_fd,
        'inputMean': torch.tensor(j['inputMean'], dtype=dtype, device=device),
        'inputStd': torch.tensor(j['inputStd'], dtype=dtype, device=device),
        'outputScale': float(j['outputScale']),
        'clipMagnitude': float(j.get('clipMagnitude', 0.05)),
        'layers': layers,
    }


def nn_forward(features: torch.Tensor, weights: dict) -> torch.Tensor:
    """Single-policy forward.

    features: (B, fd_or_more). weights: result of load_weights() — W has
    shape (in, out), b has shape (out,). Returns (B, out_dim).
    """
    fd = weights['featureDim']
    x = (features[:, :fd] - weights['inputMean']) / weights['inputStd']
    for L in weights['layers']:
        x = x @ L['W'] + L['b']
        if L['activation'] == 'relu':
            x = torch.relu(x)
    x = x * weights['outputScale']
    cm = weights['clipMagnitude']
    if cm > 0:
        # Match JS predator_nn.js: alpha-max-beta-min approximation
        # mag ≈ max(|x|,|y|)*0.96 + min(|x|,|y|)*0.398. NOT exact sqrt.
        ax = torch.abs(x[:, 0])
        ay = torch.abs(x[:, 1])
        mag = torch.maximum(ax, ay) * 0.96 + torch.minimum(ax, ay) * 0.398
        s = torch.where(mag > cm, cm / torch.clamp(mag, min=1e-12), torch.ones_like(mag))
        x = x * s.unsqueeze(1)
    return x


def nn_forward_batched(features: torch.Tensor, weights: dict) -> torch.Tensor:
    """Multi-policy forward.

    weights is the result of `stack_weights(...)` and has layers with
    W of shape (K, in, out), b of shape (K, out); inputMean/inputStd
    have shape (K, fd). features: (B=K*S, fd_or_more), with each
    consecutive S elements belonging to one policy. Returns (B, out_dim).
    """
    K = weights['K']
    S = features.shape[0] // K
    fd = weights['featureDim']
    x = features[:, :fd].view(K, S, fd)
    x = (x - weights['inputMean'].unsqueeze(1)) / weights['inputStd'].unsqueeze(1)
    for L in weights['layers']:
        x = torch.bmm(x, L['W']) + L['b'].unsqueeze(1)
        if L['activation'] == 'relu':
            x = torch.relu(x)
    x = x * weights['outputScale']
    cm = weights['clipMagnitude']
    if cm > 0:
        ax = torch.abs(x[..., 0])
        ay = torch.abs(x[..., 1])
        mag = torch.maximum(ax, ay) * 0.96 + torch.minimum(ax, ay) * 0.398
        s = torch.where(mag > cm, cm / torch.clamp(mag, min=1e-12), torch.ones_like(mag))
        x = x * s.unsqueeze(-1)
    return x.reshape(K * S, -1)


def stack_weights(weights_list: list[dict]) -> dict:
    """Stack K loaded weight dicts into one batched dict for use with
    nn_forward_batched. Assumes identical architecture across policies.
    """
    K = len(weights_list)
    w0 = weights_list[0]
    layers = []
    for li in range(len(w0['layers'])):
        W = torch.stack([w['layers'][li]['W'] for w in weights_list], dim=0)
        b = torch.stack([w['layers'][li]['b'] for w in weights_list], dim=0)
        layers.append({'W': W, 'b': b, 'activation': w0['layers'][li]['activation']})
    return {
        'featureDim': w0['featureDim'],
        'inputMean': torch.stack([w['inputMean'] for w in weights_list]),
        'inputStd': torch.stack([w['inputStd'] for w in weights_list]),
        'outputScale': w0['outputScale'],
        'clipMagnitude': w0['clipMagnitude'],
        'layers': layers,
        'K': K,
    }


def build_features(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive,
                   auto_target, feature_dim, device, dtype=torch.float64):
    B = pred_pos.shape[0]
    out = torch.zeros((B, feature_dim), dtype=dtype, device=device)
    out[:, 0] = pred_vel[:, 0]
    out[:, 1] = pred_vel[:, 1]
    dxA = auto_target[:, 0] - pred_pos[:, 0]
    dyA = auto_target[:, 1] - pred_pos[:, 1]
    out[:, 2] = dxA
    out[:, 3] = dyA
    dA = torch.sqrt(dxA * dxA + dyA * dyA)
    safe = dA > 1e-9
    out[:, 4] = torch.where(safe, dxA / torch.where(safe, dA, torch.ones_like(dA)), torch.zeros_like(dA))
    out[:, 5] = torch.where(safe, dyA / torch.where(safe, dA, torch.ones_like(dA)), torch.zeros_like(dA))
    out[:, 6] = dA

    dx = boid_pos[:, :, 0] - pred_pos[:, None, 0]    # (B, N)
    dy = boid_pos[:, :, 1] - pred_pos[:, None, 1]
    d2 = dx * dx + dy * dy
    d2_masked = torch.where(boid_alive, d2, torch.full_like(d2, float('inf')))

    K = POLICY_K
    # Sort by d2 along axis 1
    sorted_d2, order = torch.sort(d2_masked, dim=1)
    near_idx = order[:, :K]
    rows = torch.arange(B, device=device).unsqueeze(1)
    near_dx = dx.gather(1, near_idx)
    near_dy = dy.gather(1, near_idx)
    near_d2 = d2_masked.gather(1, near_idx)
    near_alive = boid_alive.gather(1, near_idx)
    near_d = torch.sqrt(torch.where(near_alive, near_d2, torch.zeros_like(near_d2)))
    near_bvx = boid_vel[:, :, 0].gather(1, near_idx)
    near_bvy = boid_vel[:, :, 1].gather(1, near_idx)

    pad_x = torch.full_like(near_dx, POLICY_PAD)
    pad_y = torch.full_like(near_dy, POLICY_PAD)
    for k in range(K):
        is_real = near_alive[:, k]
        dx_k = torch.where(is_real, near_dx[:, k], pad_x[:, k])
        dy_k = torch.where(is_real, near_dy[:, k], pad_y[:, k])
        d_k = torch.where(is_real, near_d[:, k], pad_x[:, k])  # PAD value
        out[:, 7 + 2 * k] = dx_k
        out[:, 8 + 2 * k] = dy_k
        safe_k = is_real & (near_d[:, k] > 1e-9)
        out[:, 15 + 2 * k] = torch.where(
            safe_k,
            near_dx[:, k] / torch.where(safe_k, near_d[:, k], torch.ones_like(near_d[:, k])),
            torch.where(is_real, torch.zeros_like(is_real, dtype=dtype),
                        torch.ones_like(is_real, dtype=dtype)))
        out[:, 16 + 2 * k] = torch.where(
            safe_k,
            near_dy[:, k] / torch.where(safe_k, near_d[:, k], torch.ones_like(near_d[:, k])),
            torch.zeros((B,), dtype=dtype, device=device))
        out[:, 23 + k] = d_k

    dx1 = out[:, 7]; dy1 = out[:, 8]; d1 = out[:, 23]
    in_range_real = (d1 < PREDATOR_RANGE) & (dx1 != POLICY_PAD)

    dx0, dy0 = fast_set_magnitude(dx1, dy1, PREDATOR_MAX_SPEED)
    sx = dx0 - pred_vel[:, 0]
    sy = dy0 - pred_vel[:, 1]
    sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
    out[:, 29] = sx
    out[:, 30] = sy

    dx0, dy0 = fast_set_magnitude(dxA, dyA, PREDATOR_MAX_SPEED)
    sx = dx0 - pred_vel[:, 0]
    sy = dy0 - pred_vel[:, 1]
    sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
    out[:, 31] = sx
    out[:, 32] = sy

    t = (PREDATOR_RANGE - d1) / 10.0 + 0.5
    out[:, 33] = torch.clamp(t, 0.0, 1.0)
    out[:, 34] = in_range_real.float()

    if feature_dim >= 43:
        for k in range(K):
            is_real = near_alive[:, k]
            zero = torch.zeros((B,), dtype=dtype, device=device)
            out[:, 35 + 2 * k] = torch.where(is_real, near_bvx[:, k], zero)
            out[:, 36 + 2 * k] = torch.where(is_real, near_bvy[:, k], zero)

    if feature_dim >= 45:
        bvx1 = out[:, 35]; bvy1 = out[:, 36]
        tx = dx1 + PREDICT_ALPHA * (bvx1 - pred_vel[:, 0])
        ty = dy1 + PREDICT_ALPHA * (bvy1 - pred_vel[:, 1])
        dx0, dy0 = fast_set_magnitude(tx, ty, PREDATOR_MAX_SPEED)
        sx = dx0 - pred_vel[:, 0]
        sy = dy0 - pred_vel[:, 1]
        sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
        z = torch.zeros((B,), dtype=dtype, device=device)
        out[:, 43] = torch.where(in_range_real, sx, z)
        out[:, 44] = torch.where(in_range_real, sy, z)

    return out


class Sim:
    def __init__(self, seeds, weights, num_boids=N_BOIDS,
                 auto_target='flock_centroid', device='cpu', sequential=False,
                 auto_target_opts=None):
        self.seeds = list(seeds)
        self.B = len(self.seeds)
        self.N = num_boids
        self.weights = weights
        self.auto_target_mode = auto_target
        self.auto_target_opts = auto_target_opts or {}
        self.device = device
        self.sequential = sequential
        self._initialize()

    def _initialize(self):
        d = self.device
        dt = torch.float64
        self.boid_pos = torch.zeros((self.B, self.N, 2), dtype=dt, device=d)
        self.boid_vel = torch.zeros((self.B, self.N, 2), dtype=dt, device=d)
        self.boid_alive = torch.ones((self.B, self.N), dtype=torch.bool, device=d)
        self.pred_pos = torch.full((self.B, 2), 0.0, dtype=dt, device=d)
        self.pred_pos[:, 0] = CANVAS_W / 2
        self.pred_pos[:, 1] = CANVAS_H / 2
        self.pred_vel = torch.zeros((self.B, 2), dtype=dt, device=d)
        self.pred_size = torch.full((self.B,), PREDATOR_BASE_SIZE, dtype=dt, device=d)
        self.pred_auto = self.pred_pos.clone()
        self.pred_last_feed_ms = torch.zeros((self.B,), dtype=dt, device=d)
        self.catches = torch.zeros((self.B,), dtype=torch.int32, device=d)

        for bi, seed in enumerate(self.seeds):
            r = mulberry32_seq(seed, 2 + self.N + 2)
            start_x = float(int(r[0] * CANVAS_W))
            start_y = float(int(r[1] * CANVAS_H))
            self.boid_pos[bi, :, 0] = start_x
            self.boid_pos[bi, :, 1] = start_y
            angles = r[2:2 + self.N] * 2 * np.pi
            self.boid_vel[bi, :, 0] = torch.tensor(np.cos(angles), dtype=dt, device=d)
            self.boid_vel[bi, :, 1] = torch.tensor(np.sin(angles), dtype=dt, device=d)
            self.pred_vel[bi, 0] = r[2 + self.N] * 2 - 1
            self.pred_vel[bi, 1] = r[2 + self.N + 1] * 2 - 1

        self.frame = 0
        # GPU-resident scalars + per-batch row index. Pre-allocated so the
        # full step() is graph-safe (no fresh tensor allocations per frame).
        self._frame_ms = torch.zeros((), dtype=dt, device=d)
        self._row_idx = torch.arange(self.B, device=d)
        self._max_size_t = torch.tensor(PREDATOR_MAX_SIZE, dtype=dt, device=d)
        self._base_size_t = torch.tensor(PREDATOR_BASE_SIZE, dtype=dt, device=d)
        self._wrap_w_max = torch.tensor(CANVAS_W + 20.0, dtype=dt, device=d)
        self._wrap_h_max = torch.tensor(CANVAS_H + 20.0, dtype=dt, device=d)
        self._wrap_neg20 = torch.tensor(-20.0, dtype=dt, device=d)
        # Boid wrap (uses BORDER_OFFSET=10) — different scalars from predator wrap (20)
        self._wrap_b_w_max = torch.tensor(CANVAS_W + BORDER_OFFSET, dtype=dt, device=d)
        self._wrap_b_h_max = torch.tensor(CANVAS_H + BORDER_OFFSET, dtype=dt, device=d)
        self._wrap_neg_b = torch.tensor(-BORDER_OFFSET, dtype=dt, device=d)
        self._inf_t = torch.tensor(float('inf'), dtype=dt, device=d)

        if self.sequential:
            # Acceleration accumulator — matches JS Oracle which calls
            # sim.render() only (skipping the live-page's sim.tick()), so each
            # boid gets exactly ONE flock pass per frame inside boid.run().
            self.boid_accel = torch.zeros((self.B, self.N, 2), dtype=dt, device=d)

    def _compute_boid_acceleration(self):
        B, N = self.B, self.N
        d = self.device
        pos = self.boid_pos
        vel = self.boid_vel
        alive = self.boid_alive

        delta = pos[:, None, :, :] - pos[:, :, None, :]      # (B, N, N, 2)
        dist = torch.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2) + EPSILON

        eye = torch.eye(N, dtype=torch.bool, device=d).unsqueeze(0)   # (1, N, N)
        alive_j = alive[:, None, :] & ~eye

        # Cohesion
        coh_mask = alive_j & (dist <= NEIGHBOR_DISTANCE)
        coh_count = coh_mask.sum(dim=2)
        coh_pos_sum = (pos[:, None, :, :] * coh_mask.unsqueeze(-1)).sum(dim=2)
        coh_has = coh_count > 0
        ones_count = torch.ones_like(coh_count, dtype=torch.float64)
        coh_avg = torch.where(
            coh_has.unsqueeze(-1),
            coh_pos_sum / torch.where(coh_has.unsqueeze(-1), coh_count.unsqueeze(-1).double(), ones_count.unsqueeze(-1)),
            pos.double())
        seek_x = coh_avg[..., 0] - pos[..., 0]
        seek_y = coh_avg[..., 1] - pos[..., 1]
        dx0, dy0 = fast_set_magnitude(seek_x, seek_y, MAX_SPEED)
        cx = dx0 - vel[..., 0]
        cy = dy0 - vel[..., 1]
        cx, cy = fast_limit(cx, cy, MAX_FORCE)
        cx = torch.where(coh_has, cx, torch.zeros_like(cx))
        cy = torch.where(coh_has, cy, torch.zeros_like(cy))

        # Separation
        sep_mask = alive_j & (dist < DESIRED_SEPARATION) & (dist > 0)
        sep_count = sep_mask.sum(dim=2)
        delta_neg = -delta
        true_mag = torch.sqrt(delta_neg[..., 0] ** 2 + delta_neg[..., 1] ** 2) + EPSILON
        unit_x = delta_neg[..., 0] / true_mag
        unit_y = delta_neg[..., 1] / true_mag
        scaled_x = unit_x / dist
        scaled_y = unit_y / dist
        sep_x = (scaled_x * sep_mask).sum(dim=2)
        sep_y = (scaled_y * sep_mask).sum(dim=2)
        sep_has = sep_count > 0
        sep_x_avg = torch.where(sep_has, sep_x / torch.where(sep_has, sep_count.double(), torch.ones_like(sep_count, dtype=torch.float64)), torch.zeros_like(sep_x))
        sep_y_avg = torch.where(sep_has, sep_y / torch.where(sep_has, sep_count.double(), torch.ones_like(sep_count, dtype=torch.float64)), torch.zeros_like(sep_y))
        m = fast_mag(sep_x_avg, sep_y_avg)
        applied = m > 0
        sx, sy = fast_set_magnitude(sep_x_avg, sep_y_avg, MAX_SPEED)
        sx = sx - vel[..., 0]
        sy = sy - vel[..., 1]
        sx, sy = fast_limit(sx, sy, MAX_FORCE)
        sep_x_final = torch.where(applied, sx, torch.zeros_like(sx))
        sep_y_final = torch.where(applied, sy, torch.zeros_like(sy))

        # Alignment
        ali_mask = alive_j & (dist < NEIGHBOR_DISTANCE) & (dist > 0)
        ali_count = ali_mask.sum(dim=2)
        ali_vel_sum = (vel[:, None, :, :] * ali_mask.unsqueeze(-1)).sum(dim=2)
        ali_has = ali_count > 0
        ali_avg = torch.where(ali_has.unsqueeze(-1),
                              ali_vel_sum / torch.where(ali_has.unsqueeze(-1), ali_count.unsqueeze(-1).double(), torch.ones_like(ali_count.unsqueeze(-1), dtype=torch.float64)),
                              torch.zeros_like(ali_vel_sum))
        ax_, ay_ = fast_set_magnitude(ali_avg[..., 0], ali_avg[..., 1], MAX_SPEED)
        ax_ = ax_ - vel[..., 0]
        ay_ = ay_ - vel[..., 1]
        ax_, ay_ = fast_limit(ax_, ay_, MAX_FORCE)
        ax_ = torch.where(ali_has, ax_, torch.zeros_like(ax_))
        ay_ = torch.where(ali_has, ay_, torch.zeros_like(ay_))

        # Predator avoidance
        pdx = pos[..., 0] - self.pred_pos[:, None, 0]
        pdy = pos[..., 1] - self.pred_pos[:, None, 1]
        pdist = torch.sqrt(pdx * pdx + pdy * pdy) + EPSILON
        in_pr = pdist < PREDATOR_RANGE
        fm = fast_mag(pdx, pdy)
        fm_safe = torch.where(fm > 0, fm, torch.ones_like(fm))
        avx = pdx / fm_safe
        avy = pdy / fm_safe
        strength = (PREDATOR_RANGE - pdist) / PREDATOR_RANGE
        avx = avx * strength * PREDATOR_TURN_FACTOR
        avy = avy * strength * PREDATOR_TURN_FACTOR
        avx, avy = fast_limit(avx, avy, MAX_FORCE * 1.5)
        avx = torch.where(in_pr, avx, torch.zeros_like(avx))
        avy = torch.where(in_pr, avy, torch.zeros_like(avy))

        acc_x = (cx * COH_MULT) + (sep_x_final * SEP_MULT) + (ax_ * ALI_MULT) + avx
        acc_y = (cy * COH_MULT) + (sep_y_final * SEP_MULT) + (ay_ * ALI_MULT) + avy
        return acc_x, acc_y

    def _step_boids(self):
        ax, ay = self._compute_boid_acceleration()
        new_vx = self.boid_vel[..., 0] + ax
        new_vy = self.boid_vel[..., 1] + ay
        new_vx, new_vy = fast_limit(new_vx, new_vy, MAX_SPEED)
        self.boid_vel[..., 0] = new_vx
        self.boid_vel[..., 1] = new_vy
        self.boid_pos[..., 0] += new_vx
        self.boid_pos[..., 1] += new_vy
        # Wrap — use pre-allocated scalars (graph-capture-safe; creating tensors
        # inline here previously broke CUDA graph capture in parallel mode).
        neg_b = self._wrap_neg_b
        pos_mw = self._wrap_b_w_max
        pos_mh = self._wrap_b_h_max
        self.boid_pos[..., 0] = torch.where(self.boid_pos[..., 0] > pos_mw, neg_b, self.boid_pos[..., 0])
        self.boid_pos[..., 0] = torch.where(self.boid_pos[..., 0] < neg_b, pos_mw, self.boid_pos[..., 0])
        self.boid_pos[..., 1] = torch.where(self.boid_pos[..., 1] > pos_mh, neg_b, self.boid_pos[..., 1])
        self.boid_pos[..., 1] = torch.where(self.boid_pos[..., 1] < neg_b, pos_mh, self.boid_pos[..., 1])

    def _compute_single_boid_acceleration(self, i):
        """Compute (ax, ay) for boid i across all B batches, using CURRENT
        boid positions (which may include in-frame updates from boids 0..i-1
        when called from the sequential loop).

        Returns: ax (B,), ay (B,) — one acceleration delta from one flock pass
        for boid i. Mirrors getCohesionVector + getSeparationVector +
        getAlignmentVector + getPredatorAvoidanceVector from js/boid.js.
        """
        B, N = self.B, self.N
        d = self.device
        pos = self.boid_pos          # (B, N, 2)
        vel = self.boid_vel
        alive = self.boid_alive

        pos_ix = pos[:, i, 0]        # (B,)
        pos_iy = pos[:, i, 1]
        vel_ix = vel[:, i, 0]
        vel_iy = vel[:, i, 1]

        delta_x = pos[..., 0] - pos_ix.unsqueeze(1)   # (B, N) other - self
        delta_y = pos[..., 1] - pos_iy.unsqueeze(1)
        dist = torch.sqrt(delta_x ** 2 + delta_y ** 2) + EPSILON

        other_mask = torch.ones((B, N), dtype=torch.bool, device=d)
        other_mask[:, i] = False
        alive_j = alive & other_mask

        # Cohesion
        coh_mask = alive_j & (dist <= NEIGHBOR_DISTANCE)
        coh_count = coh_mask.sum(dim=1)
        coh_pos_sum_x = (pos[..., 0] * coh_mask).sum(dim=1)
        coh_pos_sum_y = (pos[..., 1] * coh_mask).sum(dim=1)
        coh_has = coh_count > 0
        ones_c = torch.ones_like(coh_count, dtype=torch.float64)
        denom = torch.where(coh_has, coh_count.double(), ones_c)
        coh_avg_x = torch.where(coh_has, coh_pos_sum_x / denom, pos_ix)
        coh_avg_y = torch.where(coh_has, coh_pos_sum_y / denom, pos_iy)
        seek_x = coh_avg_x - pos_ix
        seek_y = coh_avg_y - pos_iy
        dx0, dy0 = fast_set_magnitude(seek_x, seek_y, MAX_SPEED)
        cx = dx0 - vel_ix
        cy = dy0 - vel_iy
        cx, cy = fast_limit(cx, cy, MAX_FORCE)
        cx = torch.where(coh_has, cx, torch.zeros_like(cx))
        cy = torch.where(coh_has, cy, torch.zeros_like(cy))

        # Separation
        sep_mask = alive_j & (dist < DESIRED_SEPARATION) & (dist > 0)
        sep_count = sep_mask.sum(dim=1)
        delta_neg_x = -delta_x
        delta_neg_y = -delta_y
        true_mag = torch.sqrt(delta_neg_x ** 2 + delta_neg_y ** 2) + EPSILON
        unit_x = delta_neg_x / true_mag
        unit_y = delta_neg_y / true_mag
        scaled_x = unit_x / dist
        scaled_y = unit_y / dist
        sep_sx = (scaled_x * sep_mask).sum(dim=1)
        sep_sy = (scaled_y * sep_mask).sum(dim=1)
        sep_has = sep_count > 0
        ones_s = torch.ones_like(sep_count, dtype=torch.float64)
        denom_s = torch.where(sep_has, sep_count.double(), ones_s)
        sep_x_avg = torch.where(sep_has, sep_sx / denom_s, torch.zeros_like(sep_sx))
        sep_y_avg = torch.where(sep_has, sep_sy / denom_s, torch.zeros_like(sep_sy))
        m_sep = fast_mag(sep_x_avg, sep_y_avg)
        applied = m_sep > 0
        sx, sy = fast_set_magnitude(sep_x_avg, sep_y_avg, MAX_SPEED)
        sx = sx - vel_ix
        sy = sy - vel_iy
        sx, sy = fast_limit(sx, sy, MAX_FORCE)
        sep_x_final = torch.where(applied, sx, torch.zeros_like(sx))
        sep_y_final = torch.where(applied, sy, torch.zeros_like(sy))

        # Alignment
        ali_mask = alive_j & (dist < NEIGHBOR_DISTANCE) & (dist > 0)
        ali_count = ali_mask.sum(dim=1)
        ali_sum_x = (vel[..., 0] * ali_mask).sum(dim=1)
        ali_sum_y = (vel[..., 1] * ali_mask).sum(dim=1)
        ali_has = ali_count > 0
        ones_a = torch.ones_like(ali_count, dtype=torch.float64)
        denom_a = torch.where(ali_has, ali_count.double(), ones_a)
        ali_avg_x = torch.where(ali_has, ali_sum_x / denom_a, torch.zeros_like(ali_sum_x))
        ali_avg_y = torch.where(ali_has, ali_sum_y / denom_a, torch.zeros_like(ali_sum_y))
        ax_, ay_ = fast_set_magnitude(ali_avg_x, ali_avg_y, MAX_SPEED)
        ax_ = ax_ - vel_ix
        ay_ = ay_ - vel_iy
        ax_, ay_ = fast_limit(ax_, ay_, MAX_FORCE)
        ax_ = torch.where(ali_has, ax_, torch.zeros_like(ax_))
        ay_ = torch.where(ali_has, ay_, torch.zeros_like(ay_))

        # Predator avoidance
        pdx = pos_ix - self.pred_pos[:, 0]
        pdy = pos_iy - self.pred_pos[:, 1]
        pdist = torch.sqrt(pdx * pdx + pdy * pdy) + EPSILON
        in_pr = pdist < PREDATOR_RANGE
        fm = fast_mag(pdx, pdy)
        fm_safe = torch.where(fm > 0, fm, torch.ones_like(fm))
        avx = pdx / fm_safe
        avy = pdy / fm_safe
        strength = (PREDATOR_RANGE - pdist) / PREDATOR_RANGE
        avx = avx * strength * PREDATOR_TURN_FACTOR
        avy = avy * strength * PREDATOR_TURN_FACTOR
        avx, avy = fast_limit(avx, avy, MAX_FORCE * 1.5)
        avx = torch.where(in_pr, avx, torch.zeros_like(avx))
        avy = torch.where(in_pr, avy, torch.zeros_like(avy))

        acc_x = (cx * COH_MULT) + (sep_x_final * SEP_MULT) + (ax_ * ALI_MULT) + avx
        acc_y = (cy * COH_MULT) + (sep_y_final * SEP_MULT) + (ay_ * ALI_MULT) + avy
        return acc_x, acc_y

    def _step_boids_sequential(self):
        """One-pass sequential boid update, matching dev/oracle.js's render-only
        step (Oracle skips sim.tick(); fullTick=false is the default — see
        oracle.js step() line 348-352).

        Per-frame, per-boid in order:
          (a) flock pass — compute forces from CURRENT (pre-update) positions
              that include in-frame updates from boids 0..i-1, ADD to accel.
          (b) velocity += accel; fastLimit. position += velocity; wrap.
              accel = 0.

        IMPORTANT: this is single-flock-pass per frame to match Oracle. The
        live page (js/simulation.js:run) does TWO flock passes (tick + render),
        but all our eval and training data is generated via Oracle, so we
        match that — otherwise the boid dynamics diverge after 1 frame.
        """
        # Pre-allocated scalars (see _initialize) — using them directly so
        # the step is graph-safe.
        neg_b = self._wrap_neg_b
        pos_mw = self._wrap_b_w_max
        pos_mh = self._wrap_b_h_max

        # Sequential per-boid flock + update pass
        for i in range(self.N):
            ax_i, ay_i = self._compute_single_boid_acceleration(i)
            self.boid_accel[:, i, 0] += ax_i
            self.boid_accel[:, i, 1] += ay_i

            new_vx = self.boid_vel[:, i, 0] + self.boid_accel[:, i, 0]
            new_vy = self.boid_vel[:, i, 1] + self.boid_accel[:, i, 1]
            new_vx, new_vy = fast_limit(new_vx, new_vy, MAX_SPEED)
            self.boid_vel[:, i, 0] = new_vx
            self.boid_vel[:, i, 1] = new_vy

            new_px = self.boid_pos[:, i, 0] + new_vx
            new_py = self.boid_pos[:, i, 1] + new_vy
            new_px = torch.where(new_px > pos_mw, neg_b, new_px)
            new_px = torch.where(new_px < neg_b, pos_mw, new_px)
            new_py = torch.where(new_py > pos_mh, neg_b, new_py)
            new_py = torch.where(new_py < neg_b, pos_mh, new_py)
            self.boid_pos[:, i, 0] = new_px
            self.boid_pos[:, i, 1] = new_py

            self.boid_accel[:, i, 0] = 0
            self.boid_accel[:, i, 1] = 0

    def _update_auto_target(self):
        if self.auto_target_mode == 'random':
            return

        dx = self.boid_pos[..., 0] - self.pred_pos[:, None, 0]
        dy = self.boid_pos[..., 1] - self.pred_pos[:, None, 1]
        d = torch.sqrt(dx * dx + dy * dy)
        d_masked = torch.where(self.boid_alive, d, self._inf_t)
        any_in_range = (d_masked < PREDATOR_RANGE).any(dim=1)
        any_alive = self.boid_alive.any(dim=1)

        alive_f = self.boid_alive.double()
        n_alive = alive_f.sum(dim=1)
        n_safe = torch.where(n_alive > 0, n_alive, torch.ones_like(n_alive))

        mode = self.auto_target_mode
        if mode == 'flock_centroid':
            cx = (self.boid_pos[..., 0] * alive_f).sum(dim=1) / n_safe
            cy = (self.boid_pos[..., 1] * alive_f).sum(dim=1) / n_safe
        elif mode == 'weighted_centroid':
            # JS: w = 1/sqrt(d^2 + 1); centroid = sum(w*pos)/sum(w)
            w = 1.0 / torch.sqrt(dx * dx + dy * dy + 1.0)
            w = w * alive_f
            wsum = w.sum(dim=1)
            wsafe = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
            cx = (self.boid_pos[..., 0] * w).sum(dim=1) / wsafe
            cy = (self.boid_pos[..., 1] * w).sum(dim=1) / wsafe
        elif mode == 'predicted_centroid':
            lookahead = float(self.auto_target_opts.get('lookahead', 30))
            cx0 = (self.boid_pos[..., 0] * alive_f).sum(dim=1) / n_safe
            cy0 = (self.boid_pos[..., 1] * alive_f).sum(dim=1) / n_safe
            vx0 = (self.boid_vel[..., 0] * alive_f).sum(dim=1) / n_safe
            vy0 = (self.boid_vel[..., 1] * alive_f).sum(dim=1) / n_safe
            cx = cx0 + lookahead * vx0
            cy = cy0 + lookahead * vy0
        elif mode == 'weighted_predicted':
            # Density-weighted centroid + lookahead × density-weighted mean velocity.
            # weight_pow controls the distance falloff: w = 1/(d^2+1)^(weight_pow/2).
            # weight_pow=1 → 1/sqrt(d^2+1) (default, the +1.77 winner);
            # weight_pow=2 → 1/(d^2+1); weight_pow=0.5 → gentler falloff.
            lookahead = float(self.auto_target_opts.get('lookahead', 5))
            weight_pow = float(self.auto_target_opts.get('weight_pow', 1.0))
            w = (dx * dx + dy * dy + 1.0) ** (-weight_pow / 2.0)
            w = w * alive_f
            wsum = w.sum(dim=1)
            wsafe = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
            cx0 = (self.boid_pos[..., 0] * w).sum(dim=1) / wsafe
            cy0 = (self.boid_pos[..., 1] * w).sum(dim=1) / wsafe
            vx0 = (self.boid_vel[..., 0] * w).sum(dim=1) / wsafe
            vy0 = (self.boid_vel[..., 1] * w).sum(dim=1) / wsafe
            cx = cx0 + lookahead * vx0
            cy = cy0 + lookahead * vy0
        elif mode == 'weighted_adaptive':
            # Like weighted_predicted but the lookahead is ADAPTIVE: instead of
            # a fixed 5 frames, lead = (dist from predator to the weighted
            # centroid / predator_max_speed) * lead_scale, capped at lead_max.
            # Far clusters get more lead (predator needs more frames to reach
            # them); near clusters get less. Principled because the predator is
            # slower than the boids, so the right lead depends on travel time.
            lead_scale = float(self.auto_target_opts.get('lead_scale', 1.0))
            lead_max = float(self.auto_target_opts.get('lead_max', 40.0))
            weight_pow = float(self.auto_target_opts.get('weight_pow', 1.0))
            w = (dx * dx + dy * dy + 1.0) ** (-weight_pow / 2.0)
            w = w * alive_f
            wsum = w.sum(dim=1)
            wsafe = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
            cx0 = (self.boid_pos[..., 0] * w).sum(dim=1) / wsafe
            cy0 = (self.boid_pos[..., 1] * w).sum(dim=1) / wsafe
            vx0 = (self.boid_vel[..., 0] * w).sum(dim=1) / wsafe
            vy0 = (self.boid_vel[..., 1] * w).sum(dim=1) / wsafe
            ddx = cx0 - self.pred_pos[:, 0]
            ddy = cy0 - self.pred_pos[:, 1]
            dcent = torch.sqrt(ddx * ddx + ddy * ddy)
            lead = torch.clamp(dcent / PREDATOR_MAX_SPEED * lead_scale, 0.0, lead_max)
            cx = cx0 + lead * vx0
            cy = cy0 + lead * vy0
        elif mode == 'weighted_intercept':
            # Solve the intercept quadratic: aim at C + t*Vc where t is the
            # time for the predator (moving at PREDATOR_MAX_SPEED toward the
            # aim point) to reach it. C = weighted centroid, Vc = weighted mean
            # boid velocity. If no positive real root (cluster recedes faster
            # than predator can chase), fall back to adaptive lead.
            lead_max = float(self.auto_target_opts.get('lead_max', 60.0))
            w = (dx * dx + dy * dy + 1.0) ** (-0.5)
            w = w * alive_f
            wsum = w.sum(dim=1)
            wsafe = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
            cx0 = (self.boid_pos[..., 0] * w).sum(dim=1) / wsafe
            cy0 = (self.boid_pos[..., 1] * w).sum(dim=1) / wsafe
            vx0 = (self.boid_vel[..., 0] * w).sum(dim=1) / wsafe
            vy0 = (self.boid_vel[..., 1] * w).sum(dim=1) / wsafe
            Dx = cx0 - self.pred_pos[:, 0]
            Dy = cy0 - self.pred_pos[:, 1]
            Vp = PREDATOR_MAX_SPEED
            a = vx0 * vx0 + vy0 * vy0 - Vp * Vp
            b = 2.0 * (Dx * vx0 + Dy * vy0)
            c = Dx * Dx + Dy * Dy
            disc = b * b - 4.0 * a * c
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            a_safe = torch.where(a.abs() > 1e-9, a, torch.full_like(a, 1e-9))
            # two roots; want smallest positive
            t1 = (-b - sqrt_disc) / (2.0 * a_safe)
            t2 = (-b + sqrt_disc) / (2.0 * a_safe)
            big = torch.full_like(t1, 1e9)
            t1p = torch.where(t1 > 0, t1, big)
            t2p = torch.where(t2 > 0, t2, big)
            t = torch.minimum(t1p, t2p)
            # adaptive-lead fallback when no valid intercept (disc<0 or t huge)
            dcent = torch.sqrt(Dx * Dx + Dy * Dy)
            fallback = dcent / Vp
            valid = (disc >= 0) & (t < 1e8)
            t = torch.where(valid, t, fallback)
            t = torch.clamp(t, 0.0, lead_max)
            cx = cx0 + t * vx0
            cy = cy0 + t * vy0
        elif mode == 'nearest_cluster':
            # Densest-cluster centroid + adaptive lead. Find the live boid with
            # the most live neighbors within cluster_r, target the centroid (and
            # mean velocity) of that boid's neighborhood, then lead adaptively by
            # travel time. Addresses the failure where a density-weighted
            # centroid lands in the empty gap between two separate clusters.
            cluster_r = float(self.auto_target_opts.get('cluster_r', 60.0))
            lead_scale = float(self.auto_target_opts.get('lead_scale', 0.6))
            lead_max = float(self.auto_target_opts.get('lead_max', 40.0))
            bx = self.boid_pos[..., 0]
            by = self.boid_pos[..., 1]
            ddx_ij = bx.unsqueeze(2) - bx.unsqueeze(1)   # (B,N,N): i - j
            ddy_ij = by.unsqueeze(2) - by.unsqueeze(1)
            dist_ij = torch.sqrt(ddx_ij * ddx_ij + ddy_ij * ddy_ij)
            pair_ok = (dist_ij < cluster_r) & self.boid_alive.unsqueeze(1) & self.boid_alive.unsqueeze(2)
            ncount = pair_ok.double().sum(dim=2)         # (B,N) neighbors of i
            ncount = torch.where(self.boid_alive, ncount, torch.full_like(ncount, -1.0))
            dense_idx = ncount.argmax(dim=1)             # (B,)
            # neighborhood mask of the densest boid, graph-safe via gather
            sel = dense_idx.view(self.B, 1, 1).expand(self.B, 1, self.N)
            mask = torch.gather(pair_ok, 1, sel).squeeze(1).double()   # (B,N)
            # optional density weighting within the cluster: weight each member
            # by its own neighbor count ^ centroid_pow (pulls the target toward
            # the densest sub-region). centroid_pow=0 → uniform (default).
            centroid_pow = float(self.auto_target_opts.get('centroid_pow', 0.0))
            if centroid_pow != 0.0:
                nc_pos = torch.clamp(ncount, min=0.0)
                mask = mask * (nc_pos ** centroid_pow)
            msum = mask.sum(dim=1)
            msafe = torch.where(msum > 0, msum, torch.ones_like(msum))
            cx0 = (bx * mask).sum(dim=1) / msafe
            cy0 = (by * mask).sum(dim=1) / msafe
            vx0 = (self.boid_vel[..., 0] * mask).sum(dim=1) / msafe
            vy0 = (self.boid_vel[..., 1] * mask).sum(dim=1) / msafe
            ddx2 = cx0 - self.pred_pos[:, 0]
            ddy2 = cy0 - self.pred_pos[:, 1]
            dcent = torch.sqrt(ddx2 * ddx2 + ddy2 * ddy2)
            lead = torch.clamp(dcent / PREDATOR_MAX_SPEED * lead_scale, 0.0, lead_max)
            cx = cx0 + lead * vx0
            cy = cy0 + lead * vy0
        elif mode == 'nearest_K_centroid':
            # Centroid of the K nearest live boids.
            K = int(self.auto_target_opts.get('K', 8))
            K = min(K, self.N)
            d_masked2 = torch.where(self.boid_alive, d * d, self._inf_t)
            _, idx = torch.topk(d_masked2, K, dim=1, largest=False)
            sel_x = torch.gather(self.boid_pos[..., 0], 1, idx)
            sel_y = torch.gather(self.boid_pos[..., 1], 1, idx)
            sel_alive = torch.gather(alive_f, 1, idx)
            k_sum = sel_alive.sum(dim=1)
            k_safe = torch.where(k_sum > 0, k_sum, torch.ones_like(k_sum))
            cx = (sel_x * sel_alive).sum(dim=1) / k_safe
            cy = (sel_y * sel_alive).sum(dim=1) / k_safe
        elif mode == 'evolved':
            # Unified, evolvable patrol target. Generalizes the whole lineage:
            # weight each live boid by (local_density^dens_pow) * exp(-dist_pred
            # / reach_scale), normalize by the per-env max and raise to `sharp`
            # (sharp->inf picks the single densest-reachable boid like
            # nearest_cluster; sharp->0 gives the broad/global centroid), then
            # take the weighted centroid + mean velocity and lead by travel time.
            # reach_scale is the NEW axis: the predator is slow, so a closer but
            # slightly-less-dense cluster can beat a distant denser one.
            o = self.auto_target_opts
            Rc = float(o.get('cluster_r', 150.0))
            dens_pow = float(o.get('dens_pow', 1.0))
            reach_scale = float(o.get('reach_scale', 1e9))
            sharp = float(o.get('sharp', 6.0))
            lead_scale = float(o.get('lead_scale', 0.4))
            lead_max = float(o.get('lead_max', 120.0))
            bx = self.boid_pos[..., 0]
            by = self.boid_pos[..., 1]
            ddx_ij = bx.unsqueeze(2) - bx.unsqueeze(1)
            ddy_ij = by.unsqueeze(2) - by.unsqueeze(1)
            dist_ij = torch.sqrt(ddx_ij * ddx_ij + ddy_ij * ddy_ij)
            pair_ok = (dist_ij < Rc) & self.boid_alive.unsqueeze(1) & self.boid_alive.unsqueeze(2)
            ncount = pair_ok.double().sum(dim=2)            # (B,N) neighbors incl self
            attract = (ncount + 1.0).pow(dens_pow) * torch.exp(-d / reach_scale)
            attract = torch.where(self.boid_alive, attract, torch.zeros_like(attract))
            amax = attract.max(dim=1, keepdim=True).values.clamp_min(1e-12)
            w = (attract / amax).pow(sharp)
            w = torch.where(self.boid_alive, w, torch.zeros_like(w))
            wsum = w.sum(dim=1).clamp_min(1e-12)
            cx0 = (bx * w).sum(dim=1) / wsum
            cy0 = (by * w).sum(dim=1) / wsum
            vx0 = (self.boid_vel[..., 0] * w).sum(dim=1) / wsum
            vy0 = (self.boid_vel[..., 1] * w).sum(dim=1) / wsum
            ddx2 = cx0 - self.pred_pos[:, 0]
            ddy2 = cy0 - self.pred_pos[:, 1]
            dcent = torch.sqrt(ddx2 * ddx2 + ddy2 * ddy2)
            lead = torch.clamp(dcent / PREDATOR_MAX_SPEED * lead_scale, 0.0, lead_max)
            cx = cx0 + lead * vx0
            cy = cy0 + lead * vy0
        else:
            raise NotImplementedError(f"autoTargetMode={mode}")

        cond = (~any_in_range) & any_alive
        self.pred_auto[:, 0] = torch.where(cond, cx, self.pred_auto[:, 0])
        self.pred_auto[:, 1] = torch.where(cond, cy, self.pred_auto[:, 1])

    def _step_predator(self):
        self._update_auto_target()
        # Build features as float32 because the NN weights are float32 (JS
        # uses Float32Array for the NN — see js/predator_nn.js). Boid state
        # itself stays float64 to match JS Number semantics elsewhere.
        feats = build_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self.weights['featureDim'], self.device,
            dtype=torch.float32,
        )
        if 'K' in self.weights:
            steering = nn_forward_batched(feats, self.weights).double()
        else:
            steering = nn_forward(feats, self.weights).double()
        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = new_vx
        self.pred_vel[:, 1] = new_vy
        self.pred_pos[:, 0] += new_vx
        self.pred_pos[:, 1] += new_vy
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > self._wrap_w_max, self._wrap_neg20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < self._wrap_neg20, self._wrap_w_max, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > self._wrap_h_max, self._wrap_neg20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < self._wrap_neg20, self._wrap_h_max, self.pred_pos[:, 1])

    def _check_catches(self):
        # cur_ms lives on GPU as a tensor scalar so the whole step() can be
        # captured in a CUDA graph (graphs can't read Python state during
        # replay, only GPU tensors).
        cur_ms = self._frame_ms
        cooldown_done = (cur_ms - self.pred_last_feed_ms) >= PREDATOR_FEED_COOLDOWN_MS
        dx = self.boid_pos[..., 0] - self.pred_pos[:, None, 0]
        dy = self.boid_pos[..., 1] - self.pred_pos[:, None, 1]
        d = torch.sqrt(dx * dx + dy * dy)
        catch_radius = (self.pred_size * 0.7).unsqueeze(1)
        in_catch = (d < catch_radius) & self.boid_alive
        catch_avail = in_catch & cooldown_done.unsqueeze(1)
        any_catch = catch_avail.any(dim=1)
        first_idx = catch_avail.int().argmax(dim=1)
        rows = self._row_idx
        # In-place: clear the bit at (row, first_idx) where any_catch.
        # No clone(), no attribute reassignment — graph-safe.
        cur = self.boid_alive[rows, first_idx]
        self.boid_alive[rows, first_idx] = cur & ~any_catch
        self.pred_size.copy_(torch.where(
            any_catch,
            torch.minimum(self.pred_size + PREDATOR_GROWTH, self._max_size_t),
            self.pred_size))
        self.pred_last_feed_ms.copy_(torch.where(any_catch, cur_ms.expand_as(self.pred_last_feed_ms), self.pred_last_feed_ms))
        self.catches += any_catch.int()

    def _decay_size(self):
        self.pred_size.copy_(torch.where(
            self.pred_size > PREDATOR_BASE_SIZE,
            torch.maximum(self.pred_size - PREDATOR_DECAY, self._base_size_t),
            self.pred_size))

    def step(self):
        if self.sequential:
            self._step_boids_sequential()
        else:
            self._step_boids()
        self._step_predator()
        self._check_catches()
        self._decay_size()
        self.frame += 1
        self._frame_ms += FRAME_MS

    def run(self, max_frames):
        for _ in range(max_frames):
            self.step()
        return {
            'mean_catches': float(self.catches.float().mean().item()),
            'per_seed_catches': self.catches.cpu().tolist(),
            'seeds': self.seeds,
        }

    def run_graph(self, max_frames, warmup=3):
        """Capture step() as a CUDA graph and replay it `max_frames` times.

        Cuts the ~0.47s per-frame Python+kernel-launch overhead down to a
        single graph launch per frame. Requires that all state mutations
        in step() be in-place (no `self.X = new_tensor`) and that no
        tensor allocations depend on CPU-side state. Both are true after
        the refactor in this file.

        Falls back to ordinary `run()` if CUDA is unavailable.
        """
        if not (isinstance(self.device, str) and self.device.startswith('cuda')) and self.device != 'cuda':
            return self.run(max_frames)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.step()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.step()

        remaining = max_frames - warmup - 1
        for _ in range(remaining):
            g.replay()
        torch.cuda.synchronize()
        return {
            'mean_catches': float(self.catches.float().mean().item()),
            'per_seed_catches': self.catches.cpu().tolist(),
            'seeds': self.seeds,
        }


if __name__ == '__main__':
    import sys
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'js/predator_weights.json'
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    sequential = '--sequential' in sys.argv
    seeds = list(range(100, 100 + n_seeds))
    weights = load_weights(weights_path, device=device)
    print(f"loaded weights featureDim={weights['featureDim']} on {device}  sequential={sequential}")
    sim = Sim(seeds=seeds, weights=weights, device=device, sequential=sequential)
    t0 = time.time()
    out = sim.run(max_frames)
    elapsed = time.time() - t0
    print(f"per_seed_catches: {out['per_seed_catches']}")
    print(f"mean_catches: {out['mean_catches']:.2f}")
    print(f"elapsed: {elapsed:.1f}s  ({n_seeds}×{max_frames} = {n_seeds*max_frames} frames @ {n_seeds*max_frames/elapsed:.0f}/s)")
