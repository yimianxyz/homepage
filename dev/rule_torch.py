"""Python ports of dev/policy_spec.js rule policies, vectorized over B
batch elements and runnable inside sim_torch. Keeps the same per-frame
semantics as the JS rule (single nearest target, k=4 nearest, alpha-
lookahead, smart target selection by closing speed, etc.) so that
sim_torch evals of any rule policy are directly comparable to JS evals
of the same rule.

The features tensor shape is (B, 45):
  [0..1]   pred vel
  [2..3]   auto_target offset
  [4..5]   unit dir to auto_target
  [6]      dA
  [7..14]  K=4 nearest boid relative offsets (dx, dy)
  [15..22] unit dir to nearest K
  [23..26] distances d_k
  [27..28] legacy padding (0)
  [29..30] precomputed seek_boid_xy
  [31..32] precomputed seek_auto_xy
  [33]     inRange smooth
  [34]     inRange binary
  [35..42] K nearest boid velocities (raw)
  [43..44] seek_v2 precomputed
"""
import math
import torch

POLICY_R = 80.0
POLICY_K = 4
POLICY_PAD = 2000.0
PREDATOR_MAX_SPEED = 2.5
PREDATOR_MAX_FORCE = 0.05
PREDATOR_RANGE = 80.0
PREDATOR_TURN_FACTOR = 0.3
BOID_MAX_FORCE_AVOID = 0.15
BOID_MAX_SPEED = 6.0


def fast_mag(x, y):
    ax = torch.abs(x); ay = torch.abs(y)
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


def _seek_steering(tx, ty, vx, vy):
    """Compute fastLimit(fastSetMag(t, MAX_SPEED) - v, MAX_FORCE)."""
    dx0, dy0 = fast_set_magnitude(tx, ty, PREDATOR_MAX_SPEED)
    sx = dx0 - vx
    sy = dy0 - vy
    sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
    return sx, sy


def rule_v1_torch(features):
    """Plain "head to nearest within range" rule. features: (B, ≥35)."""
    vx = features[:, 0]; vy = features[:, 1]
    dxA = features[:, 2]; dyA = features[:, 3]
    dx1 = features[:, 7]; dy1 = features[:, 8]
    d1 = features[:, 23]
    in_range = (d1 < POLICY_R) & (dx1 != POLICY_PAD)
    tx = torch.where(in_range, dx1, dxA)
    ty = torch.where(in_range, dy1, dyA)
    return torch.stack(_seek_steering(tx, ty, vx, vy), dim=-1)


def rule_v2_torch(features, alpha):
    """Hunt branch aims α frames ahead. features: (B, ≥43)."""
    vx = features[:, 0]; vy = features[:, 1]
    dxA = features[:, 2]; dyA = features[:, 3]
    dx1 = features[:, 7]; dy1 = features[:, 8]
    d1 = features[:, 23]
    bvx1 = features[:, 35]; bvy1 = features[:, 36]
    in_range = (d1 < POLICY_R) & (dx1 != POLICY_PAD)
    tx_hunt = dx1 + alpha * (bvx1 - vx)
    ty_hunt = dy1 + alpha * (bvy1 - vy)
    tx = torch.where(in_range, tx_hunt, dxA)
    ty = torch.where(in_range, ty_hunt, dyA)
    return torch.stack(_seek_steering(tx, ty, vx, vy), dim=-1)


def rule_v3_torch(features, mode='score_minus_dist', distW=0.05, alpha=0.0,
                   buffers=None):
    """Smart target selection. features: (B, ≥43).
    buffers (optional): dict with pre-allocated 'best_score', 'best_tx',
    'best_ty', 'any_valid', 'neg_inf'. Avoids allocations inside CUDA
    graph capture.
    """
    B = features.shape[0]
    device = features.device
    vx = features[:, 0]; vy = features[:, 1]
    dxA = features[:, 2]; dyA = features[:, 3]
    EPS = 0.05

    if buffers is None:
        best_score = torch.full((B,), -float('inf'), dtype=features.dtype, device=device)
        best_tx = torch.zeros(B, dtype=features.dtype, device=device)
        best_ty = torch.zeros(B, dtype=features.dtype, device=device)
        any_valid = torch.zeros(B, dtype=torch.bool, device=device)
    else:
        best_score = buffers['best_score'].fill_(-float('inf'))
        best_tx = buffers['best_tx'].fill_(0.0)
        best_ty = buffers['best_ty'].fill_(0.0)
        any_valid = buffers['any_valid'].fill_(False)

    for k in range(POLICY_K):
        dxk = features[:, 7 + 2*k]
        dyk = features[:, 8 + 2*k]
        dk = features[:, 23 + k]
        bvxk = features[:, 35 + 2*k]
        bvyk = features[:, 36 + 2*k]

        real = (dxk != POLICY_PAD) & (dk < POLICY_R)

        if alpha != 0:
            tx_k = dxk + alpha * (bvxk - vx)
            ty_k = dyk + alpha * (bvyk - vy)
            d_k_eff = torch.sqrt(tx_k * tx_k + ty_k * ty_k).clamp_min(1e-9)
        else:
            tx_k = dxk
            ty_k = dyk
            d_k_eff = dk.clamp_min(1e-9)

        closing = ((vx - bvxk) * tx_k + (vy - bvyk) * ty_k) / d_k_eff

        if mode == 'closing_only':
            score = closing
        elif mode == 'time_to_catch':
            score = -d_k_eff / torch.clamp_min(closing, EPS)
        else:  # score_minus_dist
            score = closing - distW * d_k_eff

        better = real & (score > best_score)
        best_score = torch.where(better, score, best_score)
        best_tx = torch.where(better, tx_k, best_tx)
        best_ty = torch.where(better, ty_k, best_ty)
        any_valid = any_valid | better | (any_valid & real)

    tx = torch.where(any_valid, best_tx, dxA)
    ty = torch.where(any_valid, best_ty, dyA)
    return torch.stack(_seek_steering(tx, ty, vx, vy), dim=-1)


def rule_v4_torch(features, distW=0.0, buffers=None):
    """Perfect-intercept. features: (B, ≥43)."""
    B = features.shape[0]
    device = features.device
    vx = features[:, 0]; vy = features[:, 1]
    dxA = features[:, 2]; dyA = features[:, 3]
    sp = PREDATOR_MAX_SPEED
    sp2 = sp * sp

    if buffers is None:
        best_t = torch.full((B,), float('inf'), dtype=features.dtype, device=device)
        best_tx = torch.zeros(B, dtype=features.dtype, device=device)
        best_ty = torch.zeros(B, dtype=features.dtype, device=device)
        any_valid = torch.zeros(B, dtype=torch.bool, device=device)
    else:
        best_t = buffers['best_t'].fill_(float('inf'))
        best_tx = buffers['best_tx'].fill_(0.0)
        best_ty = buffers['best_ty'].fill_(0.0)
        any_valid = buffers['any_valid'].fill_(False)

    for k in range(POLICY_K):
        dxk = features[:, 7 + 2*k]
        dyk = features[:, 8 + 2*k]
        dk = features[:, 23 + k]
        bvxk = features[:, 35 + 2*k]
        bvyk = features[:, 36 + 2*k]

        real = (dxk != POLICY_PAD) & (dk < POLICY_R)
        v2 = bvxk * bvxk + bvyk * bvyk
        a = v2 - sp2
        b = 2 * (dxk * bvxk + dyk * bvyk)
        c = dxk * dxk + dyk * dyk

        # General quadratic. For a≈0, use linear. We approximate by adding
        # a tiny epsilon to keep things vectorized.
        disc = b * b - 4 * a * c
        valid_quad = disc >= 0
        sqd = torch.sqrt(torch.clamp_min(disc, 0))
        denom = 2 * a
        # Choose smallest positive root
        t1 = (-b - sqd) / torch.where(denom != 0, denom, torch.ones_like(denom))
        t2 = (-b + sqd) / torch.where(denom != 0, denom, torch.ones_like(denom))
        # Pick min positive
        t1p = torch.where(t1 > 0, t1, torch.full_like(t1, float('inf')))
        t2p = torch.where(t2 > 0, t2, torch.full_like(t2, float('inf')))
        t = torch.minimum(t1p, t2p)
        # Handle the degenerate a≈0 case via linear: t = -c / b
        is_lin = torch.abs(a) < 1e-9
        t_lin = -c / torch.where(b != 0, b, torch.ones_like(b))
        t_lin_p = torch.where(t_lin > 0, t_lin, torch.full_like(t_lin, float('inf')))
        t = torch.where(is_lin, t_lin_p, t)
        # Mask non-real (out-of-range or imaginary roots)
        t_ok = (~torch.isinf(t)) & valid_quad & real
        t_adj = t + distW * dk
        better = t_ok & (t_adj < best_t)
        best_t = torch.where(better, t_adj, best_t)
        best_tx = torch.where(better, dxk + bvxk * t, best_tx)
        best_ty = torch.where(better, dyk + bvyk * t, best_ty)
        any_valid = any_valid | better

    tx = torch.where(any_valid, best_tx, dxA)
    ty = torch.where(any_valid, best_ty, dyA)
    return torch.stack(_seek_steering(tx, ty, vx, vy), dim=-1)


def rule_v5_torch(features, steps=5, distW=0.0, buffers=None):
    """Multi-step prediction with boid-avoidance acceleration.
    features: (B, ≥43). Returns (B, 2) steering.
    """
    B = features.shape[0]
    device = features.device
    vx = features[:, 0]; vy = features[:, 1]
    dxA = features[:, 2]; dyA = features[:, 3]

    if buffers is None:
        best_score = torch.full((B,), float('inf'), dtype=features.dtype, device=device)
        best_tx = torch.zeros(B, dtype=features.dtype, device=device)
        best_ty = torch.zeros(B, dtype=features.dtype, device=device)
        any_valid = torch.zeros(B, dtype=torch.bool, device=device)
    else:
        best_score = buffers['best_score'].fill_(float('inf'))
        best_tx = buffers['best_tx'].fill_(0.0)
        best_ty = buffers['best_ty'].fill_(0.0)
        any_valid = buffers['any_valid'].fill_(False)

    for k in range(POLICY_K):
        dxk = features[:, 7 + 2*k]
        dyk = features[:, 8 + 2*k]
        dk = features[:, 23 + k]
        bvxk0 = features[:, 35 + 2*k]
        bvyk0 = features[:, 36 + 2*k]

        real = (dxk != POLICY_PAD) & (dk < POLICY_R)
        rx = dxk
        ry = dyk
        bvx = bvxk0
        bvy = bvyk0
        # Forward-simulate T steps. We compute predator steering toward
        # the CURRENT relative offset at each step.
        for t in range(steps):
            dnow = torch.sqrt(rx * rx + ry * ry).clamp_min(1e-9)
            in_range = dnow < POLICY_R
            strength = ((POLICY_R - dnow) / POLICY_R * PREDATOR_TURN_FACTOR)
            ux = rx / dnow
            uy = ry / dnow
            # Avoidance acceleration on boid
            ax_av = ux * strength
            ay_av = uy * strength
            # fastLimit avoidance to MAX_FORCE * 1.5
            mag_a = fast_mag(ax_av, ay_av)
            cap = mag_a > BOID_MAX_FORCE_AVOID
            s = torch.where(cap, BOID_MAX_FORCE_AVOID / torch.where(cap, mag_a, torch.ones_like(mag_a)), torch.ones_like(mag_a))
            ax_av = ax_av * s
            ay_av = ay_av * s
            # Only apply within range
            ax_av = torch.where(in_range, ax_av, torch.zeros_like(ax_av))
            ay_av = torch.where(in_range, ay_av, torch.zeros_like(ay_av))
            # Update boid velocity
            bvx = bvx + ax_av
            bvy = bvy + ay_av
            # Cap boid speed to BOID_MAX_SPEED
            bm = fast_mag(bvx, bvy)
            cap = bm > BOID_MAX_SPEED
            s2 = torch.where(cap, BOID_MAX_SPEED / torch.where(cap, bm, torch.ones_like(bm)), torch.ones_like(bm))
            bvx = bvx * s2
            bvy = bvy * s2
            # Predator step: head toward (rx, ry) at MAX_SPEED
            sm = torch.sqrt(rx * rx + ry * ry).clamp_min(1e-9)
            pvx = rx / sm * PREDATOR_MAX_SPEED
            pvy = ry / sm * PREDATOR_MAX_SPEED
            # Relative offset update
            rx = rx + bvx - pvx
            ry = ry + bvy - pvy

        d_pred = torch.sqrt(rx * rx + ry * ry)
        score = d_pred + distW * dk
        better = real & (score < best_score)
        best_score = torch.where(better, score, best_score)
        best_tx = torch.where(better, rx, best_tx)
        best_ty = torch.where(better, ry, best_ty)
        any_valid = any_valid | better

    tx = torch.where(any_valid, best_tx, dxA)
    ty = torch.where(any_valid, best_ty, dyA)
    return torch.stack(_seek_steering(tx, ty, vx, vy), dim=-1)


def predator_steering(features, kind, opts=None, buffers=None):
    """Top-level dispatcher. Returns steering tensor (B, 2)."""
    opts = opts or {}
    if kind == 'rule_v1' or kind == 'rule':
        return rule_v1_torch(features)
    if kind == 'rule_v2':
        return rule_v2_torch(features, opts.get('alpha', 5.0))
    if kind == 'rule_v3':
        return rule_v3_torch(features,
                              mode=opts.get('mode', 'score_minus_dist'),
                              distW=opts.get('distW', 0.05),
                              alpha=opts.get('alpha', 0.0),
                              buffers=buffers)
    if kind == 'rule_v4':
        return rule_v4_torch(features, distW=opts.get('distW', 0.0),
                              buffers=buffers)
    if kind == 'rule_v5':
        return rule_v5_torch(features,
                              steps=opts.get('steps', 5),
                              distW=opts.get('distW', 0.0),
                              buffers=buffers)
    raise ValueError(f"unknown rule kind: {kind}")


def make_rule_buffers(B, device, dtype=torch.float64):
    """Pre-allocate scratch buffers for graph-safe rule policies."""
    return {
        'best_score': torch.empty((B,), dtype=dtype, device=device),
        'best_t':     torch.empty((B,), dtype=dtype, device=device),
        'best_tx':    torch.empty((B,), dtype=dtype, device=device),
        'best_ty':    torch.empty((B,), dtype=dtype, device=device),
        'any_valid':  torch.empty((B,), dtype=torch.bool, device=device),
    }
