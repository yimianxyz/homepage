"""TD-MPC state-value student: SIMPLER + far cheaper to eval than the per-candidate
value net.

Per-candidate V + max-bootstrap recomputes 19 features (incl. a 90-step ballistic
sim) at B*K*K terminal candidates every decision -> ~24x the rollout cost, which
is what throttles every eval to ~20 min. Here the bootstrap is a STATE value:

  deploy: every D frames, roll each of K candidates Hs frames (M nearest boids),
          score[k] = (catches during Hs) + Vs(state_obs at terminal_k); argmax.

Vs is a tiny MLP on GLOBAL state features only (predator vel, E3D-target-rel, M
nearest boids rel pos/vel, frac_alive, centroid -- planner_obs). No per-candidate
features, no ballistic anywhere -> the rollout (which actually simulates the
chase) subsumes what the ballistic feature approximated. ~20x faster eval, simpler.

  state_obs(sim)                       -> (B,F) global features
  run_log_state(...)                   -> (obs, vtarget) ; Vs target = catches over
                                          next W frames under the planner
  StateValueNet                        -> MLP F->h->h->1
  run_state_student(..., vs, Hs, M)    -> per-seed catches
"""
import numpy as np
import torch
import torch.nn as nn

import planner_probe as pp
import sim_torch as st
from sim_torch import Sim

MOBS = 8          # nearest boids included in the state observation
_PS = 200.0


def state_obs(sim):
    """Global state features (planner_obs): pred vel, E3D-target rel, MOBS nearest
    boids rel pos/vel, frac_alive, alive-centroid rel. Returns (B,F) float32."""
    e3d_rel = pp._e3d_target(sim) - sim.pred_pos
    return pp.planner_obs(sim, MOBS, e3d_rel).float()


def feat_dim():
    return 2 + 2 + 4 * MOBS + 1 + 2


def run_log_state(seeds, frames, device, K, H, D, W):
    """Run the planner (full H rollout argmax), logging state_obs each frame and
    cumulative catches; Vs target[t] = catches over the next W frames = cat[t+W]-cat[t]."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    rows = torch.arange(B, device=device)
    held = None
    obs_log, cat_log = [], []
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            base = pp._save_state(sim)
            pp._load_state(roll, pp._tile_state(base, K))
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            c0 = roll.catches.clone()
            h = min(H, frames - f)
            for _ in range(h):
                pp._step_with_target(roll, roll_tgt)
            gain = pp.rollout_gain(roll, c0, B, K)
            held = cand[rows, gain.argmax(dim=1)]
        obs_log.append(state_obs(sim).cpu())
        cat_log.append(sim.catches.clone().float().cpu())
        pp._step_with_target(sim, held)
        f += 1
    obs = torch.stack(obs_log, 0)       # (T,B,F)
    cat = torch.stack(cat_log, 0)       # (T,B)
    T = obs.shape[0]
    idx = torch.clamp(torch.arange(T) + W, max=T - 1)
    vtarget = cat[idx] - cat            # (T,B) catches over next W
    obs = obs.reshape(-1, obs.shape[-1]).numpy()
    vtarget = vtarget.reshape(-1).numpy()
    return obs, vtarget, sim.catches.cpu().numpy()


class StateValueNet(nn.Module):
    def __init__(self, fdim, hidden=48, depth=2):
        super().__init__()
        layers = [nn.Linear(fdim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeployVs(nn.Module):
    """Wrap StateValueNet with saved standardization (raw obs -> value)."""
    def __init__(self, blob, device):
        super().__init__()
        self.m = StateValueNet(blob['fdim'], blob['hidden'], blob['depth'])
        self.m.load_state_dict(blob['state']); self.m.to(device).eval()
        self.mu = blob['mu'].to(device); self.sd = blob['sd'].to(device)

    def forward(self, obs):
        return self.m((obs - self.mu) / self.sd)


def run_state_student(seeds, frames, device, vs, K, D, Hs, roll_M=0, bias0=0.0):
    """Deploy: roll each candidate Hs frames (M nearest boids if roll_M>0), score =
    catches during Hs + Vs(terminal state); argmax."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    rows = torch.arange(B, device=device)
    held = None
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            base = pp._save_state(sim)
            pp._load_state(roll, pp._tile_state(base, K))
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            c0 = roll.catches.clone()
            if roll_M and roll_M < sim.N:
                dx = roll.boid_pos[..., 0] - roll.pred_pos[:, None, 0]
                dy = roll.boid_pos[..., 1] - roll.pred_pos[:, None, 1]
                d2 = torch.where(roll.boid_alive, dx * dx + dy * dy,
                                 torch.full_like(roll.boid_pos[..., 0], float('inf')))
                order = torch.argsort(d2, dim=1)
                active = torch.zeros_like(roll.boid_alive)
                active.scatter_(1, order[:, :roll_M],
                                torch.ones_like(order[:, :roll_M], dtype=active.dtype))
                frozen = ~active
                for _ in range(Hs):
                    sp = roll.boid_pos.clone(); sv = roll.boid_vel.clone()
                    roll._step_boids()
                    roll.boid_pos[frozen] = sp[frozen]; roll.boid_vel[frozen] = sv[frozen]
                    pp._analytic_steer(roll, roll_tgt); roll._check_catches()
            else:
                for _ in range(Hs):
                    pp._step_with_target(roll, roll_tgt)
            c_near = (roll.catches - c0).reshape(B, K).float()
            with torch.no_grad():
                vterm = vs(state_obs(roll).to(device)).reshape(B, K)
            score = c_near + vterm
            if bias0 != 0.0:
                score = score.clone(); score[:, 0] = score[:, 0] + bias0
            held = cand[rows, score.argmax(dim=1)]
        pp._step_with_target(sim, held)
        f += 1
    return sim.catches.cpu().numpy()
