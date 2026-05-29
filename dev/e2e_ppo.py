"""PPO for the end-to-end predator policy on the parallel GPU sim (vec-env).

The batched sim IS the vectorized environment (B parallel envs, one shared
policy). Stochastic Gaussian policy over 2D steering + value baseline; reward =
per-step catch delta (optionally + small proximity shaping); GAE; clipped PPO.

This is the sample-efficient escalation after ES plateaued (~5.4 vs
nearest_cluster 7.6 @1500f). PPO's value baseline cuts gradient variance, so the
*central* policy should converge, not just lucky perturbations.

  python3 dev/e2e_ppo.py --B 1024 --rollout 128 --iters 200 --device cuda \
      --out ~/ckpt/ppo1
"""
import argparse, json, time, math, sys, os
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import (Sim, fast_limit, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE,
                       PREDATOR_FEED_COOLDOWN_MS, load_weights, nn_forward, build_features)
from e2e_obs import build_obs_egocentric, build_obs_augmented, AUG_EXTRA

A, R = 8, 3
OBS_RAW = 3 * A * R + 3      # 75
OBS_AUG = OBS_RAW + AUG_EXTRA  # 83
# nearest_cluster patrol opts matching the deployed production policy
NC_OPTS = {'cluster_r': 150.0, 'lead_scale': 0.4, 'lead_max': 120.0}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_RAW, hidden=64, act_dim=2):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        for m in [self.mu, self.v]:
            nn.init.orthogonal_(m.weight, 0.01); nn.init.zeros_(m.bias)

    def forward(self, obs):
        h = self.body(obs)
        return self.mu(h), self.v(h).squeeze(-1)


class PPOSim(Sim):
    """Sim where the predator is driven by an external action set each step.

    augment=True injects the nearest_cluster patrol target + nearest-boid
    features into the observation (obs_dim 83) so PPO starts from the known-good
    hand-crafted signal and learns to improve on it.
    """
    def __init__(self, seeds, device='cuda', augment=False,
                 residual=False, base_weights=None, resid_scale=0.05,
                 cooldown_obs=False):
        w = {'featureDim': 45, 'inputMean': None, 'inputStd': None,
             'outputScale': 1.0, 'clipMagnitude': 0.0, 'layers': []}
        # residual mode rides on top of the deployed nearest_cluster policy, so
        # it always uses the cluster patrol target + augmented obs.
        self.residual = residual
        self.augment = augment or residual
        self._base_w = base_weights
        self.resid_scale = resid_scale
        self.cooldown_obs = cooldown_obs
        at = 'nearest_cluster' if self.augment else 'random'
        ato = NC_OPTS if self.augment else None
        super().__init__(seeds=seeds, weights=w, device=device,
                         sequential=False, auto_target=at, auto_target_opts=ato)
        self._action = torch.zeros((self.B, 2), dtype=torch.float64, device=device)

    def _base_steer(self):
        """Deployed production steering (nearest_cluster + distilled NN),
        already magnitude-clipped to clipMagnitude=0.05. pred_auto must be
        fresh (updated this frame)."""
        feats = build_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self._base_w['featureDim'], self.device,
            dtype=torch.float32)
        return nn_forward(feats, self._base_w).double()

    def reset(self, new_seeds):
        """Re-init episode state with fresh seeds (boids don't respawn, so each
        rollout segment past the horizon must start a fresh episode)."""
        self.seeds = list(new_seeds)
        self._initialize()

    def _cooldown_frac(self):
        elapsed = self._frame_ms - self.pred_last_feed_ms
        return torch.clamp((PREDATOR_FEED_COOLDOWN_MS - elapsed) / PREDATOR_FEED_COOLDOWN_MS, 0.0, 1.0)

    def current_obs(self):
        if self.augment:
            self._update_auto_target()
            cd = self._cooldown_frac() if self.cooldown_obs else None
            return build_obs_augmented(self.pred_pos, self.pred_vel,
                                       self.boid_pos, self.boid_vel,
                                       self.boid_alive, self.pred_auto, cooldown=cd)
        return build_obs_egocentric(self.pred_pos, self.pred_vel,
                                    self.boid_pos, self.boid_vel, self.boid_alive)

    def min_boid_dist(self):
        dx = self.boid_pos[..., 0] - self.pred_pos[:, None, 0]
        dy = self.boid_pos[..., 1] - self.pred_pos[:, None, 1]
        d = torch.sqrt(dx * dx + dy * dy)
        d = torch.where(self.boid_alive, d, torch.full_like(d, 1e9))
        return d.min(dim=1).values

    def _step_predator(self):
        if self.residual:
            self._update_auto_target()
            base = self._base_steer()                       # (B,2), already clipped to 0.05
            rx = base[:, 0] + self._action[:, 0] * self.resid_scale
            ry = base[:, 1] + self._action[:, 1] * self.resid_scale
            sx, sy = fast_limit(rx, ry, PREDATOR_MAX_FORCE)  # total capped to 0.05 (resid=0 -> base)
        else:
            sx, sy = fast_limit(self._action[:, 0] * PREDATOR_MAX_FORCE,
                                self._action[:, 1] * PREDATOR_MAX_FORCE, PREDATOR_MAX_FORCE)
        nvx = self.pred_vel[:, 0] + sx
        nvy = self.pred_vel[:, 1] + sy
        nvx, nvy = fast_limit(nvx, nvy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = nvx; self.pred_vel[:, 1] = nvy
        self.pred_pos[:, 0] += nvx; self.pred_pos[:, 1] += nvy
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > self._wrap_w_max, self._wrap_neg20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < self._wrap_neg20, self._wrap_w_max, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > self._wrap_h_max, self._wrap_neg20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < self._wrap_neg20, self._wrap_h_max, self.pred_pos[:, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=1024, help='parallel envs')
    p.add_argument('--rollout', type=int, default=128, help='steps per rollout')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatches', type=int, default=8)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent', type=float, default=0.003)
    p.add_argument('--vf', type=float, default=0.5)
    p.add_argument('--shape', type=float, default=0.0, help='proximity shaping weight')
    p.add_argument('--episode', type=int, default=1500, help='episode horizon (env resets here; boids do not respawn)')
    p.add_argument('--seedStart', type=int, default=100000, help='training seed pool start (kept disjoint from holdout 9000-9999)')
    p.add_argument('--holdout', type=int, default=64)
    p.add_argument('--eval_frames', type=int, default=1500)
    p.add_argument('--augment', action='store_true', help='inject nearest_cluster + nearest-boid features (obs_dim 83)')
    p.add_argument('--residual', action='store_true', help='learn a correction on top of the deployed policy (guaranteed 7.63 floor)')
    p.add_argument('--base_weights', default='js/predator_weights.json', help='deployed weights for residual base steering')
    p.add_argument('--resid_scale', type=float, default=0.05, help='residual action scale (force units)')
    p.add_argument('--cooldown_obs', action='store_true', help='add feed-cooldown remaining to obs (info the base ignores)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    use_aug = a.augment or a.residual
    OBS_DIM = (OBS_AUG if use_aug else OBS_RAW) + (1 if (use_aug and a.cooldown_obs) else 0)
    base_w = load_weights(a.base_weights, device=a.device) if a.residual else None
    mk = lambda seeds: PPOSim(seeds, device=a.device, augment=a.augment,
                              residual=a.residual, base_weights=base_w,
                              resid_scale=a.resid_scale, cooldown_obs=a.cooldown_obs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'ppo_log.jsonl', 'a')
    def log(o): print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()
    dev = a.device
    torch.manual_seed(a.seed)

    ac = ActorCritic(obs_dim=OBS_DIM, hidden=a.hidden).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)

    # training env (B envs). Boids don't respawn, so we run fixed-horizon
    # episodes and reset with fresh seeds at the horizon. next_seed walks a
    # wide pool disjoint from the holdout (9000-9999).
    next_seed = a.seedStart
    def fresh_seeds():
        nonlocal next_seed
        s = list(range(next_seed, next_seed + a.B))
        next_seed += a.B
        if next_seed > a.seedStart + 500000:
            next_seed = a.seedStart
        return s
    env = mk(fresh_seeds())
    ep_frame = 0  # frames since last reset (all B envs synchronized)
    DNORM = 1.0 / 200.0

    def policy_step(collect=True):
        with torch.no_grad():
            obs = env.current_obs()  # (B,75) f32
            mu, val = ac(obs)
            std = torch.exp(ac.log_std)
            if collect:
                noise = torch.randn_like(mu)
                act = mu + std * noise
                logp = (-0.5 * (((act - mu) / std) ** 2 + 2 * ac.log_std + math.log(2 * math.pi))).sum(-1)
            else:
                act = mu; logp = None
        env._action = act.detach().double()  # raw; _step_predator applies scaling/clip
        prev = env.catches.clone()
        prev_d = env.min_boid_dist() if a.shape > 0 else None
        env.step()
        rew = (env.catches - prev).float()
        if a.shape > 0:
            rew = rew + a.shape * (prev_d - env.min_boid_dist()).float() * DNORM
        return obs, act, logp, val, rew

    best_eval = -1.0
    gstep = 0
    for it in range(a.iters):
        t0 = time.time()
        obs_b, act_b, logp_b, val_b, rew_b, done_b = [], [], [], [], [], []
        ep_catches = []  # completed-episode catch totals seen this rollout
        for t in range(a.rollout):
            o, ac_, lp, v, r = policy_step(collect=True)
            ep_frame += 1
            done = ep_frame >= a.episode
            obs_b.append(o); act_b.append(ac_); logp_b.append(lp); val_b.append(v); rew_b.append(r)
            done_b.append(1.0 if done else 0.0)
            gstep += 1
            if done:
                ep_catches.append(env.catches.float().mean().item())
                env.reset(fresh_seeds()); ep_frame = 0
        with torch.no_grad():
            _, last_v = ac(env.current_obs())
        obs_b = torch.stack(obs_b); act_b = torch.stack(act_b)
        logp_b = torch.stack(logp_b); val_b = torch.stack(val_b); rew_b = torch.stack(rew_b)  # (T,B)
        done_t = torch.tensor(done_b, device=dev)  # (T,)
        # GAE with episodic termination (no bootstrap across a reset)
        adv = torch.zeros_like(rew_b); lastgae = torch.zeros(a.B, device=dev)
        for t in reversed(range(a.rollout)):
            mask = 1.0 - done_t[t]
            nextv = (last_v if t == a.rollout - 1 else val_b[t + 1]) * mask
            delta = rew_b[t] + a.gamma * nextv - val_b[t]
            lastgae = delta + a.gamma * a.lam * mask * lastgae
            adv[t] = lastgae
        ret = adv + val_b
        # flatten
        N = a.rollout * a.B
        O = obs_b.reshape(N, OBS_DIM); Ac = act_b.reshape(N, 2)
        LP = logp_b.reshape(N); ADV = adv.reshape(N); RET = ret.reshape(N)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        idx = torch.randperm(N, device=dev)
        mb = N // a.minibatches
        for ep in range(a.epochs):
            for k in range(a.minibatches):
                j = idx[k * mb:(k + 1) * mb]
                mu, v = ac(O[j]); std = torch.exp(ac.log_std)
                lp = (-0.5 * (((Ac[j] - mu) / std) ** 2 + 2 * ac.log_std + math.log(2 * math.pi))).sum(-1)
                ratio = torch.exp(lp - LP[j])
                s1 = ratio * ADV[j]; s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADV[j]
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = ((v - RET[j]) ** 2).mean()
                ent = (ac.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
                loss = pol_loss + a.vf * v_loss - a.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()

        # periodic deterministic holdout eval (fresh env)
        if it % 5 == 0 or it == a.iters - 1:
            hs = list(range(9000, 9000 + a.holdout))
            heval = mk(hs)
            with torch.no_grad():
                for _ in range(a.eval_frames):
                    o = heval.current_obs(); mu, _ = ac(o)
                    heval._action = mu.double()
                    heval.step()
            hscore = heval.catches.float().mean().item()
            if hscore > best_eval:
                best_eval = hscore
                torch.save({'state': ac.state_dict(), 'hidden': a.hidden}, out / 'best.pt')
            log({'iter': it, 'gstep': gstep, 'holdout': hscore, 'best': best_eval,
                 'ep_catch': (sum(ep_catches) / len(ep_catches)) if ep_catches else None,
                 'mean_rew': rew_b.sum(0).mean().item(), 'it_s': time.time() - t0,
                 'log_std': ac.log_std.detach().tolist()})
        else:
            log({'iter': it, 'gstep': gstep,
                 'ep_catch': (sum(ep_catches) / len(ep_catches)) if ep_catches else None,
                 'mean_rew': rew_b.sum(0).mean().item(), 'it_s': time.time() - t0})
    log({'phase': 'done', 'best_holdout': best_eval})


if __name__ == '__main__':
    main()
