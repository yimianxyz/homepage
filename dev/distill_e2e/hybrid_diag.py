"""Decompose the e2e behavioural gap: which regime (patrol vs chase) loses catches?

Runs a hybrid predator that uses the e2e net for ONE regime and the full production
pipeline for the other, gated per-seed by in-range (any boid within PREDATOR_RANGE):

  patrol_e2e : e2e steers when OUT of range (patrol), production when in range (chase)
  chase_e2e  : e2e steers when IN range (chase),  production when out (patrol)
  full       : e2e everywhere (sanity, == eval_e2e)

Compare mean catches vs pure production. If patrol_e2e ~ production, patrol is fine and
chase is the weakness (and vice versa).

  python3 hybrid_diag.py --net net_h64x32.pt --seeds 512 --device cuda
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')
from sim_torch import (Sim, load_weights, build_features, nn_forward,        # noqa: E402
                       nn_forward_batched, fast_limit, PREDATOR_MAX_SPEED, PREDATOR_RANGE)
from raw_obs import raw_obs                                                   # noqa: E402
from eval_e2e import load_net, E3D                                           # noqa: E402


class HybridSim(Sim):
    def __init__(self, *a, net=None, mode='full', G=9, K=8, **kw):
        super().__init__(*a, **kw)
        self.e2e = net.eval() if net is not None else None
        self.mode, self.G, self.K = mode, G, K

    def _prod_force(self):
        feats = build_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self.weights['featureDim'], self.device, dtype=torch.float32)
        s = nn_forward_batched(feats, self.weights) if 'K' in self.weights else nn_forward(feats, self.weights)
        return s.double()

    def _e2e_force(self):
        with torch.no_grad():
            obs, _ = raw_obs(self.pred_pos, self.pred_vel, self.boid_pos,
                             self.boid_vel, self.boid_alive, G=self.G, K=self.K)
            return self.e2e(obs).double()

    def _step_predator(self):
        self._update_auto_target()
        prod = self._prod_force()
        if self.mode == 'prod':
            steering = prod
        else:
            e2e = self._e2e_force()
            dx = self.boid_pos[..., 0] - self.pred_pos[:, None, 0]
            dy = self.boid_pos[..., 1] - self.pred_pos[:, None, 1]
            d = torch.sqrt(dx * dx + dy * dy)
            d = torch.where(self.boid_alive, d, self._inf_t)
            in_range = (d < PREDATOR_RANGE).any(dim=1, keepdim=True)
            if self.mode == 'full':
                steering = e2e
            elif self.mode == 'patrol_e2e':
                steering = torch.where(in_range, prod, e2e)
            elif self.mode == 'chase_e2e':
                steering = torch.where(in_range, e2e, prod)
        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = new_vx; self.pred_vel[:, 1] = new_vy
        self.pred_pos[:, 0] += new_vx; self.pred_pos[:, 1] += new_vy
        for j, mx in ((0, self._wrap_w_max), (1, self._wrap_h_max)):
            self.pred_pos[:, j] = torch.where(self.pred_pos[:, j] > mx, self._wrap_neg20, self.pred_pos[:, j])
            self.pred_pos[:, j] = torch.where(self.pred_pos[:, j] < self._wrap_neg20, mx, self.pred_pos[:, j])


def run(seeds, weights, dev, net, mode, frames, G, K):
    sim = HybridSim(seeds=seeds, weights=weights, device=dev, auto_target='evolved',
                    auto_target_opts=dict(E3D), net=net, mode=mode, G=G, K=K)
    return sim.run(frames)['mean_catches']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--seeds', type=int, default=512)
    ap.add_argument('--seedStart', type=int, default=70000)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--G', type=int, default=9)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    seeds = list(range(args.seedStart, args.seedStart + args.seeds))
    dev = args.device
    weights = load_weights(args.weights, device=dev)
    net, _ = load_net(args.net, dev)
    for mode in ['prod', 'full', 'patrol_e2e', 'chase_e2e']:
        mc = run(seeds, weights, dev, net, mode, args.frames, args.G, args.K)
        print(f"{mode:>11}: {mc:.3f}")


if __name__ == '__main__':
    main()
