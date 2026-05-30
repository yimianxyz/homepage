"""M0/M3 — behavioral eval harness for an e2e net (the north-star metric).

Plugs a trained E2ENet into sim_torch in place of the whole production pipeline
(no evolved patrol, no 35-feat builder, no chase NN) and measures behavioral
equivalence to production on the SAME held-out seeds:

  * mean catches  (e2e vs production)        <- primary
  * per-seed mean |diff| and catch correlation
  * sorted-distribution gap (behavioural, since exact per-seed is impossible — see M0)

Progressive precision: run with small --seeds to screen many candidates fast, then a
large block to verify the survivors.

Run:
  python3 eval_e2e.py --net net_h16.pt --seeds 512 --seedStart 70000 --device cuda
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')
from sim_torch import Sim, load_weights, fast_limit, PREDATOR_MAX_SPEED            # noqa: E402
from raw_obs import raw_obs                                                        # noqa: E402
from e2e_net import E2ENet                                                         # noqa: E402

E3D = dict(cluster_r=178.09, dens_pow=2.373, reach_scale=1515.0, sharp=9.25,
           lead_scale=0.454, lead_max=230.6, nbhd=0.461, momentum=0.0)


class E2ESim(Sim):
    """Predator steering comes entirely from the e2e net on raw_obs."""
    def __init__(self, *a, net=None, G=9, K=8, **kw):
        super().__init__(*a, **kw)
        self.e2e = net.eval()
        self.G, self.K = G, K

    def _step_predator(self):
        with torch.no_grad():
            obs, _ = raw_obs(self.pred_pos, self.pred_vel, self.boid_pos,
                             self.boid_vel, self.boid_alive, G=self.G, K=self.K)
            steering = self.e2e(obs).double()
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


def load_net(path, device):
    ck = torch.load(path, map_location=device)
    net = E2ENet(ck['in_dim'], hidden=tuple(ck['hidden']), act=ck['act'],
                 head=ck.get('head', 'force'))
    net.load_state_dict(ck['state_dict'])
    return net.to(device), ck.get('meta', {})


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
    net, meta = load_net(args.net, dev)

    t0 = time.time()
    prod = Sim(seeds=seeds, weights=weights, device=dev,
               auto_target='evolved', auto_target_opts=dict(E3D)).run(args.frames)
    e2e = E2ESim(seeds=seeds, weights=weights, device=dev,
                 auto_target='evolved', auto_target_opts=dict(E3D),
                 net=net, G=args.G, K=args.K).run(args.frames)
    dt = time.time() - t0

    pc = torch.tensor(prod['per_seed_catches']).float()
    ec = torch.tensor(e2e['per_seed_catches']).float()
    diff = ec - pc
    corr = torch.corrcoef(torch.stack([pc, ec]))[0, 1].item() if pc.std() > 0 else float('nan')
    dist_gap = (ec.sort().values - pc.sort().values).abs().mean().item()
    se = diff.std().item() / (len(seeds) ** 0.5)
    res = dict(net=os.path.basename(args.net), n_params=meta.get('n_params'),
               hidden=meta.get('hidden'), seeds=len(seeds), seedStart=args.seedStart,
               frames=args.frames,
               mean_prod=pc.mean().item(), mean_e2e=ec.mean().item(),
               mean_delta=diff.mean().item(), delta_se=se,
               mean_abs_diff=diff.abs().mean().item(),
               per_seed_corr=corr, dist_gap=dist_gap, eval_seconds=round(dt, 1))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
