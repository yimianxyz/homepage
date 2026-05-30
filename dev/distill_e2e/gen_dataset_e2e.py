"""M1 — on-policy dataset: raw obs -> production steering force.

Runs the SHIPPED production policy (E3D evolved patrol + 35-feat builder + 35->4->2
NN) in sim_torch on GPU and, at every captured frame, records:
    obs   = raw_obs(current state)            (B, obs_dim)   -- the e2e NN input
    force = production steering (NN output)   (B, 2)         -- the regression target
    d1    = torus distance to nearest boid    (B,)           -- for chase/patrol strata

On-policy: states are visited by the production policy itself, so the distilled net
is trained on exactly the distribution it must reproduce. Frames are subsampled by
--stride (consecutive frames are near-duplicates). Output: dataset_<tag>.pt with
obs/force/d1 tensors + meta.

Run (GPU):
  python3 gen_dataset_e2e.py --seeds 512 --frames 1500 --stride 5 \
      --device cuda --weights predator_weights.json --tag train --seedStart 50000
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')
from sim_torch import (Sim, load_weights, build_features, nn_forward,           # noqa: E402
                       nn_forward_batched, fast_limit, PREDATOR_MAX_SPEED)
from raw_obs import raw_obs, obs_dim                                            # noqa: E402

E3D = dict(cluster_r=178.09, dens_pow=2.373, reach_scale=1515.0, sharp=9.25,
           lead_scale=0.454, lead_max=230.6, nbhd=0.461, momentum=0.0)


class CaptureSim(Sim):
    def __init__(self, *a, G=9, K=8, stride=5, **kw):
        super().__init__(*a, **kw)
        self.G, self.K, self.stride = G, K, stride
        self._obs, self._force, self._d1 = [], [], []
        self._auto, self._ppos = [], []

    def _step_predator(self):
        self._update_auto_target()
        feats = build_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self.weights['featureDim'], self.device,
            dtype=torch.float32)
        if 'K' in self.weights:
            steering = nn_forward_batched(feats, self.weights).double()
        else:
            steering = nn_forward(feats, self.weights).double()

        if self.frame % self.stride == 0:
            with torch.no_grad():
                obs, d1 = raw_obs(self.pred_pos, self.pred_vel,
                                  self.boid_pos, self.boid_vel, self.boid_alive,
                                  G=self.G, K=self.K)
            self._obs.append(obs.cpu())
            self._force.append(steering.float().cpu())
            self._d1.append(d1.cpu())
            self._auto.append(self.pred_auto.float().cpu())
            self._ppos.append(self.pred_pos.float().cpu())

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--seeds', type=int, default=512)
    ap.add_argument('--seedStart', type=int, default=50000)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--G', type=int, default=9)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='train')
    ap.add_argument('--outdir', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    seeds = list(range(args.seedStart, args.seedStart + args.seeds))
    weights = load_weights(args.weights, device=args.device)
    t0 = time.time()
    sim = CaptureSim(seeds=seeds, weights=weights, device=args.device,
                     auto_target='evolved', auto_target_opts=dict(E3D),
                     G=args.G, K=args.K, stride=args.stride)
    out = sim.run(args.frames)
    obs = torch.cat(sim._obs, 0)
    force = torch.cat(sim._force, 0)
    d1 = torch.cat(sim._d1, 0)
    dt = time.time() - t0

    meta = dict(seeds=len(seeds), seedStart=args.seedStart, frames=args.frames,
                stride=args.stride, G=args.G, K=args.K, obs_dim=obs_dim(args.G, args.K),
                n_samples=int(obs.shape[0]), mean_catches=out['mean_catches'],
                in_range_frac=float((torch.isfinite(d1) & (d1 < 80.0)).float().mean()),
                gen_seconds=round(dt, 1))
    path = os.path.join(args.outdir, f'dataset_{args.tag}.pt')
    auto = torch.cat(sim._auto, 0)
    ppos = torch.cat(sim._ppos, 0)
    torch.save(dict(obs=obs, force=force, d1=d1, auto=auto, ppos=ppos, meta=meta), path)
    print(json.dumps(meta, indent=2))
    print(f"# wrote {path}  obs={tuple(obs.shape)} force={tuple(force.shape)}")


if __name__ == '__main__':
    main()
