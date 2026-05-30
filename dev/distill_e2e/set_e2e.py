"""Set-encoder e2e pipeline: gen | train | decompose  (parallel to the grid/raw_obs path).

  python3 set_e2e.py gen   --seeds 768 --seedStart 50000 --tag train --device cuda
  python3 set_e2e.py gen   --seeds 192 --seedStart 60000 --tag val   --device cuda
  python3 set_e2e.py train --train setds_train.pt --val setds_val.pt --mode attn \
                           --d 32 --rho 64 --epochs 250 --tag attn --device cuda
  python3 set_e2e.py decompose --net setnet_attn.pt --seeds 512 --seedStart 70000 --device cuda
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')
from sim_torch import (Sim, load_weights, build_features, nn_forward,             # noqa: E402
                       nn_forward_batched, fast_limit, PREDATOR_MAX_SPEED, PREDATOR_RANGE)
from set_obs import set_obs, FEAT_DIM                                             # noqa: E402
from set_net import SetNet                                                        # noqa: E402

E3D = dict(cluster_r=178.09, dens_pow=2.373, reach_scale=1515.0, sharp=9.25,
           lead_scale=0.454, lead_max=230.6, nbhd=0.461, momentum=0.0)


def _prod_steer(sim):
    feats = build_features(
        sim.pred_pos.float(), sim.pred_vel.float(),
        sim.boid_pos.float(), sim.boid_vel.float(), sim.boid_alive,
        sim.pred_auto.float(), sim.weights['featureDim'], sim.device, dtype=torch.float32)
    s = nn_forward_batched(feats, sim.weights) if 'K' in sim.weights else nn_forward(feats, sim.weights)
    return s.double()


def _advance(sim, steering):
    nvx = sim.pred_vel[:, 0] + steering[:, 0]
    nvy = sim.pred_vel[:, 1] + steering[:, 1]
    nvx, nvy = fast_limit(nvx, nvy, PREDATOR_MAX_SPEED)
    sim.pred_vel[:, 0] = nvx; sim.pred_vel[:, 1] = nvy
    sim.pred_pos[:, 0] += nvx; sim.pred_pos[:, 1] += nvy
    for j, mx in ((0, sim._wrap_w_max), (1, sim._wrap_h_max)):
        sim.pred_pos[:, j] = torch.where(sim.pred_pos[:, j] > mx, sim._wrap_neg20, sim.pred_pos[:, j])
        sim.pred_pos[:, j] = torch.where(sim.pred_pos[:, j] < sim._wrap_neg20, mx, sim.pred_pos[:, j])


class SetCaptureSim(Sim):
    def __init__(self, *a, stride=5, density_radii=None, **kw):
        super().__init__(*a, **kw)
        self.stride = stride
        self.density_radii = density_radii
        self.F, self.M, self.PV, self.FO, self.D1 = [], [], [], [], []
        self.AU, self.PP = [], []

    def _step_predator(self):
        self._update_auto_target()
        steering = _prod_steer(self)
        if self.frame % self.stride == 0:
            with torch.no_grad():
                feats, mask, pvel, d1 = set_obs(self.pred_pos, self.pred_vel,
                                                self.boid_pos, self.boid_vel, self.boid_alive,
                                                density_radii=self.density_radii)
            self.F.append(feats.half().cpu()); self.M.append(mask.bool().cpu())
            self.PV.append(pvel.cpu()); self.FO.append(steering.float().cpu())
            self.D1.append(d1.cpu())
            self.AU.append(self.pred_auto.float().cpu()); self.PP.append(self.pred_pos.float().cpu())
        _advance(self, steering)


class SetHybridSim(Sim):
    def __init__(self, *a, net=None, mode='full', density_radii=None, **kw):
        super().__init__(*a, **kw)
        self.net = net.eval() if net is not None else None
        self.mode = mode
        self.density_radii = density_radii

    def _e2e(self):
        with torch.no_grad():
            feats, mask, pvel, _ = set_obs(self.pred_pos, self.pred_vel,
                                           self.boid_pos, self.boid_vel, self.boid_alive,
                                           density_radii=self.density_radii)
            return self.net(feats, mask, pvel).double()

    def _step_predator(self):
        self._update_auto_target()
        prod = _prod_steer(self)
        if self.mode == 'prod':
            steering = prod
        else:
            e2e = self._e2e()
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
        _advance(self, steering)


# ---------------- gen ----------------
def _parse_radii(s):
    return [float(x) for x in s.split(',')] if s else None


def cmd_gen(a):
    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    weights = load_weights(a.weights, device=a.device)
    radii = _parse_radii(a.density_radii)
    t0 = time.time()
    sim = SetCaptureSim(seeds=seeds, weights=weights, device=a.device,
                        auto_target='evolved', auto_target_opts=dict(E3D), stride=a.stride,
                        density_radii=radii)
    out = sim.run(a.frames)
    feats = torch.cat(sim.F, 0); mask = torch.cat(sim.M, 0)
    pvel = torch.cat(sim.PV, 0); force = torch.cat(sim.FO, 0); d1 = torch.cat(sim.D1, 0)
    auto = torch.cat(sim.AU, 0); ppos = torch.cat(sim.PP, 0)
    meta = dict(seeds=len(seeds), seedStart=a.seedStart, frames=a.frames, stride=a.stride,
                density_radii=radii,
                n=int(feats.shape[0]), N=int(feats.shape[1]), mean_catches=out['mean_catches'],
                gen_seconds=round(time.time() - t0, 1))
    path = f'setds_{a.tag}.pt'
    torch.save(dict(feats=feats, mask=mask, pvel=pvel, force=force, d1=d1,
                    auto=auto, ppos=ppos, meta=meta), path)
    print(json.dumps(meta, indent=2))
    print(f"# wrote {path} feats={tuple(feats.shape)} ({feats.dtype})")


# ---------------- train ----------------
def angerr(pred, tgt):
    pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
    tn = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-9)
    cos = (pn * tn).sum(1).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos))


def cmd_train(a):
    dev = a.device
    tr = torch.load(a.train, map_location='cpu'); va = torch.load(a.val, map_location='cpu')
    Ftr, Mtr, Ptr, Ytr, Dtr = tr['feats'].float(), tr['mask'].float(), tr['pvel'].float(), tr['force'].float(), tr['d1']
    Fva, Mva, Pva, Yva, Dva = va['feats'].float(), va['mask'].float(), va['pvel'].float(), va['force'].float(), va['d1']
    in_dim = Ftr.shape[-1]                                   # FEAT_DIM (+ density cols if present)
    density_radii = tr.get('meta', {}).get('density_radii')
    rho = tuple(int(x) for x in a.rho.split(',')) if a.rho else ()
    net = SetNet(in_dim=in_dim, d=a.d, rho=rho, mode=a.mode, heads=a.heads, act=a.act,
                 nblocks=a.nblocks, pool=a.pool).to(dev)
    net.set_standardizer(Ftr, Mtr)                          # CPU stats -> buffers (no full set on GPU)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    n = Ftr.shape[0]; bs = a.bs
    pat_tr = ((Dtr > 80) | ~torch.isfinite(Dtr)).float()    # 1=patrol, 0=chase (train, on CPU)
    pat_va = (Dva > 80) | ~torch.isfinite(Dva); chs_va = ~pat_va
    # E3D selection is softmax(9.25*log a): the gate needs a SHARP temperature. log_tau init
    # = -log(9.25) ~ -2.22 puts tau in the right basin instead of the too-soft tau=1 default.
    if a.pool == 'gate' and a.tauinit is not None:
        with torch.no_grad():
            net.gatepool.log_tau.fill_(a.tauinit)

    def eval_val():                                          # batched: attention is O(N^2) per row
        outs = []
        for j in range(0, Fva.shape[0], bs):
            pv = net(Fva[j:j + bs].to(dev), Mva[j:j + bs].to(dev), Pva[j:j + bs].to(dev))
            outs.append(pv.cpu())
        return torch.cat(outs, 0)

    def cosmed(pred, tgt, msk):
        pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
        tn = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-9)
        c = (pn * tn).sum(1).clamp(-1, 1)[msk]
        return c.median().item(), (c > 0.99).float().mean().item() * 100

    best = -1.0; best_state = None                          # select on PATROL cos_med (the hard regime)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            f = Ftr[idx].to(dev); m = Mtr[idx].to(dev); p = Ptr[idx].to(dev); y = Ytr[idx].to(dev)
            pat_b = pat_tr[idx].to(dev)
            if a.patrolonly:
                w = pat_b                                       # train ONLY on patrol (chase is analytic seek-nearest)
            else:
                w = 1.0 + (a.patrolw - 1.0) * pat_b             # upweight patrol frames
            pred = net(f, m, p)
            mse = ((pred - y) ** 2).sum(1)
            pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
            tn = y / (y.norm(dim=1, keepdim=True) + 1e-9)
            dircos = 1 - (pn * tn).sum(1)
            per = (0.0 if a.dironly else mse) + a.dirw * dircos
            loss = (w * per).sum() / w.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = eval_val()
            pcm, pp99 = cosmed(pv, Yva, pat_va)
            ccm, _ = cosmed(pv, Yva, chs_va)
        if pcm > best:
            best = pcm; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if not a.quiet or ep == a.epochs - 1 or ep % 25 == 0:
            tau = (net.gatepool.log_tau.exp().item() if a.pool == 'gate' else 0.0)
            print(f"ep{ep:3d} pat_cosM={pcm:.4f} pat%>99={pp99:5.1f} chs_cosM={ccm:.4f} "
                  f"tau={tau:.3f} params={net.n_params()}")
    path = f'setnet_{a.tag}.pt'
    torch.save(dict(state_dict=best_state, in_dim=in_dim, d=a.d, rho=rho, mode=a.mode,
                    heads=a.heads, act=a.act, nblocks=a.nblocks, pool=a.pool, best_pat_cosmed=best,
                    density_radii=density_radii,
                    meta=dict(epochs=a.epochs, params=net.n_params())), path)
    print(f"# wrote {path}  best_pat_cosM={best:.4f} params={net.n_params()}")


def load_setnet(path, dev):
    ck = torch.load(path, map_location=dev)
    net = SetNet(in_dim=ck['in_dim'], d=ck['d'], rho=ck['rho'], mode=ck['mode'],
                 heads=ck['heads'], act=ck['act'],
                 nblocks=ck.get('nblocks', 1), pool=ck.get('pool', 'mean')).to(dev)
    net.load_state_dict(ck['state_dict']); net.eval()
    return net, ck


# ---------------- decompose ----------------
def cmd_decompose(a):
    dev = a.device
    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    weights = load_weights(a.weights, device=dev)
    net, ck = load_setnet(a.net, dev)
    radii = ck.get('density_radii')
    print(f"# net mode={ck['mode']} d={ck['d']} rho={ck['rho']} params={ck['meta']['params']} density_radii={radii}")
    for mode in ['prod', 'full', 'patrol_e2e', 'chase_e2e']:
        sim = SetHybridSim(seeds=seeds, weights=weights, device=dev, auto_target='evolved',
                           auto_target_opts=dict(E3D), net=net, mode=mode, density_radii=radii)
        mc = sim.run(a.frames)['mean_catches']
        print(f"{mode:>11}: {mc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--weights', default='predator_weights.json'); g.add_argument('--seeds', type=int, default=768)
    g.add_argument('--seedStart', type=int, default=50000); g.add_argument('--frames', type=int, default=1500)
    g.add_argument('--stride', type=int, default=5); g.add_argument('--tag', default='train')
    g.add_argument('--density-radii', dest='density_radii', default=None,
                   help='comma-sep raw-unit radii for per-boid local density features, e.g. 80,178')
    g.add_argument('--device', default='cuda')
    t = sub.add_parser('train')
    t.add_argument('--train', required=True); t.add_argument('--val', required=True)
    t.add_argument('--mode', default='attn'); t.add_argument('--d', type=int, default=32)
    t.add_argument('--rho', default='64'); t.add_argument('--heads', type=int, default=2)
    t.add_argument('--nblocks', type=int, default=1); t.add_argument('--pool', default='mean')
    t.add_argument('--act', default='relu'); t.add_argument('--epochs', type=int, default=250)
    t.add_argument('--bs', type=int, default=4096); t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--wd', type=float, default=0.0); t.add_argument('--dirw', type=float, default=1.0)
    t.add_argument('--patrolw', type=float, default=1.0, help='loss multiplier on patrol frames')
    t.add_argument('--patrolonly', action='store_true', help='train ONLY on patrol frames (chase handled analytically)')
    t.add_argument('--dironly', action='store_true', help='pure direction (cosine) loss, drop MSE')
    t.add_argument('--tauinit', type=float, default=None, help='gate log_tau init (e.g. -2.22 = 1/9.25)')
    t.add_argument('--tag', default='attn'); t.add_argument('--device', default='cuda')
    t.add_argument('--quiet', action='store_true')
    d = sub.add_parser('decompose')
    d.add_argument('--net', required=True); d.add_argument('--weights', default='predator_weights.json')
    d.add_argument('--seeds', type=int, default=512); d.add_argument('--seedStart', type=int, default=70000)
    d.add_argument('--frames', type=int, default=1500); d.add_argument('--device', default='cuda')
    a = ap.parse_args()
    {'gen': cmd_gen, 'train': cmd_train, 'decompose': cmd_decompose}[a.cmd](a)


if __name__ == '__main__':
    main()
