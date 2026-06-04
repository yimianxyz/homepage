"""Deploy + score the value-net student vs planner and E3D (the 99.9% gate).

Loads net_value.pt, wraps it with the saved feature standardization, and runs
feat_planner.run_value_student over n seeds (optionally with a short rollout Hs).
Reports student mean catches, planner mean, E3D mean, and student/planner ratio.
"""
import argparse, json
import numpy as np
import torch

import feat_planner as fp
import planner_probe as pp
import sim_torch as st
from train_value import ValueNet


class Deploy(torch.nn.Module):
    """Wraps ValueNet to accept RAW (feat, ctx) and standardize internally."""
    def __init__(self, blob, device):
        super().__init__()
        self.m = ValueNet(blob['fc'], blob['fctx'], blob['hidden'], blob['depth'])
        self.m.load_state_dict(blob['state'])
        self.m.to(device).eval()
        self.fmu = blob['fmu'].to(device); self.fsd = blob['fsd'].to(device)
        self.xmu = blob['xmu'].to(device); self.xsd = blob['xsd'].to(device)

    def forward(self, feat, ctx):
        return self.m((feat - self.fmu) / self.fsd, (ctx - self.xmu) / self.xsd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='net_value.pt')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seedStart', type=int, default=200000)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--Hs', type=int, default=0, help='short rollout depth (0=pure value net)')
    ap.add_argument('--bias_sweep', default=None,
                    help='comma list of E3D-bias values to sweep, e.g. "0,0.5,1,2,4,8"')
    ap.add_argument('--bias0', type=float, default=0.0)
    ap.add_argument('--lookahead', action='store_true',
                    help='depth-1 rollout + max-V bootstrap (needs calibrated absval net)')
    ap.add_argument('--roll_M', type=int, default=0,
                    help='if >0 with --lookahead: cheap rollout of only M nearest boids')
    ap.add_argument('--no_value', action='store_true',
                    help='zero the value net: roll top-K_roll, non-rolled score 0 (E3D fallback)')
    ap.add_argument('--prune_by', default='v', choices=['v', 'ball'],
                    help='which candidates get rolled: v=value prior, ball=ballistic catchability')
    ap.add_argument('--K_roll', type=int, default=0,
                    help='if >0: only top-K_roll candidates (by prior V) get the rollout')
    ap.add_argument('--H', type=int, default=120, help='planner reference horizon')
    ap.add_argument('--twopass', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='../js/predator_weights.json')
    ap.add_argument('--skip-planner', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.TWO_PASS = args.twopass
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    seeds = list(range(args.seedStart, args.seedStart + args.n))

    blob = torch.load(args.net, map_location='cpu')
    model = Deploy(blob, device)

    def deploy(bias0):
        if args.lookahead and args.roll_M > 0:
            return fp.run_value_lookahead_cheap(seeds, args.frames, device, model, args.K,
                                                args.D, args.Hs, args.roll_M, bias0=bias0,
                                                K_roll=args.K_roll, prune_by=args.prune_by,
                                                no_value=args.no_value)
        if args.lookahead:
            return fp.run_value_lookahead(seeds, args.frames, device, model, args.K,
                                          args.D, args.Hs, bias0=bias0)
        return fp.run_value_student(seeds, args.frames, device, model, args.K, args.D,
                                    Hs=args.Hs, bias0=bias0)

    e3d = pp.run_e3d(seeds, args.frames, device)
    if args.bias_sweep:
        biases = [float(x) for x in args.bias_sweep.split(',')]
        sweep = []
        for b in biases:
            s = deploy(b)
            sweep.append(dict(bias0=b, student_mean=float(s.mean()),
                              student_se=float(s.std(ddof=1)/np.sqrt(len(s)))))
            print(json.dumps(dict(bias0=b, student_mean=round(float(s.mean()), 3),
                                  e3d=round(float(e3d.mean()), 2))), flush=True)
        best = max(sweep, key=lambda d: d['student_mean'])
        res = dict(net=args.net, nparams=blob.get('nparams'), n=args.n, frames=args.frames,
                   K=args.K, D=args.D, Hs=args.Hs, two_pass=args.twopass,
                   e3d_mean=float(e3d.mean()), bias_sweep=sweep, best=best)
        print(json.dumps(dict(BEST_BIAS=best, e3d=round(float(e3d.mean()), 2))))
        if args.out:
            json.dump(res, open(args.out, 'w'))
        return
    stu = deploy(args.bias0)
    res = dict(net=args.net, nparams=blob.get('nparams'), n=args.n, frames=args.frames,
               K=args.K, D=args.D, Hs=args.Hs, lookahead=args.lookahead,
               two_pass=args.twopass, bias0=args.bias0,
               student_mean=float(stu.mean()), student_se=float(stu.std(ddof=1)/np.sqrt(len(stu))),
               e3d_mean=float(e3d.mean()))
    if not args.skip_planner:
        plan, _ = pp.run_planner(seeds, args.frames, device, args.K, args.H, args.D)
        res['planner_mean'] = float(plan.mean())
        res['ratio_student_planner'] = float(stu.mean() / plan.mean())
        res['student_per'] = stu.astype(float).round(0).tolist()
        res['planner_per'] = plan.astype(float).round(0).tolist()
    print(json.dumps({k: v for k, v in res.items() if not k.endswith('_per')}))
    if args.out:
        json.dump(res, open(args.out, 'w'))


if __name__ == '__main__':
    main()
