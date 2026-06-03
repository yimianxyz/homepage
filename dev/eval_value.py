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

    stu = fp.run_value_student(seeds, args.frames, device, model, args.K, args.D, Hs=args.Hs)
    e3d = pp.run_e3d(seeds, args.frames, device)
    res = dict(net=args.net, nparams=blob.get('nparams'), n=args.n, frames=args.frames,
               K=args.K, D=args.D, Hs=args.Hs, two_pass=args.twopass,
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
