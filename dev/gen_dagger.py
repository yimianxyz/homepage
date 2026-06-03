"""DAgger relabel pass: run the current student, log the states IT visits with
true planner gains. Output appends to the training set to fix distribution shift.

  python3 gen_dagger.py --net net_value_v1.pt --n 64 --seedStart 211000 \
      --frames 5000 --K 16 --H 120 --D 8 --bias0 0 --device cuda --out ds_dagger1.pt
"""
import argparse, json, time
import torch
import feat_planner as fp
import planner_probe as pp
import sim_torch as st
from eval_value import Deploy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--seedStart', type=int, default=211000)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--bias0', type=float, default=0.0)
    ap.add_argument('--twopass', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='../js/predator_weights.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.TWO_PASS = args.twopass
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    blob = torch.load(args.net, map_location='cpu')
    model = Deploy(blob, device)
    seeds = list(range(args.seedStart, args.seedStart + args.n))
    t0 = time.time()
    feat, ctx, gain, cats = fp.run_dagger_feat(seeds, args.frames, device, model,
                                               args.K, args.H, args.D, bias0=args.bias0)
    torch.save(dict(feat=torch.from_numpy(feat), ctx=torch.from_numpy(ctx),
                    gain=torch.from_numpy(gain), student_catches=cats.tolist(),
                    meta=dict(dagger=True, net=args.net, n=args.n, seedStart=args.seedStart,
                              frames=args.frames, K=args.K, H=args.H, D=args.D,
                              bias0=args.bias0, fc=fp.FC, fctx=fp.FCTX)), args.out)
    print(json.dumps(dict(saved=args.out, rows=int(feat.shape[0]),
                          student_mean=float(cats.mean()), secs=round(time.time() - t0, 1))))


if __name__ == '__main__':
    main()
