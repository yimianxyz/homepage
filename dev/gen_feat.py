"""Generate a value-net training dataset (engineered features + per-candidate
true H-frame gain) by running the planner with feature logging.

Use HELD-OUT seeds (default 210000+) distinct from the eval bank (200000-200255)
so the value net never trains on eval scenes. Writes a .pt with feat/ctx/gain.

  python3 gen_feat.py --n 256 --seedStart 210000 --frames 5000 --K 16 --H 120 \
      --D 8 --device cuda --weights ../js/predator_weights.json --out ds_feat_train.pt
"""
import argparse, json, time
import torch
import feat_planner as fp
import planner_probe as pp
import sim_torch as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seedStart', type=int, default=210000)
    ap.add_argument('--frames', type=int, default=5000)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--twopass', action='store_true')
    ap.add_argument('--fast_twopass', action='store_true',
                    help='use the fast vectorized 2x-accel two-pass (data-gen speed)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='../js/predator_weights.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    pp.TWO_PASS = args.twopass
    st.FAST_TWO_PASS = args.fast_twopass
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    seeds = list(range(args.seedStart, args.seedStart + args.n))
    t0 = time.time()
    feat, ctx, gain, cats = fp.run_log_feat(seeds, args.frames, device, args.K, args.H, args.D)
    blob = dict(feat=torch.from_numpy(feat), ctx=torch.from_numpy(ctx),
                gain=torch.from_numpy(gain), planner_catches=cats.tolist(),
                meta=dict(n=args.n, seedStart=args.seedStart, frames=args.frames,
                          K=args.K, H=args.H, D=args.D, two_pass=args.twopass,
                          fc=fp.FC, fctx=fp.FCTX))
    torch.save(blob, args.out)
    print(json.dumps(dict(saved=args.out, rows=int(feat.shape[0]), K=args.K,
                          planner_mean=float(cats.mean()), secs=round(time.time() - t0, 1))))


if __name__ == '__main__':
    main()
