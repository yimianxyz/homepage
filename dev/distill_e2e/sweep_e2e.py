"""Architecture/loss sweep: train many E2ENets, rank by validation patrol+chase
angular error (the behavioural bottleneck), save each. Fast (train ~secs on L4).
Behavioural eval (slow) is run separately on the top survivors.

  python3 sweep_e2e.py --epochs 80 --dirw 1.0 --device cuda
"""
import argparse, json, os, sys, subprocess

ARCHS = ['', '16', '32', '64', '128', '32,32', '64,64', '64,32', '128,64']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--dirw', type=float, default=1.0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for h in ARCHS:
        tag = 'lin' if h == '' else 'h' + h.replace(',', 'x')
        cmd = [sys.executable, '-u', os.path.join(here, 'train_e2e.py'),
               '--hidden', h, '--epochs', str(args.epochs), '--dirw', str(args.dirw),
               '--device', args.device, '--tag', tag, '--quiet']
        out = subprocess.run(cmd, capture_output=True, text=True)
        try:
            meta = json.loads(out.stdout[out.stdout.index('{'):out.stdout.rindex('}') + 1])
            v = meta['val']
            rows.append((tag, meta['n_params'], v['ang_chase'], v['ang_patrol'],
                         v['ang_med'], v['mse']))
        except Exception:
            print('FAIL', tag, out.stdout[-400:], out.stderr[-400:])
    print(f"\n{'arch':>8} {'params':>7} {'ang_chase':>10} {'ang_patrol':>11} {'ang_med':>8} {'mse':>10}")
    for r in sorted(rows, key=lambda x: x[3]):
        print(f"{r[0]:>8} {r[1]:>7} {r[2]:>10.2f} {r[3]:>11.2f} {r[4]:>8.2f} {r[5]:>10.3e}")


if __name__ == '__main__':
    main()
