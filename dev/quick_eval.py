"""Quick GPU eval of a patrol policy. Prints mean catches + timing.

  python3 dev/quick_eval.py --auto_target nearest_cluster --seeds 512 --frames 3000
  python3 dev/quick_eval.py --auto_target evolved --opts '{"sharp":12,"reach_scale":2000}' ...
"""
import argparse, json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sim import evaluate

NC = {"cluster_r": 150.0, "lead_scale": 0.4, "lead_max": 120.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--auto_target', default='nearest_cluster')
    p.add_argument('--opts', default=None, help='JSON dict of auto_target_opts')
    p.add_argument('--weights', default='js/predator_weights.json')
    p.add_argument('--seeds', type=int, default=512)
    p.add_argument('--seedStart', type=int, default=5000)
    p.add_argument('--frames', type=int, default=3000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no_graph', action='store_true')
    p.add_argument('--sequential', action='store_true')
    a = p.parse_args()
    opts = json.loads(a.opts) if a.opts else (NC if a.auto_target == 'nearest_cluster' else {})
    t = time.time()
    r = evaluate(a.weights, seeds=range(a.seedStart, a.seedStart + a.seeds),
                 frames=a.frames, device=a.device, use_graph=not a.no_graph,
                 sequential=a.sequential, auto_target=a.auto_target, auto_target_opts=opts)
    per = r['per_seed_catches']
    n = len(per); mean = r['mean_catches']
    sd = (sum((c - mean) ** 2 for c in per) / n) ** 0.5
    print(json.dumps({'auto_target': a.auto_target, 'opts': opts, 'seeds': a.seeds,
                      'frames': a.frames, 'mean_catches': mean, 'sd': sd,
                      'se': sd / (n ** 0.5), 'elapsed_s': round(time.time() - t, 1)}))


if __name__ == '__main__':
    main()
