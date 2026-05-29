"""Evaluate a PPO checkpoint (ActorCritic best.pt) deterministically.

Runs the mean action (no exploration noise) over N fresh seeds for `frames`,
matching the training holdout but on a held-out seed block and any horizon.

  python3 dev/e2e_ppo_eval.py --ckpt ~/ckpt/ppo_aug/best.pt --augment \
      --seeds 256 --seedStart 5000 --frames 1500
"""
import argparse, json, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_ppo import PPOSim, ActorCritic, OBS_RAW, OBS_AUG, PREDATOR_MAX_FORCE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--seeds', type=int, default=256)
    p.add_argument('--seedStart', type=int, default=5000)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    hidden = ck['hidden']
    obs_dim = OBS_AUG if a.augment else OBS_RAW
    ac = ActorCritic(obs_dim=obs_dim, hidden=hidden).to(a.device)
    ac.load_state_dict(ck['state'])
    ac.eval()
    seeds = list(range(a.seedStart, a.seedStart + a.seeds))
    env = PPOSim(seeds, device=a.device, augment=a.augment)
    with torch.no_grad():
        for _ in range(a.frames):
            mu, _ = ac(env.current_obs())
            env._action = mu.double() * PREDATOR_MAX_FORCE
            env.step()
    per = env.catches.float()
    m = per.mean().item(); sd = per.std().item(); se = sd / (a.seeds ** 0.5)
    print(json.dumps({'ckpt': a.ckpt, 'augment': a.augment, 'seeds': a.seeds,
                      'frames': a.frames, 'mean_catches': m, 'sd': sd, 'se': se}))


if __name__ == '__main__':
    main()
