"""Reconcile the 'base' (deployed policy) score across eval paths at identical
seeds, to resolve the residual-path (7.84) vs target-path (8.12) discrepancy."""
import sys, os, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights
from e2e_ppo import PPOSim, ActorCritic, OBS_AUG, NC_OPTS

dev = 'cuda'
SEEDS = list(range(5000, 5000 + 512))
FRAMES = 1500
bw = load_weights('js/predator_weights.json', device=dev)


def run(env, frames):
    for _ in range(frames):
        env.step()
    return env.catches.float().mean().item()


# A) PPOSim residual base (scale 0): policy zeroed
ac = ActorCritic(obs_dim=OBS_AUG, hidden=64).to(dev)
envA = PPOSim(SEEDS, device=dev, residual=True, base_weights=bw, resid_scale=0.0)
with torch.no_grad():
    for _ in range(FRAMES):
        mu, _ = ac(envA.current_obs()); envA._action = mu.double(); envA.step()
A = envA.catches.float().mean().item()

# B) PPOSim target base (offset 0)
envB = PPOSim(SEEDS, device=dev, target_residual=True, base_weights=bw, target_scale=0.0)
with torch.no_grad():
    for _ in range(FRAMES):
        mu, _ = ac(envB.current_obs()); envB._action = mu.double(); envB.step()
B = envB.catches.float().mean().item()

# C) Canonical Sim, parallel boids (sequential=False), nearest_cluster
simC = Sim(seeds=SEEDS, weights=bw, device=dev, sequential=False,
           auto_target='nearest_cluster', auto_target_opts=NC_OPTS)
C = run(simC, FRAMES)

# D) Canonical Sim, sequential boids (faithful Oracle), nearest_cluster
simD = Sim(seeds=SEEDS, weights=bw, device=dev, sequential=True,
           auto_target='nearest_cluster', auto_target_opts=NC_OPTS)
D = run(simD, FRAMES)

print(json.dumps({'A_residual_base': A, 'B_target_base': B,
                  'C_canonical_parallel': C, 'D_canonical_sequential': D}))
