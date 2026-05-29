"""Dump sim_torch 'evolved' patrol state + target for JS-parity checking.

Runs the sim a few frames, then for every env currently in PATROL mode (no prey
in PREDATOR_RANGE, some boid alive) records: predator position, the list of
ALIVE boids (pos+vel) — exactly what predator.js sees — and the patrol target
sim_torch computes. dev/check_parity.js recomputes the target in JS and asserts
they match. Writes dev/parity_state.json."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights, PREDATOR_RANGE

OPTS = {
    "cluster_r": 178.09, "dens_pow": 2.373, "reach_scale": 1515.0,
    "sharp": 9.25, "lead_scale": 0.454, "lead_max": 230.6, "nbhd": 0.461,
    "momentum": 0.0,
}
WARMUP = 120

def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    w = load_weights('js/predator_weights.json', device=dev)
    seeds = list(range(50000, 50128))
    sim = Sim(seeds=seeds, weights=w, device=dev, sequential=False,
              auto_target='evolved', auto_target_opts=OPTS)
    # step WARMUP frames so clusters form and some boids get caught
    for _ in range(WARMUP):
        sim.step()

    dx = sim.boid_pos[..., 0] - sim.pred_pos[:, None, 0]
    dy = sim.boid_pos[..., 1] - sim.pred_pos[:, None, 1]
    d = torch.sqrt(dx * dx + dy * dy)
    inf = torch.full_like(d, float('inf'))
    d_masked = torch.where(sim.boid_alive, d, inf)
    any_in_range = (d_masked < PREDATOR_RANGE).any(dim=1)
    any_alive = sim.boid_alive.any(dim=1)
    cond = (~any_in_range) & any_alive

    # capture target sim_torch computes. Seed pred_auto with the predator
    # position (a finite value, like the real sim) so the momentum term
    # (0*prev) stays finite; momentum=0 makes the prev value irrelevant anyway.
    sim.pred_auto[:, 0] = sim.pred_pos[:, 0]
    sim.pred_auto[:, 1] = sim.pred_pos[:, 1]
    sim._update_auto_target()
    tgt = sim.pred_auto.detach().cpu().numpy()

    cases = []
    bp = sim.boid_pos.detach().cpu().numpy()
    bv = sim.boid_vel.detach().cpu().numpy()
    ba = sim.boid_alive.detach().cpu().numpy()
    pp = sim.pred_pos.detach().cpu().numpy()
    condn = cond.detach().cpu().numpy()
    import math
    for e in range(sim.B):
        if not condn[e]:
            continue
        if not (math.isfinite(tgt[e, 0]) and math.isfinite(tgt[e, 1])):
            continue
        boids = []
        for i in range(sim.N):
            if ba[e, i]:
                boids.append({"position": {"x": float(bp[e, i, 0]), "y": float(bp[e, i, 1])},
                              "velocity": {"x": float(bv[e, i, 0]), "y": float(bv[e, i, 1])}})
        cases.append({
            "seed": seeds[e],
            "pred": {"x": float(pp[e, 0]), "y": float(pp[e, 1])},
            "n_alive": len(boids),
            "boids": boids,
            "target": {"x": float(tgt[e, 0]), "y": float(tgt[e, 1])},
        })
    out = {"opts": OPTS, "warmup": WARMUP, "cases": cases}
    with open('dev/parity_state.json', 'w') as f:
        json.dump(out, f)
    print(f"dumped {len(cases)} patrol-mode cases (of {sim.B} envs) to dev/parity_state.json")

if __name__ == '__main__':
    main()
