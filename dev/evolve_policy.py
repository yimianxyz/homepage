#!/usr/bin/env python3
"""AlphaEvolve-style policy search: evolve the cheap predator's continuous params
against the device-mix catch metric, evaluated on the 3-VM fleet (dev/fleet_eval.py).

The shipped EVOLVED_PATROL params + POLICY_R were CEM-tuned on the 1680 square; this
re-tunes them on the realistic phone/iPad/laptop mix. Simple CMA-lite ES: sample a
population around a Gaussian, eval each candidate on the device mix via the fleet,
recombine from the elite fraction.

  python3 dev/evolve_policy.py --pop 8 --gens 6 --seeds 48 --fixed '{"wrapAuto":true,"K_roll":6,"Hs":120,"D":12}'
"""
import sys, json, subprocess, math, argparse, os

HERE = os.path.dirname(os.path.abspath(__file__))

# param name -> (init, lo, hi)  (init = shipped EVOLVED_PATROL + POLICY_R)
PARAMS = {
    "cluster_r":   (178.09, 60.0, 400.0),
    "dens_pow":    (2.373,  0.5,  5.0),
    "reach_scale": (1515.0, 200.0, 4000.0),
    "sharp":       (9.25,   1.0,  20.0),
    "lead_scale":  (0.454,  0.0,  2.0),
    "lead_max":    (230.6,  20.0, 600.0),
    "nbhd":        (0.461,  0.0,  1.0),
    "POLICY_R":    (80.0,   30.0, 250.0),
}
PNAMES = list(PARAMS.keys())

# device mix (W,H,weight) — phone-heavy, then laptop, then iPad
DEVICES = [(390, 844, 0.45), (1440, 900, 0.30), (820, 1180, 0.25)]


def vec_to_config(vec, fixed):
    patrol = {}
    cfg = dict(fixed)
    for i, name in enumerate(PNAMES):
        if name == "POLICY_R":
            cfg["POLICY_R"] = vec[i]
        else:
            patrol[name] = vec[i]
    cfg["patrol"] = patrol
    return cfg


def clip(vec):
    return [min(max(vec[i], PARAMS[n][1]), PARAMS[n][2]) for i, n in enumerate(PNAMES)]


def eval_population(pop, fixed, seeds, frames, seed0):
    """Build one big jobs list (pop x devices), call fleet_eval, return weighted score per candidate."""
    jobs = []
    for ci, vec in enumerate(pop):
        cfg = vec_to_config(vec, fixed)
        for (W, H, w) in DEVICES:
            jobs.append(dict(label=f"c{ci}_{W}x{H}", policy="exp", config=cfg,
                             W=W, H=H, seeds=seeds, frames=frames, seed0=seed0))
    p = subprocess.run([sys.executable, os.path.join(HERE, "fleet_eval.py")],
                       input=json.dumps(jobs), capture_output=True, text=True, timeout=7200)
    res = json.loads(p.stdout)
    by = {r["label"]: r for r in res}
    scores = []
    for ci in range(len(pop)):
        s, wsum = 0.0, 0.0
        for (W, H, w) in DEVICES:
            r = by.get(f"c{ci}_{W}x{H}", {})
            m = r.get("mean")
            if m is None:
                m = 0.0
            s += w * m; wsum += w
        scores.append(s / wsum)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--fixed", default='{"wrapAuto":true}')
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out", default="/tmp/evolve_log.json")
    args = ap.parse_args()
    fixed = json.loads(args.fixed)

    # deterministic-ish sampling without Math.random: use a fixed LCG seeded by gen/idx
    state = [12345]
    def rnd():
        state[0] = (1103515245 * state[0] + 12345) & 0x7fffffff
        return state[0] / 0x7fffffff
    def gauss():
        u1 = max(rnd(), 1e-9); u2 = rnd()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    mu = [PARAMS[n][0] for n in PNAMES]
    sigma = [(PARAMS[n][2] - PARAMS[n][1]) * 0.15 for n in PNAMES]  # 15% of range

    # baseline (shipped params) score, as candidate 0 of gen 0
    log = {"params": PNAMES, "gens": []}
    best = {"score": -1, "vec": mu[:]}
    for g in range(args.gens):
        pop = [mu[:]]  # always include current mean
        for _ in range(args.pop - 1):
            pop.append(clip([mu[i] + sigma[i] * gauss() for i in range(len(mu))]))
        scores = eval_population(pop, fixed, args.seeds, args.frames, args.seed0)
        ranked = sorted(range(len(pop)), key=lambda i: -scores[i])
        elite = ranked[:args.elite]
        new_mu = [sum(pop[i][d] for i in elite) / len(elite) for d in range(len(mu))]
        new_sigma = []
        for d in range(len(mu)):
            var = sum((pop[i][d] - new_mu[d]) ** 2 for i in elite) / len(elite)
            floor = (PARAMS[PNAMES[d]][2] - PARAMS[PNAMES[d]][1]) * 0.03
            new_sigma.append(max(math.sqrt(var), floor))
        if scores[ranked[0]] > best["score"]:
            best = {"score": scores[ranked[0]], "vec": pop[ranked[0]][:]}
        gen_rec = dict(gen=g, best_score=round(scores[ranked[0]], 3),
                       mean_score=round(sum(scores) / len(scores), 3),
                       mu_score=round(scores[0], 3),
                       best_vec={PNAMES[i]: round(pop[ranked[0]][i], 3) for i in range(len(mu))})
        log["gens"].append(gen_rec)
        print(json.dumps(gen_rec), flush=True)
        mu, sigma = new_mu, new_sigma
    log["best"] = {"score": round(best["score"], 3),
                   "config": vec_to_config(best["vec"], fixed)}
    print("BEST", json.dumps(log["best"]), flush=True)
    json.dump(log, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
