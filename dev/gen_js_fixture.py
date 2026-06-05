"""Generate a JS-parity fixture for the cheap ballistic deploy policy.

Dumps, for a few real decision states: the RAW sim state (boids+predator) plus
the expected candidate_features (19), ctx (4), value-net V (16), ballistic
(t_catch,bmin,caught), and the kr=1 chosen target. The JS port must reproduce
these from the raw state alone. Mirrors run_value_lookahead_cheap's per-decision
math (prune_by=ball, K_roll=1) up to candidate scoring (the rollout/bootstrap is
verified separately via the existing rolloutFlat selftest).
"""
import json
import numpy as np
import torch

import planner_probe as pp
import feat_planner as fp
import sim_torch as st
from sim_torch import Sim
from train_value import ValueNet


class Deploy(torch.nn.Module):
    def __init__(self, blob):
        super().__init__()
        self.m = ValueNet(blob['fc'], blob['fctx'], blob['hidden'], blob['depth'])
        self.m.load_state_dict(blob['state']); self.m.eval()
        self.fmu = blob['fmu']; self.fsd = blob['fsd']
        self.xmu = blob['xmu']; self.xsd = blob['xsd']

    def forward(self, feat, ctx):
        return self.m((feat - self.fmu) / self.fsd, (ctx - self.xmu) / self.xsd)


def main():
    dev = 'cpu'
    pp.WEIGHTS = st.load_weights('../js/predator_weights.json', device=dev)
    blob = torch.load('/tmp/net_v2_absval.pt', map_location='cpu')
    model = Deploy(blob)
    K, D = 16, 8
    sim = Sim(seeds=[200000], weights=pp.WEIGHTS, device=dev,
              auto_target='evolved', auto_target_opts=dict(pp.E3D), two_pass=pp.TWO_PASS)
    fixtures = []
    held = None
    for f in range(0, 200):
        if f % D == 0 and 24 <= f:   # skip warmup, capture a few mid-episode states
            cand = pp._candidate_targets(sim, K)            # (1,K,2)
            feat, ctx = fp.candidate_features(sim, cand)     # (1,K,19),(1,4)
            with torch.no_grad():
                v = model(feat, ctx)                         # (1,K)
            balli = feat[0, :, [16, 17, 18]]                 # t_catch,bmin,caught (already norm)
            pscore = (feat[0, :, 18] - feat[0, :, 16])
            top1 = int(pscore.argmax())
            n = int(sim.boid_alive[0].sum())
            fixtures.append(dict(
                frame=f,
                state=dict(
                    px=float(sim.pred_pos[0, 0]), py=float(sim.pred_pos[0, 1]),
                    pvx=float(sim.pred_vel[0, 0]), pvy=float(sim.pred_vel[0, 1]),
                    psize=float(sim.pred_size[0]),
                    bx=sim.boid_pos[0, :, 0][sim.boid_alive[0]].tolist(),
                    by=sim.boid_pos[0, :, 1][sim.boid_alive[0]].tolist(),
                    bvx=sim.boid_vel[0, :, 0][sim.boid_alive[0]].tolist(),
                    bvy=sim.boid_vel[0, :, 1][sim.boid_alive[0]].tolist(),
                    n_alive=n,
                ),
                cand=cand[0].tolist(),
                feat=feat[0].tolist(),
                ctx=ctx[0].tolist(),
                v=v[0].tolist(),
                ballistic=balli.tolist(),
                top1_ballistic=top1,
            ))
            if len(fixtures) >= 4:
                break
            held = cand[0, v[0].argmax()][None]
        if held is None:
            held = pp._candidate_targets(sim, K)[:, 0]
        pp._step_with_target(sim, held.expand(1, 2).contiguous() if held.dim() == 1 else held)
    out = dict(
        net=dict(fc=blob['fc'], fctx=blob['fctx'], hidden=blob['hidden'], depth=blob['depth'],
                 fmu=blob['fmu'].tolist(), fsd=blob['fsd'].tolist(),
                 xmu=blob['xmu'].tolist(), xsd=blob['xsd'].tolist()),
        consts=dict(PS=fp._PS, VS=fp._VS, RHO=fp._RHO, HB=fp._HB,
                    PRED_MAX_SPEED=float(st.PREDATOR_MAX_SPEED),
                    PRED_MAX_FORCE=float(st.PREDATOR_MAX_FORCE)),
        fixtures=fixtures,
    )
    json.dump(out, open('js_fixture.json', 'w'))
    print('wrote js_fixture.json with', len(fixtures), 'decisions; n_alive=',
          [fx['state']['n_alive'] for fx in fixtures])
    print('top1_ballistic per decision:', [fx['top1_ballistic'] for fx in fixtures])


if __name__ == '__main__':
    main()
