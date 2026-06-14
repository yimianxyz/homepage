#!/usr/bin/env python3
"""Dump torch MoE forward on sample decisions -> JSON, for the JS parity check
(moe_parity.js). Confirms moeForward.js reproduces the trained model bit-closely.
  python3 moe_parity.py --pack pack --ckpt moe_ckpt.pt --n 200 --out parity.json
"""
import argparse, json
import numpy as np
import torch
from moe_model import MoEPolicy, PLANNER_DIM, ENDGAME_DIM, NSLOT
from moe_train import load_pack, Pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--out', default='parity.json')
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu')
    model = MoEPolicy(d=ck['args']['d'])
    model.load_state_dict(ck['model']); model.eval()
    d, pm, em = load_pack(args.pack)
    pk = Pack(d, torch.device('cpu'))
    g = torch.Generator().manual_seed(123)

    samples = []
    # planner samples
    pidx = pk.p_va[torch.randint(0, len(pk.p_va), (args.n,), generator=g).numpy()]
    eidx = pk.e_va[torch.randint(0, len(pk.e_va), (args.n,), generator=g).numpy()]
    for which, idx in (('planner', pidx), ('endgame', eidx)):
        for j in idx[:args.n]:
            if which == 'planner':
                b = pk.assemble(torch.tensor([j]), torch.empty(0, dtype=torch.long))
            else:
                b = pk.assemble(torch.empty(0, dtype=torch.long), torch.tensor([j]))
            with torch.no_grad():
                logit, gg = model(b['pb'], b['eb'], b['gf'], b['sv'], b['pv'], b['ev'])
            samples.append(dict(
                regime=which,
                planner_block=b['pb'][0].tolist(), endgame_block=b['eb'][0].tolist(),
                gate_feat=b['gf'][0].tolist(), slot_valid=b['sv'][0].int().tolist(),
                p_valid=b['pv'][0].int().tolist(), e_valid=b['ev'][0].int().tolist(),
                torch_logit=logit[0].tolist(), torch_g=float(gg.reshape(-1)[0]),
            ))
    json.dump(samples, open(args.out, 'w'))
    print('wrote', args.out, len(samples), 'samples')


if __name__ == '__main__':
    main()
