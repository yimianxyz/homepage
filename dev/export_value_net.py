"""Export a trained ValueNet (.pt blob) to js/value_net.json for the browser.

Format matches js/cheap_planner.js cp_value: {fc,fctx,hidden,depth, fmu,fsd,
xmu,xsd, layers:[{w,b}x3]} where layers are the 3 Linear layers (GELU between).
  python3 export_value_net.py <net.pt> <out.json>
"""
import json
import sys
import torch
from train_value import ValueNet


def main():
    src, out = sys.argv[1], sys.argv[2]
    blob = torch.load(src, map_location='cpu')
    m = ValueNet(blob['fc'], blob['fctx'], blob['hidden'], blob['depth'])
    m.load_state_dict(blob['state']); m.eval()
    layers = []
    for mod in m.net:
        if isinstance(mod, torch.nn.Linear):
            layers.append({'w': mod.weight.detach().tolist(), 'b': mod.bias.detach().tolist(),
                           'in': mod.in_features, 'out': mod.out_features})
    d = {
        'fc': blob['fc'], 'fctx': blob['fctx'], 'hidden': blob['hidden'], 'depth': blob['depth'],
        'fmu': blob['fmu'].tolist(), 'fsd': blob['fsd'].tolist(),
        'xmu': blob['xmu'].tolist(), 'xsd': blob['xsd'].tolist(),
        'layers': layers,
    }
    json.dump(d, open(out, 'w'))
    print('wrote', out, 'layers', [(L['in'], L['out']) for L in layers])


if __name__ == '__main__':
    main()
