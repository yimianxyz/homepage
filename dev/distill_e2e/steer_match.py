"""Fast steer-match eval — the new inner-loop metric for the >99% steering goal.

Loads a saved net + a matching val dataset, does ONE batched forward pass (no rollout),
and reports the SHARP direction-match statistics that the slow catch-count rollout hides:

    cos      = cosine(pred_force, production_force)   per sample
    reported : mean/median cos, %cos>0.99, %cos>0.999, median angle (deg)
    split    : patrol (d1>=80 or no boid) vs chase (d1<80)

Why cosine, not catches: the per-frame force is a DETERMINISTIC function of state; matching
it to >99% is a clean supervised target (universal approximation applies). Catch-count is the
chaotic downstream metric and is useless for fast iteration.

Supports both net families:
  --kind set  : set_net.SetNet  (feats/mask/pvel datasets, setds_*.pt)   [default if loadable]
  --kind grid : e2e_net.E2ENet  (obs datasets, dataset_*.pt)

Usage:
  python3 steer_match.py --net setnet_gate_d48a.pt --val setds_densA_val.pt --device cuda
  python3 steer_match.py --kind grid --net net_gb9h256-128_a.pt --val dataset_gb9_val.pt --device cuda
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')


def cos_stats(pred, tgt, mask):
    pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
    tn = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-12)
    cos = (pn * tn).sum(1).clamp(-1, 1)
    if mask is not None:
        cos = cos[mask]
    if cos.numel() == 0:
        return None
    ang = torch.rad2deg(torch.arccos(cos))
    relL2 = ((pred - tgt).norm(dim=1) / (tgt.norm(dim=1) + 1e-12))
    if mask is not None:
        relL2 = relL2[mask]
    return dict(n=int(cos.numel()),
                cos_mean=cos.mean().item(), cos_med=cos.median().item(),
                pct_gt99=(cos > 0.99).float().mean().item() * 100,
                pct_gt999=(cos > 0.999).float().mean().item() * 100,
                ang_med=ang.median().item(), ang_mean=ang.mean().item(),
                relL2_med=relL2.median().item())


def report(name, s):
    if s is None:
        print(f"  {name:8s}  (no samples)")
        return
    print(f"  {name:8s} n={s['n']:7d}  cos_med={s['cos_med']:.4f} cos_mean={s['cos_mean']:.4f}  "
          f"%>.99={s['pct_gt99']:5.1f} %>.999={s['pct_gt999']:5.1f}  "
          f"ang_med={s['ang_med']:5.2f}deg relL2_med={s['relL2_med']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--kind', default='set', choices=['set', 'grid'])
    ap.add_argument('--bs', type=int, default=16384)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = a.device

    va = torch.load(a.val, map_location='cpu')
    if a.kind == 'set':
        from set_e2e import load_setnet
        net, ck = load_setnet(a.net, dev)
        F_, M_, P_, Y_, D_ = va['feats'], va['mask'], va['pvel'], va['force'], va['d1']
        in_dim = F_.shape[-1]
        net_in = net.fmean.numel()
        assert in_dim == net_in, f"feat dim mismatch: val {in_dim} vs net {net_in} (density_radii?)"
        preds = []
        for j in range(0, F_.shape[0], a.bs):
            f = F_[j:j + a.bs].float().to(dev)
            m = M_[j:j + a.bs].float().to(dev)
            p = P_[j:j + a.bs].float().to(dev)
            with torch.no_grad():
                preds.append(net(f, m, p).float().cpu())
        pred = torch.cat(preds, 0)
        tgt = Y_.float()
        d1 = D_
        hdr = (f"net={os.path.basename(a.net)} mode={ck['mode']} pool={ck.get('pool')} "
               f"d={ck['d']} rho={ck['rho']} params={ck['meta']['params']} "
               f"density_radii={ck.get('density_radii')}")
    else:
        from e2e_net import E2ENet
        ck = torch.load(a.net, map_location='cpu')
        net = E2ENet(ck['in_dim'], hidden=tuple(ck['hidden']), act=ck['act'], head=ck['head']).to(dev)
        net.load_state_dict(ck['state_dict']); net.eval()
        X_, Y_, D_ = va['obs'], va['force'], va['d1']
        preds = []
        for j in range(0, X_.shape[0], a.bs):
            with torch.no_grad():
                preds.append(net(X_[j:j + a.bs].float().to(dev)).float().cpu())
        pred = torch.cat(preds, 0)
        tgt = Y_.float()
        d1 = D_
        hdr = f"net={os.path.basename(a.net)} hidden={ck['hidden']} head={ck['head']} in_dim={ck['in_dim']}"

    chase = (torch.isfinite(d1) & (d1 < 80.0))
    patrol = ~chase
    print(hdr)
    report('all', cos_stats(pred, tgt, None))
    report('patrol', cos_stats(pred, tgt, patrol))
    report('chase', cos_stats(pred, tgt, chase))
    print(f"  frac_chase={chase.float().mean().item():.3f}")


if __name__ == '__main__':
    main()
