import torch
for split, n in [("train", 50000), ("val", 10000)]:
    d = torch.load(f"setds_densA_{split}.pt", map_location="cpu")
    tot = d["feats"].shape[0]
    keys = [k for k, v in d.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == tot]
    idx = torch.randperm(tot)[:n]
    out = dict(d)
    for k in keys:
        out[k] = d[k][idx]
    torch.save(out, f"setds_densAsm_{split}.pt")
    print(split, "sliced", tot, "->", n, "keys=", keys)
