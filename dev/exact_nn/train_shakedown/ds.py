#!/usr/bin/env python3
"""Dataset/loader for EXACT-NN plan-decision records (SPEC 6.4 schema).

JSONL.gz shards (line 0 = shard header) -> packed numpy arrays -> torch
tensors. Carries the derived-config vector through to the model inputs
(omitting it aliases labels across devices -- SPEC section 1) and computes the
candidate coordinate-dedup equivalence classes (SPEC section 3: candidates with
bitwise-equal (x,y) are ONE class; canonical id = lowest member index).

Split is by seed (game id) range: --val-seed-min style threshold, never by
record index, so all plans of one game land on one side.

In-memory packing is the shakedown path (fits 1e6 decisions ~ 10 GB host RAM;
the 1e7 path shards the same pack() per file and memmaps -- noted in REPORT).
"""
import glob, gzip, json, os
import numpy as np

MAXB = 120   # max boids carried in the packed tensor (schema: <=120)
K = 16
NFEAT = 19   # cp_features per-candidate dim (fc=19; +4 shared ctx = 23 net input)
CP_PS, CP_VS = 200.0, 6.0


def iter_shard_paths(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, '*.jsonl.gz')))


def pack_shard(path):
    """One JSONL.gz shard -> (header, dict of numpy arrays)."""
    cols = {k: [] for k in ('seed', 'n', 'boids', 'bmask', 'pred', 'cfg', 'cands',
                            'cand_kind', 'cand_bref', 'feat', 'ctx', 'vprior',
                            'pidx', 'rolled', 'score', 'bi')}
    with gzip.open(path, 'rt') as f:
        header = json.loads(f.readline())
        assert header.get('shard_header'), '%s: missing shard header line' % path
        for line in f:
            r = json.loads(line)
            b = np.asarray(r['boids'], dtype=np.float64)        # (n,4)
            n = b.shape[0]
            assert n <= MAXB, 'record with %d boids > MAXB' % n
            bp = np.zeros((MAXB, 4)); bp[:n] = b
            m = np.zeros(MAXB, dtype=bool); m[:n] = True
            p = r['pred']
            src = np.asarray(r['cand_src'], dtype=np.int64)     # (16,2)
            cols['seed'].append(r['seed'])
            cols['n'].append(n)
            cols['boids'].append(bp)
            cols['bmask'].append(m)
            cols['pred'].append([p['x'], p['y'], p['vx'], p['vy'], p['size']])
            c = r['cfg']
            cols['cfg'].append([c['W'], c['Hc'], c['PREDATOR_RANGE'], c['NUM_BOIDS']])
            cols['cands'].append(np.asarray(r['cands'], dtype=np.float64))
            cols['cand_kind'].append(src[:, 0])
            cols['cand_bref'].append(src[:, 1])
            cols['feat'].append(np.asarray(r['feat'], dtype=np.float64))
            cols['ctx'].append(np.asarray(r['ctx'], dtype=np.float64))
            cols['vprior'].append(np.asarray(r['vprior'], dtype=np.float64))
            cols['pidx'].append(np.asarray(r['pidx'], dtype=np.int64))
            # JS JSON.stringify(non-finite) -> null. Prod's extermination path
            # (rollout kills all boids -> bootstrap -Infinity) makes rolled /
            # score entries -Infinity; restore them as -inf here. (NaN would
            # also arrive as null; -inf is the only non-finite prod emits in
            # these fields -- flagged as an oracle schema question.)
            cols['rolled'].append(np.asarray([-np.inf if v is None else v
                                              for v in r['rolled']], dtype=np.float64))
            cols['score'].append(np.asarray([-np.inf if v is None else v
                                             for v in r['score']], dtype=np.float64))
            cols['bi'].append(r['bi'])
    out = {k: np.asarray(v) for k, v in cols.items()}
    out['seed'] = out['seed'].astype(np.int64)
    out['n'] = out['n'].astype(np.int64)
    out['bi'] = out['bi'].astype(np.int64)
    return header, out


def dedup_classes(cands):
    """(B,16,2) float64 -> (B,16) int64 class ids: class = lowest candidate
    index among bitwise-equal (x,y). Equality is on the raw float64 bit
    pattern (void view), per SPEC section 3 -- NOT approximate."""
    B = cands.shape[0]
    raw = np.ascontiguousarray(cands).view('V16').reshape(B, K)  # 2*f64 = 16 bytes
    cls = np.empty((B, K), dtype=np.int64)
    for b in range(B):
        first = {}
        row = raw[b]
        for k in range(K):
            key = row[k].tobytes()
            if key not in first:
                first[key] = k
            cls[b, k] = first[key]
    return cls


def load_packed(data_dir, seed_min=None, seed_max=None, max_records=None):
    """Load + concat all shards whose seed range intersects [seed_min, seed_max],
    then filter records by seed. Returns dict of numpy arrays + dedup classes."""
    parts, headers = [], []
    total = 0
    for path in iter_shard_paths(data_dir):
        hdr, arr = pack_shard(path)
        sel = np.ones(arr['seed'].shape[0], dtype=bool)
        if seed_min is not None:
            sel &= arr['seed'] >= seed_min
        if seed_max is not None:
            sel &= arr['seed'] <= seed_max
        if not sel.any():
            continue
        arr = {k: v[sel] for k, v in arr.items()}
        parts.append(arr); headers.append(hdr)
        total += arr['seed'].shape[0]
        if max_records is not None and total >= max_records:
            break
    if not parts:
        raise RuntimeError('no records in %s for seed range [%s,%s]'
                           % (data_dir, seed_min, seed_max))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if max_records is not None:
        out = {k: v[:max_records] for k, v in out.items()}
    out['cls'] = dedup_classes(out['cands'])
    out['bi_cls'] = out['cls'][np.arange(out['bi'].shape[0]), out['bi']]
    out['headers'] = headers
    return out


def build_inputs(arr):
    """Model-input featurization (float32) from the packed float64 arrays.
    Positions are predator-relative / CP_PS; velocities / CP_VS; the config
    vector rides in the global feature. Targets stay float64 (cast at loss
    time to the head dtype)."""
    pred = arr['pred']                                   # (B,5) x y vx vy size
    px, py = pred[:, 0:1], pred[:, 1:2]
    boids = arr['boids']                                 # (B,120,4)
    bmask = arr['bmask']
    bt = np.stack([(boids[:, :, 0] - px) / CP_PS, (boids[:, :, 1] - py) / CP_PS,
                   boids[:, :, 2] / CP_VS, boids[:, :, 3] / CP_VS], axis=2)
    bt[~bmask] = 0.0
    cfg = arr['cfg']                                     # W Hc R NB
    n = arr['n'].astype(np.float64)
    glob = np.stack([pred[:, 2] / CP_VS, pred[:, 3] / CP_VS, pred[:, 4] / 20.0,
                     n / 120.0, n / cfg[:, 3],
                     cfg[:, 0] / 1680.0, cfg[:, 1] / 1680.0,
                     cfg[:, 2] / 80.0, cfg[:, 3] / 120.0,
                     px[:, 0] / cfg[:, 0], py[:, 0] / cfg[:, 1]], axis=1)
    cands = arr['cands']
    kind = arr['cand_kind']                              # (B,16) 0/1/2
    onehot = np.eye(3)[kind]                             # (B,16,3)
    ct = np.concatenate([
        np.stack([(cands[:, :, 0] - px) / CP_PS, (cands[:, :, 1] - py) / CP_PS], axis=2),
        arr['feat'],                                     # 19, already normalized
        arr['vprior'][:, :, None] / 3.0,
        onehot,
    ], axis=2)                                           # (B,16,25)
    return {
        'boid_tok': bt.astype(np.float32), 'bmask': bmask,
        'glob': glob.astype(np.float32), 'cand_tok': ct.astype(np.float32),
        # targets / bookkeeping (float64 -- SPEC: labels are f64 JS numbers)
        'vprior': arr['vprior'], 'pidx': arr['pidx'], 'rolled': arr['rolled'],
        'score': arr['score'], 'bi': arr['bi'], 'cls': arr['cls'],
        'bi_cls': arr['bi_cls'], 'seed': arr['seed'],
    }

GLOB_DIM = 11
CAND_DIM = 2 + NFEAT + 1 + 3   # 25


def to_torch(inp, device):
    import torch
    out = {}
    for k, v in inp.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device)
    return out


class BatchSampler:
    """Random-index batches over GPU-resident packed tensors (shakedown path)."""
    def __init__(self, tensors, batch_size, seed=0):
        import torch
        self.t = tensors
        self.bs = batch_size
        self.n = tensors['bi'].shape[0]
        self.g = torch.Generator(device='cpu').manual_seed(seed)

    def state_dict(self):
        return {'g': self.g.get_state()}

    def load_state_dict(self, sd):
        self.g.set_state(sd['g'])

    def next(self):
        import torch
        idx = torch.randint(0, self.n, (self.bs,), generator=self.g)
        idx = idx.to(self.t['bi'].device)
        return {k: v[idx] for k, v in self.t.items()}
