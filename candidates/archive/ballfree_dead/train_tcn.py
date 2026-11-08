#!/usr/bin/env python3
"""Ball-free dead/alive TCN (dilated 1-D convolutions over 5 Hz player-only feature sequences).

    python -I candidates/ballfree_dead/train_tcn.py --seq <dir> --folds <dir> [--out <dir>]

torch only (never imported together with lightgbm). Leave-one-DFL-game-out: for each held-out
game gi a model is trained on the other six (all three degradation seeds) and saved as
<folds>/fold_tcn_<gi>.npz; the held-out loss is recorded every EVAL_EVERY iterations. The final
all-games model (tcn_bfd.npz) is trained for the argmin of the mean held-out curve; the fold
files are trained for MAX_ITERS but saved at that same iteration count (a second pass is not
needed: every fold keeps a snapshot at each evaluation point and the chosen one is written).
Inference is plain numpy (tcn_np.py). Deterministic: fixed seeds, 2 threads.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
C = 32
DIL = (1, 2, 4, 8, 16, 32, 64, 128)       # receptive field +-255 steps = +-51 s at 5 Hz
CROP, BATCH, LR, WD, DROP = 1024, 16, 3e-3, 1e-4, 0.1
MAX_ITERS, EVAL_EVERY = 800, 100


class TCN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.inp = nn.Conv1d(k, C, 1)
        self.convs = nn.ModuleList([nn.Conv1d(C, C, 3, dilation=d, padding=d) for d in DIL])
        self.mix = nn.ModuleList([nn.Conv1d(C, C, 1) for _ in DIL])
        self.head = nn.Conv1d(C, 1, 1)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):                     # x (B, K, T) -> logits (B, T)
        h = self.inp(x)
        for conv, mix in zip(self.convs, self.mix):
            h = h + mix(self.drop(torch.relu(conv(torch.relu(h)))))
        return self.head(torch.relu(h))[:, 0]


def load(seq):
    paths = sorted(glob.glob(os.path.join(seq, "DFL-MAT-*_*.npz")))
    matches = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in paths})
    data = []
    for p in paths:
        z = np.load(p)
        data.append((matches.index(os.path.basename(p).rsplit("_", 1)[0]), z["X"], z["y"]))
    return data, matches, [str(n) for n in z["names"]]


def norm_stats(data):
    X = np.concatenate([x for _, x, _ in data])
    med = np.nanmedian(X, 0)
    q1, q3 = np.nanpercentile(X, 25, 0), np.nanpercentile(X, 75, 0)
    scale = np.where(q3 - q1 > 1e-6, q3 - q1, np.nanstd(X, 0) + 1e-3)
    return med.astype(np.float32), scale.astype(np.float32)


def prep(x, med, scale):
    z = (x - med) / scale
    return np.clip(np.nan_to_num(z, nan=0.0), -5, 5).astype(np.float32)


def export(model, med, scale, names, iters):
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    return dict(sd, med=med, scale=scale, dil=np.array(DIL), names=np.array(names),
                iters=np.array(iters))


def fit(train, val, k, iters_list, seed=0):
    """Train; snapshot (exported arrays) at every iteration in iters_list; val loss curve."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    med, scale = norm_stats(train)
    tr = [(prep(x, med, scale), y) for _, x, y in train]
    va = [(prep(x, med, scale), y) for _, x, y in val]
    model = TCN(k)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = max(iters_list)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.1)
    lens = np.array([len(y) for _, y in tr], float)
    curve, snaps = [], {}
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for it in range(1, total + 1):
        model.train()
        xb, yb = [], []
        for _ in range(BATCH):
            i = rng.choice(len(tr), p=lens / lens.sum())
            x, y = tr[i]
            s = int(rng.integers(0, len(y) - CROP))
            xb.append(x[s:s + CROP].T)
            yb.append(y[s:s + CROP])
        xb = torch.from_numpy(np.stack(xb))
        yb = np.stack(yb)
        mask = torch.from_numpy((yb != 3).astype(np.float32))
        tgt = torch.from_numpy((yb == 2).astype(np.float32))
        loss = (bce(model(xb), tgt) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % EVAL_EVERY == 0:
            model.eval()
            vl = []
            if va:
                with torch.no_grad():
                    num = den = 0.0
                    for x, y in va:
                        lo = model(torch.from_numpy(x.T[None]))[0]
                        m = torch.from_numpy((y != 3).astype(np.float32))
                        t = torch.from_numpy((y == 2).astype(np.float32))
                        num += float((bce(lo, t) * m).sum())
                        den += float(m.sum())
                    vl = num / den
            curve.append(vl)
            if it in iters_list:
                snaps[it] = export(model, med, scale, None, it)
    return curve, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    data, matches, names = load(a.seq)
    k = data[0][1].shape[1]
    os.makedirs(a.folds, exist_ok=True)
    evals = list(range(EVAL_EVERY, MAX_ITERS + 1, EVAL_EVERY))
    curves, snaps = [], []
    t0 = time.time()
    for gi in range(len(matches)):
        tr = [d for d in data if d[0] != gi]
        va = [d for d in data if d[0] == gi]
        c, s = fit(tr, va, k, evals)
        curves.append(c)
        snaps.append(s)
        print("fold {} curve {} ({:.0f}s)".format(gi, [round(v, 4) for v in c], time.time() - t0),
              flush=True)
    mean = np.mean(curves, 0)
    best = evals[int(np.argmin(mean))]
    print("best iters {} LOGO logloss {:.4f} per game {}".format(
        best, mean.min(), [round(c[int(np.argmin(mean))], 4) for c in curves]), flush=True)
    for gi, s in enumerate(snaps):
        arr = dict(s[best], names=np.array(names))
        np.savez(os.path.join(a.folds, "fold_tcn_{}.npz".format(gi)), **arr)
    _, s = fit(data, [], k, [best])
    arr = dict(s[best], names=np.array(names))
    path = os.path.join(a.out, "tcn_bfd.npz")
    np.savez(path, **arr)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = dict(iters=best, logo_logloss=float(mean.min()), curve=[round(float(v), 5) for v in mean],
                matches=matches, names=names, channels=C, dilations=list(DIL), crop=CROP,
                batch=BATCH, lr=LR, wd=WD, dropout=DROP, torch=torch.__version__, sha256=sha)
    with open(os.path.join(a.out, "tcn_bfd_meta.json"), "w") as h:
        json.dump(meta, h, indent=1)
    print("sha256", sha[:12], "total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
