#!/usr/bin/env python3
"""Train the temporal TCN (dilated non-causal 1-D convolutions) on degraded DFL/IDSSE sequences.

    OMP_NUM_THREADS=2 python -I candidates/temporal_tcn/train_tcn.py --folds all \
        --data <idsse_r1/ds> <idsse_r2d/ds> --raw <idsse/npz> --fold-dir <scratch>/folds \
        --out candidates/temporal_tcn/tcn_weights.npz

Input data: learned_dfl_evidence/make_dataset.py output (features.py columns of every graded
frame whose index is = seed (mod 5), seeds 11/12/13, i.e. a 5 fps point-sampled sequence per
degradation; round-1 and round-2 degradations). The bench scores DFL with seed 2027, never
seen here. Each degradation x half is one sequence (half from the raw npz).

Model (exported to .npz, inference in numpy, see tcn_np.py):
    z = clip((F - mu) / sd, +-6), NaN -> 0, plus missing indicators      (Din)
    x = W_in z                                                            (C)
    for d in dilations:  x = x + W_b relu(conv_k3_dil_d(relu(x)))        residual blocks
    heads = W_out relu(x): 3 logits (H, A, D) + 1 logit (hand-over within +-4 frames)
Team-swap augmentation: with p = 0.5 a training crop is replaced by its H/A mirror
(features.swap_index; labels H <-> A). Normalisation stats are computed on the union of the
training games and their mirrors, so the mirror commutes with the normalisation.

Leave-one-game-out: --folds 0,1,..,6 trains on the other 6 games and writes
<fold-dir>/fold_tcn_<gi>.npz (gi = index in sorted DFL match list, as bench.py); 'full' trains
on all 7 and writes --out. 'all' = the 7 folds then full.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fe  # noqa: E402


def load(ds_dirs, raw_dir):
    seqs = []           # (game index, X (T, K) float32, y (T,), yc (T,))
    names = kinds = None
    matches = sorted(f[:-4] for f in os.listdir(raw_dir) if f.endswith(".npz"))
    halves = {m: np.load(os.path.join(raw_dir, m + ".npz"))["half"] for m in matches}
    for d in ds_dirs:
        for p in sorted(glob.glob(os.path.join(d, "DFL-MAT-*_*.npz"))):
            match = os.path.basename(p).rsplit("_", 1)[0]
            z = np.load(p)
            names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
            idx = z["idx"]
            h = halves[match][idx]
            for hv in np.unique(h):
                m = h == hv
                assert np.all(np.diff(idx[m]) == 5)
                seqs.append((matches.index(match), z["X"][m].astype(np.float32),
                             z["y"][m].astype(np.int64), z["yc"][m].astype(np.float32)))
    return seqs, names, kinds, matches


def norm_stats(seqs, perm, signs, min_nan=0.005):
    K = seqs[0][1].shape[1]
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    cnt = np.zeros(K)
    nn_ = np.zeros(K)
    tot = 0
    for _, X, _, _ in seqs:
        ok = ~np.isnan(X)
        Xz = np.where(ok, X, 0.0).astype(np.float64)
        s1 += Xz.sum(0)
        s2 += (Xz ** 2).sum(0)
        cnt += ok.sum(0)
        nn_ += (~ok).sum(0)
        tot += len(X)
    # mirror: column j of the mirror is signs[j] * column perm[j]
    s1a = 0.5 * (s1 + signs * s1[perm])
    s2a = 0.5 * (s2 + s2[perm])
    ca = 0.5 * (cnt + cnt[perm])
    mu = s1a / np.maximum(ca, 1)
    sd = np.sqrt(np.maximum(s2a / np.maximum(ca, 1) - mu ** 2, 1e-8))
    sd = np.maximum(sd, 1e-3)
    nan_share = 0.5 * (nn_ + nn_[perm]) / tot
    ind = np.where(nan_share > min_nan)[0]
    return mu.astype(np.float32), sd.astype(np.float32), ind


def prep(X, mu, sd, ind):
    """(T, K) raw features -> (T, K + len(ind)) model input (numpy; same as tcn_np.prep)."""
    miss = np.isnan(X)
    z = np.clip((X - mu) / sd, -6.0, 6.0)
    z[miss] = 0.0
    return np.concatenate([z, miss[:, ind].astype(np.float32)], 1).astype(np.float32)


class Block(nn.Module):
    def __init__(self, C, d, k, drop):
        super().__init__()
        self.a = nn.Conv1d(C, C, k, dilation=d, padding=d * (k // 2))
        self.b = nn.Conv1d(C, C, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.b(self.drop(Fn.relu(self.a(Fn.relu(x)))))


class TCN(nn.Module):
    def __init__(self, din, C, dils, k=3, drop=0.1, in_drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(din, C, 1)
        self.in_drop = nn.Dropout(in_drop)
        self.blocks = nn.ModuleList([Block(C, d, k, drop) for d in dils])
        self.out = nn.Conv1d(C, 4, 1)

    def forward(self, x):                   # x (B, Din, T)
        x = self.inp(self.in_drop(x))
        for b in self.blocks:
            x = b(x)
        return self.out(Fn.relu(x))


def export(model, path, meta_arrays):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    np.savez(path, **sd, **meta_arrays)


def train_one(seqs, hold, names, kinds, args, out_path, log):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    perm, signs = fe.swap_index(names, kinds)
    perm = np.asarray(perm)
    signs = np.asarray(signs, np.float32)
    tr = [s for s in seqs if s[0] != hold]
    va = [s for s in seqs if s[0] == hold]
    mu, sd, ind = norm_stats(tr, perm, signs)
    # pre-normalise both orientations (float16 to save memory)
    def mk(s, mirror):
        X = s[1][:, perm] * signs if mirror else s[1]
        y = s[2].copy()
        if mirror:
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
        # per-step loss weight: 1 + trans_w within +-trans_hw steps of any label change
        wt = np.ones(len(y), np.float32)
        if args.trans_w > 0:
            ch = np.where(y[1:] != y[:-1])[0] + 1
            near = np.zeros(len(y), bool)
            for k in range(-args.trans_hw, args.trans_hw):
                near[np.clip(ch + k, 0, len(y) - 1)] = True
            wt[near] += args.trans_w
        return torch.from_numpy(prep(X, mu, sd, ind).T.copy()).half(), torch.from_numpy(y), \
            torch.from_numpy(np.stack([s[3], wt]))
    data = [(mk(s, False), mk(s, True)) for s in tr]
    vdata = [mk(s, False) for s in va]
    lens = np.array([d[0][1].shape[0] for d in data])
    w = lens / lens.sum()
    din = data[0][0][0].shape[0]
    dils = [int(v) for v in args.dils.split(",")]
    model = TCN(din, args.C, dils, drop=args.drop, in_drop=args.in_drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters,
                                                pct_start=0.1)
    L = args.crop
    t0 = time.time()
    cw = torch.tensor([1.0, 1.0, args.dead_w])
    for it in range(args.iters):
        model.train()
        xs, ys, cs = [], [], []
        for b in range(args.batch):
            j = int(rng.choice(len(data), p=w))
            mir = int(rng.random() < 0.5)
            X, y, c = data[j][mir]
            T = X.shape[1]
            s = int(rng.integers(0, max(1, T - L)))
            xs.append(X[:, s:s + L])
            ys.append(y[s:s + L])
            cs.append(c[:, s:s + L])
        Tm = min(x.shape[1] for x in xs)
        x = torch.stack([v[:, :Tm] for v in xs]).float()
        y = torch.stack([v[:Tm] for v in ys])
        c = torch.stack([v[:, :Tm] for v in cs])
        out = model(x)
        wt = c[:, 1]
        ce = Fn.cross_entropy(out[:, :3], y, weight=cw, reduction="none")
        loss = (ce * wt).sum() / wt.sum() + \
            args.c_w * Fn.binary_cross_entropy_with_logits(out[:, 3], c[:, 0])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % args.log_every == 0 or it + 1 == args.iters:
            msg = "  it {} loss {:.4f} {:.0f}s".format(it + 1, loss.item(), time.time() - t0)
            if vdata:
                model.eval()
                with torch.no_grad():
                    vl, vn, acc = 0.0, 0, 0
                    for X, y, c in vdata:
                        o = model(X.float()[None])[0]
                        vl += Fn.cross_entropy(o[:3].T, y, reduction="sum").item()
                        acc += (o[:3].argmax(0) == y).sum().item()
                        vn += len(y)
                msg += " | val ce {:.4f} acc {:.4f}".format(vl / vn, acc / vn)
            log(msg)
    meta = dict(mu=mu, sd=sd, ind=ind.astype(np.int64), dils=np.array(dils, np.int64),
                names=np.array(names), kinds=np.array(kinds),
                change_prior=np.array(float(np.mean(np.concatenate([s[3] for s in tr])))),
                hold=np.array(hold))
    export(model, out_path, meta)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--kind", default="tcn")
    ap.add_argument("--out", default=os.path.join(HERE, "tcn_weights.npz"))
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--dils", default="1,2,4,8,16,32,64,128,1,2,4,8")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--in-drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--dead-w", type=float, default=1.0)
    ap.add_argument("--c-w", type=float, default=0.5)
    ap.add_argument("--trans-w", type=float, default=0.0)
    ap.add_argument("--trans-hw", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.fold_dir, exist_ok=True)
    t0 = time.time()
    seqs, names, kinds, matches = load(args.data, args.raw)
    print("loaded {} sequences, {} steps, {:.0f}s".format(len(seqs), sum(len(s[2]) for s in seqs),
                                                         time.time() - t0), flush=True)
    folds = list(range(len(matches))) + ["full"] if args.folds == "all" else \
        [f if f == "full" else int(f) for f in args.folds.split(",")]

    def log(m):
        print(m, flush=True)
    for f in folds:
        t1 = time.time()
        if f == "full":
            path, hold = args.out, -1
        else:
            path, hold = os.path.join(args.fold_dir, "fold_{}_{}.npz".format(args.kind, f)), f
        log("fold {} (hold {}) -> {}".format(f, matches[hold] if hold >= 0 else "-", path))
        train_one(seqs, hold, names, kinds, args, path, log)
        log("  done {:.0f}s".format(time.time() - t1))
    with open(os.path.join(args.fold_dir, "train_args_{}.json".format(args.kind)), "w") as h:
        json.dump(vars(args), h, indent=1)


if __name__ == "__main__":
    main()
