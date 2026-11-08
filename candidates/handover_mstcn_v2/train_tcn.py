#!/usr/bin/env python3
"""Train the ASRF-style MS-TCN (class branch + boundary-regression branch) on the degraded DFL
sequences from prep_data.py, leave-one-game-out, and export numpy weights.

    OMP_NUM_THREADS=2 python -I train_tcn.py --data DATA --out FOLDS_DIR --folds 0,1,...,6[,all]

Each fold model (held-out game index gi = position in the sorted DFL match list) is written to
FOLDS_DIR/fold_tcn_<gi>.npz; --folds all trains on all 7 games -> FOLDS_DIR/full_tcn.npz.
torch is used for training only; pipeline.py runs the exported weights in plain numpy.

Architecture (Farha & Gall, MS-TCN, CVPR 2019; Ishikawa et al., ASRF, WACV 2021):
  input  standardised features (NaN -> 0), 5 fps (one stride-5 phase of the 25 fps frames)
  shared backbone: 1x1 conv K -> C, L dilated residual layers (dilation 2^i, kernel 3,
                   acausal), heads: 1x1 conv -> 3 class logits (H, A, D) and -> 2 boundary
                   logits (H<->A change, alive<->dead change)
  class branch:    S_c refinement stages (MS-TCN) on softmax of the previous stage
  boundary branch: S_b refinement stages (ASRF) on sigmoid of the previous stage
Loss per stage: CE + lambda_tmse * truncated MSE of log-prob differences (tau 4), boundary BCE
on Gaussian soft targets with pos_weight. Team-swap augmentation (features.swap_index, H<->A
label swap) with probability 0.5 per crop.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fe  # noqa: E402

CFG = dict(C=64, L=10, Cr=32, Lr=10, S_c=2, S_b=1, lr=5e-4, epochs=12, crop=1024, batch=8,
           lam_tmse=0.15, tau_tmse=4.0, w_b=1.0, pos_weight=10.0, dropout=0.5, seed=0,
           crops_per_epoch=600, wd=0.0)


class DRL(nn.Module):
    def __init__(self, d, c, p):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.out = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.out(F.relu(self.conv(x))))


class Stage(nn.Module):
    def __init__(self, cin, c, L, cout, p):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.layers = nn.ModuleList([DRL(2 ** i, c, p) for i in range(L)])
        self.heads = nn.ModuleList([nn.Conv1d(c, k, 1) for k in cout])

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers:
            h = l(h)
        return [hd(h) for hd in self.heads]


class ASRF(nn.Module):
    def __init__(self, K, cfg):
        super().__init__()
        p = cfg["dropout"]
        self.backbone = Stage(K, cfg["C"], cfg["L"], [3, 2], p)
        self.cref = nn.ModuleList([Stage(3, cfg["Cr"], cfg["Lr"], [3], p) for _ in range(cfg["S_c"])])
        self.bref = nn.ModuleList([Stage(2, cfg["Cr"], cfg["Lr"], [2], p) for _ in range(cfg["S_b"])])

    def forward(self, x):
        c, b = self.backbone(x)
        cs, bs = [c], [b]
        for st in self.cref:
            cs.append(st(F.softmax(cs[-1], 1))[0])
        for st in self.bref:
            bs.append(st(torch.sigmoid(bs[-1]))[0])
        return cs, bs


def export(model, mean, std, names, cfg, path):
    w = {}
    for k, v in model.state_dict().items():
        w["w/" + k] = v.detach().cpu().numpy().astype(np.float32)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             names=np.array(names), cfg=json.dumps(cfg), **w)


def load_data(data_dir):
    games = {}
    names = kinds = None
    for p in sorted(glob.glob(os.path.join(data_dir, "DFL-MAT-*_*.npz"))):
        match = os.path.basename(p)[:-4].rsplit("_", 1)[0]
        z = np.load(p)
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
        for s in np.unique(z["seq"]):
            m = z["seq"] == s
            games.setdefault(match, []).append((z["X"][m], z["y"][m].astype(np.int64),
                                                np.stack([z["bha"][m], z["bd"][m]], 0)))
    return games, names, kinds


def train_fold(games, names, kinds, hold, out_path, cfg, log):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    matches = sorted(games)
    seqs = [s for m in matches if m != hold for s in games[m]]
    perm, signs = fe.swap_index(names, kinds)
    allX = np.concatenate([s[0] for s in seqs])
    allX = np.concatenate([allX, allX[:, perm] * signs])        # symmetric stats
    mean = np.nanmean(allX, 0)
    std = np.nanstd(allX, 0)
    mean = np.nan_to_num(mean)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    del allX
    normed = []
    for X, y, b in seqs:
        Xn = np.nan_to_num((X - mean) / std).astype(np.float32)
        Xs = np.nan_to_num((X[:, perm] * signs - mean) / std).astype(np.float32)
        ys = np.where(y == 0, 1, np.where(y == 1, 0, y))
        normed.append((Xn, Xs, y, ys, b))
    lens = np.array([len(s[0]) for s in normed], float)
    model = ASRF(len(names), cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    T = cfg["crop"]
    pw = torch.tensor(cfg["pos_weight"])
    hold_seqs = games.get(hold, [])
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        tot = []
        for it in range(cfg["crops_per_epoch"] // cfg["batch"]):
            xb, yb, bb = [], [], []
            for _ in range(cfg["batch"]):
                i = int(rng.choice(len(normed), p=lens / lens.sum()))
                Xn, Xs, y, ys, b = normed[i]
                s0 = int(rng.integers(0, max(1, len(y) - T)))
                sw = rng.random() < 0.5
                xb.append((Xs if sw else Xn)[s0:s0 + T].T)
                yb.append((ys if sw else y)[s0:s0 + T])
                bb.append(b[:, s0:s0 + T])
            x = torch.from_numpy(np.stack(xb))
            y = torch.from_numpy(np.stack(yb))
            b = torch.from_numpy(np.stack(bb))
            cs, bs = model(x)
            valid = y < 3
            loss = 0
            for c in cs:
                lp = F.log_softmax(c, 1)
                loss = loss + F.nll_loss(lp, y.clamp(max=2), reduction="none")[valid].mean()
                d = (lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2
                d = torch.clamp(d, max=cfg["tau_tmse"] ** 2)
                vm = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)
                loss = loss + cfg["lam_tmse"] * (d * vm).sum() / (vm.sum() * 3 + 1)
            for bl in bs:
                l = F.binary_cross_entropy_with_logits(bl, b, pos_weight=pw, reduction="none")
                loss = loss + cfg["w_b"] * (l * valid.unsqueeze(1)).sum() / (valid.sum() * 2 + 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append(float(loss.detach()))
        msg = "hold {} ep {} loss {:.4f} {:.0f}s".format(hold, ep, np.mean(tot), time.time() - t0)
        if hold_seqs and (ep % 3 == 2 or ep == cfg["epochs"] - 1):
            model.eval()
            with torch.no_grad():
                nll, n, acc = 0.0, 0, 0
                for X, y, b in hold_seqs[:2]:
                    x = torch.from_numpy(np.nan_to_num((X - mean) / std).astype(np.float32).T[None])
                    cs, bs = model(x)
                    lp = F.log_softmax(cs[-1][0], 0).numpy()
                    v = y < 3
                    nll += -lp[y[v], np.where(v)[0]].sum()
                    acc += (lp.argmax(0)[v] == y[v]).sum()
                    n += v.sum()
            msg += "  held-out nll {:.4f} acc {:.4f}".format(nll / n, acc / n)
        log(msg)
    export(model, mean, std, names, cfg, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = dict(CFG, **json.loads(a.cfg))
    games, names, kinds = load_data(a.data)
    matches = sorted(games)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.txt"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    log("cfg " + json.dumps(cfg))
    for f in a.folds.split(","):
        if f == "all":
            train_fold(games, names, kinds, None, os.path.join(a.out, "full_tcn.npz"), cfg, log)
        else:
            gi = int(f)
            train_fold(games, names, kinds, matches[gi],
                       os.path.join(a.out, "fold_tcn_{}.npz".format(gi)), cfg, log)


if __name__ == "__main__":
    main()
