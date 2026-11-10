#!/usr/bin/env python3
"""Train the temporal TCN (temporal_tcn_v2 recipe) on PFF + DFL; exports the temporal_tcn numpy format.

    OMP_NUM_THREADS=2 python -I candidates/nn_pff/train_tcn.py --ds $MTAH_WORK/ds_pff/ds \
        --out $MTAH_WORK/arch/nn_pff/<config> --specs pff0,pff1,pff2,pff3 [--cfg '{...}']

Same model, input preparation, loss and export as candidates/temporal_tcn_v2/train_tcn.py (C 64,
12 dilated residual blocks, missing-value indicators, 3 class logits + 1 hand-over logit; CE +
c_w * BCE; OneCycle AdamW), so candidates/nn_pff_tcn (= temporal_tcn_v3's numpy code and LDE
decoder) runs the exported file. Data, folds and target = data.py (PFF touch labels, DFL raw flag;
pff<k> / dfl<g> / full specs). Knobs beyond the recipe (defaults = the recipe): sched (onecycle |
cosine), ema (export an EMA of the weights), focal (gamma), ls (label smoothing), trans_w / trans_hw
(extra CE weight within +-trans_hw steps of a label change, temporal_tcn_v2's --trans-w), init (a
config folder of the same architecture: fine-tune its model of the same spec) and src (all | pff |
dfl). Team-swap augmentation p = 0.5, applied to the prepared input with
the permutation that the symmetric normalisation makes exact. Files: fold_pff_<k>.npz,
fold_tcn_<g>.npz (bench dfl_folds name), full.npz, each with `train_games` and `spec`.
"""

import argparse
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
import data as D       # noqa: E402
import features as fe  # noqa: E402

CFG = dict(C=64, dils="1,2,4,8,16,32,64,128,1,2,4,8", drop=0.1, in_drop=0.1, lr=2e-3, iters=4000,
           batch=16, crop=512, dead_w=1.0, c_w=0.5, seed=0, w_dfl=0.15, wd=1e-4,
           sched="onecycle", ema=0.0, focal=0.0, ls=0.0, trans_w=0.0, trans_hw=2, init="", src="all")


def norm_stats(seqs, perm, signs, min_nan=0.005):
    K = seqs[0][1].shape[1]
    s1, s2, cnt, nn_ = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    tot = 0
    for _, X, _, _ in seqs:
        ok = ~np.isnan(X)
        Xz = np.where(ok, X, 0.0).astype(np.float64)
        s1 += Xz.sum(0)
        s2 += (Xz ** 2).sum(0)
        cnt += ok.sum(0)
        nn_ += (~ok).sum(0)
        tot += len(X)
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

    def forward(self, x):
        x = self.inp(self.in_drop(x))
        for b in self.blocks:
            x = b(x)
        return self.out(Fn.relu(x))


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device="cpu"):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec)
    assert not D.excluded(spec) & set(games)
    if cfg["src"] != "all":
        keep = "DFL" if cfg["src"] == "dfl" else "PFF"
        seqs = [q for q in seqs if q[0].startswith(keep)]
        games = [g for g in games if g.startswith(keep)]
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mu, sd, ind = norm_stats(seqs, perm, signs)
    init = None
    if cfg["init"]:
        ip = out_name(cfg["init"], spec) if os.path.isdir(cfg["init"]) else cfg["init"]
        init = np.load(ip)
        assert str(init["spec"]) == spec, (ip, spec)
        mu, sd, ind = init["mu"], init["sd"], init["ind"]          # keep the pretrained input scale
        games = sorted(set(games) | {str(g) for g in init["train_games"]})
    pos = {int(j): i for i, j in enumerate(ind)}
    assert all(int(perm[j]) in pos for j in ind), "missing indicators not swap-symmetric"
    permf = np.concatenate([perm, [len(perm) + pos[int(perm[j])] for j in ind]])
    signf = np.concatenate([signs, np.ones(len(ind), np.float32)])
    data = []
    for g, X, y, b in seqs:
        wt = np.ones(len(y), np.float32)
        if cfg["trans_w"] > 0:
            ch = np.where(y[1:] != y[:-1])[0] + 1
            near = np.zeros(len(y), bool)
            for k in range(-cfg["trans_hw"], cfg["trans_hw"]):
                near[np.clip(ch + k, 0, len(y) - 1)] = True
            wt[near] += cfg["trans_w"]
        data.append((g.startswith("DFL"), prep(X, mu, sd, ind).astype(np.float16), y,
                     np.stack([b[2].astype(np.float32), wt])))
    del seqs
    lens = np.array([len(d[2]) for d in data], float)
    isd = np.array([d[0] for d in data])
    w = lens.copy()
    if 0 < cfg["w_dfl"] < 1 and isd.any() and (~isd).any():
        w[isd] *= cfg["w_dfl"] / w[isd].sum()
        w[~isd] *= (1 - cfg["w_dfl"]) / w[~isd].sum()
    w /= w.sum()
    log("spec {} games {} (PFF {}, DFL {}) seqs {} steps {} load {:.0f}s".format(
        spec, len(games), sum(g.startswith("PFF") for g in games), sum(g.startswith("DFL") for g in games),
        len(data), int(lens.sum()), time.time() - t0))
    torch.set_num_threads(threads)
    din = data[0][1].shape[1]
    dils = [int(v) for v in cfg["dils"].split(",")]
    model = TCN(din, cfg["C"], dils, drop=cfg["drop"], in_drop=cfg["in_drop"])
    if init is not None:
        model.load_state_dict({k: torch.from_numpy(init[k]) for k in model.state_dict()})
    dev = torch.device(device)
    model.to(dev)
    log("params {} device {}".format(sum(p.numel() for p in model.parameters()), device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    if cfg["sched"] == "cosine":
        n_it, warm = cfg["iters"], max(1, int(0.03 * cfg["iters"]))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["iters"], pct_start=0.1)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg["ema"] > 0 else None
    L = cfg["crop"]
    cw = torch.tensor([1.0, 1.0, cfg["dead_w"]], device=dev)
    permt, signt = torch.from_numpy(permf), torch.from_numpy(signf)[:, None]
    t1 = time.time()
    run = []
    for it in range(cfg["iters"]):
        model.train()
        xs, ys, cs = [], [], []
        for _ in range(cfg["batch"]):
            j = int(rng.choice(len(data), p=w))
            _, Z, y, c = data[j]
            s = int(rng.integers(0, max(1, len(y) - L)))
            x = torch.from_numpy(Z[s:s + L].T.astype(np.float32))
            yy = torch.from_numpy(y[s:s + L])
            if rng.random() < 0.5:
                x = x[permt] * signt
                yy = torch.where(yy == 0, 1, torch.where(yy == 1, 0, yy))
            xs.append(x)
            ys.append(yy)
            cs.append(torch.from_numpy(c[:, s:s + L]))
        Tm = min(v.shape[1] for v in xs)
        x = torch.stack([v[:, :Tm] for v in xs]).to(dev)
        y = torch.stack([v[:Tm] for v in ys]).to(dev)
        c = torch.stack([v[:, :Tm] for v in cs]).to(dev)
        out = model(x)
        valid = y < 3
        ce = Fn.cross_entropy(out[:, :3], y.clamp(max=2), weight=cw, reduction="none",
                              label_smoothing=cfg["ls"])
        if cfg["focal"] > 0:
            pt = torch.gather(Fn.softmax(out[:, :3], 1), 1, y.clamp(max=2)[:, None]).squeeze(1)
            ce = ce * (1 - pt).pow(cfg["focal"])
        wt = c[:, 1] * valid
        loss = (ce * wt).sum() / wt.sum() + cfg["c_w"] * Fn.binary_cross_entropy_with_logits(out[:, 3], c[:, 0])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(cfg["ema"]).add_(v.detach(), alpha=1 - cfg["ema"])
        if it % 25 == 0:
            run.append(float(loss.detach()))
        if (it + 1) % 250 == 0:
            log("  {} it {} loss {:.4f} {:.0f}s".format(spec, it + 1, np.mean(run[-10:]), time.time() - t1))
    if ema is not None:
        model.load_state_dict(ema)
    model.to("cpu")
    cfg = dict(cfg, device=device, pff_holdout_trained=D.allow_holdout())
    sdict = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    meta = dict(mu=mu, sd=sd, ind=ind.astype(np.int64), dils=np.array(dils, np.int64), names=np.array(names),
                kinds=np.array(kinds), change_prior=np.array(float(np.mean(np.concatenate([d[3][0] for d in data])))),
                hold=np.array(-1), train_games=np.array(games), spec=np.array(spec), train_cfg=json.dumps(cfg),
                pff_holdout_trained=np.array(D.allow_holdout()))
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, **sdict, **meta)
    os.replace(tmp, out_path)
    log("{} -> {} ({:.0f} min)".format(spec, out_path, (time.time() - t0) / 60))


def out_name(out, spec):
    if spec == "full":
        return os.path.join(out, "full.npz")
    if spec.startswith("pff"):
        return os.path.join(out, "fold_pff_{}.npz".format(int(spec[3:])))
    return os.path.join(out, "fold_tcn_{}.npz".format(int(spec[3:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="pff0,pff1,pff2,pff3")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU; float32)")
    a = ap.parse_args()
    cfg = dict(CFG, **json.loads(a.cfg))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_cfg.json"), "w") as h:
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), model="tcn"), h, indent=1)
    logf = open(os.path.join(a.out, "train_log.txt"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    log("cfg " + json.dumps(cfg) + " specs " + a.specs)
    for spec in a.specs.split(","):
        p = out_name(a.out, spec)
        if os.path.isfile(p):
            log("{} exists, skipped".format(p))
            continue
        train_spec(a.ds, spec, p, cfg, a.threads, log, a.device)


if __name__ == "__main__":
    main()
