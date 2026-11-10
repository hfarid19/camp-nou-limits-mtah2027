#!/usr/bin/env python3
"""nn_pff TCN + auxiliary targets (round 5, R5-4). Same model, data, folds, loss and export as
candidates/nn_pff/train_tcn.py; adds an aux head (aux_common.AuxHead) on the TCN's last hidden layer
(the input of its output conv), trained jointly with aux_common.aux_loss on make_aux.py's targets. The
aux head is not exported, so nn_pff_tcn's numpy code runs the file unchanged.

    python -I candidates/nn_aux/train_tcn_aux.py --device cuda --ds $MTAH_WORK/ds_pff2/ds --out <dir> \
        --specs pff0,... --cfg '{"ls":0.05,"aux_lam":0.3}'
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
CAND = os.path.dirname(HERE)
sys.path.append(os.path.join(CAND, "nn_pff"))
sys.path.insert(0, HERE)
import aux_common as AC   # noqa: E402
import aux_data as AD     # noqa: E402
import data as D          # noqa: E402
import features as fe     # noqa: E402
import train_tcn as TT    # noqa: E402

CFG = dict(TT.CFG, **AC.AUX_CFG)


def fwd(model, x):
    x = model.inp(model.in_drop(x))
    for b in model.blocks:
        x = b(x)
    h = Fn.relu(x)
    return model.out(h), h


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device="cpu"):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = AD.load(ds_dir, spec)
    assert not D.excluded(spec) & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mu, sd, ind = TT.norm_stats([s[:4] for s in seqs], perm, signs)
    pos = {int(j): i for i, j in enumerate(ind)}
    assert all(int(perm[j]) in pos for j in ind), "missing indicators not swap-symmetric"
    permf = np.concatenate([perm, [len(perm) + pos[int(perm[j])] for j in ind]])
    signf = np.concatenate([signs, np.ones(len(ind), np.float32)])
    data = []
    for g, X, y, b, a in seqs:
        wt = np.ones(len(y), np.float32)
        if cfg["trans_w"] > 0:
            ch = np.where(y[1:] != y[:-1])[0] + 1
            near = np.zeros(len(y), bool)
            for k in range(-cfg["trans_hw"], cfg["trans_hw"]):
                near[np.clip(ch + k, 0, len(y) - 1)] = True
            wt[near] += cfg["trans_w"]
        data.append((g.startswith("DFL"), TT.prep(X, mu, sd, ind).astype(np.float16), y,
                     np.stack([b[2].astype(np.float32), wt]), a))
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
    dev = torch.device(device)
    model = TT.TCN(din, cfg["C"], dils, drop=cfg["drop"], in_drop=cfg["in_drop"]).to(dev)
    aux = AC.AuxHead(cfg["C"], cfg["aux_hidden"]).to(dev)
    log("params {} + aux {} device {}".format(sum(p.numel() for p in model.parameters()),
                                             sum(p.numel() for p in aux.parameters()), device))
    params = list(model.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    if cfg["sched"] == "cosine":
        n_it, warm = cfg["iters"], max(1, int(0.03 * cfg["iters"]))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["iters"], pct_start=0.1)
    L = cfg["crop"]
    cw = torch.tensor([1.0, 1.0, cfg["dead_w"]], device=dev)
    permt, signt = torch.from_numpy(permf), torch.from_numpy(signf)[:, None]
    t1 = time.time()
    run, parts = [], []
    for it in range(cfg["iters"]):
        model.train()
        aux.train()
        xs, ys, cs, ab = [], [], [], []
        for _ in range(cfg["batch"]):
            j = int(rng.choice(len(data), p=w))
            _, Z, y, c, a = data[j]
            s = int(rng.integers(0, max(1, len(y) - L)))
            x = torch.from_numpy(Z[s:s + L].T.astype(np.float32))
            yy = torch.from_numpy(y[s:s + L])
            aa = torch.from_numpy(a[:, s:s + L])
            if rng.random() < 0.5:
                x = x[permt] * signt
                yy = torch.where(yy == 0, 1, torch.where(yy == 1, 0, yy))
                aa = AC.swap_aux(aa)
            xs.append(x)
            ys.append(yy)
            cs.append(torch.from_numpy(c[:, s:s + L]))
            ab.append(aa)
        Tm = min(v.shape[1] for v in xs)
        x = torch.stack([v[:, :Tm] for v in xs]).to(dev)
        y = torch.stack([v[:Tm] for v in ys]).to(dev)
        c = torch.stack([v[:, :Tm] for v in cs]).to(dev)
        at = torch.stack([v[:, :Tm] for v in ab]).to(dev)
        out, h = fwd(model, x)
        valid = y < 3
        ce = Fn.cross_entropy(out[:, :3], y.clamp(max=2), weight=cw, reduction="none", label_smoothing=cfg["ls"])
        wt = c[:, 1] * valid
        loss = (ce * wt).sum() / wt.sum() + cfg["c_w"] * Fn.binary_cross_entropy_with_logits(out[:, 3], c[:, 0])
        la, pa = AC.aux_loss(aux(h), at, valid, cfg)
        loss = loss + la
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            run.append(float(loss.detach()))
            parts.append([pa["ball"], pa["time"], pa["rst"]])
        if (it + 1) % 250 == 0:
            pm = np.mean(parts[-10:], 0)
            log("  {} it {} loss {:.4f} aux ball {:.4f} time {:.4f} rst {:.4f} {:.0f}s".format(
                spec, it + 1, np.mean(run[-10:]), pm[0], pm[1], pm[2], time.time() - t1))
    model.to("cpu")
    cfg = dict(cfg, device=device, pff_holdout_trained=D.allow_holdout(), trainer="nn_aux/train_tcn_aux.py")
    sdict = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    meta = dict(mu=mu, sd=sd, ind=ind.astype(np.int64), dils=np.array(dils, np.int64), names=np.array(names),
                kinds=np.array(kinds), change_prior=np.array(float(np.mean(np.concatenate([d[3][0] for d in data])))),
                hold=np.array(-1), train_games=np.array(games), spec=np.array(spec), train_cfg=json.dumps(cfg),
                pff_holdout_trained=np.array(D.allow_holdout()))
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, **sdict, **meta)
    os.replace(tmp, out_path)
    log("{} -> {} ({:.0f} min)".format(spec, out_path, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="pff0,pff1,pff2,pff3")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = dict(CFG, **json.loads(a.cfg))
    assert set(cfg) == set(CFG), set(cfg) ^ set(CFG)
    assert not cfg["init"] and cfg["ema"] == 0 and cfg["focal"] == 0 and cfg["src"] == "all", "not supported here"
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
        p = TT.out_name(a.out, spec)
        if os.path.isfile(p):
            log("{} exists, skipped".format(p))
            continue
        train_spec(a.ds, spec, p, cfg, a.threads, log, a.device)


if __name__ == "__main__":
    main()
