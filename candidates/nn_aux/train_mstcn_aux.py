#!/usr/bin/env python3
"""nn_pff MS-TCN (ASRF) + auxiliary targets (round 5, R5-4). Same model, data, folds, loss and export as
candidates/nn_pff/train_mstcn.py; adds an aux head (aux_common.AuxHead) on the backbone's last hidden
layer, trained jointly with aux_common.aux_loss on make_aux.py's targets. The aux head is not exported:
the npz has the same keys as train_mstcn.py's, so nn_pff/tcn_np.py + pipeline_cfg.py run it unchanged.

    python -I candidates/nn_aux/train_mstcn_aux.py --device cuda --ds $MTAH_WORK/ds_pff2/ds --out <dir> \
        --specs pff0,... --cfg '{"cosine":0,"clip":0,"C":128,"Cr":64,"aux_lam":0.3}'
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.append(os.path.join(CAND, "nn_pff"))
sys.path.insert(0, HERE)
import aux_common as AC   # noqa: E402
import aux_data as AD     # noqa: E402
import data as D          # noqa: E402
import features as fe     # noqa: E402
import train_mstcn as TM  # noqa: E402

CFG = dict(TM.CFG, **AC.AUX_CFG)


def backbone_fwd(model, x):
    """ASRF forward that also returns the backbone's last hidden layer."""
    st = model.backbone
    h = st.inp(model.in_drop(x))
    for l in st.layers:
        h = l(h)
    c, b = [hd(h) for hd in st.heads]
    cs, bs = [c], [b]
    for s in model.cref:
        cs.append(s(F.softmax(cs[-1], 1))[0])
    for s in model.bref:
        bs.append(s(torch.sigmoid(bs[-1]))[0])
    return cs, bs, h


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device="cpu"):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = AD.load(ds_dir, spec, cfg["dfl_fold_s"], cfg["pff_fold_s"])
    ex = D.excluded(spec)
    assert not ex & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mean, std = TM.sym_stats([s[:4] for s in seqs], perm, signs)
    normed = []
    for g, X, y, b, a in seqs:
        Xn = np.nan_to_num((X - mean) / std).astype(np.float16)
        normed.append((g.startswith("DFL"), Xn, y, b[:2], a))
    del seqs
    lens = np.array([len(s[2]) for s in normed], float)
    isd = np.array([s[0] for s in normed])
    w = lens.copy()
    if 0 < cfg["w_dfl"] < 1 and isd.any() and (~isd).any():
        w[isd] *= cfg["w_dfl"] / w[isd].sum()
        w[~isd] *= (1 - cfg["w_dfl"]) / w[~isd].sum()
    w /= w.sum()
    log("spec {} games {} (PFF {}, DFL {}) seqs {} steps {} load {:.0f}s".format(
        spec, len(games), sum(g.startswith("PFF") for g in games), sum(g.startswith("DFL") for g in games),
        len(normed), int(lens.sum()), time.time() - t0))
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = TM.ASRF(len(names), cfg).to(dev)
    aux = AC.AuxHead(cfg["C"], cfg["aux_hidden"]).to(dev)
    log("params {} + aux {}".format(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in aux.parameters())))
    params = list(model.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"]) if cfg["wd"] > 0 else \
        torch.optim.Adam(params, lr=cfg["lr"])
    n_it = cfg["epochs"] * (cfg["crops_per_epoch"] // cfg["batch"])
    sched = None
    if cfg["cosine"]:
        warm = max(1, int(cfg["warm"] * n_it))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    T = cfg["crop"]
    pw = torch.tensor(cfg["pos_weight"], device=dev)
    cw = torch.tensor([1.0, 1.0, cfg["cw_d"]], device=dev)
    permt = torch.from_numpy(perm)
    signst = torch.from_numpy(signs)[:, None]
    for ep in range(cfg["epochs"]):
        model.train()
        aux.train()
        t1 = time.time()
        tot, parts = [], []
        for _ in range(cfg["crops_per_epoch"] // cfg["batch"]):
            xb, yb, bb, ab = [], [], [], []
            for _ in range(cfg["batch"]):
                i = int(rng.choice(len(normed), p=w))
                _, Xn, y, b, a = normed[i]
                s0 = int(rng.integers(0, max(1, len(y) - T)))
                x = torch.from_numpy(Xn[s0:s0 + T].T.astype(np.float32))
                yy = torch.from_numpy(y[s0:s0 + T])
                aa = torch.from_numpy(a[:, s0:s0 + T])
                if rng.random() < 0.5:
                    x = x[permt] * signst
                    yy = torch.where(yy == 0, 1, torch.where(yy == 1, 0, yy))
                    aa = AC.swap_aux(aa)
                xb.append(x)
                yb.append(yy)
                bb.append(torch.from_numpy(b[:, s0:s0 + T]))
                ab.append(aa)
            Tm = min(v.shape[1] for v in xb)
            x = torch.stack([v[:, :Tm] for v in xb]).to(dev)
            y = torch.stack([v[:Tm] for v in yb]).to(dev)
            b = torch.stack([v[:, :Tm] for v in bb]).to(dev)
            at = torch.stack([v[:, :Tm] for v in ab]).to(dev)
            cs, bs, h = backbone_fwd(model, x)
            valid = y < 3
            loss = 0
            for c in cs:
                lp = F.log_softmax(c, 1)
                yt = y.clamp(max=2)
                nll = F.nll_loss(lp, yt, weight=cw, reduction="none")
                if cfg["ls"] > 0:
                    nll = (1 - cfg["ls"]) * nll - cfg["ls"] * (lp * cw[None, :, None]).mean(1)
                loss = loss + nll[valid].mean()
                d = (lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2
                d = torch.clamp(d, max=cfg["tau_tmse"] ** 2)
                vm = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)
                loss = loss + cfg["lam_tmse"] * (d * vm).sum() / (vm.sum() * 3 + 1)
            for bl in bs:
                l = F.binary_cross_entropy_with_logits(bl, b, pos_weight=pw, reduction="none")
                loss = loss + cfg["w_b"] * (l * valid.unsqueeze(1)).sum() / (valid.sum() * 2 + 1)
            la, pa = AC.aux_loss(aux(h), at, valid, cfg)
            loss = loss + la
            opt.zero_grad()
            loss.backward()
            if cfg["clip"] > 0:
                nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            if sched is not None:
                sched.step()
            tot.append(float(loss.detach()))
            parts.append([pa["ball"], pa["time"], pa["rst"]])
        pm = np.mean(parts, 0)
        log("{} ep {} loss {:.4f} aux ball {:.4f} time {:.4f} rst {:.4f} {:.0f}s".format(
            spec, ep, np.mean(tot), pm[0], pm[1], pm[2], time.time() - t1))
    model.to("cpu")
    cfg = dict(cfg, device=device, pff_holdout_trained=D.allow_holdout(), trainer="nn_aux/train_mstcn_aux.py")
    TM.export(model, mean, std, names, cfg, out_path, games, spec)
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
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), model="mstcn_aux"), h, indent=1)
    logf = open(os.path.join(a.out, "train_log.txt"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    log("cfg " + json.dumps(cfg) + " specs " + a.specs)
    for spec in a.specs.split(","):
        p = TM.out_name(a.out, spec)
        if os.path.isfile(p):
            log("{} exists, skipped".format(p))
            continue
        train_spec(a.ds, spec, p, cfg, a.threads, log, a.device)


if __name__ == "__main__":
    main()
