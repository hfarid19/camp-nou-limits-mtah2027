#!/usr/bin/env python3
"""MET-FT fine-tune (round 5 prereg addendum 2026-11-10 11:50 B): start from a finished parent model (same spec) and
train briefly on Metrica sample games (Metrica-rule controlled-possession labels, $MTAH_WORK/ds_met, met_data.py)
mixed 1:1 with the parent's own training data.

    OMP_NUM_THREADS=2 python -I candidates/met_ft/ft.py --kind ptr|mstcn --parent <config dir> --spec full|dfl<g> \
        --met MET-G1,MET-G2[,MET-G3] --out <file.npz> [--threads 2]

Fixed settings (prereg, no tuning): lr = 0.1 x the parent's peak lr; 5 epochs of the parents' unit (1200 crops) =
750 iterations of batch 8; each crop is a Metrica crop with probability 0.5 (uniform over the allowed Metrica frames),
else a crop of the parent's training data for that spec drawn with the parent's sampling weights (DFL share w_dfl);
everything else as the parent (loss, label smoothing, dropout, team-swap augmentation, optimiser type, clip;
schedule shape: pointer warm-up + cosine over the 750 iterations, MS-TCN constant when the parent's cosine = 0; no
EMA in either parent). The parent's input normalisation is kept; the pointer's decoder prior change_prior is kept.
The output has the parent's file layout (same numpy inference code), spec = the parent's spec, train_games = the
parent's games + the Metrica games used. The parent's training games for the spec are re-derived with the same
data.py rules and must equal the parent's recorded train_games (leak guard). Uses only the trainers' default paths
(no opt-in env); the trainers themselves are imported, not modified.
"""

import argparse
import json
import multiprocessing.pool    # noqa: F401  (stdlib queue before nn_pff/queue.py is on sys.path)
import multiprocessing.queues  # noqa: F401
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "nn_pff"), os.path.join(CAND, "nn_pointer")]
import data as D           # noqa: E402
import features as fe      # noqa: E402
import train_mstcn as TM   # noqa: E402
import train_ptr as TP     # noqa: E402
import ptr_net as PN       # noqa: E402

WORK = os.environ.get("MTAH_WORK")
EPOCH_CROPS, EPOCHS, LR_MULT, MET_SHARE = 1200, 5, 0.1, 0.5


def weights(lens, isd, w_dfl):
    w = lens.astype(float).copy()
    if 0 < w_dfl < 1 and isd.any() and (~isd).any():
        w[isd] *= w_dfl / w[isd].sum()
        w[~isd] *= (1 - w_dfl) / w[~isd].sum()
    return w / w.sum()


def check_games(games, parent, met):
    pg = {str(g) for g in parent["train_games"]}
    assert set(games) == pg, ("training games differ from the parent's", set(games) ^ pg)
    assert not any(g.startswith("MET-") for g in pg), "parent already saw Metrica"
    return sorted(pg | set(met))


def ft_ptr(a, log):
    pp = TM.out_name(a.parent, a.spec)
    parent = np.load(pp)
    assert str(parent["spec"]) == a.spec, (pp, str(parent["spec"]))
    pcfg = json.loads(str(parent["cfg"]))
    cfg = {k: pcfg[k] for k in TP.CFG}
    assert cfg["ballc"] == 1
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    seqs, names, kinds, games = TP.load(a.ds, a.spec, True)
    mseqs, mnames, _, mgames = TP.load(a.ds_met, "full", True)
    assert mnames == names == [str(s) for s in parent["names"]]
    mseqs = [s for s in mseqs if s["g"] in a.met]
    assert {s["g"] for s in mseqs} == set(a.met), (a.met, mgames)
    train_games = check_games(games, parent, a.met)
    mean, std = parent["mean"].astype(np.float64), parent["std"].astype(np.float64)
    for s in seqs + mseqs:
        s["Xn"] = np.nan_to_num((s.pop("X") - mean) / std).astype(np.float16)
    w0 = weights(np.array([len(s["y"]) for s in seqs]), np.array([s["g"].startswith("DFL") for s in seqs]), cfg["w_dfl"])
    wm = np.array([len(s["y"]) for s in mseqs], float)
    w = np.concatenate([(1 - MET_SHARE) * w0, MET_SHARE * wm / wm.sum()])
    allseqs = seqs + mseqs
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    log("ptr spec {} parent {} games {} + Metrica {} ({} seqs, {} steps) load {:.0f}s".format(
        a.spec, pp, len(games), a.met, len(mseqs), int(wm.sum()), time.time() - t0))
    torch.set_num_threads(a.threads)
    model = PN.PointerNet(len(names), cfg)
    model.load_state_dict({k[2:]: torch.from_numpy(parent[k].astype(np.float32)) for k in parent.files if k.startswith("w/")})
    lr = LR_MULT * cfg["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
    n_it = EPOCHS * EPOCH_CROPS // cfg["batch"]
    warm = max(1, int(cfg["warm"] * n_it))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    ls = cfg["ls"]
    t1 = time.time()
    acc = []
    for it in range(n_it):
        model.train()
        b = TP.make_batch(allseqs, w, rng, cfg, perm, signs)
        tb = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
        o = model(tb["xs"], tb["ms"], tb["xf"], tb["X"])
        y = tb["y"]
        valid = y < 3
        lt, lsm = PN.team_probs(o["logits"], o["team"], tb["ms"])
        lt = torch.logaddexp(lt, torch.full_like(lt, float(np.log(1e-3))))    # = train_ptr (ballc)
        yt = y.clamp(max=2)
        nll = -(1 - ls) * torch.gather(lt, -1, yt[..., None])[..., 0] - ls * lt.mean(-1)
        l_team = nll[valid].mean()
        own = tb["own"].clone()
        own[~valid] = -1
        l_ptr = F.nll_loss(lsm.reshape(-1, lsm.shape[-1]), own.reshape(-1), ignore_index=-1) \
            if (own >= 0).any() else torch.zeros(())
        l_ho = F.binary_cross_entropy_with_logits(o["ho"], tb["yc"])
        l_dir = (F.cross_entropy(o["dir"], yt, reduction="none", label_smoothing=ls)[valid]).mean()
        loss = l_team + cfg["lam_ptr"] * l_ptr + cfg["c_w"] * l_ho + cfg["lam_dir"] * l_dir
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        if it % 10 == 0:
            acc.append([float(v.detach()) for v in (l_team, l_ptr, l_ho, l_dir)])
        if (it + 1) % 50 == 0:
            m = np.mean(acc[-5:], 0)
            log("  it {} team {:.4f} ptr {:.4f} ho {:.4f} dir {:.4f} lr {:.2e} {:.0f}s".format(
                it + 1, *m, opt.param_groups[0]["lr"], time.time() - t1))
    out_cfg = dict(pcfg, met_ft=dict(parent=pp, met_games=a.met, lr=lr, iters=n_it, met_share=MET_SHARE, seed=a.seed))
    W = {"w/" + k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    tmp = a.out[:-4] + ".tmp.npz"
    np.savez(tmp, mean=parent["mean"], std=parent["std"], names=parent["names"], kinds=parent["kinds"],
             cfg=json.dumps(out_cfg), train_games=np.array(train_games), spec=np.array(a.spec),
             change_prior=parent["change_prior"], pff_holdout_trained=parent["pff_holdout_trained"], **W)
    os.replace(tmp, a.out)


def ft_mstcn(a, log):
    pp = TM.out_name(a.parent, a.spec)
    parent = np.load(pp)
    assert str(parent["spec"]) == a.spec, (pp, str(parent["spec"]))
    cfg = dict(TM.CFG, **json.loads(str(parent["train_cfg"])))
    assert not cfg["init"] and cfg["src"] == "all" and cfg["dfl_fold_s"] == 0 and cfg["pff_fold_s"] == 0
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    seqs, names, kinds, games = D.load(a.ds, a.spec)
    mseqs, mnames, _, mgames = D.load(a.ds_met, "full")
    assert mnames == names == [str(s) for s in parent["names"]]
    mseqs = [s for s in mseqs if s[0] in a.met]
    assert {s[0] for s in mseqs} == set(a.met), (a.met, mgames)
    train_games = check_games(games, parent, a.met)
    mean, std = parent["mean"].astype(np.float64), parent["std"].astype(np.float64)
    normed = [(g.startswith("DFL"), np.nan_to_num((X - mean) / std).astype(np.float16), y, b[:2])
              for g, X, y, b in seqs + mseqs]
    n0 = len(seqs)
    del seqs
    lens = np.array([len(s[2]) for s in normed], float)
    w0 = weights(lens[:n0], np.array([s[0] for s in normed[:n0]]), cfg["w_dfl"])
    wm = lens[n0:]
    w = np.concatenate([(1 - MET_SHARE) * w0, MET_SHARE * wm / wm.sum()])
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    log("mstcn spec {} parent {} games {} + Metrica {} ({} seqs, {} steps) load {:.0f}s".format(
        a.spec, pp, len(games), a.met, len(normed) - n0, int(wm.sum()), time.time() - t0))
    torch.set_num_threads(a.threads)
    model = TM.ASRF(len(names), cfg)
    model.load_state_dict({k[2:]: torch.from_numpy(parent[k].astype(np.float32)) for k in parent.files if k.startswith("w/")})
    assert cfg["ema"] == 0
    lr = LR_MULT * cfg["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"]) if cfg["wd"] > 0 else \
        torch.optim.Adam(model.parameters(), lr=lr)
    n_it = EPOCHS * EPOCH_CROPS // cfg["batch"]
    sched = None
    if cfg["cosine"]:
        warm = max(1, int(cfg["warm"] * n_it))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    T = cfg["crop"]
    pw = torch.tensor(cfg["pos_weight"])
    cw = torch.tensor([1.0, 1.0, cfg["cw_d"]])
    permt = torch.from_numpy(perm)
    signst = torch.from_numpy(signs)[:, None]
    t1 = time.time()
    tot = []
    for it in range(n_it):
        model.train()
        xb, yb, bb = [], [], []
        for _ in range(cfg["batch"]):
            i = int(rng.choice(len(normed), p=w))
            _, Xn, y, b = normed[i]
            s0 = int(rng.integers(0, max(1, len(y) - T)))
            x = torch.from_numpy(Xn[s0:s0 + T].T.astype(np.float32))
            yy = torch.from_numpy(y[s0:s0 + T])
            if rng.random() < 0.5:
                x = x[permt] * signst
                yy = torch.where(yy == 0, 1, torch.where(yy == 1, 0, yy))
            xb.append(x)
            yb.append(yy)
            bb.append(torch.from_numpy(b[:, s0:s0 + T]))
        Tm = min(v.shape[1] for v in xb)
        x = torch.stack([v[:, :Tm] for v in xb])
        y = torch.stack([v[:Tm] for v in yb])
        b = torch.stack([v[:, :Tm] for v in bb])
        cs, bs = model(x)
        valid = y < 3
        loss = 0
        for c in cs:                                   # = train_mstcn.train_spec
            lp = F.log_softmax(c, 1)
            yt = y.clamp(max=2)
            nll = F.nll_loss(lp, yt, weight=cw, reduction="none")
            if cfg["ls"] > 0:
                nll = (1 - cfg["ls"]) * nll - cfg["ls"] * (lp * cw[None, :, None]).mean(1)
            if cfg["focal"] > 0:
                pt = torch.gather(lp, 1, yt[:, None]).squeeze(1).exp()
                nll = nll * (1 - pt).pow(cfg["focal"])
            loss = loss + nll[valid].mean()
            d = (lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2
            d = torch.clamp(d, max=cfg["tau_tmse"] ** 2)
            vm = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)
            loss = loss + cfg["lam_tmse"] * (d * vm).sum() / (vm.sum() * 3 + 1)
        for bl in bs:
            l = F.binary_cross_entropy_with_logits(bl, b, pos_weight=pw, reduction="none")
            loss = loss + cfg["w_b"] * (l * valid.unsqueeze(1)).sum() / (valid.sum() * 2 + 1)
        opt.zero_grad()
        loss.backward()
        if cfg["clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        if sched is not None:
            sched.step()
        tot.append(float(loss.detach()))
        if (it + 1) % 50 == 0:
            log("  it {} loss {:.4f} lr {:.2e} {:.0f}s".format(it + 1, np.mean(tot[-50:]), opt.param_groups[0]["lr"],
                                                               time.time() - t1))
    cfg = dict(cfg, device="cpu", pff_holdout_trained=bool(parent["pff_holdout_trained"]),
               met_ft=dict(parent=pp, met_games=a.met, lr=lr, iters=n_it, met_share=MET_SHARE, seed=a.seed))
    TM.export(model, mean, std, names, cfg, a.out, train_games, a.spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["ptr", "mstcn"])
    ap.add_argument("--parent", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--met", required=True, help="comma-separated MET-G<k> games to fine-tune on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", default=os.path.join(WORK or "", "ds_pff2", "ds"))
    ap.add_argument("--ds-met", default=os.path.join(WORK or "", "ds_met", "ds"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.met = sorted(a.met.split(","))
    assert a.spec == "full" or a.spec.startswith("dfl"), a.spec
    for k in ("NN_PFF_ALLOW_HOLDOUT", "NN_PFF_SKC_CV"):
        assert not os.environ.get(k), k
    if os.path.isfile(a.out):
        print("exists", a.out)
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    logf = open(a.out[:-4] + ".log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    t0 = time.time()
    log("MET-FT {} {} spec {} met {} -> {}".format(a.kind, a.parent, a.spec, a.met, a.out))
    (ft_ptr if a.kind == "ptr" else ft_mstcn)(a, log)
    log("done {} ({:.1f} min)".format(a.out, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
