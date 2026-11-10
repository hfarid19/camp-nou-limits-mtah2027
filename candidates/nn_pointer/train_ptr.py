#!/usr/bin/env python3
"""Train the possessor-pointer model (round 5, R5-3; ptr_net.py) on nn_pff's PFF + DFL data and export numpy
weights (ptr_np.py runs them).

    python -I candidates/nn_pointer/train_ptr.py --device cuda --ds $MTAH_WORK/ds_pff2/ds --out <dir> \
        --specs pff0,pff1,pff2,pff3 [--cfg '{...}']

Data per ds file: nn_pff's features X, labels y and hand-over target (data.targets), the slot inputs
(prep_ptr.py: <ds>/../ptr) and the possessor slot (make_owner.py: <ds>/../owner). Fold specs, holdout
refusals and file names = nn_pff (data.py; fold_pff_<k>.npz, fold_tcn_<g>.npz, full.npz, each with
train_games and spec).
Loss = team NLL of the pointer's team marginals (label smoothing ls) + lam_ptr * pointer CE (where the
possessor slot is known; dead frames -> the dead class) + c_w * hand-over BCE + lam_dir * direct team
head CE. Team-swap augmentation p = 0.5.
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
import ptr_net as PN      # noqa: E402
import data as D          # noqa: E402
import features as fe     # noqa: E402
import train_mstcn as TM  # noqa: E402

CFG = dict(D=64, heads=4, nsab=2, C=128, L=10, Dp=32, Lp=5, dropout=0.3, dropout_s=0.1, in_drop=0.0,
           lr=5e-4, wd=1e-4, iters=6000, batch=8, crop=1024, warm=0.03, clip=1.0, seed=0, w_dfl=0.15,
           ls=0.05, lam_ptr=0.5, c_w=0.5, lam_dir=0.3, mix=0.0, noX=0, ballc=0, FS=11, FF=10)
SMAX = 48


def remap_owner(own, sel):
    """slot index -> position among the kept (ball-centred) players; -1 if the possessor is not kept; -2 dead stays."""
    out = np.where(own == -2, -2, -1).astype(np.int64)
    hit = (sel == own[:, None]) & (own[:, None] >= 0)
    has = hit.any(1)
    out[has] = np.argmax(hit[has], 1)
    return out


def load(ds_dir, spec, ballc=False):
    hold = D.holdout()
    if D.allow_holdout():
        if spec.startswith("pff"):
            raise ValueError("NN_PFF_ALLOW_HOLDOUT=1: only full / dfl<g>")
        hold = set()
    shold = D.skc_holdout()
    ex = D.excluded(spec)
    base = os.path.dirname(os.path.abspath(ds_dir))
    seqs, games, names, kinds = [], set(), None, None
    import glob
    for p in sorted(glob.glob(os.path.join(ds_dir, "*_*.npz"))):
        g = D.game_of(p)
        assert g[4:] not in hold or not g.startswith("PFF-"), p
        assert g not in shold, p
        if g in ex:
            continue
        bn = os.path.basename(p)
        y, bha, bd, yc, idx, half = D.targets(p, os.path.join(base, "raw", bn), os.path.join(base, "tgt"))
        z = np.load(p)
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
        pz = np.load(os.path.join(base, "ptrb" if ballc else "ptr", bn))
        oz = np.load(os.path.join(base, "owner", bn))
        assert np.array_equal(pz["idx"], idx) and np.array_equal(oz["idx"], idx), p
        X, xs, ms, xf, own = z["X"], pz["xs"], pz["ms"], pz["xf"], oz["own"]
        if ballc:
            own = remap_owner(own.astype(np.int64), pz["sel"].astype(np.int64))
        brk = np.concatenate(([True], (np.diff(idx) != 5) | (np.diff(half) != 0)))
        sid = np.cumsum(brk) - 1
        for s in np.unique(sid):
            m = sid == s
            if m.sum() < 64:
                continue
            seqs.append(dict(g=g, X=X[m].astype(np.float32), y=y[m].astype(np.int64), yc=yc[m].astype(np.float32),
                             xs=xs[m], ms=ms[m], xf=xf[m], own=own[m].astype(np.int64)))
        games.add(g)
    return seqs, names, kinds, sorted(games)


def make_batch(data, w, rng, cfg, perm, signs):
    T = cfg["crop"]
    items = []
    for _ in range(cfg["batch"]):
        d = data[int(rng.choice(len(data), p=w))]
        n = len(d["y"])
        s0 = int(rng.integers(0, max(1, n - T)))
        sl = slice(s0, s0 + T)
        it = dict(X=d["Xn"][sl].astype(np.float32), y=d["y"][sl].copy(), yc=d["yc"][sl], xs=d["xs"][sl].astype(np.float32),
                  ms=d["ms"][sl].astype(np.float32), xf=d["xf"][sl].astype(np.float32), own=d["own"][sl])
        if rng.random() < 0.5:
            it["X"] = it["X"][:, perm] * signs
            it["xs"] = it["xs"].copy()
            it["xf"] = it["xf"].copy()
            if cfg["ballc"]:
                it["xs"][..., 0:5] *= -1.0
                it["xf"][:, [0, 1, 10, 11]] *= -1.0
            else:
                it["xs"][..., 0:7] *= -1.0
                it["xf"][:, 0:2] *= -1.0
            y = it["y"]
            it["y"] = np.where(y == 0, 1, np.where(y == 1, 0, y))
        items.append(it)
    Tm = min(len(it["y"]) for it in items)
    S = max(it["xs"].shape[1] for it in items)
    out = dict(X=np.stack([it["X"][:Tm].T for it in items]), y=np.stack([it["y"][:Tm] for it in items]),
               yc=np.stack([it["yc"][:Tm] for it in items]), xf=np.stack([it["xf"][:Tm] for it in items]))
    xs = np.zeros((len(items), Tm, S, items[0]["xs"].shape[2]), np.float32)
    ms = np.zeros((len(items), Tm, S), np.float32)
    own = np.stack([it["own"][:Tm] for it in items])
    for i, it in enumerate(items):
        k = it["xs"].shape[1]
        xs[i, :, :k] = it["xs"][:Tm]
        ms[i, :, :k] = it["ms"][:Tm]
    keep = ms.max((0, 1)) > 0
    kept = np.where(keep)[0]
    remap = -np.ones(S, np.int64)
    remap[kept] = np.arange(len(kept))
    Sk = len(kept)
    tgt = np.where(own == -2, Sk, np.where(own >= 0, remap[np.clip(own, 0, S - 1)], -1))
    out.update(xs=xs[:, :, keep], ms=ms[:, :, keep], own=tgt)
    return out


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = load(ds_dir, spec, bool(cfg["ballc"]))
    assert not D.excluded(spec) & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mean, std = TM.sym_stats([(s["g"], s["X"], None, None) for s in seqs], perm, signs)
    for s in seqs:
        s["Xn"] = np.nan_to_num((s.pop("X") - mean) / std).astype(np.float16)
    lens = np.array([len(s["y"]) for s in seqs], float)
    isd = np.array([s["g"].startswith("DFL") for s in seqs])
    w = lens.copy()
    if 0 < cfg["w_dfl"] < 1 and isd.any() and (~isd).any():
        w[isd] *= cfg["w_dfl"] / w[isd].sum()
        w[~isd] *= (1 - cfg["w_dfl"]) / w[~isd].sum()
    w /= w.sum()
    own_all = np.concatenate([s["own"] for s in seqs])
    yall = np.concatenate([s["y"] for s in seqs])
    log("spec {} games {} (PFF {}, DFL {}) seqs {} steps {} owner known (alive) {:.3f} load {:.0f}s".format(
        spec, len(games), sum(g.startswith("PFF") for g in games), sum(g.startswith("DFL") for g in games),
        len(seqs), int(lens.sum()), float((own_all[yall < 2] >= 0).mean()), time.time() - t0))
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = PN.PointerNet(len(names), cfg).to(dev)
    log("params {}".format(sum(p.numel() for p in model.parameters())))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n_it = cfg["iters"]
    warm = max(1, int(cfg["warm"] * n_it))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    ls = cfg["ls"]
    t1 = time.time()
    acc = []
    for it in range(n_it):
        model.train()
        b = make_batch(seqs, w, rng, cfg, perm, signs)
        tb = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in b.items()}
        o = model(tb["xs"], tb["ms"], tb["xf"], tb["X"])
        y = tb["y"]
        valid = y < 3
        lt, lsm = PN.team_probs(o["logits"], o["team"], tb["ms"])
        if cfg["ballc"]:
            # with only the K nearest players kept, a team can be absent from a frame (log P = -1e4): floor at 1e-3
            lt = torch.logaddexp(lt, torch.full_like(lt, float(np.log(1e-3))))
        yt = y.clamp(max=2)
        nll = -(1 - ls) * torch.gather(lt, -1, yt[..., None])[..., 0] - ls * lt.mean(-1)
        l_team = nll[valid].mean()
        own = tb["own"].clone()
        own[~valid] = -1
        l_ptr = F.nll_loss(lsm.reshape(-1, lsm.shape[-1]), own.reshape(-1), ignore_index=-1) \
            if (own >= 0).any() else torch.zeros((), device=dev)
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
        if (it + 1) % 250 == 0:
            a = np.mean(acc[-25:], 0)
            log("  {} it {} team {:.4f} ptr {:.4f} ho {:.4f} dir {:.4f} {:.0f}s".format(spec, it + 1, *a, time.time() - t1))
    model.to("cpu")
    cfg = dict(cfg, device=device, pff_holdout_trained=D.allow_holdout())
    W = {"w/" + k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, mean=mean.astype(np.float32), std=std.astype(np.float32), names=np.array(names), kinds=np.array(kinds),
             cfg=json.dumps(cfg), train_games=np.array(games), spec=np.array(spec),
             change_prior=np.array(float(np.mean(np.concatenate([s["yc"] for s in seqs])))),
             pff_holdout_trained=np.array(D.allow_holdout()), **W)
    os.replace(tmp, out_path)
    log("{} -> {} ({:.0f} min)".format(spec, out_path, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="pff0,pff1,pff2,pff3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = dict(CFG, **json.loads(a.cfg))
    assert set(cfg) == set(CFG), set(cfg) ^ set(CFG)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_cfg.json"), "w") as h:
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), model="pointer"), h, indent=1)
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
