#!/usr/bin/env python3
"""Fine-tune MOMENT-1-small (model.MomentTCN) on nn_pff's dataset and protocol; exports numpy weights.

    OMP_NUM_THREADS=2 PYTORCH_ENABLE_MPS_FALLBACK=1 python -I candidates/nn_tsfm/train.py \
        --ds $MTAH_WORK/ds_pff/ds --out $MTAH_WORK/arch/nn_tsfm/<config> --device mps \
        --specs pff0,pff1,pff2,pff3,dfl0,...,dfl6,full [--cfg '{...}']

Data, folds, targets, normalisation and input preparation are candidates/nn_pff's (data.load,
train_tcn.norm_stats / prep: z-scores clipped at 6, NaN -> 0, missing indicators), unchanged, so
this model sees exactly the input of nn_pff's TCN (e3_tcn). Sampling = train_tcn: sequences
drawn with probability proportional to length, DFL mass w_dfl, random crops of `crop` steps
(multiple of 8), team-swap augmentation p 0.5 (column permutation + signs of features.swap_index,
labels H<->A). Loss = train_tcn: class-weighted CE on H/A/D (graded steps) + c_w * BCE on the
hand-over target. Optimiser AdamW, two learning rates (backbone = T5 encoder + value embedding,
head = everything new), OneCycle, grad clip 1.

cfg knobs: pretrained (1: load MOMENT weights, 0: random init of the same architecture),
freeze (1: encoder frozen), Cm / dp / Ch / Cs / Chh / dils (architecture), lr_bb / lr_head,
iters / batch / crop, w_dfl, c_w, dead_w, wd, seed, drop / in_drop / t5_drop.

Output: <out>/fold_pff_<k>.npz, fold_tsfm_<g>.npz (DFL LOGO, bench.py kind "tsfm"), full.npz:
the model's float32 tensors (model.export) + mu, sd, ind, names, kinds, change_prior, train_games,
spec, arch (json), train_cfg (json). Only the exported file is used at inference (moment_np.py).
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
sys.path[:0] = [os.path.join(CAND, "nn_pff"), HERE]
import data as D            # noqa: E402  (candidates/nn_pff/data.py)
import features as fe       # noqa: E402  (candidates/nn_pff/features.py)
import train_tcn as TT      # noqa: E402  (norm_stats, prep)
import model as M           # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(os.path.dirname(CAND), "external", "work")
SNAPSHOT = os.path.join(WORK, "arch", "nn_tsfm", "hf", "MOMENT-1-small")

CFG = dict(pretrained=1, freeze=0, Cm=16, dp=32, Ch=96, Cs=64, Chh=96, dils="1,2,4,8", drop=0.1,
           in_drop=0.1, t5_drop=0.1, lr_bb=1e-4, lr_head=1e-3, iters=2000, batch=8, crop=512,
           dead_w=1.0, c_w=0.5, seed=0, w_dfl=0.15, wd=1e-4, snapshot=SNAPSHOT)


def out_name(out, spec):
    if spec == "full":
        return os.path.join(out, "full.npz")
    if spec.startswith("pff"):
        return os.path.join(out, "fold_pff_{}.npz".format(int(spec[3:])))
    return os.path.join(out, "fold_tsfm_{}.npz".format(int(spec[3:])))


def build(din, cfg, kw, used):
    dils = tuple(int(v) for v in cfg["dils"].split(","))
    m = M.MomentTCN(din, kw, Cm=cfg["Cm"], dp=cfg["dp"], Ch=cfg["Ch"], Cs=cfg["Cs"], Chh=cfg["Chh"],
                    dils=dils, drop=cfg["drop"], in_drop=cfg["in_drop"], t5_dropout=cfg["t5_drop"],
                    pe=used["pe"], max_patches=1024)
    if cfg["pretrained"]:
        m.load_pretrained(used)
    if cfg["freeze"]:
        for p in m.encoder.parameters():
            p.requires_grad_(False)
        m.value_embedding.weight.requires_grad_(False)
    return m


def train_spec(ds_dir, spec, out_path, cfg, log, device):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec)
    assert not D.excluded(spec) & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mu, sd, ind = TT.norm_stats(seqs, perm, signs)
    pos = {int(j): i for i, j in enumerate(ind)}
    assert all(int(perm[j]) in pos for j in ind), "missing indicators not swap-symmetric"
    permf = np.concatenate([perm, [len(perm) + pos[int(perm[j])] for j in ind]])
    signf = np.concatenate([signs, np.ones(len(ind), np.float32)])
    data = [(g.startswith("DFL"), TT.prep(X, mu, sd, ind).astype(np.float16), y, b[2].astype(np.float32))
            for g, X, y, b in seqs]
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
    din = data[0][1].shape[1]
    kw, used = M.load_moment(cfg["snapshot"])
    model = build(din, cfg, kw, used)
    dev = torch.device(device)
    model.to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("params {} trainable {} device {} pretrained {} freeze {}".format(
        sum(p.numel() for p in model.parameters()), n_tr, device, cfg["pretrained"], cfg["freeze"]))
    groups = model.param_groups(cfg["lr_bb"], cfg["lr_head"], cfg["wd"])
    groups = [dict(g, params=[p for p in g["params"] if p.requires_grad]) for g in groups]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=cfg["iters"], pct_start=0.1)
    L = cfg["crop"]
    assert L % M.PATCH == 0
    cw = torch.tensor([1.0, 1.0, cfg["dead_w"]], device=dev)
    permt, signt = torch.from_numpy(permf), torch.from_numpy(signf)[:, None]
    t1 = time.time()
    run = []
    for it in range(cfg["iters"]):
        model.train()
        xs = np.zeros((cfg["batch"], din, L), np.float32)
        ys = np.full((cfg["batch"], L), 3, np.int64)
        cs = np.zeros((cfg["batch"], L), np.float32)
        vs = np.zeros((cfg["batch"], L), bool)
        for i in range(cfg["batch"]):
            j = int(rng.choice(len(data), p=w))
            _, Z, y, c = data[j]
            s = int(rng.integers(0, max(1, len(y) - L)))
            n = min(L, len(y) - s)
            x = torch.from_numpy(Z[s:s + n].T.astype(np.float32))
            yy = y[s:s + n].copy()
            if rng.random() < 0.5:
                x = x[permt] * signt
                yy = np.where(yy == 0, 1, np.where(yy == 1, 0, yy))
            xs[i, :, :n], ys[i, :n], cs[i, :n], vs[i, :n] = x.numpy(), yy, c[s:s + n], True
        x = torch.from_numpy(xs).to(dev)
        y = torch.from_numpy(ys).to(dev)
        c = torch.from_numpy(cs).to(dev)
        valid = torch.from_numpy(vs).to(dev)
        out = model(x, valid)
        graded = (y < 3) & valid
        ce = Fn.cross_entropy(out[:, :3], y.clamp(max=2), weight=cw, reduction="none")
        wt = graded.float()
        bce = Fn.binary_cross_entropy_with_logits(out[:, 3], c, reduction="none")
        vf = valid.float()
        loss = (ce * wt).sum() / wt.sum() + cfg["c_w"] * (bce * vf).sum() / vf.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            run.append(float(loss.detach()))
        if (it + 1) % 250 == 0:
            log("  {} it {} loss {:.4f} {:.0f}s".format(spec, it + 1, np.mean(run[-10:]), time.time() - t1))
    model.to("cpu")
    model.eval()
    sdict = model.export()
    arch = dict(din=din, Cm=cfg["Cm"], dp=cfg["dp"], Ch=cfg["Ch"], Cs=cfg["Cs"], Chh=cfg["Chh"],
                dils=[int(v) for v in cfg["dils"].split(",")], t5=kw)
    meta = dict(mu=mu, sd=sd, ind=ind.astype(np.int64), names=np.array(names), kinds=np.array(kinds),
                change_prior=np.array(float(np.mean(np.concatenate([d[3] for d in data])))),
                train_games=np.array(games), spec=np.array(spec), arch=json.dumps(arch),
                train_cfg=json.dumps(dict(cfg, device=device)))
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, **sdict, **meta)
    os.replace(tmp, out_path)
    log("{} -> {} ({:.0f} min, {:.1f} MB)".format(spec, out_path, (time.time() - t0) / 60,
                                                   os.path.getsize(out_path) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="pff0,pff1,pff2,pff3")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--device", default="cpu", help="cpu or mps")
    a = ap.parse_args()
    cfg = dict(CFG, **json.loads(a.cfg))
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_cfg.json"), "w") as h:
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), model="moment_tcn"), h, indent=1)
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
        train_spec(a.ds, spec, p, cfg, log, a.device)


if __name__ == "__main__":
    main()
