#!/usr/bin/env python3
"""Round 5, R5-5: segment prediction on the nn_pff MS-TCN trunk (m5c_W recipe). Same model, data, folds and
loss as candidates/nn_pff/train_mstcn.py, plus an anchor-free segment head (ActionFormer-style) on the
backbone's last hidden layer: at every 5 fps step t it regresses the distances (in steps, log scale) to the
start and the end of the spell that contains t, trained with a 1-D IoU loss
    IoU = (min(ps, s) + min(pe, e)) / (max(ps, s) + max(pe, e)),  loss = lam_seg * (1 - IoU)
on graded steps whose spell lies inside the training sequence (spells cut by a sequence edge are censored).
Spells = maximal runs of one code (H, A, D) of the 5 fps label sequence.

Export = train_mstcn.py's npz (the members' numpy code still runs it) + the segment head under "seg/" keys
(seg_np.py), read by pipeline.py, which turns the predicted segments into boundary votes for the ASRF decoder.

    python -I candidates/nn_seg/train_seg.py --device cuda --ds $MTAH_WORK/ds_pff2/ds --out <dir> --specs ... \
        --cfg '{"cosine":0,"clip":0,"C":128,"Cr":64}'
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
sys.path.append(os.path.join(CAND, "nn_aux"))
import data as D          # noqa: E402
import features as fe     # noqa: E402
import train_mstcn as TM  # noqa: E402
import train_mstcn_aux as TA  # noqa: E402  (backbone_fwd)

CFG = dict(TM.CFG, lam_seg=1.0, seg_hidden=64, seg_cap=600.0)


class SegHead(nn.Module):
    def __init__(self, C, hidden):
        super().__init__()
        self.c1 = nn.Conv1d(C, hidden, 1)
        self.c2 = nn.Conv1d(hidden, 2, 1)

    def forward(self, h):
        return self.c2(F.relu(self.c1(h)))


def seg_targets(y, cap):
    """(2, T) float32 distances (steps) to the start / end of each step's spell; NaN = censored or ungraded."""
    n = len(y)
    ch = np.concatenate(([0], np.where(y[1:] != y[:-1])[0] + 1, [n]))
    out = np.full((2, n), np.nan, np.float32)
    for a, b in zip(ch[:-1], ch[1:]):
        if a == 0 or b == n or y[a] == 3:
            continue
        t = np.arange(a, b)
        out[0, a:b] = np.minimum(t - a + 0.5, cap)
        out[1, a:b] = np.minimum(b - t - 0.5, cap)
    return out


def iou_loss(o, tg):
    m = ~torch.isnan(tg[:, 0])
    if not m.any():
        return o.sum() * 0.0
    p = torch.exp(o.clamp(-3.0, 7.0))
    t = torch.nan_to_num(tg)
    inter = torch.minimum(p[:, 0], t[:, 0]) + torch.minimum(p[:, 1], t[:, 1])
    union = torch.maximum(p[:, 0], t[:, 0]) + torch.maximum(p[:, 1], t[:, 1])
    return (1 - inter / union)[m].mean()


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec)
    assert not D.excluded(spec) & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mean, std = TM.sym_stats(seqs, perm, signs)
    normed = []
    for g, X, y, b in seqs:
        normed.append((g.startswith("DFL"), np.nan_to_num((X - mean) / std).astype(np.float16), y, b[:2],
                       seg_targets(y, cfg["seg_cap"])))
    del seqs
    lens = np.array([len(s[2]) for s in normed], float)
    isd = np.array([s[0] for s in normed])
    w = lens.copy()
    if 0 < cfg["w_dfl"] < 1 and isd.any() and (~isd).any():
        w[isd] *= cfg["w_dfl"] / w[isd].sum()
        w[~isd] *= (1 - cfg["w_dfl"]) / w[~isd].sum()
    w /= w.sum()
    log("spec {} games {} seqs {} load {:.0f}s".format(spec, len(games), len(normed), time.time() - t0))
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = TM.ASRF(len(names), cfg).to(dev)
    seg = SegHead(cfg["C"], cfg["seg_hidden"]).to(dev)
    params = list(model.parameters()) + list(seg.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"]) if cfg["wd"] > 0 else torch.optim.Adam(params, lr=cfg["lr"])
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
        seg.train()
        t1 = time.time()
        tot, ls_ = [], []
        for _ in range(cfg["crops_per_epoch"] // cfg["batch"]):
            xb, yb, bb, sb = [], [], [], []
            for _ in range(cfg["batch"]):
                _, Xn, y, b, st = normed[int(rng.choice(len(normed), p=w))]
                s0 = int(rng.integers(0, max(1, len(y) - T)))
                x = torch.from_numpy(Xn[s0:s0 + T].T.astype(np.float32))
                yy = torch.from_numpy(y[s0:s0 + T])
                if rng.random() < 0.5:
                    x = x[permt] * signst
                    yy = torch.where(yy == 0, 1, torch.where(yy == 1, 0, yy))
                xb.append(x)
                yb.append(yy)
                bb.append(torch.from_numpy(b[:, s0:s0 + T]))
                sb.append(torch.from_numpy(st[:, s0:s0 + T]))
            Tm = min(v.shape[1] for v in xb)
            x = torch.stack([v[:, :Tm] for v in xb]).to(dev)
            y = torch.stack([v[:Tm] for v in yb]).to(dev)
            b = torch.stack([v[:, :Tm] for v in bb]).to(dev)
            stg = torch.stack([v[:, :Tm] for v in sb]).to(dev)
            cs, bs, h = TA.backbone_fwd(model, x)
            valid = y < 3
            loss = 0
            for c in cs:
                lp = F.log_softmax(c, 1)
                yt = y.clamp(max=2)
                nll = F.nll_loss(lp, yt, weight=cw, reduction="none")
                if cfg["ls"] > 0:
                    nll = (1 - cfg["ls"]) * nll - cfg["ls"] * (lp * cw[None, :, None]).mean(1)
                loss = loss + nll[valid].mean()
                d = torch.clamp((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2, max=cfg["tau_tmse"] ** 2)
                vm = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)
                loss = loss + cfg["lam_tmse"] * (d * vm).sum() / (vm.sum() * 3 + 1)
            for bl in bs:
                l = F.binary_cross_entropy_with_logits(bl, b, pos_weight=pw, reduction="none")
                loss = loss + cfg["w_b"] * (l * valid.unsqueeze(1)).sum() / (valid.sum() * 2 + 1)
            lseg = iou_loss(seg(h), stg)
            loss = loss + cfg["lam_seg"] * lseg
            opt.zero_grad()
            loss.backward()
            if cfg["clip"] > 0:
                nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            if sched is not None:
                sched.step()
            tot.append(float(loss.detach()))
            ls_.append(float(lseg.detach()))
        log("{} ep {} loss {:.4f} seg(1-IoU) {:.4f} {:.0f}s".format(spec, ep, np.mean(tot), np.mean(ls_), time.time() - t1))
    model.to("cpu")
    seg.to("cpu")
    cfg = dict(cfg, device=device, pff_holdout_trained=D.allow_holdout(), trainer="nn_seg/train_seg.py")
    TM.export(model, mean, std, names, cfg, out_path, games, spec)
    z = dict(np.load(out_path))
    for k, v in seg.state_dict().items():
        z["seg/" + k] = v.numpy().astype(np.float32)
    tmp = out_path[:-4] + ".tmp2.npz"
    np.savez(tmp, **z)
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
    assert not cfg["init"] and cfg["ema"] == 0 and cfg["focal"] == 0 and cfg["src"] == "all"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_cfg.json"), "w") as h:
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), model="mstcn_seg"), h, indent=1)
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
