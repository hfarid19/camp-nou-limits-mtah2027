#!/usr/bin/env python3
"""Train the nn_pff MS-TCN (ASRF: class branch + boundary branch) on PFF + DFL, export numpy weights.

    OMP_NUM_THREADS=2 python -I candidates/nn_pff/train_mstcn.py --ds $MTAH_WORK/ds_pff/ds \
        --out $MTAH_WORK/arch/nn_pff/<config> --specs pff0,pff1,pff2,pff3 [--cfg '{...}']

Each spec (data.py: pff<k>, dfl<g>, full) trains one model on the data that spec allows and writes
<out>/fold_<kind>_<i>.npz (pff<k> -> fold_pff_<k>.npz, dfl<g> -> fold_tcn_<g>.npz, the name
bench.py's dfl_folds.resolve looks for) or <out>/full.npz. Every file stores `train_games` (the
games it saw) and `spec`, which the PFF out-of-fold runner checks.

Architecture = handover_mstcn_v2/train_tcn.py (Farha & Gall MS-TCN; Ishikawa et al. ASRF) so the
exported file runs in that member's numpy code (tcn_np.Single): backbone of L dilated residual
layers (kernel 3, dilation 2^i) with a 3-class head (H, A, D) and a 2-logit boundary head
(H<->A, alive<->dead); S_c class refinement stages, S_b boundary refinement stages. Loss per stage:
CE (+ optional class weights) + lam_tmse * truncated MSE, boundary BCE with pos_weight. Team-swap
augmentation (features.swap_index; labels H<->A) with p = 0.5 per crop, done on the normalised
features (the normalisation statistics are symmetric under the swap, so it commutes).

Changes from handover_mstcn_v2 (all in CFG): crops are drawn with per-source weights (w_dfl:
total sampling share of the DFL files), cosine learning-rate decay with warm-up, weight decay,
input feature dropout (in_drop), and the length / width of the network. Optimisation / loss knobs
(off by default except clip): clip (gradient-norm clip, 0 = off as in handover_mstcn_v2), ema (export
an exponential moving average of the weights with this decay), focal (focal-loss gamma on the class
CE), ls (label smoothing), init (start from an exported npz of the same shape: fine-tuning) and
src (all / pff / dfl: which training games are used).
torch is used for training only.
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
sys.path.insert(0, HERE)
import data as D       # noqa: E402
import features as fe  # noqa: E402  (nn_pff/features.py = learned_dfl_evidence/features.py)

CFG = dict(C=64, L=10, Cr=32, Lr=10, S_c=2, S_b=1, lr=5e-4, epochs=40, crop=1024, batch=8,
           lam_tmse=0.15, tau_tmse=4.0, w_b=1.0, pos_weight=10.0, dropout=0.5, seed=0,
           crops_per_epoch=1200, wd=0.0, in_drop=0.0, w_dfl=0.15, cosine=1, warm=0.03,
           dfl_fold_s=0.0, pff_fold_s=0.0, cw_d=1.0, clip=5.0, ema=0.0, focal=0.0, ls=0.0,
           init="", src="all")


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
        self.in_drop = nn.Dropout(cfg["in_drop"])
        self.backbone = Stage(K, cfg["C"], cfg["L"], [3, 2], p)
        self.cref = nn.ModuleList([Stage(3, cfg["Cr"], cfg["Lr"], [3], p) for _ in range(cfg["S_c"])])
        self.bref = nn.ModuleList([Stage(2, cfg["Cr"], cfg["Lr"], [2], p) for _ in range(cfg["S_b"])])

    def forward(self, x):
        c, b = self.backbone(self.in_drop(x))
        cs, bs = [c], [b]
        for st in self.cref:
            cs.append(st(F.softmax(cs[-1], 1))[0])
        for st in self.bref:
            bs.append(st(torch.sigmoid(bs[-1]))[0])
        return cs, bs


def export(model, mean, std, names, cfg, path, train_games, spec):
    w = {}
    for k, v in model.state_dict().items():
        w["w/" + k] = v.detach().cpu().numpy().astype(np.float32)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, mean=mean.astype(np.float32), std=std.astype(np.float32), names=np.array(names),
             cfg=json.dumps({k: cfg[k] for k in ("C", "L", "Cr", "Lr", "S_c", "S_b")}),
             train_games=np.array(train_games), spec=np.array(spec), train_cfg=json.dumps(cfg), **w)
    os.replace(tmp, path)


def sym_stats(seqs, perm, signs):
    """NaN-aware mean / std over the training rows and their team-swap mirrors (float64 sums)."""
    K = seqs[0][1].shape[1]
    s1, s2, cnt = np.zeros(K), np.zeros(K), np.zeros(K)
    for _, X, _, _ in seqs:
        ok = ~np.isnan(X)
        Xz = np.where(ok, X, 0.0).astype(np.float64)
        s1 += Xz.sum(0)
        s2 += (Xz ** 2).sum(0)
        cnt += ok.sum(0)
    s1a = s1 + signs * s1[perm]
    s2a = s2 + s2[perm]
    ca = np.maximum(cnt + cnt[perm], 1)
    mean = s1a / ca
    std = np.sqrt(np.maximum(s2a / ca - mean ** 2, 0))
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return np.nan_to_num(mean), std


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device="cpu"):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec, cfg["dfl_fold_s"], cfg["pff_fold_s"])
    if cfg["src"] != "all":
        keep = "DFL" if cfg["src"] == "dfl" else "PFF"
        seqs = [s for s in seqs if s[0].startswith(keep)]
        games = [g for g in games if g.startswith(keep)]
    ex = D.excluded(spec)
    assert not ex & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mean, std = sym_stats(seqs, perm, signs)
    init = None
    if cfg["init"]:
        ip = out_name(cfg["init"], spec) if os.path.isdir(cfg["init"]) else cfg["init"]   # same spec: no leak
        init = np.load(ip)
        assert str(init["spec"]) == spec, (ip, str(init["spec"]), spec)
        mean, std = init["mean"].astype(np.float64), init["std"].astype(np.float64)   # keep the input scale
        games = sorted(set(games) | {str(g) for g in init["train_games"]})
    normed = []
    for g, X, y, b in seqs:
        Xn = np.nan_to_num((X - mean) / std).astype(np.float16)
        normed.append((g.startswith("DFL"), Xn, y, b[:2]))
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
    model = ASRF(len(names), cfg)
    if init is not None:
        sd = {k[2:]: torch.from_numpy(init[k].astype(np.float32)) for k in init.files if k.startswith("w/")}
        model.load_state_dict(sd)
    model.to(dev)
    ema = None
    if cfg["ema"] > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log("params {}".format(sum(p.numel() for p in model.parameters())))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) if cfg["wd"] > 0 else \
        torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_it = cfg["epochs"] * (cfg["crops_per_epoch"] // cfg["batch"])
    if cfg["cosine"]:
        warm = max(1, int(cfg["warm"] * n_it))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    else:
        sched = None
    T = cfg["crop"]
    pw = torch.tensor(cfg["pos_weight"], device=dev)
    cw = torch.tensor([1.0, 1.0, cfg["cw_d"]], device=dev)
    permt = torch.from_numpy(perm)
    signst = torch.from_numpy(signs)[:, None]
    it = 0
    for ep in range(cfg["epochs"]):
        model.train()
        t1 = time.time()
        tot = []
        for _ in range(cfg["crops_per_epoch"] // cfg["batch"]):
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
            x = torch.stack([v[:, :Tm] for v in xb]).to(dev)
            y = torch.stack([v[:Tm] for v in yb]).to(dev)
            b = torch.stack([v[:, :Tm] for v in bb]).to(dev)
            cs, bs = model(x)
            valid = y < 3
            loss = 0
            for c in cs:
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
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(cfg["ema"]).add_(v.detach(), alpha=1 - cfg["ema"])
                        else:
                            ema[k].copy_(v)
            if sched is not None:
                sched.step()
            it += 1
            if it % 25 == 0:
                tot.append(float(loss.detach()))
        log("{} ep {} loss {:.4f} lr {:.2e} {:.0f}s".format(spec, ep, np.mean(tot), opt.param_groups[0]["lr"],
                                                              time.time() - t1))
    if ema is not None:
        model.load_state_dict(ema)
    model.to("cpu")
    cfg = dict(cfg, device=device)
    export(model, mean, std, names, cfg, out_path, games, spec)
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
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds)), h, indent=1)
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
