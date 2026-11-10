#!/usr/bin/env python3
"""Train an nn_asformer network (model.py: asformer | global | temba) on PFF + DFL and export numpy
weights for asf_np.py.

    PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 python -I candidates/nn_asformer/train.py \
        --device mps --ds $MTAH_WORK/ds_pff/ds --out $MTAH_WORK/arch/nn_asformer/<config> \
        --specs pff0,pff1,pff2,pff3,dfl0,...,dfl6,full [--cfg '{...}']

Data, fold specs, targets and file names are the NN trainer's (candidates/nn_pff/data.py is imported
unchanged: pff<k> leaves out PFF selection fold k, dfl<g> leaves out DFL game g, full = everything;
the 16 holdout games are never in the dataset). Output names follow nn_pff: fold_pff_<k>.npz,
fold_tcn_<g>.npz (the name dfl_folds.resolve looks for), full.npz.

Recipe (all in CFG): crops of `crop` steps drawn with per-source weights (w_dfl = DFL sampling
share), team-swap augmentation with p = 0.5 per crop (features.swap_index; labels H<->A) on the
normalised features, loss per stage = CE + lam_tmse * truncated MSE on log-probs + boundary BCE
(pos_weight), AdamW (weight decay on weight matrices / conv kernels only), linear warm-up + cosine
decay, gradient-norm clipping, dropout + stochastic depth (model.py), EMA of the weights (exported),
float32 throughout (no mixed precision on MPS).

Early stopping / model selection on a TRAINING-side validation split: `val_games` PFF games are
drawn (seeded) from the spec's own training games and removed from training; every epoch the EMA
model's class CE on their sequences is measured and the best epoch's weights are exported (patience
in epochs, 0 = export the last epoch). The selection fold / DFL game / holdout are never used.

Checkpointing: after every epoch <out>/<spec>.ckpt.pt holds model, EMA, optimiser, scheduler, RNG
states, epoch, best-so-far and history; a run started with the same --out / --specs resumes from it
(removed after the export). A resumed run is not bit-identical to an uninterrupted one (MPS RNG).
"""

import argparse
import copy
import importlib.util
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
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import features as fe  # noqa: E402

_spec = importlib.util.spec_from_file_location("nn_pff_data", os.path.join(CAND, "nn_pff", "data.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

CFG = dict(M.CFG, lr=5e-4, epochs=40, crop=2048, batch=4, lam_tmse=0.15, tau_tmse=4.0, w_b=1.0,
           pos_weight=10.0, seed=0, crops_per_epoch=1200, wd=0.05, w_dfl=0.15, warm=0.05, clip=1.0,
           ema=0.999, val_games=4, patience=10, src="all")
MODEL_KEYS = tuple(M.CFG)


def export(state, mean, std, names, cfg, path, train_games, spec, stats):
    w = {"w/" + k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, mean=mean.astype(np.float32), std=std.astype(np.float32), names=np.array(names),
             cfg=json.dumps({k: cfg[k] for k in MODEL_KEYS}), train_games=np.array(train_games),
             spec=np.array(spec), train_cfg=json.dumps(cfg), stats=json.dumps(stats), **w)
    os.replace(tmp, path)


def sym_stats(seqs, perm, signs):
    """NaN-aware mean / std over the training rows and their team-swap mirrors (= nn_pff)."""
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


def losses(cs, bs, y, b, cfg, pw):
    valid = y < 3
    yt = y.clamp(max=2)
    loss = 0
    for c in cs:
        lp = F.log_softmax(c, -1)
        nll = F.nll_loss(lp.transpose(1, 2), yt, reduction="none")
        loss = loss + nll[valid].mean()
        d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
        d = torch.clamp(d, max=cfg["tau_tmse"] ** 2)
        vm = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
        loss = loss + cfg["lam_tmse"] * (d * vm).sum() / (vm.sum() * 3 + 1)
    for bl in bs:
        l = F.binary_cross_entropy_with_logits(bl, b, pos_weight=pw, reduction="none")
        loss = loss + cfg["w_b"] * (l * valid.unsqueeze(-1)).sum() / (valid.sum() * 2 + 1)
    return loss


def val_loss(model, val, cfg, dev):
    """Class CE of the last stage over the validation sequences, in non-overlapping crops."""
    model.eval()
    T = cfg["crop"]
    tot, n = 0.0, 0
    with torch.no_grad():
        for Xn, y in val:
            for s0 in range(0, len(y), T):
                x = torch.from_numpy(Xn[s0:s0 + T].astype(np.float32))[None].to(dev)
                yy = torch.from_numpy(y[s0:s0 + T])[None].to(dev)
                cs, _ = model(x)
                valid = yy < 3
                if valid.sum() == 0:
                    continue
                nll = F.cross_entropy(cs[-1].transpose(1, 2), yy.clamp(max=2), reduction="none")
                tot += float(nll[valid].sum())
                n += int(valid.sum())
    model.train()
    return tot / max(n, 1)


def train_spec(ds_dir, spec, out_path, cfg, threads, log, device):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec)
    if cfg["src"] != "all":
        keep = "DFL" if cfg["src"] == "dfl" else "PFF"
        seqs = [s for s in seqs if s[0].startswith(keep)]
        games = [g for g in games if g.startswith(keep)]
    assert not D.excluded(spec) & set(games)
    hold = D.holdout()
    assert not any(g.startswith("PFF-") and g[4:] in hold for g in games)
    # training-side validation split: PFF games of THIS spec's training set
    vrng = np.random.default_rng(cfg["seed"] * 1000 + 7)
    pff_games = sorted(g for g in games if g.startswith("PFF-"))
    val_set = set(vrng.choice(pff_games, size=min(cfg["val_games"], len(pff_games)), replace=False).tolist()) \
        if cfg["val_games"] > 0 else set()
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    train_seqs = [s for s in seqs if s[0] not in val_set]
    mean, std = sym_stats(train_seqs, perm, signs)
    normed, val = [], []
    for g, X, y, b in seqs:
        Xn = np.nan_to_num((X - mean) / std).astype(np.float16)
        if g in val_set:
            val.append((Xn, y))
        else:
            normed.append((g.startswith("DFL"), Xn, y, b[:2].T.copy()))
    del seqs
    lens = np.array([len(s[2]) for s in normed], float)
    isd = np.array([s[0] for s in normed])
    w = lens.copy()
    if 0 < cfg["w_dfl"] < 1 and isd.any() and (~isd).any():
        w[isd] *= cfg["w_dfl"] / w[isd].sum()
        w[~isd] *= (1 - cfg["w_dfl"]) / w[~isd].sum()
    w /= w.sum()
    log("spec {} games {} (PFF {}, DFL {}) val {} seqs {} steps {} load {:.0f}s".format(
        spec, len(games), sum(g.startswith("PFF") for g in games), sum(g.startswith("DFL") for g in games),
        sorted(val_set), len(normed), int(lens.sum()), time.time() - t0))
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = M.Net(len(names), cfg).to(dev)
    log("params {}".format(M.n_params(model)))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if (p.ndim >= 2 and n.endswith("weight")) else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.98))
    per_ep = cfg["crops_per_epoch"] // cfg["batch"]
    n_it = cfg["epochs"] * per_ep
    warm = max(1, int(cfg["warm"] * n_it))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg["ema"] > 0 else None
    T = cfg["crop"]
    pw = torch.tensor(cfg["pos_weight"], device=dev)
    permt, signst = torch.from_numpy(perm), torch.from_numpy(signs)[None, :]
    best, best_ep, best_state, bad = float("inf"), -1, None, 0
    hist = []
    it = 0
    ep0 = 0
    ckpt = out_path[:-4] + ".ckpt.pt"
    if os.path.isfile(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert ck["spec"] == spec and ck["cfg"] == {k: cfg[k] for k in cfg if k != "device"}, "checkpoint from another run"
        model.load_state_dict(ck["model"])
        model.to(dev)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema = {k: v.to(dev) for k, v in ck["ema"].items()}
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        best, best_ep, bad, hist, it = ck["best"], ck["best_ep"], ck["bad"], ck["hist"], ck["it"]
        best_state = ck["best_state"]
        ep0 = ck["epoch"] + 1
        log("{} resumed from {} at epoch {} (best val_ce {:.4f} at ep {})".format(spec, ckpt, ep0, best, best_ep))
    for ep in range(ep0, cfg["epochs"]):
        model.train()
        t1 = time.time()
        tot = []
        for _ in range(per_ep):
            xb, yb, bb = [], [], []
            for _ in range(cfg["batch"]):
                i = int(rng.choice(len(normed), p=w))
                _, Xn, y, b = normed[i]
                s0 = int(rng.integers(0, max(1, len(y) - T)))
                x = torch.from_numpy(Xn[s0:s0 + T].astype(np.float32))
                yy = torch.from_numpy(y[s0:s0 + T])
                if rng.random() < 0.5:
                    x = x[:, permt] * signst
                    yy = torch.where(yy == 0, 1, torch.where(yy == 1, 0, yy))
                xb.append(x)
                yb.append(yy)
                bb.append(torch.from_numpy(b[s0:s0 + T]))
            Tm = min(v.shape[0] for v in xb)
            x = torch.stack([v[:Tm] for v in xb]).to(dev)
            y = torch.stack([v[:Tm] for v in yb]).to(dev)
            b = torch.stack([v[:Tm] for v in bb]).to(dev)
            cs, bs = model(x)
            loss = losses(cs, bs, y, b, cfg, pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(cfg["ema"]).add_(v.detach(), alpha=1 - cfg["ema"])
                        else:
                            ema[k].copy_(v)
            it += 1
            if it % 10 == 0:
                tot.append(float(loss.detach()))
        tr = float(np.mean(tot))
        vl = float("nan")
        if val:
            evalm = M.Net(len(names), cfg).to(dev)
            evalm.load_state_dict(ema if ema is not None else model.state_dict())
            vl = val_loss(evalm, val, cfg, dev)
            del evalm
        hist.append((ep, tr, vl))
        log("{} ep {} loss {:.4f} val_ce {:.4f} lr {:.2e} {:.0f}s".format(spec, ep, tr, vl, opt.param_groups[0]["lr"],
                                                                      time.time() - t1))
        if val and cfg["patience"] > 0:
            if vl < best - 1e-4:
                best, best_ep, bad = vl, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in (ema if ema is not None else model.state_dict()).items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log("{} early stop at epoch {} (best {} val_ce {:.4f})".format(spec, ep, best_ep, best))
                    break
        torch.save(dict(spec=spec, cfg={k: cfg[k] for k in cfg if k != "device"}, epoch=ep, it=it,
                        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        ema=None if ema is None else {k: v.detach().cpu() for k, v in ema.items()},
                        opt=opt.state_dict(), sched=sched.state_dict(), np_rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), best=best, best_ep=best_ep, bad=bad, hist=hist,
                        best_state=best_state), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in (ema if ema is not None else model.state_dict()).items()}
        best_ep = hist[-1][0]
    stats = dict(params=M.n_params(model), epochs_run=len(hist), best_epoch=best_ep, best_val_ce=best,
                 final_train_loss=hist[-1][1], train_loss_at_best=hist[best_ep][1] if best_ep >= 0 else hist[-1][1],
                 val_games=sorted(val_set), hist=hist, minutes=(time.time() - t0) / 60, device=device)
    export(best_state, mean, std, names, dict(cfg, device=device), out_path, games, spec, stats)
    if os.path.isfile(ckpt):
        os.remove(ckpt)
    log("{} -> {} ({:.0f} min, best epoch {}, val_ce {:.4f}, train loss {:.4f})".format(
        spec, out_path, stats["minutes"], best_ep, best, stats["train_loss_at_best"]))


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
    assert cfg["seed"] not in (2027, 2028, 2029)
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
