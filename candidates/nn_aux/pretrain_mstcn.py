#!/usr/bin/env python3
"""Round 5, R5-6a: self-supervised pretraining (masked trajectory-feature modelling) of the nn_pff MS-TCN
backbone, per fold spec, then fine-tuning with train_mstcn.py's `init` (same spec, same input scale).

    python -I candidates/nn_aux/pretrain_mstcn.py --device cuda --ds $MTAH_WORK/ds_pff2/ds \
        --extra $MTAH_WORK/ds_pff64/ds --out <dir> --specs pff0,... --cfg '{"C":128,"Cr":64}'
    python -I candidates/nn_pff/train_mstcn.py ... --cfg '{..., "init": "<dir>"}'   # fine-tune

Unlabelled inputs = the spec's own training files (ds_pff2: every game the spec may train on) + the 16 spent
PFF holdout games' input features from --extra (labels never read). No bench game the spec scores is used
(pff<k>: fold k excluded; dfl<g>: game g excluded; no SkillCorner, no Metrica, no match). Pretext: random time
spans (span_min..span_max steps, mask_frac of each crop) are replaced by 0 (= the feature mean after
normalisation) and the backbone's last hidden layer reconstructs the normalised features there through a
1x1 conv decoder (MSE on the masked, originally non-missing entries). Team-swap augmentation as in training.
Output = train_mstcn.py's npz format (backbone pretrained; refinement stages and heads at their initial
values; train_games = the spec's labelled games + the unlabelled holdout games, so the fine-tuned model
records them; the decoder is dropped).
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
CAND = os.path.dirname(HERE)
sys.path.append(os.path.join(CAND, "nn_pff"))
import data as D          # noqa: E402
import features as fe     # noqa: E402
import train_mstcn as TM  # noqa: E402

CFG = dict(TM.CFG, pt_iters=4000, pt_lr=5e-4, mask_frac=0.3, span_min=5, span_max=60)


def unlabelled(extra_dir, names_ref):
    """input features of the 16 PFF holdout games in extra_dir (no labels read)."""
    hold = D.holdout()
    out, games = [], set()
    for p in sorted(glob.glob(os.path.join(extra_dir, "PFF-*_*.npz"))):
        g = D.game_of(p)
        if g[4:] not in hold:
            continue
        z = np.load(p)
        assert [str(s) for s in z["names"]] == names_ref, p
        idx = z["idx"].astype(np.int64)
        brk = np.concatenate(([True], np.diff(idx) != 5))
        sid = np.cumsum(brk) - 1
        X = z["X"]
        for s in np.unique(sid):
            m = sid == s
            if m.sum() >= 64:
                out.append((g, X[m].astype(np.float32)))
        games.add(g)
    return out, sorted(games)


def span_mask(T, rng, frac, lo, hi):
    m = np.zeros(T, bool)
    while m.mean() < frac:
        L = int(rng.integers(lo, hi + 1))
        s = int(rng.integers(0, max(1, T - L)))
        m[s:s + L] = True
    return m


def pretrain_spec(ds_dir, extra_dir, spec, out_path, cfg, threads, log, device):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec)
    assert not D.excluded(spec) & set(games)
    un, ugames = unlabelled(extra_dir, names) if extra_dir else ([], [])
    assert not set(ugames) & D.excluded(spec)
    allx = [(g, X, None, None) for g, X, _, _ in seqs] + [(g, X, None, None) for g, X in un]
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    mean, std = TM.sym_stats(allx, perm, signs)
    data = []
    for g, X, _, _ in allx:
        data.append((np.nan_to_num((X - mean) / std).astype(np.float16), (~np.isnan(X)).astype(np.bool_)))
    del seqs, un, allx
    lens = np.array([len(d[0]) for d in data], float)
    w = lens / lens.sum()
    log("pretrain spec {} labelled games {} + unlabelled {} seqs {} steps {} load {:.0f}s".format(
        spec, len(games), len(ugames), len(data), int(lens.sum()), time.time() - t0))
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = TM.ASRF(len(names), cfg).to(dev)
    dec = nn.Conv1d(cfg["C"], len(names), 1).to(dev)
    params = list(model.backbone.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["pt_lr"], weight_decay=1e-4)
    n_it = cfg["pt_iters"]
    warm = max(1, int(0.03 * n_it))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
    permt, signst = torch.from_numpy(perm), torch.from_numpy(signs)[:, None]
    T = cfg["crop"]
    t1 = time.time()
    run = []
    for it in range(n_it):
        model.train()
        xs, ts, ms = [], [], []
        for _ in range(cfg["batch"]):
            Z, ok = data[int(rng.choice(len(data), p=w))]
            s0 = int(rng.integers(0, max(1, len(Z) - T)))
            x = torch.from_numpy(Z[s0:s0 + T].T.astype(np.float32))
            o = torch.from_numpy(ok[s0:s0 + T].T.copy())
            if rng.random() < 0.5:
                x = x[permt] * signst
                o = o[permt]
            mk = torch.from_numpy(span_mask(x.shape[1], rng, cfg["mask_frac"], cfg["span_min"], cfg["span_max"]))
            ts.append(x)
            xs.append(x * (~mk).float()[None])
            ms.append(o & mk[None])
        Tm = min(v.shape[1] for v in xs)
        x = torch.stack([v[:, :Tm] for v in xs]).to(dev)
        tg = torch.stack([v[:, :Tm] for v in ts]).to(dev)
        mm = torch.stack([v[:, :Tm] for v in ms]).to(dev).float()
        st = model.backbone
        h = st.inp(x)
        for l in st.layers:
            h = l(h)
        rec = dec(h)
        loss = ((rec - tg) ** 2 * mm).sum() / mm.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            run.append(float(loss.detach()))
        if (it + 1) % 250 == 0:
            log("  {} pt it {} masked mse {:.4f} {:.0f}s".format(spec, it + 1, np.mean(run[-10:]), time.time() - t1))
    model.to("cpu")
    cfg = dict(cfg, device=device, unlabelled_games=ugames, trainer="nn_aux/pretrain_mstcn.py")
    TM.export(model, mean, std, names, cfg, out_path, sorted(set(games) | set(ugames)), spec)
    log("{} -> {} ({:.0f} min)".format(spec, out_path, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--extra", default="")
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
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), extra=a.extra, model="mstcn_pretrain"), h, indent=1)
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
        pretrain_spec(a.ds, a.extra, spec, p, cfg, a.threads, log, a.device)


if __name__ == "__main__":
    main()
