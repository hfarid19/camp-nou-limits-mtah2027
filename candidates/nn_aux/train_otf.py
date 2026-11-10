#!/usr/bin/env python3
"""Round 5, R5-6b: nn_pff MS-TCN / TCN trained on ON-THE-FLY degradations (a fresh randomised degrade_v2 per
crop) instead of ds_pff2's 3 fixed seeds per game. Model, loss, folds and export = train_mstcn.py /
train_tcn.py (the exported npz runs in the members' numpy code unchanged).

    python -I candidates/nn_aux/train_otf.py --model mstcn|tcn --device cuda --ds $MTAH_WORK/ds_pff2/ds \
        --out <dir> --specs pff0,... --cfg '{...}' [--workers 12]

Per crop (DataLoader workers): pick a training game (length-weighted, DFL share w_dfl), a half, a 5*crop
frame window plus MARGIN_S seconds of context on both sides; draw make_dataset_v2.draw_params + seeds,
degrade_v2 the window's clean tracks (+ make_dataset_v2's AR(1) ball offset), features.compute on the window,
keep the rows of one random stride-5 phase inside the crop. Labels / boundary targets from the game's codes
(learned_dfl_evidence make_dataset.labels_blocks = the ds files' codes; data.py's soft targets).
Clean tracks: $MTAH_WORK/pff_wc2022/npz (48 selection games; holdout refused) and $MTAH_WORK/idsse/npz;
attack sign = the ds_pff2 raw file's (kick-off rule). Normalisation statistics = the trainer's own, from
the spec's ds_pff2 files (the same feature distribution). p_otf < 1 mixes in stored ds_pff2 crops.
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
ROOT = os.path.dirname(CAND)
sys.path.append(os.path.join(CAND, "nn_pff"))
sys.path.append(os.path.join(CAND, "learned_dfl_evidence"))
sys.path.append(CAND)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "evaluation"))
import data as D          # noqa: E402
import features as fe     # noqa: E402  (nn_pff/features.py, first on the path: nn_pff appended first)
import train_mstcn as TM  # noqa: E402
import train_tcn as TT    # noqa: E402

_NNP = os.path.join(CAND, "nn_pff")      # the trainers put nn_pff first; its queue.py shadows the stdlib
sys.path = [p for p in sys.path if os.path.abspath(p) != _NNP] + [_NNP]

WORK = os.environ.get("MTAH_WORK")
MARGIN_S = 60.0
FPS, PITCH = 25.0, (105.0, 68.0)
OTF_CFG = dict(p_otf=1.0, otf_seed=0)


def clean_path(g):
    return os.path.join(WORK, "pff_wc2022", "npz", g + ".npz") if g.startswith("PFF-") else \
        os.path.join(WORK, "idsse", "npz", g + ".npz")


def labels_blocks(g):
    """= learned_dfl_evidence/make_dataset.labels_blocks (copied: that module imports pandas via tune_on_sample_data)."""
    st, po, half = g["status"], g["poss"], g["half"]
    codes = np.full(len(st), 3, np.int8)
    codes[(st == 1) & (po == 1)] = 0
    codes[(st == 1) & (po == 2)] = 1
    codes[st == 0] = 2
    blocks = []
    for h in (1, 2):
        idx = np.where((half == h) & (codes != 3))[0]
        if len(idx):
            blocks.append((h, int(idx[0]), int(idx[-1])))
    return codes, blocks


def draw_params(rng):
    """= make_dataset_v2.draw_params (copied for the same reason; verified identical on seeds 0-4)."""
    return dict(lead_s=rng.uniform(0.5, 4.0), reveal_s=rng.uniform(0.0, 3.0),
                p_keep_dead=rng.uniform(0.1, 0.35), p_split=rng.uniform(0.15, 0.5),
                split_s=rng.uniform(0.2, 1.5), alive_gaps_per90=rng.uniform(10, 60),
                alive_gap_s=rng.uniform(0.3, 1.5), crowd_boost=rng.uniform(1.5, 4.0),
                tracklet_s=rng.uniform(40, 160), player_gap_s=rng.uniform(3, 12),
                slots_per_team=int(rng.integers(21, 25)), smooth_frames=int(rng.choice([7, 9, 11, 13, 15, 19])),
                ball_smooth_frames=int(rng.choice([1, 3, 5])), ball_offset_sd=rng.uniform(0.0, 0.5))


class Games(object):
    """Clean games of a spec, with codes, blocks, attack sign and soft targets at every frame."""

    def __init__(self, ds_dir, games):
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(ds_dir)), "raw")
        self.g = []
        for name in games:
            z = np.load(clean_path(name))
            codes, blocks = labels_blocks(z)
            rz = np.load(os.path.join(raw_dir, name + "_41.npz"))
            assert np.array_equal(rz["codes"], codes), name
            ha, dd = D.change_positions(codes)
            fr = np.arange(len(codes), dtype=np.float64)
            tg = np.stack([D.soft_target(fr, ha), D.soft_target(fr, dd), D.change_target(codes)]).astype(np.float32)
            self.g.append(dict(name=name, players=z["players"].astype(np.float32), tp=z["team_of_player"].astype(int),
                               ball=z["ball"].astype(np.float32), status=z["status"], codes=codes,
                               blocks=[tuple(int(v) for v in b) for b in blocks], sign=float(rz["sign"]), tg=tg))


def otf_crop(G, rng, crop, w_dfl):
    import degrade_v2 as dv2
    L = np.array([sum(b - a + 1 for _, a, b in g["blocks"]) for g in G.g], float)
    isd = np.array([g["name"].startswith("DFL") for g in G.g])
    w = L.copy()
    if 0 < w_dfl < 1 and isd.any() and (~isd).any():
        w[isd] *= w_dfl / w[isd].sum()
        w[~isd] *= (1 - w_dfl) / w[~isd].sum()
    w /= w.sum()
    g = G.g[int(rng.choice(len(G.g), p=w))]
    bl = [b for b in g["blocks"]]
    bw = np.array([b - a + 1 for _, a, b in bl], float)
    h, a, b = bl[int(rng.choice(len(bl), p=bw / bw.sum()))]
    span = 5 * crop
    s = int(rng.integers(a, max(a + 1, b + 1 - span)))
    e = min(b + 1, s + span)
    m = int(MARGIN_S * FPS)
    wa, wb = max(a, s - m), min(b + 1, e + m)
    sl = slice(wa, wb)
    p = draw_params(rng)
    codes = g["codes"][sl]
    slots, st, ball = dv2.degrade_v2(g["players"][sl].astype(np.float64), g["tp"], g["ball"][sl].astype(np.float64),
                                     g["status"][sl], codes != 3, np.random.default_rng(rng.integers(1 << 62)), p)
    n = len(ball)
    aa = np.exp(-1.0 / FPS)
    eps = rng.normal(0, 1, (n, 2)) * p["ball_offset_sd"] * np.sqrt(1 - aa * aa)
    off = np.zeros((n, 2))
    for t in range(1, n):
        off[t] = aa * off[t - 1] + eps[t]
    ball = (ball + off).astype(np.float32)
    sign = g["sign"] if h == 1 else -g["sign"]
    Fw, names, kinds = fe.compute(slots, st, ball, np.full(n, sign), PITCH, FPS)
    ph = int(rng.integers(0, 5))
    rows = np.arange(s + ph, e, 5)
    X = Fw[rows - wa].astype(np.float32)
    y = g["codes"][rows].astype(np.int64)
    tg = g["tg"][:, rows]
    return X, y, tg


class OTFData(torch.utils.data.IterableDataset):
    def __init__(self, G, crop, w_dfl, seed):
        self.G, self.crop, self.w_dfl, self.seed = G, crop, w_dfl, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0, 777])
        while True:
            X, y, tg = otf_crop(self.G, rng, self.crop, self.w_dfl)
            if len(y) >= 64:
                yield X, y, tg


def loader(G, cfg, crop, workers):
    ds = OTFData(G, crop, cfg["w_dfl"], cfg["otf_seed"] * 1000 + cfg["seed"])
    # spawn: worker command lines do not repeat the trainer's ("--device cuda"), which the GPU queue counts
    return iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=8,
                                            persistent_workers=True, multiprocessing_context="spawn"))


def train_spec(model_kind, ds_dir, spec, out_path, cfg, threads, log, device, workers):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    seqs, names, kinds, games = D.load(ds_dir, spec)
    assert not D.excluded(spec) & set(games)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float32)
    G = Games(ds_dir, games)
    log("spec {} games {} (PFF {}, DFL {}) load {:.0f}s".format(
        spec, len(games), sum(g.startswith("PFF") for g in games), sum(g.startswith("DFL") for g in games), time.time() - t0))
    if model_kind == "mstcn":
        mean, std = TM.sym_stats(seqs, perm, signs)
        prepf = lambda X: np.nan_to_num((X - mean) / std).astype(np.float32)  # noqa: E731
        permf, signf = perm, signs
    else:
        mu, sd, ind = TT.norm_stats(seqs, perm, signs)
        prepf = lambda X: TT.prep(X, mu, sd, ind)  # noqa: E731
        pos = {int(j): i for i, j in enumerate(ind)}
        permf = np.concatenate([perm, [len(perm) + pos[int(perm[j])] for j in ind]])
        signf = np.concatenate([signs, np.ones(len(ind), np.float32)])
    stored = [(g.startswith("DFL"), prepf(X).astype(np.float16), y, b) for g, X, y, b in seqs]
    change_prior = float(np.mean(np.concatenate([s[3][2] for s in stored])))
    del seqs
    lens = np.array([len(s[2]) for s in stored], float)
    isd = np.array([s[0] for s in stored])
    w = lens.copy()
    if 0 < cfg["w_dfl"] < 1 and isd.any() and (~isd).any():
        w[isd] *= cfg["w_dfl"] / w[isd].sum()
        w[~isd] *= (1 - cfg["w_dfl"]) / w[~isd].sum()
    w /= w.sum()
    torch.set_num_threads(threads)
    dev = torch.device(device)
    T = cfg["crop"]
    it_otf = loader(G, cfg, T, workers)
    permt, signt = torch.from_numpy(permf), torch.from_numpy(np.asarray(signf, np.float32))[:, None]

    def crop():
        if rng.random() < cfg["p_otf"]:
            X, y, tg = next(it_otf)
            X, y, tg = X.numpy(), y.numpy(), tg.numpy()
            x = torch.from_numpy(prepf(X).T.copy())
            return x, torch.from_numpy(y), torch.from_numpy(tg)
        i = int(rng.choice(len(stored), p=w))
        _, Z, y, b = stored[i]
        s0 = int(rng.integers(0, max(1, len(y) - T)))
        return torch.from_numpy(Z[s0:s0 + T].T.astype(np.float32)), torch.from_numpy(y[s0:s0 + T]), \
            torch.from_numpy(b[:, s0:s0 + T].astype(np.float32))

    def batch(B):
        xs, ys, bs = [], [], []
        for _ in range(B):
            x, y, b = crop()
            if rng.random() < 0.5:
                x = x[permt] * signt
                y = torch.where(y == 0, 1, torch.where(y == 1, 0, y))
            xs.append(x)
            ys.append(y)
            bs.append(b)
        Tm = min(v.shape[1] for v in xs)
        return (torch.stack([v[:, :Tm] for v in xs]).to(dev), torch.stack([v[:Tm] for v in ys]).to(dev),
                torch.stack([v[:, :Tm] for v in bs]).to(dev))

    if model_kind == "mstcn":
        model = TM.ASRF(len(names), cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) if cfg["wd"] > 0 else \
            torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        n_it = cfg["epochs"] * (cfg["crops_per_epoch"] // cfg["batch"])
        sched = None
        if cfg["cosine"]:
            warm = max(1, int(cfg["warm"] * n_it))
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, n_it) / n_it)))
        pw = torch.tensor(cfg["pos_weight"], device=dev)
        cw = torch.tensor([1.0, 1.0, cfg["cw_d"]], device=dev)
    else:
        dils = [int(v) for v in cfg["dils"].split(",")]
        model = TT.TCN(len(permf), cfg["C"], dils, drop=cfg["drop"], in_drop=cfg["in_drop"]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        n_it = cfg["iters"]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=n_it, pct_start=0.1)
        cw = torch.tensor([1.0, 1.0, cfg["dead_w"]], device=dev)
    log("params {} iters {} workers {}".format(sum(p.numel() for p in model.parameters()), n_it, workers))
    t1 = time.time()
    run = []
    for it in range(n_it):
        model.train()
        x, y, b = batch(cfg["batch"])
        valid = y < 3
        yt = y.clamp(max=2)
        if model_kind == "mstcn":
            cs, bs = model(x)
            loss = 0
            for c in cs:
                lp = F.log_softmax(c, 1)
                nll = F.nll_loss(lp, yt, weight=cw, reduction="none")
                if cfg["ls"] > 0:
                    nll = (1 - cfg["ls"]) * nll - cfg["ls"] * (lp * cw[None, :, None]).mean(1)
                loss = loss + nll[valid].mean()
                d = torch.clamp((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2, max=cfg["tau_tmse"] ** 2)
                vm = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)
                loss = loss + cfg["lam_tmse"] * (d * vm).sum() / (vm.sum() * 3 + 1)
            for bl in bs:
                l = F.binary_cross_entropy_with_logits(bl, b[:, :2], pos_weight=pw, reduction="none")
                loss = loss + cfg["w_b"] * (l * valid.unsqueeze(1)).sum() / (valid.sum() * 2 + 1)
        else:
            out = model(x)
            ce = F.cross_entropy(out[:, :3], yt, weight=cw, reduction="none", label_smoothing=cfg["ls"])
            wt = valid.float()
            loss = (ce * wt).sum() / wt.sum() + cfg["c_w"] * F.binary_cross_entropy_with_logits(out[:, 3], b[:, 2])
        opt.zero_grad()
        loss.backward()
        clip = cfg["clip"] if model_kind == "mstcn" else 1.0
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if sched is not None:
            sched.step()
        if it % 25 == 0:
            run.append(float(loss.detach()))
        if (it + 1) % 250 == 0:
            log("  {} it {} loss {:.4f} {:.0f}s".format(spec, it + 1, np.mean(run[-10:]), time.time() - t1))
    model.to("cpu")
    cfg = dict(cfg, device=device, pff_holdout_trained=D.allow_holdout(), trainer="nn_aux/train_otf.py " + model_kind)
    if model_kind == "mstcn":
        TM.export(model, mean, std, names, cfg, out_path, games, spec)
    else:
        sdict = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
        meta = dict(mu=mu, sd=sd, ind=ind.astype(np.int64), dils=np.array(dils, np.int64), names=np.array(names),
                    kinds=np.array(kinds), change_prior=np.array(change_prior), hold=np.array(-1),
                    train_games=np.array(games), spec=np.array(spec), train_cfg=json.dumps(cfg),
                    pff_holdout_trained=np.array(D.allow_holdout()))
        tmp = out_path[:-4] + ".tmp.npz"
        np.savez(tmp, **sdict, **meta)
        os.replace(tmp, out_path)
    log("{} -> {} ({:.0f} min)".format(spec, out_path, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mstcn", "tcn"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="pff0,pff1,pff2,pff3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    base = TM.CFG if a.model == "mstcn" else TT.CFG
    CFG = dict(base, **OTF_CFG)
    cfg = dict(CFG, **json.loads(a.cfg))
    assert set(cfg) == set(CFG), set(cfg) ^ set(CFG)
    assert not cfg["init"] and cfg["ema"] == 0 and cfg["focal"] == 0 and cfg["src"] == "all"
    if a.model == "tcn":
        assert cfg["trans_w"] == 0 and cfg["sched"] == "onecycle"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_cfg.json"), "w") as h:
        json.dump(dict(cfg=cfg, ds=os.path.abspath(a.ds), model=a.model + "_otf"), h, indent=1)
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
        train_spec(a.model, a.ds, spec, p, cfg, a.threads, log, a.device, a.workers)


if __name__ == "__main__":
    main()
