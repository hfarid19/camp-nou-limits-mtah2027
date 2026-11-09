#!/usr/bin/env python3
"""ball_radar_pff: ball_radar_v3/train_radar.py with fold specs over DFL + PFF games (--spec pff<k> /
dfl<g> / full, brp_common), --device (cpu / mps), float16 storage of the training sequences in
memory, and a <out>.games.json sidecar listing the training games. Model, loss, optimiser and
schedule unchanged (--iters scaled for the larger data, see README).

Train the Ball Radar network (torch, CPU) and export numpy weights (.npz).

    python -I candidates/ball_radar/train_radar.py --seq <scratch>/arch/ball_radar/seq \
        --exclude 3 --out <scratch>/arch/ball_radar/folds/fold_radar_3.npz

--exclude takes game indices (positions in the sorted DFL game list) that must not be seen:
none = the full model, one = a leave-one-game-out fold model, two = a nested model (used to
make out-of-fold radar features for the stacked LightGBM). Never imports lightgbm (libomp).

Architecture (radar_common.forward is the numpy mirror): per-slot MLP -> one set-attention block
over all slots of a frame (permutation-equivariant) -> per-team mean/max pooling (partially
permutation-invariant context) -> player possession classifier: slot-wise Bi-GRU over time with
shared weights + a dead / no-player class -> ball trajectory regressor: per-team pooling of
[slot embedding, slot GRU state, possession prob] + expected possessor position -> frame Bi-GRU
-> ball (x, y) and a team head (H, A, D). Loss = SmoothL1(ball) + l_ce * CE(owner) +
l_team * CE(team) + l_real * reality loss (Kim et al. eq. 17).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import radar_common as rc  # noqa: E402
import brp_common as bc    # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")


class BiGRU(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.f = nn.GRU(i, h, batch_first=True)
        self.b = nn.GRU(i, h, batch_first=True)

    def forward(self, x):                       # (B, T, I)
        hf, _ = self.f(x)
        hb, _ = self.b(torch.flip(x, [1]))
        return torch.cat([hf, torch.flip(hb, [1])], -1)


def pool(e, ms, team):
    outs = []
    for side in (team > 0, team < 0):
        mk = ms * side.float()
        cnt = mk.sum(2, keepdim=True)
        mean = (e * mk[..., None]).sum(2) / cnt.clamp(min=1.0)
        mx = torch.where(mk[..., None] > 0, e, torch.full_like(e, -1e4)).max(2).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        outs += [mean, mx]
    return torch.cat(outs, -1)


class Radar(nn.Module):
    def __init__(self, D=32, Hp=32, Hb=64, C=64, heads=4):
        super().__init__()
        self.heads = heads
        self.phi1 = nn.Linear(rc.FS, D)
        self.phi2 = nn.Linear(D, D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff1, self.ff2 = nn.Linear(D, 2 * D), nn.Linear(2 * D, D)
        self.ctx = nn.Linear(4 * D + rc.FF, C)
        self.pgru = BiGRU(D + C, Hp)
        self.plog = nn.Linear(2 * Hp + rc.FS, 1)
        self.dlog = nn.Linear(2 * Hp + C, 1)
        self.bt1 = nn.Linear(D + 2 * Hp + 1, D)
        self.bt2 = nn.Linear(4 * D + 2 + 3 + C + rc.FF, C)
        self.bgru = BiGRU(C, Hb)
        self.ball = nn.Linear(2 * Hb, 2)
        self.team = nn.Linear(2 * Hb, 3)

    def forward(self, xs, ms, pos, xf):
        B, T, S, _ = xs.shape
        team = xs[..., 4]
        e = Fn.relu(self.phi2(Fn.relu(self.phi1(xs))))
        D = e.shape[-1]
        h, dh = self.heads, D // self.heads
        q = self.q(e).view(B, T, S, h, dh).transpose(2, 3)
        k = self.k(e).view(B, T, S, h, dh).transpose(2, 3)
        v = self.v(e).view(B, T, S, h, dh).transpose(2, 3)
        sc = q @ k.transpose(-1, -2) / np.sqrt(dh)
        sc = sc.masked_fill(ms[:, :, None, None, :] <= 0, -1e4)
        a = torch.softmax(sc, -1)
        o = (a @ v).transpose(2, 3).reshape(B, T, S, D)
        e = self.ln1(e + self.o(o))
        e = self.ln2(e + self.ff2(Fn.relu(self.ff1(e))))
        e = e * ms[..., None]
        ctx = Fn.relu(self.ctx(torch.cat([pool(e, ms, team), xf], -1)))      # (B, T, C)
        Xp = torch.cat([e, ctx[:, :, None, :].expand(B, T, S, ctx.shape[-1])], -1)
        Xp = Xp.permute(0, 2, 1, 3).reshape(B * S, T, -1)
        hp = self.pgru(Xp).view(B, S, T, -1).permute(0, 2, 1, 3)            # (B, T, S, 2Hp)
        lp = self.plog(torch.cat([hp, xs], -1))[..., 0]
        lp = lp.masked_fill(ms <= 0, -1e4)
        hmax = torch.where(ms[..., None] > 0, hp, torch.full_like(hp, -1e4)).max(2).values
        hmax = torch.where(ms.sum(2, keepdim=True) > 0, hmax, torch.zeros_like(hmax))
        ld = self.dlog(torch.cat([hmax, ctx], -1))
        logits = torch.cat([lp, ld], -1)                                     # (B, T, S+1)
        g = torch.softmax(logits, -1)
        gs = g[..., :S]
        xt = Fn.relu(self.bt1(torch.cat([e, hp, gs[..., None]], -1))) * ms[..., None]
        z = pool(xt, ms, team)
        epos = (gs[..., None] * pos).sum(2)
        gh = (gs * (team > 0).float()).sum(2, keepdim=True)
        ga = (gs * (team < 0).float()).sum(2, keepdim=True)
        zin = Fn.relu(self.bt2(torch.cat([z, epos, gh, ga, g[..., S:], ctx, xf], -1)))
        hb = self.bgru(zin)
        return dict(ball=xf[..., 0:2] + self.ball(hb), logits=logits, team=self.team(hb), epos=epos)


def export(model, path):
    W = {}
    for k, v in model.state_dict().items():
        k2 = k.replace("_l0", "")
        W[k2] = v.detach().cpu().numpy().astype(np.float32)
    np.savez(path, **W)


def load_seqs(seq_dir, names, seeds):
    out = []
    for m in names:
        for seed in seeds:
            z = np.load(os.path.join(seq_dir, "{}_{}.npz".format(m, seed)))
            for h in (1, 2):
                if "h{}_xs".format(h) not in z.files:
                    continue
                d = {k: z["h{}_{}".format(h, k)] for k in ("xs", "ms", "pos", "xf", "ball", "code", "owner")}
                for k in ("xs", "pos"):
                    d[k] = d[k].astype(np.float16)
                out.append(d)
    return out


def batch(seqs, rng, B, T, swap_p=0.5):
    cols = {k: [] for k in ("xs", "ms", "pos", "xf", "ball", "code", "owner")}
    lens = np.array([len(s["code"]) for s in seqs], float)
    for _ in range(B):
        s = seqs[rng.choice(len(seqs), p=lens / lens.sum())]
        t0 = int(rng.integers(0, len(s["code"]) - T))
        d = {k: s[k][t0:t0 + T] for k in cols}
        if rng.random() < swap_p:            # home <-> away (exact symmetry, see radar_common)
            sw = rc.swap(d)
            d = dict(d, **sw)
            d["ball"] = -d["ball"]
            c = d["code"].copy()
            c[d["code"] == 0], c[d["code"] == 1] = 1, 0
            d["code"] = c
        for k in cols:
            cols[k].append(d[k])
    out = {k: np.stack(v) for k, v in cols.items()}
    for k in ("xs", "pos"):
        out[k] = out[k].astype(np.float32)
    keep = out["ms"].max((0, 1)) > 0           # drop slots empty in the whole batch
    S = out["ms"].shape[2]
    remap = -np.ones(S + 1, np.int64)
    idx = np.where(keep)[0]
    remap[idx] = np.arange(len(idx))
    remap[S] = len(idx)
    for k in ("xs", "ms", "pos"):
        out[k] = out[k][:, :, keep]
    own = out["owner"].astype(np.int64)
    own = np.where(own >= 0, remap[np.clip(own, 0, S)], -1)
    Sk = len(idx)                               # owner slot empty at that frame -> ignore
    ok = (own == Sk) | (own < 0)
    sel = (own >= 0) & (own < Sk)
    bb, tt = np.nonzero(sel)
    okv = np.zeros_like(sel)
    okv[bb, tt] = out["ms"][bb, tt, own[bb, tt]] > 0
    own = np.where(ok | okv, own, -1)
    out["owner"] = own
    return out


def reality_loss(ball, pos, ms):
    """Kim et al. eq. 17 (sign flipped to a penalty): tanh(turn angle) * distance to nearest
    tracked player, so the ball only changes direction near a player."""
    v = ball[:, 1:] - ball[:, :-1]
    v1, v2 = v[:, :-1], v[:, 1:]
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
    theta = torch.acos(cos.clamp(-0.999, 0.999))
    d = (ball[:, :, None, :] - pos).norm(dim=-1)
    d = torch.where(ms > 0, d, torch.full_like(d, 1e3)).min(-1).values[:, 1:-1]
    return (torch.tanh(theta) * d.clamp(max=3.0)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--spec", required=True, help="pff<k>, dfl<g> or full (brp_common)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l_ce", type=float, default=1.0)
    ap.add_argument("--l_team", type=float, default=1.0)
    ap.add_argument("--l_real", type=float, default=0.0)
    ap.add_argument("--w_seen", type=float, default=0.05)
    ap.add_argument("--w_deadgap", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = bc.train_names(args.spec)
    seqs = load_seqs(args.seq, names, args.seeds)
    dev = torch.device(args.device)
    model = Radar().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters,
                                                pct_start=0.1)
    print("spec", args.spec, "games", len(names), "sequences", len(seqs),
          "params", sum(p.numel() for p in model.parameters()), flush=True)
    t0 = time.time()
    acc = np.zeros(5)
    for it in range(1, args.iters + 1):
        b = batch(seqs, rng, args.batch, args.T)
        tt = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in b.items()}
        out = model(tt["xs"].float(), tt["ms"].float(), tt["pos"].float(), tt["xf"].float())
        valid = tt["code"] < 3
        # ball loss weighted towards the frames that need imputation (raw ball missing), alive
        # play first; seen frames keep a small weight (the pipeline keeps the observed ball there)
        seen = tt["xf"][..., 2] > 0.5
        wb = torch.where(seen, torch.full_like(seen, args.w_seen, dtype=torch.float32),
                         torch.where(tt["code"] < 2, torch.ones_like(seen, dtype=torch.float32),
                                     torch.full_like(seen, args.w_deadgap, dtype=torch.float32)))
        wb = wb * valid.float()
        lb = (Fn.smooth_l1_loss(out["ball"], tt["ball"].float(), beta=0.5, reduction="none").sum(-1)
              * wb).sum() / wb.sum().clamp(min=1.0)
        own = tt["owner"].clone()
        own[~valid] = -1
        lo = Fn.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]), own.reshape(-1),
                              ignore_index=-1)
        code = tt["code"].long().clone()
        code[~valid] = -1
        lt = Fn.cross_entropy(out["team"].reshape(-1, 3), code.reshape(-1), ignore_index=-1)
        loss = lb + args.l_ce * lo + args.l_team * lt
        lr_ = torch.zeros(())
        if args.l_real > 0:
            lr_ = reality_loss(out["ball"], tt["pos"].float(), tt["ms"].float())
            loss = loss + args.l_real * lr_
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 20 == 0:
            acc += [lb.item(), lo.item(), lt.item(), float(lr_), 1]
        if it % 200 == 0:
            a = acc[:4] / acc[4]
            print("it {} ball {:.3f} owner {:.3f} team {:.3f} real {:.3f}  {:.0f}s".format(
                it, *a, time.time() - t0), flush=True)
            acc[:] = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.to("cpu")
    export(model, args.out)
    import json
    with open(args.out + ".games.json", "w") as handle:
        json.dump(names, handle)
    print("saved", args.out, "{:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
