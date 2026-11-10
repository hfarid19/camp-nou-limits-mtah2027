#!/usr/bin/env python3
"""Steps 2-4: corrupt the true touches, build labels in every setting, score them against PFF touch labels.

    python candidates/touch_ceiling/sim.py grid  [--jobs 4]   # break-even grid  -> grid.pkl
    python candidates/touch_ceiling/sim.py real  [--jobs 4]   # real-detector neighbourhood + r_chg sweep -> real.pkl
    python candidates/touch_ceiling/sim.py smoke               # 3 games, a few configs

Detector model (knobs, per game, common random numbers across configs so surfaces are monotone):
    r      each true touch (restarts and same-team touches) is detected w.p. r
    rchg   open-play team-changing touches (interceptions, tackles, won challenges, deflections
           that flip the PFF touch team) are detected w.p. rchg; None = same as r
    p      false touches are added so that E[precision] = p: round(n_detected_true * (1-p)/p) of them
           at uniform random truth-alive rows; team = opponent of the true team w.p. 0.5
    sigma  Gaussian timing jitter (frames) on detected true touches, rounded, kept inside the half
    a      a detected true touch keeps its team w.p. a (else flipped)

Settings (team for alive frames; v6 = ensemble_ctrl_v6 out-of-fold codes, shipped rule):
    A      dead from the truth; team = last detected touch (labels.py convention before the first)
    B      alive/dead from v6; team = last detected touch on a v6-alive row of the current v6-alive
           run, v6's team before the run's first detected touch
    C1_k   v6, but each v6 H<->A switch is kept only if a detected touch of the new team lies within
           k frames of it (and is then moved to that touch); unconfirmed switches are dropped
    C1s_k  as C1_k but an unconfirmed switch is kept where v6 put it (retime only, never drop)
    C2     v6 where its three members agree; on v6-alive rows where they disagree, team from B
    C3r    v6, a v6 spell shorter than 3 s (not the first of its alive run) is kept only if a detected
           touch of its team lies inside it (+-5 frames); else it takes the previous team
    C3     C3r + insertion: a detected touch of team U inside a v6 spell of team T opens a U spell
           until the next detected touch of T, if that is < 3 s later and inside the same spell
    C13    C1s_12 then C3 (retime confirmed switches, prune unconfirmed short spells, insert short ones)
    C123   C13, then C2's member-disagreement override
Scores per game: spellF1 vs raw touch labels (the task metric) and vs control.fold T0.5, T1
(fine3 = mean of the three). Results: $MTAH_WORK/arch/touch_ceiling/<name>.pkl,
{(cfg, setting): array (games, seeds, 3)}.
"""

import argparse
import itertools
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
from touches import build_labels  # noqa: E402

SHORT = 75          # C3: "short" spell, frames (3 s)
TOL3 = 5            # C3: touch tolerance around a short spell
C1_K = (5, 12, 25)
SETTINGS = (("A", "B") + tuple("C1_{}".format(k) for k in C1_K) + tuple("C1s_{}".format(k) for k in C1_K)
            + ("C2", "C3r", "C3", "C13", "C123"))
POOL_MULT = 3.0     # false-touch candidate pool: 3 x n touches (enough for p >= 0.25 at r = 1)


# ------------------------------------------------------------------------------------------------
# per-game context
# ------------------------------------------------------------------------------------------------

_G = {}


def game_ctx(g):
    if g in _G:
        return _G[g]
    d = pickle.load(open(os.path.join(C.OUT, "touches.pkl"), "rb"))[g]
    T = C.targets()[g][1]
    v = C.v6_codes(g)
    m = C.member_codes(g)
    gt = T["raw"]
    halves = d["halves"]
    v_alive = v != C.D
    # v6-alive runs, inside halves
    runs = []
    for h0, h1 in halves:
        w = np.concatenate(([0], v_alive[h0:h1].astype(np.int8), [0]))
        e = np.diff(w)
        runs += [(h0 + a, h0 + b) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    run_start = np.full(len(v), -1, np.int64)
    for a, b in runs:
        run_start[a:b] = a
    half_of = np.zeros(d["n"], np.int64)
    for i, (h0, h1) in enumerate(halves):
        half_of[h0:h1] = i
    ctx = dict(d, v=v, v_alive=v_alive, runs=runs, run_start=run_start,
               dis=~((m[0] == m[1]) & (m[1] == m[2])) & v_alive,
               gsp={k: C.gt_spells(T[k]) for k in C.TARGETS}, gt=gt, half_of=half_of,
               alive_rows=np.flatnonzero(d["alive"]))
    _G[g] = ctx
    return ctx


def crn(ctx, seed, g):
    """Common random numbers for one (game, seed)."""
    rng = np.random.default_rng([seed, int(g)])
    n = len(ctx["rows"])
    npool = int(POOL_MULT * n) + 10
    fr = rng.choice(ctx["alive_rows"], npool)
    true_team = ctx["gt"][fr]
    opp = np.where(true_team == C.H, C.A, C.H)
    ft = np.where(rng.random(npool) < 0.5, opp, true_team).astype(np.int8)
    return dict(u_keep=rng.random(n), z=rng.standard_normal(n), u_flip=rng.random(n), f_rows=fr, f_teams=ft)


def corrupt(ctx, R, r, rchg, p, a, sigma):
    rows, teams, kind = ctx["rows"], ctx["teams"], ctx["kind"]
    rc = r if rchg is None else rchg
    keep = R["u_keep"] < np.where(kind == 1, rc, r)
    kr = rows[keep]
    if sigma > 0:
        hs = ctx["halves"]
        h = ctx["half_of"][kr]
        lo = np.array([hs[i][0] for i in h], np.int64)
        hi = np.array([hs[i][1] - 1 for i in h], np.int64)
        kr = np.clip(kr + np.rint(R["z"][keep] * sigma).astype(np.int64), lo, hi)
    kt = teams[keep].copy()
    flip = R["u_flip"][keep] >= a
    kt[flip] = np.where(kt[flip] == C.H, C.A, C.H)
    nf = int(round(keep.sum() * (1 - p) / p)) if p < 1 else 0
    assert nf <= len(R["f_rows"]), "false-touch pool too small"
    tr = np.concatenate((kr, R["f_rows"][:nf]))
    tt = np.concatenate((kt, R["f_teams"][:nf])).astype(np.int8)
    o = np.argsort(tr, kind="stable")
    return tr[o], tt[o]


# ------------------------------------------------------------------------------------------------
# settings
# ------------------------------------------------------------------------------------------------

def setting_A(ctx, tr, tt):
    return build_labels(tr, tt, ctx["alive"], ctx["halves"], ctx["n"])


def setting_B(ctx, tr, tt):
    v = ctx["v"]
    ok = ctx["v_alive"][tr]
    tr, tt = tr[ok], tt[ok]
    out = v.copy()
    if not len(tr):
        return out
    idx = np.searchsorted(tr, np.arange(len(v)), side="right") - 1
    last_row = np.where(idx >= 0, tr[np.maximum(idx, 0)], -1)
    use = ctx["v_alive"] & (last_row >= ctx["run_start"])
    out[use] = tt[idx[use]]
    return out


def setting_C1(ctx, tr, tt, k, keep_unconfirmed=False):
    v = ctx["v"]
    out = v.copy()
    for a, b in ctx["runs"]:
        w = v[a:b]
        sw = a + np.flatnonzero(w[1:] != w[:-1]) + 1
        if not len(sw):
            continue
        cur, last = v[a], a
        pts = []
        for s in sw:
            T = v[s]
            if T == cur:
                continue
            lo = np.searchsorted(tr, max(s - k, last + 1), side="left")
            hi = np.searchsorted(tr, min(s + k, b - 1), side="right")
            best = None
            for i in range(lo, hi):
                if tt[i] == T and (best is None or abs(tr[i] - s) < abs(best - s)):
                    best = int(tr[i])
            if best is not None:
                pts.append((best, T))
                cur, last = T, best
            elif keep_unconfirmed:
                pts.append((int(s), T))
                cur, last = T, int(s)
        out[a:b] = v[a]
        for u, T in pts:
            out[u:b] = T
    return out


def setting_C2(ctx, tr, tt, B):
    out = ctx["v"].copy()
    out[ctx["dis"]] = B[ctx["dis"]]
    return out


def setting_C3(ctx, tr, tt, insert, base_codes=None):
    v = ctx["v"] if base_codes is None else base_codes
    out = v.copy()
    for a, b in ctx["runs"]:
        w = v[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = a + np.concatenate(([0], br))
        en = a + np.concatenate((br, [len(w)]))
        for s, e in zip(st[1:], en[1:]):            # never the first spell of the run
            if e - s >= SHORT:
                continue
            K = v[s]
            lo = np.searchsorted(tr, s - TOL3, side="left")
            hi = np.searchsorted(tr, e + TOL3, side="left")
            if not (tt[lo:hi] == K).any():
                out[s:e] = out[s - 1]
        if insert:
            lo = np.searchsorted(tr, a, side="left")
            hi = np.searchsorted(tr, b, side="left")
            rr, tl = tr[lo:hi], tt[lo:hi]
            base = out[a:b].copy()
            for i in range(len(rr)):
                u, U = int(rr[i]), tl[i]
                T = base[u - a]
                if U == T:
                    continue
                j = i + 1
                while j < len(rr) and tl[j] != T and rr[j] - u < SHORT:
                    j += 1
                if j < len(rr) and tl[j] == T and rr[j] - u < SHORT:
                    e = int(rr[j])
                    if (base[u - a:e - a] == T).all():
                        out[u:e] = U
    return out


def labels_all(ctx, tr, tt, settings=SETTINGS):
    out = {}
    B = setting_B(ctx, tr, tt) if set(settings) & {"B", "C2", "C123"} else None
    for s in settings:
        if s == "A":
            out[s] = setting_A(ctx, tr, tt)
        elif s == "B":
            out[s] = B
        elif s.startswith("C1_"):
            out[s] = setting_C1(ctx, tr, tt, int(s[3:]))
        elif s.startswith("C1s_"):
            out[s] = setting_C1(ctx, tr, tt, int(s[4:]), keep_unconfirmed=True)
        elif s in ("C13", "C123"):
            c13 = setting_C3(ctx, tr, tt, True, setting_C1(ctx, tr, tt, 12, keep_unconfirmed=True))
            if s == "C123":
                c13[ctx["dis"]] = B[ctx["dis"]]
            out[s] = c13
        elif s == "C2":
            out[s] = setting_C2(ctx, tr, tt, B)
        elif s == "C3r":
            out[s] = setting_C3(ctx, tr, tt, False)
        elif s == "C3":
            out[s] = setting_C3(ctx, tr, tt, True)
    return out


def score(ctx, lab):
    return [C.spell_f1(ctx["gsp"][k], lab) for k in C.TARGETS]


# ------------------------------------------------------------------------------------------------
# configs and runner
# ------------------------------------------------------------------------------------------------

GRID_R = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0)
GRID_P = GRID_R
GRID_A = (0.75, 0.85, 0.95, 1.0)
GRID_S = (0, 2, 5)
GRID_RCHG = (None, 0.2)


REAL = (0.43, 0.2, 0.475, 0.75, 4)   # video_touch at thr 0.4: R 0.431 / P 0.475 at +-5 frames, team 72-78%,
                                      # tackle/block recall 10-25%; sigma 4 matches 58% of +-5 matches being within +-2


def grid_cfgs():
    return [(r, rc, p, a, s) for r, rc, p, a, s in itertools.product(GRID_R, GRID_RCHG, GRID_P, GRID_A, GRID_S)
            if not (rc is not None and rc >= r)]


def real_cfgs():
    """Around the real detector (r 0.43, p 0.475, a 0.75, sigma ~4, rchg 0.1-0.25) and an rchg sweep."""
    cfgs = set()
    for r, p, a, s in itertools.product((0.3, 0.43, 0.6), (0.3, 0.475, 0.6), (0.65, 0.75, 0.85), (2, 4, 6)):
        for rc in (None, 0.1, 0.2):
            cfgs.add((r, rc if rc is None or rc < r else None, p, a, s))
    for r, p, a, s in itertools.product((0.43, 0.7, 0.9, 1.0), (0.475, 0.8, 0.95, 1.0), (0.75, 0.95, 1.0), (0, 4)):
        for rc in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
            cfgs.add((r, rc, p, a, s))
    # one knob at a time from perfect
    for v in (0.3, 0.43, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
        cfgs |= {(v, None, 1.0, 1.0, 0), (1.0, v, 1.0, 1.0, 0), (1.0, None, v, 1.0, 0), (1.0, None, 1.0, v, 0)}
    for s in (1, 2, 3, 4, 5, 6, 8, 10, 15):
        cfgs.add((1.0, None, 1.0, 1.0, s))
    cfgs.add((1.0, None, 1.0, 1.0, 0))
    # the real point with one knob (or r and rchg together) made perfect
    cfgs |= {REAL, (1.0, 0.2, 0.475, 0.75, 4), (0.43, None, 0.475, 0.75, 4), (1.0, None, 0.475, 0.75, 4),
             (0.43, 0.2, 1.0, 0.75, 4), (0.43, 0.2, 0.475, 1.0, 4), (0.43, 0.2, 0.475, 0.75, 0)}
    return sorted(cfgs, key=lambda c: tuple(-1 if x is None else x for x in c))


def run_game(job):
    g, cfgs, seeds = job
    ctx = game_ctx(g)
    out = np.zeros((len(cfgs), len(seeds), len(SETTINGS), 3), np.float32)
    for si, seed in enumerate(seeds):
        R = crn(ctx, seed, g)
        for ci, (r, rc, p, a, s) in enumerate(cfgs):
            tr, tt = corrupt(ctx, R, r, rc, p, a, s)
            labs = labels_all(ctx, tr, tt)
            for k, st in enumerate(SETTINGS):
                out[ci, si, k] = score(ctx, labs[st])
    base = np.array(score(ctx, ctx["v"]), np.float32)
    return g, out, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "real", "smoke"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    games = C.selection_games()
    if a.cmd == "grid":
        cfgs, seeds = grid_cfgs(), list(range(a.seeds or 2))
    elif a.cmd == "real":
        cfgs, seeds = real_cfgs(), list(range(a.seeds or 3))
    else:
        games = games[:3]
        cfgs = [(1.0, None, 1.0, 1.0, 0), (0.43, 0.2, 0.475, 0.75, 4), (0.9, None, 0.9, 0.95, 2)]
        seeds = [0]
    print("{}: {} configs x {} seeds x {} games x {} settings".format(a.cmd, len(cfgs), len(seeds), len(games),
                                                                     len(SETTINGS)), flush=True)
    t0 = time.time()
    res = {}
    with Pool(a.jobs) as pool:
        for i, (g, out, base) in enumerate(pool.imap_unordered(run_game, [(g, cfgs, seeds) for g in games])):
            res[g] = (out, base)
            print("  {:6s} done ({}/{}) {:.0f}s".format(g, i + 1, len(games), time.time() - t0), flush=True)
    order = sorted(res)
    S = np.stack([res[g][0] for g in order])           # (games, cfgs, seeds, settings, 3)
    V = np.stack([res[g][1] for g in order])           # (games, 3)
    if a.cmd == "smoke":
        for ci, c in enumerate(cfgs):
            print(c, {st: np.round(S[:, ci, :, k].mean((0, 1)), 4).tolist() for k, st in enumerate(SETTINGS)})
        print("v6", V.mean(0))
        return
    pickle.dump(dict(games=order, cfgs=cfgs, seeds=seeds, settings=SETTINGS, targets=C.TARGETS, S=S, V=V),
                open(os.path.join(C.OUT, a.cmd + ".pkl"), "wb"))
    print("saved", os.path.join(C.OUT, a.cmd + ".pkl"), "{:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
