#!/usr/bin/env python3
"""Auxiliary training targets for the nn_aux trainers (round 5, track R5-4), one file per ds file.

    python -I candidates/nn_aux/make_aux.py --ds $MTAH_WORK/ds_pff2 [--jobs 3]

For every <ds>/ds/<name>_<seed>.npz (make_dataset.py format, one stride-5 phase), writes
<ds>/aux/<name>_<seed>.npz with, at the file's idx rows:
  A  float32 (T, 8) regression targets, NaN = no target
     0 bx    clean ball x in the attack frame, (x - 52.5) * sign / 52.5      (anti: negated by the team swap)
     1 by    |y - 34| / 34                                                   (sym)
     2 bvx   clean ball x-velocity in the attack frame, m/s / 10, +-0.4 s    (anti)
     3 bsp   clean ball speed, m/s / 10                                     (sym)
     4 tnHA  log1p(seconds to the next direct H<->A change) / log1p(60)      (sym; capped at 60 s)
     5 tpHA  log1p(seconds since the last direct H<->A change) / log1p(60)
     6 tnAD  log1p(seconds to the next alive<->dead change) / log1p(60)
     7 tpAD  log1p(seconds since the last alive<->dead change) / log1p(60)
  R  int8 (T,) restart team on dead frames: 0 H, 1 A (team of the next H/A frame of the half), -1 none
  M  int8 (T,) 1 where the degraded (input) ball is missing (reconstruction under ball dropout)
Clean ball = the source game file ($MTAH_WORK/pff_wc2022/npz or idsse/npz; PFF's ball is event-placed);
sign = the raw file's attack sign (the features' frame_sign). Time targets come from the raw file's
full-resolution codes (the training labels), per half; a change beyond the half's end counts as the cap.
Targets only: nothing here is an input feature, so inference (numpy export) is unchanged.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.append(os.path.join(CAND, "nn_pff"))   # appended: nn_pff/queue.py would shadow the stdlib
import features as fe  # noqa: E402

WORK = os.environ.get("MTAH_WORK")
FPS, CAP = 25.0, 60.0
NA = 8


def src_path(game):
    if game.startswith("PFF-"):
        return os.path.join(WORK, "pff_wc2022", "npz", game + ".npz")
    if game.startswith("DFL-"):
        return os.path.join(WORK, "idsse", "npz", game + ".npz")
    raise ValueError(game)


def _dist_next(ev, n):
    """frames from t to the first event position >= t (ev sorted, change between ev-1 and ev); inf if none."""
    out = np.full(n, np.inf)
    if len(ev) == 0:
        return out
    t = np.arange(n)
    j = np.searchsorted(ev, t)
    ok = j < len(ev)
    out[ok] = ev[j[ok]] - t[ok]
    return out


def _dist_prev(ev, n):
    out = np.full(n, np.inf)
    if len(ev) == 0:
        return out
    t = np.arange(n)
    j = np.searchsorted(ev, t, side="right") - 1
    ok = j >= 0
    out[ok] = t[ok] - ev[j[ok]] + 1
    return out


def time_targets(codes, a, b):
    """targets 4-7 on frames a..b (one half) of the full-resolution codes."""
    c = codes[a:b + 1].astype(np.int64)
    n = len(c)
    x, y = c[:-1], c[1:]
    ha = np.where((x != y) & (x < 2) & (y < 2))[0] + 1
    ad = np.where((x != y) & (((x == 2) & (y < 2)) | ((x < 2) & (y == 2))))[0] + 1
    out = np.empty((n, 4))
    for k, (ev, fn) in enumerate([(ha, _dist_next), (ha, _dist_prev), (ad, _dist_next), (ad, _dist_prev)]):
        d = np.minimum(fn(ev, n) / FPS, CAP)
        out[:, k] = np.log1p(d) / np.log1p(CAP)
    return out


def restart_team(codes, a, b):
    c = codes[a:b + 1]
    n = len(c)
    r = np.full(n, -1, np.int8)
    nxt = -1
    for t in range(n - 1, -1, -1):
        if c[t] in (0, 1):
            nxt = c[t]
        elif c[t] == 2:
            r[t] = nxt
    return r


def build(ds_path, out_path):
    if os.path.isfile(out_path):
        return "exists " + out_path
    z = np.load(ds_path)
    idx = z["idx"].astype(np.int64)
    raw = np.load(os.path.join(os.path.dirname(os.path.dirname(ds_path)), "raw", os.path.basename(ds_path)))
    codes, blocks, sign = raw["codes"], raw["blocks"], float(raw["sign"])
    name = os.path.basename(ds_path)[:-4].rsplit("_", 1)[0]
    g = np.load(src_path(name))
    ball = g["ball"][:, :2].astype(np.float64)
    n = len(codes)
    assert len(ball) == n, (ds_path, len(ball), n)
    fs = fe.frame_sign(n, sign, blocks)
    A = np.full((n, NA), np.nan)
    A[:, 0] = (ball[:, 0] - 52.5) * fs / 52.5
    A[:, 1] = np.abs(ball[:, 1] - 34.0) / 34.0
    h = 10
    v = np.full((n, 2), np.nan)
    v[h:n - h] = (ball[2 * h:] - ball[:n - 2 * h]) / (2 * h / FPS)
    A[:, 2] = v[:, 0] * fs / 10.0
    A[:, 3] = np.hypot(v[:, 0], v[:, 1]) / 10.0
    R = np.full(n, -1, np.int8)
    for _, a, b in blocks:
        a, b = int(a), int(b)
        A[a:b + 1, 4:8] = time_targets(codes, a, b)
        R[a:b + 1] = restart_team(codes, a, b)
        # velocity must not straddle a half boundary
        A[a:min(a + h, b + 1), 2:4] = np.nan
        A[max(a, b + 1 - h):b + 1, 2:4] = np.nan
    big = np.abs(A[:, 3]) > 4.0          # > 40 m/s: tracking glitch
    A[big, 2:4] = np.nan
    M = np.isnan(raw["ball"][:, 0]).astype(np.int8)
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, A=A[idx].astype(np.float32), R=R[idx], M=M[idx], idx=idx)
    os.replace(tmp, out_path)
    return "{} rows {} ball-missing {:.2f} restart-known {:.2f}".format(
        os.path.basename(out_path), len(idx), M[idx].mean(), (R[idx] >= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="dataset folder holding ds/ and raw/")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.ds, "aux"), exist_ok=True)
    jobs = [(p, os.path.join(a.ds, "aux", os.path.basename(p))) for p in sorted(glob.glob(os.path.join(a.ds, "ds", "*_*.npz")))]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for m in pool.starmap(build, jobs):
            print(m, flush=True)


if __name__ == "__main__":
    main()
