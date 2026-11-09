#!/usr/bin/env python3
"""Length fingerprint, possessions per 90 and dead share of the PFF label variants (and DFL).

    python -I candidates/pff/label_stats.py [--dfl] [--games N]

Fingerprint: spellF1 (pm.evaluate) of a perfect submission shifted k = 3 / 8 / 25 frames late
inside each graded half (the first k frames repeat the half's first label). It depends only on the
truth's spell lengths; EVALUATION.md publishes 0.982 / 0.893 / 0.654 for the full-match truth
(DFL raw flag: 0.960 / 0.862 / 0.557). Reported as the mean over games, plus pooled over games.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [HERE, CAND, os.path.join(os.path.dirname(CAND), "evaluation")]
import possession_metrics as pm   # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(os.path.dirname(CAND), "external", "work")
SHIFTS = (3, 8, 25)
TRUTH = (0.982, 0.893, 0.654)


def gt_blocks(z):
    status, poss, half = z["status"], z["poss"], z["half"]
    gt = np.full(len(status), pm.X, np.int8)
    gt[(status == 1) & (poss == 1)] = pm.H
    gt[(status == 1) & (poss == 2)] = pm.A
    gt[status == 0] = pm.D
    runs = []
    for h in (1, 2):
        idx = np.where((half == h) & (gt != pm.X))[0]
        if len(idx):
            runs.append((int(idx[0]), int(idx[-1]) + 1))
    return gt, runs


def shifted(gt, runs, k):
    out = gt.copy()
    for a, b in runs:
        out[a + k:b] = gt[a:b - k]
        out[a:a + k] = gt[a]
    return out


def stats(gt, runs):
    r = {}
    for k in SHIFTS:
        r["s%d" % k] = pm.evaluate(gt, shifted(gt, runs, k), bootstrap_n=0)["spell_f1_mean"]
    graded = sum(b - a for a, b in runs)
    n_sp = len(pm.possession_spells(gt, runs))
    alive = int(sum((gt[a:b] != pm.D).sum() for a, b in runs))
    r["poss90"] = n_sp / (graded / 25.0 / 60.0) * 90.0
    r["poss90_alive"] = n_sp / (alive / 25.0 / 60.0) * 90.0          # per 90 min of live ball
    r["dead"] = 1 - alive / graded
    r["mean_s"] = alive / 25.0 / max(n_sp, 1)
    r["n"] = n_sp
    r["min"] = graded / 25.0 / 60.0
    return r


def summarise(name, files):
    rows = []
    for f in files:
        z = np.load(f)
        gt, runs = gt_blocks(z)
        rows.append(stats(gt, runs))
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("{:14s} n={:2d}  shift 3/8/25: {:.3f} / {:.3f} / {:.3f}   poss/90 {:5.0f} (per 90 live {:4.0f})"
          "  dead {:.3f}  mean spell {:4.1f} s  minutes {:5.1f}".format(
              name, len(rows), m["s3"], m["s8"], m["s25"], m["poss90"], m["poss90_alive"], m["dead"],
              m["mean_s"], m["min"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", action="store_true", help="also the 7 DFL games (raw flag)")
    ap.add_argument("--games", type=int, default=0, help="only the first N PFF games")
    a = ap.parse_args()
    print("{:14s}       shift 3/8/25: {:.3f} / {:.3f} / {:.3f}   (EVALUATION.md full-match truth)".format(
        "truth", *TRUTH))
    if a.dfl:
        summarise("DFL raw", sorted(glob.glob(os.path.join(WORK, "idsse", "npz", "*.npz"))))
    import labels
    for v in labels.VARIANTS:
        files = sorted(glob.glob(os.path.join(WORK, "pff_wc2022", "variants", v, "PFF-*.npz")))
        if a.games:
            files = files[:a.games]
        if files:
            summarise("PFF " + v, files)


if __name__ == "__main__":
    main()
