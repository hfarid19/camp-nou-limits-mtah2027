#!/usr/bin/env python3
"""Paired comparison of two saved bench --dfl2 codes dirs (same game and degradation seed).

    python candidates/temporal_tcn_v2/paired_codes.py NEW_DIR OLD_DIR [--fold 2.0]

For every dfl2_<match>_<seed>_codes.npy present in both dirs: spellF1 against the raw DFL labels
and ctrl_mean (mean spellF1 against bench.controlled_labels with T = 1, 2, 3 s). --fold T also
applies control.fold(codes, T) over the graded blocks to BOTH predictions offline first (what
ensemble_ctrl does live). Reports means, paired mean difference, its SE over the 7 games (seeds
averaged per game, as bench.py) and wins over the game-seeds.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.insert(0, CAND)
import bench                     # noqa: E402
import control                   # noqa: E402
import possession_metrics as pm  # noqa: E402


def scores(pred, gt):
    raw = pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"]
    ctrl = np.mean([pm.evaluate(bench.controlled_labels(gt, T), pred, bootstrap_n=0)["spell_f1_mean"]
                    for T in (1, 2, 3)])
    return raw, float(ctrl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--fold", type=float, default=0.0)
    a = ap.parse_args()
    names = sorted(set(os.path.basename(f) for f in glob.glob(os.path.join(a.new, "dfl2_*_codes.npy"))) &
                   set(os.path.basename(f) for f in glob.glob(os.path.join(a.old, "dfl2_*_codes.npy"))))
    rows = []
    for nm in names:
        match, seed = nm[len("dfl2_"):-len("_codes.npy")].rsplit("_", 1)
        prep = bench.dfl2_prepared(match, int(seed))
        gt, blocks = prep[3], prep[4]
        runs = [(int(f), int(l) + 1) for _, f, l in blocks]
        out = [match, int(seed)]
        for d in (a.new, a.old):
            pred = np.load(os.path.join(d, nm)).astype(np.int8)
            if a.fold:
                pred = control.fold(pred, a.fold, runs)
            out += list(scores(pred, gt))
        rows.append(out)
    R = np.array([r[2:] for r in rows])          # new raw, new ctrl, old raw, old ctrl
    games = sorted({r[0] for r in rows})
    print("{} vs {}  ({} game-seeds{})".format(os.path.basename(a.new.rstrip("/")),
                                               os.path.basename(a.old.rstrip("/")), len(rows),
                                               ", control.fold {:g} s offline".format(a.fold) if a.fold else ""))
    for label, i, j in (("raw spellF1", 0, 2), ("ctrl_mean", 1, 3)):
        d = R[:, i] - R[:, j]
        pg = np.array([np.mean([d[k] for k, r in enumerate(rows) if r[0] == g]) for g in games])
        se = np.std(pg, ddof=1) / np.sqrt(len(pg))
        print("  {:12s} new {:.4f}  old {:.4f}  diff {:+.4f} (SE {:.4f} over games)  wins {}/{}  "
              "per game {}".format(label, R[:, i].mean(), R[:, j].mean(), d.mean(), se,
                                   int((d > 0).sum()), len(d), " ".join("{:+.3f}".format(v) for v in pg)))


if __name__ == "__main__":
    main()
