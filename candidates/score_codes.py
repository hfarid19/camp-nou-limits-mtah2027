#!/usr/bin/env python3
"""Rescore saved predictions offline under raw and controlled-possession DFL labels.

    python candidates/score_codes.py <codes dir> [<codes dir> ...] [--split dfl2]

Reads <split>_<match>_<seed>_codes.npy (bench.py --save-probs) and reports spellF1 against the
raw DFL labels and against bench.controlled_labels with T = 1, 2, 3 s, plus their mean (the
selection criterion from 2026-11-08) and the possession counts.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench                     # noqa: E402
import possession_metrics as pm  # noqa: E402

TS = (0, 1, 2, 3)


def score_dir(d, split):
    files = sorted(glob.glob(os.path.join(d, split + "_*_codes.npy")))
    prep = bench.dfl2_prepared if split == "dfl2" else bench.dfl_prepared
    res = {T: [] for T in TS}
    poss = []
    for f in files:
        match, seed = os.path.basename(f)[len(split) + 1:-len("_codes.npy")].rsplit("_", 1)
        gt = prep(match, int(seed))[3]
        pred = np.load(f)
        for T in TS:
            r = pm.evaluate(bench.controlled_labels(gt, T), pred, bootstrap_n=0)
            res[T].append(r["spell_f1_mean"])
        poss.append(r["n_pred_spells"])
    return {T: float(np.mean(v)) for T, v in res.items()}, int(np.sum(poss)), len(files), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--split", default="dfl2")
    args = ap.parse_args()
    print("{:28s} {:>4s} {:>8s} {:>8s} {:>8s} {:>8s} {:>9s} {:>6s}".format(
        "codes", "n", "raw", "ctrl1", "ctrl2", "ctrl3", "ctrl mean", "poss"))
    for d in args.dirs:
        m, poss, n, _ = score_dir(d, args.split)
        print("{:28s} {:4d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:9.4f} {:6d}".format(
            os.path.basename(d.rstrip("/")), n, m[0], m[1], m[2], m[3], np.mean([m[1], m[2], m[3]]), poss))


if __name__ == "__main__":
    main()
