#!/usr/bin/env python3
"""Score saved PFF codes on the 48 selection games (or --holdout), paired against a reference.

    python candidates/nn_pff/score_pff.py NAME [NAME ...] [--ref NAME] [--holdout]

NAME is a folder under $MTAH_WORK/probs/nn_pff (run_pff.py) or $MTAH_WORK/probs/pff (bench_pff.py,
full models: valid for DFL-only candidates such as ensemble_ctrl_v4, which never saw PFF).
Scoring = candidates/pff/bench_pff.score_codes: spellF1 vs the touch labels (raw) and its
control.fold readings T = 0.5 / 1 / 1.5 / 2 s, fine3 = mean(raw, T0.5, T1), the nore / seq
label variants, macroF1. Paired: mean difference, SE over games, wins/n.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND]
import bench_pff  # noqa: E402

WORK = bench_pff.WORK
ARCH = os.path.join(WORK, "arch", "nn_pff")


def games(holdout=False):
    if holdout:
        return json.load(open(os.path.join(ARCH, "holdout_games.json")))["games"]
    return sorted(g for f in json.load(open(os.path.join(ARCH, "selection_folds.json")))["folds"] for g in f)


def find(name):
    for root in (os.path.join(WORK, "probs", "nn_pff"), os.path.join(WORK, "probs", "pff")):
        if os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    if os.path.isdir(name):
        return name
    raise SystemExit("no codes for " + name)


def score_dir(name, gs):
    d = find(name)
    rows = {}
    for g in gs:
        f = os.path.join(d, "pff2_{}_2027_codes.npy".format(g))
        if os.path.isfile(f):
            rows[g] = bench_pff.score_codes(np.load(f).astype(np.int8), g)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    gs = games(a.holdout)
    res = {n: score_dir(n, gs) for n in a.names}
    if a.ref and a.ref not in res:
        res[a.ref] = score_dir(a.ref, gs)
    common = sorted(set.intersection(*[set(r) for r in res.values()]))
    if len(common) < len(gs):
        print("note: {} of {} games common to all".format(len(common), len(gs)))
    res = {n: {g: r[g] for g in common} for n, r in res.items()}
    bench_pff.print_table(res, a.ref)


if __name__ == "__main__":
    main()
