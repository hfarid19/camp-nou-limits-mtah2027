#!/usr/bin/env python3
"""Print the rule model as a slide: the dead-ball tree's leaves and the team scorecard's points.

    python candidates/decision_tree_b/explain.py [rules.npz] [--folds DIR]

With --folds, also prints each leave-one-game-out fold's thresholds / points next to the shipped
model's, to show how stable the learned numbers are.
"""

import argparse
import glob
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("decision_tree_b_pipeline_explain", os.path.join(HERE, "pipeline.py"))
dtb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtb)
LOG_FRAMES = ("ball_gap", "since_ctrl")


def fmt(name, thr):
    if any(k in name for k in LOG_FRAMES):
        return "{:.2f} s".format(np.expm1(thr) / 25.0)
    return "{:.3g}".format(thr)


def leaves(t, node=0, conds=()):
    f = int(t["feature"][node])
    if f < 0:
        c = t["count"][node]
        yield conds, float(t["prob"][node]), float(c.sum())
        return
    name = str(t["names"][f])
    miss = bool(t["missing_left"][node])
    lo = "{} <= {}{}".format(name, fmt(name, t["threshold"][node]), " or missing" if miss else "")
    hi = "{} > {}{}".format(name, fmt(name, t["threshold"][node]), "" if miss else " or missing")
    yield from leaves(t, int(t["left"][node]), conds + (lo,))
    yield from leaves(t, int(t["right"][node]), conds + (hi,))


def load(path):
    z = np.load(path)
    d = {k[2:]: z[k] for k in z.files if k.startswith("d_")}
    t = {k[2:]: z[k] for k in z.files if k.startswith("t_")}
    return d, t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rules", nargs="?", default=os.path.join(HERE, "rules.npz"))
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    d, t = load(a.rules)
    total = d["count"][0].sum()
    print("DEAD BALL (P(dead) per leaf, share of training frames)")
    for conds, p, n in leaves(d):
        print("  {:5.1%} of frames  P(dead) {:.2f}   if {}".format(n / total, p, " and ".join(conds)))
    print("\nTEAM IN PLAY: points for HOME (P(home) = 1 / (1 + exp(-sum)); away = the mirror)")
    for (name, _, lim), w in zip(dtb.SCORECARD, t["w"]):
        print("  {:18s} {:+.3f} per unit (input clipped to +-{:g})".format(name, w, lim))
    if a.folds:
        print("\nfold stability (game left out -> dead-tree root threshold, scorecard points)")
        for path in sorted(glob.glob(os.path.join(a.folds, "fold_rules_*.npz"))):
            fd, ft = load(path)
            print("  {}  root {} <= {}  points {}".format(
                os.path.basename(path), str(fd["names"][fd["feature"][0]]),
                fmt(str(fd["names"][fd["feature"][0]]), fd["threshold"][0]), np.round(ft["w"], 3)))


if __name__ == "__main__":
    main()
