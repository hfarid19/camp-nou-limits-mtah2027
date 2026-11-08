#!/usr/bin/env python3
"""Team stage as a points scorecard (logistic regression, no intercept) + the dead-stage tree.

    python candidates/decision_tree_b/train_scorecard.py [--dead D3] [--C 1.0]

The scorecard's inputs are pipeline.SCORECARD (all negated by a home <-> away swap, so P(home)
is exactly symmetric without augmentation). Trained on the in-play rows of
$MTAH_WORK/ds_v2/ds (seeds 41-43), per leave-one-game-out fold and on all 7 games. The dead-stage
tree is copied from $MTAH_WORK/arch/decision_tree_b/sweep/<dead>_T1 (train.py, same folds).
Writes $MTAH_WORK/arch/decision_tree_b/sweep_card/<dead>_TS/{fold_rules_<gi>,rules_full}.npz.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train  # noqa: E402

spec = importlib.util.spec_from_file_location("decision_tree_b_pipeline_train", os.path.join(HERE, "pipeline.py"))
dtb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtb)
ARCH = os.path.join(train.WORK, "arch", "decision_tree_b")


def fit(exclude, C):
    from sklearn.linear_model import LogisticRegression
    X, y, names, _ = train.load(exclude)
    alive = y < 2
    Z = dtb.scorecard_x(X[alive], names)
    m = LogisticRegression(fit_intercept=False, C=C, max_iter=1000)
    m.fit(Z, (y[alive] == 0).astype(int))
    return m.coef_[0].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead", default="D3")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    out = os.path.join(ARCH, "sweep_card", a.dead + "_TS")
    os.makedirs(out, exist_ok=True)
    ms = train.matches()
    for gi in list(range(len(ms))) + [None]:
        name = "rules_full.npz" if gi is None else "fold_rules_{}.npz".format(gi)
        src = np.load(os.path.join(ARCH, "sweep", a.dead + "_T1", name))
        arr = {k: src[k] for k in src.files if k.startswith("d_")}
        arr["leaf"] = src["leaf"]
        arr["t_w"] = fit(None if gi is None else ms[gi], a.C)
        arr["t_names"] = np.array([n for n, _, _ in dtb.SCORECARD])
        np.savez(os.path.join(out, name), **arr)
        print(gi, np.round(arr["t_w"], 3), flush=True)


if __name__ == "__main__":
    main()
