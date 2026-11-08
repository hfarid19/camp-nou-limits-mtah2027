#!/usr/bin/env python3
"""Fit two shallow single decision trees on the realistic DFL dataset (ds_v2) and export them to npz.

    OMP_NUM_THREADS=4 python candidates/dtree_simple/train_tree.py [--target raw|ctrl] [--out DIR]

tree_d   dead (BallStatus 0) vs alive, all graded rows (stride 5)
tree_ha  home vs away on alive rows, with team-swap augmentation (same as learned_dfl_evidence)

Targets: raw DFL flags, or (--target ctrl) H/A relabelled with control.fold(codes, 2.0 s) on the
full-resolution codes of ds_v2/raw (the challenge's controlled-possession rule).

max_depth is picked from DEPTHS by leave-one-DFL-game-out log-loss (mean over the 7 folds);
min_samples_leaf is fixed. Then fold models (fold_<kind>_<gi>.npz, gi = position in
sorted(os.listdir(idsse/npz))) and the full model (on all 7 games) are written with that depth.
sklearn is used for fitting only; inference (pipeline.py) is numpy on the exported arrays.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.dirname(HERE))
import features as fe            # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
DS = os.path.join(WORK, "ds_v2")
GAMES = sorted(os.listdir(os.path.join(WORK, "idsse", "npz")))
GAMES = [g[:-4] for g in GAMES]
DEPTHS = (3, 4, 6, 8, 10, 12)
MIN_LEAF = 200
CONTROL_T = 2.0


def ctrl_codes(match_seed):
    import control
    import possession_metrics as pm
    z = np.load(os.path.join(DS, "raw", match_seed + ".npz"))
    codes = z["codes"].astype(np.int8)
    return control.fold(codes, CONTROL_T, pm.valid_runs(codes), 25)


def load(target):
    Xs, ys, gs = [], [], []
    names = kinds = None
    for p in sorted(glob.glob(os.path.join(DS, "ds", "DFL-MAT-*_*.npz"))):
        ms = os.path.basename(p)[:-4]
        z = np.load(p)
        y = z["y"].astype(np.int8)
        if target == "ctrl":
            yc = ctrl_codes(ms)[z["idx"]]
            assert ((yc == 2) == (y == 2)).all()      # fold never touches dead ball
            y = yc
        Xs.append(z["X"].astype(np.float32))
        ys.append(y)
        gs.append(np.full(len(y), GAMES.index(ms.rsplit("_", 1)[0]), np.int8))
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), names, kinds


def task(X, y, g, names, kinds, which):
    if which == "d":
        return X, (y == 2).astype(np.int8), g
    perm, signs = fe.swap_index(names, kinds)
    alive = y < 2
    Xa, ga, ya = X[alive], g[alive], (y[alive] == 0).astype(np.int8)
    return (np.concatenate([Xa, Xa[:, perm] * signs]), np.concatenate([ya, 1 - ya]),
            np.concatenate([ga, ga]))


def fit(X, y, depth):
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=0)
    t.fit(X, y)
    return t


def export(t, path):
    tr = t.tree_
    v = tr.value[:, 0, :]
    p1 = (v[:, 1] / v.sum(1)).astype(np.float64)
    np.savez(path, left=tr.children_left.astype(np.int32), right=tr.children_right.astype(np.int32),
             feature=tr.feature.astype(np.int32), threshold=tr.threshold.astype(np.float64),
             nan_left=tr.missing_go_to_left.astype(np.int8), p1=p1, n=tr.n_node_samples.astype(np.int64))


def predict_arrays(z, X):
    """Numpy tree traversal (same as pipeline.tree_predict)."""
    node = np.zeros(len(X), np.int64)
    left, right, feat, thr, nl, p1 = z["left"], z["right"], z["feature"], z["threshold"], z["nan_left"], z["p1"]
    rows = np.arange(len(X))
    for _ in range(64):
        inner = left[node] >= 0
        if not inner.any():
            break
        r, nd = rows[inner], node[inner]
        x = X[r, feat[nd]]
        go_left = np.where(np.isnan(x), nl[nd] == 1, x <= thr[nd])
        node[r] = np.where(go_left, left[nd], right[nd])
    return p1[node]


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="raw", choices=["raw", "ctrl"])
    ap.add_argument("--out", default=os.path.join(WORK, "arch", "dtree_simple"))
    args = ap.parse_args()
    t0 = time.time()
    X, y, g, names, kinds = load(args.target)
    print("rows", len(y), "features", X.shape[1], "classes", np.bincount(y), "{:.0f}s".format(time.time() - t0))
    out_folds = os.path.join(args.out, "folds_" + args.target)
    os.makedirs(out_folds, exist_ok=True)
    meta = dict(names=names, kinds=kinds, target=args.target, min_leaf=MIN_LEAF, games=GAMES, cv={})
    for which in ("d", "ha"):
        Xt, yt, gt = task(X, y, g, names, kinds, which)
        cv = {}
        for depth in DEPTHS:
            losses = []
            for gi in range(len(GAMES)):
                tr, te = gt != gi, gt == gi
                t = fit(Xt[tr], yt[tr], depth)
                losses.append(logloss(yt[te], t.predict_proba(Xt[te])[:, 1]))
            cv[depth] = float(np.mean(losses))
            print(which, "depth", depth, "LOGO logloss {:.4f}".format(cv[depth]),
                  "{:.0f}s".format(time.time() - t0), flush=True)
        best = min(cv, key=cv.get)
        meta["cv"][which] = dict(cv=cv, depth=best)
        print(which, "-> depth", best, flush=True)
        for gi in range(len(GAMES)):
            tr = gt != gi
            export(fit(Xt[tr], yt[tr], best), os.path.join(out_folds, "fold_{}_{}.npz".format(which, gi)))
        full = fit(Xt, yt, best)
        export(full, os.path.join(args.out, "tree_{}_{}.npz".format(which, args.target)))
        z = np.load(os.path.join(args.out, "tree_{}_{}.npz".format(which, args.target)))
        assert np.allclose(predict_arrays(z, Xt[:20000]), full.predict_proba(Xt[:20000])[:, 1])
        imp = full.feature_importances_
        top = np.argsort(-imp)[:12]
        meta["cv"][which]["top_features"] = [(names[i], round(float(imp[i]), 4)) for i in top]
        meta["cv"][which]["leaves"] = int(full.get_n_leaves())
        print(which, "leaves", full.get_n_leaves(), "top", meta["cv"][which]["top_features"][:6], flush=True)
    with open(os.path.join(args.out, "meta_{}.json".format(args.target)), "w") as h:
        json.dump(meta, h, indent=1)


if __name__ == "__main__":
    main()
