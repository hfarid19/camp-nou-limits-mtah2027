#!/usr/bin/env python3
"""Fit the ball-free dead/alive LightGBM model (model_bfd.txt) on make_dataset.py rows.

    python -I candidates/ballfree_dead/train.py --ds <dir> --folds <dir> [--jobs 2]

Leave-one-DFL-game-out: one fold model per held-out game (fold_bfd_<gi>.txt, gi = index in the
sorted game list = the bench's index), each trained for MAX_ROUNDS with the held-out game's
log-loss recorded; the number of rounds of the final all-games model is the argmin of the mean
held-out curve (bench --dfl truncates every fold model to that many rounds). Deterministic
(deterministic=True, fixed seeds); at most 2 threads in total.
"""
import argparse
import glob
import hashlib
import json
import os
import time
import multiprocessing as mp

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = dict(objective="binary", deterministic=True, num_threads=1, seed=0, force_col_wise=True,
              num_leaves=31, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.5,
              feature_fraction_seed=0, bagging_fraction=1.0, bagging_freq=0, lambda_l2=1.0,
              verbose=-1)
MAX_ROUNDS = 700
_DATA = {}


def load(ds, drop=()):
    paths = sorted(glob.glob(os.path.join(ds, "DFL-MAT-*_*.npz")))
    matches = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in paths})
    Xs, ys, gs = [], [], []
    names = None
    for p in paths:
        z = np.load(p)
        Xs.append(z["X"])
        ys.append((z["y"] == 2).astype(np.float32))
        gs.append(np.full(len(z["y"]), matches.index(os.path.basename(p).rsplit("_", 1)[0])))
        names = [str(n) for n in z["names"]]
    X = np.concatenate(Xs)
    keep = [i for i, n in enumerate(names) if not any(n.startswith(d) for d in drop)]
    return X[:, keep], np.concatenate(ys), np.concatenate(gs), [names[i] for i in keep], matches


def _fold(hold):
    X, y, g, names, fold_dir, tag = _DATA["d"]
    tr, va = g != hold, g == hold
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    rec = {}
    bst = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], valid_names=["va"],
                    callbacks=[lgb.record_evaluation(rec)])
    bst.save_model(os.path.join(fold_dir, "fold_{}_{}.txt".format(tag, hold)))
    p = bst.predict(X[va], num_threads=1)
    return np.array(rec["va"]["binary_logloss"]), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--tag", default="bfd")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--drop", default="", help="comma list of feature-name prefixes to drop")
    a = ap.parse_args()
    drop = tuple(d for d in a.drop.split(",") if d)
    X, y, g, names, matches = load(a.ds, drop)
    print("rows", len(y), "features", X.shape[1], "dead share", round(float(y.mean()), 3), flush=True)
    os.makedirs(a.folds, exist_ok=True)
    _DATA["d"] = (X, y, g, names, a.folds, a.tag)
    t0 = time.time()
    holds = sorted(set(g.tolist()))
    with mp.get_context("fork").Pool(a.jobs) as pool:
        res = pool.map(_fold, holds)
    print("folds done in {:.0f}s".format(time.time() - t0), flush=True)
    curve = np.mean([r[0] for r in res], 0)
    best = int(np.argmin(curve)) + 1
    per = [round(float(r[0][best - 1]), 4) for r in res]
    accs = []
    for hold, (_, p) in zip(holds, res):
        yy = y[g == hold]
        accs.append(round(float(((p > 0.5) == (yy > 0.5)).mean()), 4))
    print("[{}] best rounds {} LOGO logloss {:.4f} per game {} (at {} rounds) frame acc@{} {}".format(
        a.tag, best, curve[best - 1], per, best, MAX_ROUNDS, accs), flush=True)
    params = dict(PARAMS, num_threads=a.jobs)
    bst = lgb.train(params, lgb.Dataset(X, y, feature_name=names), best)
    path = os.path.join(a.out, "model_{}.txt".format(a.tag))
    bst.save_model(path)
    imp = bst.feature_importance("gain")
    top = np.argsort(-imp)[:25]
    print("top features:", [names[i] for i in top], flush=True)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = dict(names=names, rounds=best, logo_logloss=float(curve[best - 1]), per_game=per,
                frame_acc_maxrounds=accs, matches=matches, params=PARAMS, drop=list(drop),
                lgb_version=lgb.__version__, sha256=sha,
                curve=[round(float(v), 5) for v in curve[::20]])
    with open(os.path.join(a.out, "model_{}_meta.json".format(a.tag)), "w") as h:
        json.dump(meta, h, indent=1)
    print("sha256", sha[:12])


if __name__ == "__main__":
    main()
