#!/usr/bin/env python3
"""Fit LDE's LightGBM frame models on DFL (degraded, make_dataset.py) + SkillCorner (make_dataset_sc.py).

    python -I train_sc.py --dfl DFL_DS_DIR --sc SC_DS_DIR --models ha --out OUT --folds FOLDS
                          [--w-sc 1.0] [--sc-share 1.0] [--max-rounds 600] [--jobs 2]

Same tasks, features, parameters and team-swap augmentation as learned_dfl_evidence/train.py:
  d   dead vs alive (all graded rows)
  ha  home vs away on alive rows (+ swapped copy, labels flipped)
  c   hand-over detector (+-4 frames of a direct H<->A flag change) on alive rows (+ swapped copy)
Leave-one-game-out stays defined on the 7 DFL games: fold gi trains on the other 6 DFL games plus
ALL SkillCorner games (weight --w-sc) and is validated on DFL game gi (log-loss curve). The full
model is trained on all 7 DFL + all SkillCorner games with the number of rounds that minimises
the mean held-out DFL curve. Fold models are written as FOLDS/fold_<kind>_<gi>.txt (gi = position
of the match in the sorted DFL list, = bench.py's game index) and kept at --max-rounds; the bench
cuts them to the full model's rounds.
"""

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fe  # noqa: E402

PARAMS = dict(objective="binary", deterministic=True, num_threads=1, seed=0,
              force_row_wise=True, num_leaves=31, learning_rate=0.05, min_data_in_leaf=200,
              feature_fraction=0.8, feature_fraction_seed=0, bagging_fraction=1.0,
              bagging_freq=0, verbose=-1)


def load_dir(ds, pattern):
    Xs, ys, cs, ms = [], [], [], []
    names = kinds = None
    for p in sorted(glob.glob(os.path.join(ds, pattern))):
        z = np.load(p)
        Xs.append(z["X"])
        ys.append(z["y"])
        cs.append(z["yc"])
        ms.append(np.full(len(z["y"]), os.path.basename(p).rsplit("_", 1)[0]))
        names, kinds = [str(v) for v in z["names"]], [str(v) for v in z["kinds"]]
    return (np.concatenate(Xs), np.concatenate(ys).astype(np.int16),
            np.concatenate(cs).astype(np.int16), np.concatenate(ms), names, kinds)


def task(X, y, yc, g, w, names, kinds, which):
    if which == "d":
        return X, (y == 2).astype(np.float32), g, w
    perm, signs = fe.swap_index(names, kinds)
    alive = y < 2
    Xa, ga, wa = X[alive], g[alive], w[alive]
    Xs = Xa[:, perm] * signs
    if which == "c":
        t = yc[alive].astype(np.float32)
        tt = np.concatenate([t, t])
    else:
        t = (y[alive] == 0).astype(np.float32)
        tt = np.concatenate([t, 1 - t])
    return np.concatenate([Xa, Xs]), tt, np.concatenate([ga, ga]), np.concatenate([wa, wa])


_DATA = {}


def _fold(job):
    which, hold = job
    X, y, g, w, names = _DATA[which]
    tr, va = g != hold, g == hold
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    rec = {}
    t0 = time.time()
    bst = lgb.train(PARAMS, dtr, _DATA["rounds"], valid_sets=[dva], valid_names=["va"],
                    callbacks=[lgb.record_evaluation(rec)])
    bst.save_model(os.path.join(_DATA["folds"], "fold_{}_{}.txt".format(which, hold)))
    print("  fold {} {} done {:.0f}s".format(which, hold, time.time() - t0), flush=True)
    return np.array(rec["va"]["binary_logloss"])


def sha(path):
    with open(path, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", required=True)
    ap.add_argument("--sc", required=True)
    ap.add_argument("--models", default="ha")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--w-sc", type=float, default=1.0)
    ap.add_argument("--sc-share", type=float, default=1.0,
                    help="keep every k-th SkillCorner row (k = round(1/share)); 0 = no SkillCorner")
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.folds, exist_ok=True)
    Xd, yd, cd, md_, names, kinds = load_dir(args.dfl, "DFL-MAT-*_*.npz")
    matches = sorted(set(md_.tolist()))
    gd = np.array([matches.index(m) for m in md_])
    if args.sc_share > 0:
        Xs, ys, cs, ms, n2, k2 = load_dir(args.sc, "SC-*_*.npz")
        assert n2 == names and k2 == kinds
        keep = np.arange(len(ys)) % int(round(1 / args.sc_share)) == 0
        Xs, ys, cs = Xs[keep], ys[keep], cs[keep]
    else:
        Xs, ys, cs = Xd[:0], yd[:0], cd[:0]
    X = np.concatenate([Xd, Xs])
    y = np.concatenate([yd, ys])
    yc = np.concatenate([cd, cs])
    g = np.concatenate([gd, np.full(len(ys), -1)])
    w = np.concatenate([np.ones(len(yd), np.float32), np.full(len(ys), args.w_sc, np.float32)])
    print("rows DFL {} SC {} features {}".format(len(yd), len(ys), X.shape[1]), flush=True)
    which_all = args.models.split(",")
    _DATA.update(rounds=args.max_rounds, folds=args.folds)
    for which in which_all:
        Xt, yt, gt, wt = task(X, y, yc, g, w, names, kinds, which)
        _DATA[which] = (Xt, yt, gt, wt, names)
    jobs = [(wh, h) for wh in which_all for h in range(len(matches))]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.jobs) as pool:
        curves = dict(zip(jobs, pool.map(_fold, jobs, chunksize=1)))
    print("folds done {:.0f}s".format(time.time() - t0), flush=True)
    meta_path = os.path.join(args.out, "train_meta.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    for which in which_all:
        cs_ = [curves[(which, h)] for h in range(len(matches))]
        curve = np.mean(cs_, 0)
        best = int(np.argmin(curve)) + 1
        Xt, yt, gt, wt, _ = _DATA[which]
        bst = lgb.train(PARAMS, lgb.Dataset(Xt, yt, weight=wt, feature_name=names), best)
        path = os.path.join(args.out, "model_{}.txt".format(which))
        bst.save_model(path)
        meta[which] = dict(rounds=best, logo_logloss=float(curve[best - 1]),
                           per_game=[round(float(c[best - 1]), 4) for c in cs_],
                           curve=[round(float(v), 5) for v in curve[::20]], w_sc=args.w_sc,
                           sc_share=args.sc_share, dfl_ds=args.dfl, sc_ds=args.sc,
                           sc_rows=int(len(ys)), dfl_rows=int(len(yd)), sha256=sha(path),
                           folds=args.folds)
        print("[{}] best rounds {} DFL LOGO logloss {:.4f} per game {}".format(
            which, best, curve[best - 1], meta[which]["per_game"]), flush=True)
    meta["names"], meta["matches"], meta["lgb_version"] = names, matches, lgb.__version__
    json.dump(meta, open(meta_path, "w"), indent=1)


if __name__ == "__main__":
    main()
