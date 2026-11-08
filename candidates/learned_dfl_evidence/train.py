#!/usr/bin/env python3
"""Fit the two LightGBM frame models on the degraded DFL/IDSSE features (never on Metrica).

    python -I candidates/learned_dfl_evidence/train.py [--cache DIR]

(a) model_d.txt   binary dead (BallStatus 0) vs alive, all graded rows
(b) model_ha.txt  binary home vs away on alive rows, with team-swap augmentation
                  (H/A feature blocks swapped, anti-symmetric columns negated, labels flipped)

The number of boosting rounds of each model is chosen by leave-one-DFL-game-out log-loss
(mean curve over folds, folds run in parallel processes with 1 thread each). The final
models are fitted on all games with that many rounds, deterministic=True, num_threads=1.
"""

import argparse
import glob
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fe  # noqa: E402

# raw data + caches: $IDSSE_CACHE, else <candidate>/external/idsse (git-ignored); --cache overrides
DEFAULT_CACHE = os.environ.get("IDSSE_CACHE", os.path.join(HERE, "external", "idsse"))
PARAMS = dict(objective="binary", deterministic=True, num_threads=1, seed=0,
              force_row_wise=True, num_leaves=31, learning_rate=0.05, min_data_in_leaf=200,
              feature_fraction=0.8, feature_fraction_seed=0, bagging_fraction=1.0,
              bagging_freq=0, verbose=-1)
MAX_ROUNDS = 800


def dataset_config(cache):
    """The degradation settings make_dataset.py recorded for <cache>/ds (provenance in meta)."""
    path = os.path.join(cache, "ds", "config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def load(cache):
    Xs, ys, gs, cs = [], [], [], []
    names = kinds = None
    paths = sorted(glob.glob(os.path.join(cache, "ds", "DFL-MAT-*_*.npz")))
    matches = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in paths})
    for p in paths:
        z = np.load(p)
        Xs.append(z["X"])
        ys.append(z["y"])
        cs.append(z["yc"])
        gs.append(np.full(len(z["y"]), matches.index(os.path.basename(p).rsplit("_", 1)[0])))
        names, kinds = list(z["names"]), list(z["kinds"])
    y = np.concatenate(ys).astype(np.int16) + 10 * np.concatenate(cs).astype(np.int16)
    return np.concatenate(Xs), y, np.concatenate(gs), names, kinds, matches


def task(X, y10, g, names, kinds, which):
    y, yc = y10 % 10, y10 // 10
    if which == "d":
        return X, (y == 2).astype(np.float32), g
    perm, signs = fe.swap_index(names, kinds)
    alive = y < 2
    Xa, ga = X[alive], g[alive]
    if which == "c":                                # hand-over detector, symmetric target
        yc = yc[alive].astype(np.float32)
        return (np.concatenate([Xa, Xa[:, perm] * signs]), np.concatenate([yc, yc]),
                np.concatenate([ga, ga]))
    ya = (y[alive] == 0).astype(np.float32)
    Xs = Xa[:, perm] * signs
    return (np.concatenate([Xa, Xs]), np.concatenate([ya, 1 - ya]), np.concatenate([ga, ga]))


_DATA = {}


def file_sha256(path):
    import hashlib
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _fold(args):
    which, hold = args
    X, y, g, names = _DATA[which]
    tr, va = g != hold, g == hold
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    rec = {}
    bst = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], valid_names=["va"],
                    callbacks=[lgb.record_evaluation(rec)])
    if _DATA.get("fold_dir"):                  # held-out-game models, for the DFL diagnostic
        bst.save_model(os.path.join(_DATA["fold_dir"], "fold_{}_{}.txt".format(which, hold)))
    return np.array(rec["va"]["binary_logloss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--models", default="d,ha,c",
                    help="subset to (re)train; the others keep their entries in <out>/model_meta.json")
    args = ap.parse_args()
    which_all = [w for w in ("d", "ha", "c") if w in args.models.split(",")]
    X, y, g, names, kinds, matches = load(args.cache)
    names = [str(n) for n in names]
    print("rows", len(y), "features", X.shape[1], "games", matches)
    print("label shares H/A/D:", [round(float((y % 10 == c).mean()), 3) for c in range(3)],
          "change-near share", round(float((y // 10).mean()), 4))
    meta = dict(names=names, kinds=[str(k) for k in kinds], matches=matches, params=PARAMS,
                change_prior=float((y // 10)[y % 10 < 2].mean()))
    old_path = os.path.join(args.out, "model_meta.json")
    if len(which_all) < 3 and os.path.isfile(old_path):    # partial retrain: keep the others
        with open(old_path) as handle:
            old = json.load(handle)
        assert old["names"] == meta["names"] and old["matches"] == meta["matches"]
        for w in ("d", "ha", "c"):
            if w not in which_all and w in old:
                meta[w] = old[w]
        if "datasets" in old:
            meta["datasets"] = old["datasets"]
    meta.setdefault("datasets", {})
    for w in which_all:
        meta["datasets"][w] = dataset_config(args.cache)
    _DATA["fold_dir"] = os.path.join(args.cache, "folds")
    os.makedirs(_DATA["fold_dir"], exist_ok=True)
    for which in which_all:
        Xt, yt, gt = task(X, y, g, names, kinds, which)
        _DATA[which] = (Xt, yt, gt, names)
    jobs = [(w, h) for w in which_all for h in sorted(set(_DATA[w][2].tolist()))]
    t0 = time.time()
    if "fork" in mp.get_all_start_methods():                  # _DATA is inherited by fork
        with mp.get_context("fork").Pool(args.jobs) as pool:
            curves = dict(zip(jobs, pool.map(_fold, jobs)))
    else:                                                     # e.g. Windows: run serially
        curves = dict(zip(jobs, map(_fold, jobs)))
    print("LOGO folds done in {:.0f}s".format(time.time() - t0), flush=True)
    for which in which_all:
        Xt, yt, gt, _ = _DATA[which]
        cs = [curves[j] for j in jobs if j[0] == which]
        curve = np.mean(cs, 0)
        best = int(np.argmin(curve)) + 1
        per = [round(float(c[best - 1]), 4) for c in cs]
        print("[{}] LOGO logloss best rounds {} mean {:.4f} per game {}".format(
            which, best, curve[best - 1], per), flush=True)
        booster = lgb.train(PARAMS, lgb.Dataset(Xt, yt, feature_name=names), best)
        booster.save_model(os.path.join(args.out, "model_{}.txt".format(which)))
        imp = booster.feature_importance("gain")
        top = np.argsort(-imp)[:15]
        print("  top features:", [names[i] for i in top], flush=True)
        meta[which] = dict(rounds=best, logo_logloss=float(curve[best - 1]), per_game=per,
                           curve=[round(float(v), 5) for v in curve[::20]])
    meta["lgb_version"] = lgb.__version__
    meta["sha256"] = {w: file_sha256(os.path.join(args.out, "model_{}.txt".format(w)))
                      for w in ("d", "ha", "c")     # asserted by pipeline._models() at load
                      if os.path.isfile(os.path.join(args.out, "model_{}.txt".format(w)))}
    with open(os.path.join(args.out, "model_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)


if __name__ == "__main__":
    main()
