#!/usr/bin/env python3
"""Train the stacked LightGBM dead / team (/ hand-over) models on features_br features
(feat_data.py output), strictly leave-one-game-out.

    python -I candidates/ball_radar/train_lgb.py --feat <scratch>/arch/ball_radar/feat \
        --folds-out <scratch>/arch/ball_radar/folds --out candidates/ball_radar [--models d,ha]

Fold k: trained on r<k>_g<j> (j != k: radar fold k, in-sample on those games) for MAX_ROUNDS,
saved as fold_br<which>_<k>.txt; its log-loss curve on r<k>_g<k> (out-of-sample radar on the
held-out game) gives the LOGO curve. Full model: r<full>_g<j> for all j, rounds = argmin of the
mean LOGO curve, saved as model_br<which>.txt (+ model_br_meta.json). The bench cuts each fold
model to the full model's rounds. Never imports torch.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features_br as fb  # noqa: E402

PARAMS = dict(objective="binary", deterministic=True, num_threads=2, seed=0,
              force_row_wise=True, num_leaves=31, learning_rate=0.08, min_data_in_leaf=200,
              feature_fraction=0.8, feature_fraction_seed=0, bagging_fraction=1.0,
              bagging_freq=0, verbose=-1)
MAX_ROUNDS = 500


def load(feat, k, games):
    Xs, ys, cs = [], [], []
    for j in games:
        z = np.load(os.path.join(feat, "r{}_g{}.npz".format(k, j)))
        Xs.append(z["X"])
        ys.append(z["y"])
        cs.append(z["yc"])
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs), names, kinds


def task(X, y, yc, names, kinds, which):
    if which == "d":
        return X, (y == 2).astype(np.float32)
    perm, signs = fb.swap_index(names, kinds)
    alive = y < 2
    Xa = X[alive]
    if which == "c":
        t = yc[alive].astype(np.float32)
        return np.concatenate([Xa, Xa[:, perm] * signs]), np.concatenate([t, t])
    ya = (y[alive] == 0).astype(np.float32)
    return np.concatenate([Xa, Xa[:, perm] * signs]), np.concatenate([ya, 1 - ya])


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="d,ha")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--radar", default=None, help="radar.npz the features came from (sha256 -> meta)")
    args = ap.parse_args()
    os.makedirs(args.folds_out, exist_ok=True)
    meta = dict(params=PARAMS, lgb_version=lgb.__version__, models={})
    for which in args.models.split(","):
        curves = []
        t0 = time.time()
        for k in range(args.n):
            tr = [j for j in range(args.n) if j != k]
            X, y, yc, names, kinds = load(args.feat, k, tr)
            Xv, yv, ycv, _, _ = load(args.feat, k, [k])
            Xt, yt = task(X, y, yc, names, kinds, which)
            Xvt, yvt = task(Xv, yv, ycv, names, kinds, which)
            dtr = lgb.Dataset(Xt, yt, feature_name=names, free_raw_data=False)
            dva = lgb.Dataset(Xvt, yvt, reference=dtr)
            rec = {}
            bst = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], valid_names=["va"],
                            callbacks=[lgb.record_evaluation(rec)])
            bst.save_model(os.path.join(args.folds_out, "fold_br{}_{}.txt".format(which, k)))
            curves.append(np.array(rec["va"]["binary_logloss"]))
            print("[{}] fold {} best {} {:.4f}  {:.0f}s".format(which, k, int(np.argmin(curves[-1])) + 1,
                                                             curves[-1].min(), time.time() - t0), flush=True)
        curve = np.mean(curves, 0)
        best = int(np.argmin(curve)) + 1
        X, y, yc, names, kinds = load(args.feat, "full", range(args.n))
        Xt, yt = task(X, y, yc, names, kinds, which)
        bst = lgb.train(PARAMS, lgb.Dataset(Xt, yt, feature_name=names), best)
        path = os.path.join(args.out, "model_br{}.txt".format(which))
        bst.save_model(path)
        imp = bst.feature_importance("gain")
        top = [names[i] for i in np.argsort(-imp)[:15]]
        print("[{}] LOGO best rounds {} logloss {:.4f} per game {}  top {}".format(
            which, best, curve[best - 1], [round(float(c[best - 1]), 4) for c in curves], top),
            flush=True)
        meta["models"][which] = dict(rounds=best, logo_logloss=float(curve[best - 1]),
                                     sha256=sha(path))
        meta["names"], meta["kinds"] = names, kinds
    mp = os.path.join(args.out, "model_br_meta.json")
    if os.path.isfile(mp):
        with open(mp) as handle:
            old = json.load(handle)
        for w, v in old.get("models", {}).items():
            meta["models"].setdefault(w, v)
    if args.radar:
        meta["radar_sha256"] = sha(args.radar)
    elif "radar_sha256" in (old if os.path.isfile(mp) else {}):
        meta["radar_sha256"] = old["radar_sha256"]
    with open(mp, "w") as handle:
        json.dump(meta, handle, indent=1)


if __name__ == "__main__":
    main()
