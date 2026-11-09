#!/usr/bin/env python3
"""ball_radar_pff: ball_radar_v3/train_lgb.py with the fold specs of brp_common (4 PFF folds + 7 DFL
LOGO folds): fold spec s trains on r<s>_<game> for the games s allows (MAX_ROUNDS) and records the
log-loss curve on the games it excludes; saved as fold_br<which>_<g>.txt (DFL, the bench's name) or
fold_pff_br<which>_<k>.txt (+ .games.json, for candidates/nn_pff/run_pff.py). The full model trains on
r<full>_<game> for all games, rounds = argmin of the mean of the 11 out-of-sample curves (the bench /
run_pff cut every fold model to those rounds). Params unchanged. --folds-only / --full-only split the
work across processes.

Train the stacked LightGBM dead / team (/ hand-over) models on features_br features
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
import brp_common as bc    # noqa: E402

PARAMS = dict(objective="binary", deterministic=True, num_threads=2, seed=0,
              force_row_wise=True, num_leaves=31, learning_rate=0.08, min_data_in_leaf=200,
              feature_fraction=0.8, feature_fraction_seed=0, bagging_fraction=1.0,
              bagging_freq=0, verbose=-1)
MAX_ROUNDS = 500


def load(feat, k, games):
    Xs, ys, cs = [], [], []
    for j in games:
        if not os.path.isfile(os.path.join(feat, "r{}_{}.npz".format(k, j))):
            raise SystemExit("missing features r{}_{}".format(k, j))
        z = np.load(os.path.join(feat, "r{}_{}.npz".format(k, j)))
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
    ap.add_argument("--radar", default=None, help="radar.npz the features came from (sha256 -> meta)")
    ap.add_argument("--specs", default=",".join(bc.SPECS))
    ap.add_argument("--folds-only", action="store_true")
    ap.add_argument("--full-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.folds_out, exist_ok=True)
    meta = dict(params=PARAMS, lgb_version=lgb.__version__, models={})
    allg = list(bc.games())
    for which in args.models.split(","):
        curves = []
        t0 = time.time()
        for k in args.specs.split(","):
            stem = "fold_br{}_{}".format(which, k[3:]) if k.startswith("dfl") else "fold_pff_br{}_{}".format(which, k[3:])
            cpath = os.path.join(args.folds_out, stem + ".curve.npy")
            if args.full_only or os.path.isfile(cpath):
                curves.append(np.load(cpath))
                continue
            tr = bc.train_names(k)
            X, y, yc, names, kinds = load(args.feat, k, tr)
            Xv, yv, ycv, _, _ = load(args.feat, k, sorted(bc.excluded(k)))
            Xt, yt = task(X, y, yc, names, kinds, which)
            Xvt, yvt = task(Xv, yv, ycv, names, kinds, which)
            del X, Xv
            dtr = lgb.Dataset(Xt, yt, feature_name=names, free_raw_data=False)
            dva = lgb.Dataset(Xvt, yvt, reference=dtr)
            rec = {}
            bst = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], valid_names=["va"],
                            callbacks=[lgb.record_evaluation(rec)])
            bst.save_model(os.path.join(args.folds_out, stem + ".txt"))
            with open(os.path.join(args.folds_out, stem + ".txt.games.json"), "w") as handle:
                json.dump(tr, handle)
            curves.append(np.array(rec["va"]["binary_logloss"]))
            np.save(cpath, curves[-1])
            print("[{}] fold {} best {} {:.4f}  {:.0f}s".format(which, k, int(np.argmin(curves[-1])) + 1,
                                                             curves[-1].min(), time.time() - t0), flush=True)
        if args.folds_only:
            continue
        curve = np.mean(curves, 0)
        best = int(np.argmin(curve)) + 1
        X, y, yc, names, kinds = load(args.feat, "full", allg)
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
    if args.folds_only:
        return
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
