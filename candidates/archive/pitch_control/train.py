#!/usr/bin/env python3
"""Train one LightGBM frame model with leave-one-DFL-game-out folds (LDE's train.py recipe).

    python -I train.py --ds DIR --which ha|c|d --feats lde+pc|pc|lde --out DIR [--max-rounds N]

ds: make_dataset.py output (X = LDE features, P = pitch-control features, stride-5 rows of the
seeds 11/12/13 degradations). Tasks as in learned_dfl_evidence/train.py: d dead vs alive; ha home
vs away on alive rows with team-swap augmentation; c hand-over (+-4 frames of an H<->A flag
change) on alive rows, swap-augmented. LDE's params (deterministic, num_leaves 31, lr 0.05,
min_data_in_leaf 200, feature_fraction 0.8). Folds run one after another with num_threads=1
(the shared machine allows 2 threads; run two trainings side by side). Each fold model is saved
as OUT/folds/fold_<which>_<gi>.txt (gi = index in the sorted match list = bench --dfl's game
index); the rounds are the argmin of the mean held-out log-loss curve, and the full model
OUT/model_<which>.txt is refit on all 7 games with that many rounds. OUT/meta_<which>.json keeps
the feature names / kinds, rounds, curve and sha256.
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fe  # noqa: E402  (swap_index)

PARAMS = dict(objective="binary", deterministic=True, num_threads=1, seed=0,
              force_row_wise=True, num_leaves=31, learning_rate=0.05, min_data_in_leaf=200,
              feature_fraction=0.8, feature_fraction_seed=0, bagging_fraction=1.0,
              bagging_freq=0, verbose=-1)


def load(ds, feats):
    paths = sorted(glob.glob(os.path.join(ds, "DFL-MAT-*_*.npz")))
    matches = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in paths})
    Xs, ys, cs, gs = [], [], [], []
    for p in paths:
        z = np.load(p)
        parts, names, kinds = [], [], []
        if "lde" in feats:
            parts.append(z["X"]); names += list(z["names"]); kinds += list(z["kinds"])
        if "pc" in feats:
            parts.append(z["P"]); names += list(z["pnames"]); kinds += list(z["pkinds"])
        Xs.append(np.concatenate(parts, 1))
        ys.append(z["y"]); cs.append(z["yc"])
        gs.append(np.full(len(z["y"]), matches.index(os.path.basename(p).rsplit("_", 1)[0])))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(gs),
            [str(n) for n in names], [str(k) for k in kinds], matches)


def task(X, y, yc, g, names, kinds, which):
    if which == "d":
        return X, (y == 2).astype(np.float32), g
    perm, signs = fe.swap_index(names, kinds)
    alive = y < 2
    Xa, ga = X[alive], g[alive]
    Xs = Xa[:, perm] * signs
    if which == "c":
        t = yc[alive].astype(np.float32)
        return np.concatenate([Xa, Xs]), np.concatenate([t, t]), np.concatenate([ga, ga])
    t = (y[alive] == 0).astype(np.float32)
    return np.concatenate([Xa, Xs]), np.concatenate([t, 1 - t]), np.concatenate([ga, ga])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--which", required=True, choices=["d", "ha", "c"])
    ap.add_argument("--feats", default="lde+pc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rounds", type=int, default=700)
    ap.add_argument("--folds-only", action="store_true")
    a = ap.parse_args()
    X, y, yc, g, names, kinds, matches = load(a.ds, a.feats)
    Xt, yt, gt = task(X, y, yc, g, names, kinds, a.which)
    del X
    print(a.which, a.feats, "rows", len(yt), "features", Xt.shape[1], flush=True)
    fold_dir = os.path.join(a.out, "folds")
    os.makedirs(fold_dir, exist_ok=True)
    curves = []
    full = lgb.Dataset(Xt, yt, feature_name=names, free_raw_data=False).construct()
    for hold in range(len(matches)):
        t0 = time.time()
        tr, va = np.where(gt != hold)[0], np.where(gt == hold)[0]
        dtr = full.subset(tr)
        dva = full.subset(va)
        rec = {}
        bst = lgb.train(PARAMS, dtr, a.max_rounds, valid_sets=[dva], valid_names=["va"],
                        callbacks=[lgb.record_evaluation(rec)])
        bst.save_model(os.path.join(fold_dir, "fold_{}_{}.txt".format(a.which, hold)))
        curves.append(np.array(rec["va"]["binary_logloss"]))
        print("  fold {} {} best {} ll {:.4f} ({:.0f}s)".format(hold, matches[hold],
              int(np.argmin(curves[-1])) + 1, curves[-1].min(), time.time() - t0), flush=True)
    curve = np.mean(curves, 0)
    best = int(np.argmin(curve)) + 1
    print("[{}] LOGO best rounds {} mean ll {:.4f} per game {}".format(
        a.which, best, curve[best - 1], [round(float(c[best - 1]), 4) for c in curves]), flush=True)
    meta = dict(which=a.which, feats=a.feats, names=names, kinds=kinds, matches=matches,
                params=PARAMS, rounds=best, logo_logloss=float(curve[best - 1]),
                per_game=[float(c[best - 1]) for c in curves],
                curve=[round(float(v), 5) for v in curve[::20]], lgb_version=lgb.__version__)
    if a.which == "c":
        meta["change_prior"] = float(yc[y < 2].mean())
    if not a.folds_only:
        t0 = time.time()
        bst = lgb.train(PARAMS, full, best)
        mp = os.path.join(a.out, "model_{}.txt".format(a.which))
        bst.save_model(mp)
        imp = bst.feature_importance("gain")
        meta["top"] = [names[i] for i in np.argsort(-imp)[:25]]
        meta["sha256"] = hashlib.sha256(open(mp, "rb").read()).hexdigest()
        print("  full model {:.0f}s; top {}".format(time.time() - t0, meta["top"][:15]), flush=True)
    with open(os.path.join(a.out, "meta_{}.json".format(a.which)), "w") as h:
        json.dump(meta, h, indent=1)


if __name__ == "__main__":
    main()
