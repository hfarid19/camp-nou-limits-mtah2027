#!/usr/bin/env python3
"""Train decision_tree_a's two trees (dead, team) and their leave-one-DFL-game-out folds.

    OMP_NUM_THREADS=2 python candidates/decision_tree_a/train.py --kind dead team --leaf 200
        [--depths 6 8 10 12] [--feats ext] [--jobs 2] [--out $MTAH_WORK/arch/decision_tree_a/sweep]

Data: the degraded DFL games of make_dataset_v2.py ($MTAH_WORK/ds_v2/raw/<match>_<seed>.npz,
randomised degrade_v2, seeds 41-43). Features are recomputed on every frame (learned_dfl_evidence
features.compute, plus extra.py's window features with --feats ext) and sampled at the same
stride-5 rows as ds_v2/ds (cached in $MTAH_WORK/arch/decision_tree_a/data).

- dead tree: binary D vs alive, on all graded rows (no swap: the label is team-symmetric; the
  inference averages the tree on the team-swapped features anyway).
- team tree: binary H vs A on alive rows only, every row also added team-swapped (H <-> A).
- joint tree: the teammate's 3-class tree (for comparison).

Each tree is grown once to max(depths); shallower trees are its top levels. Export: plain numpy
arrays (feature, threshold, left, right, missing_left, prob [Laplace alpha 1], count, impurity,
names, depth, leaf), walked by tree_walk.py.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
RAW = os.path.join(WORK, "ds_v2", "raw")
DS = os.path.join(WORK, "ds_v2", "ds")
DFL_DIR = os.path.join(WORK, "idsse", "npz")
ARCH = os.path.join(WORK, "arch", "decision_tree_a")
sys.path.insert(0, os.path.join(CAND, "learned_dfl_evidence"))
sys.path.insert(0, HERE)
import features as fe  # noqa: E402
import extra as ex     # noqa: E402

ALPHA = 1.0
PITCH, FPS = (105.0, 68.0), 25


def matches():
    return sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))


def data_file(stem):
    """Features at ds_v2's stride-5 rows of one degraded game (base + extra), cached."""
    path = os.path.join(ARCH, "data", stem + ".npz")
    if not os.path.isfile(path):
        z = np.load(os.path.join(RAW, stem + ".npz"))
        blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
        sign = fe.frame_sign(len(z["ball"]), float(z["sign"]), blocks)
        F, names, kinds = fe.compute(z["slots"], z["slot_team"], z["ball"], sign, PITCH, FPS)
        E, en, ek = ex.extra(F, names, FPS)
        idx = np.load(os.path.join(DS, stem + ".npz"))["idx"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, X=np.concatenate([F[idx], E[idx]], 1), y=z["codes"][idx].astype(np.int8),
                 names=np.array(names + en), kinds=np.array(kinds + ek), nbase=len(names))
    return path


def load(exclude, kind, feats):
    X, y = [], []
    for path in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        stem = os.path.basename(path)[:-4]
        if stem.rsplit("_", 1)[0] == exclude:
            continue
        z = np.load(data_file(stem))
        nb = int(z["nbase"])
        cols = slice(None) if feats == "ext" else slice(0, nb)
        X.append(z["X"][:, cols])
        y.append(z["y"])
        names, kinds = [str(s) for s in z["names"][cols]], [str(s) for s in z["kinds"][cols]]
    X, y = np.concatenate(X), np.concatenate(y)
    perm, signs = fe.swap_index(names, kinds)
    if kind == "dead":
        return np.ascontiguousarray(X, np.float32), (y == 2).astype(np.int8), names
    if kind == "team":
        alive = y < 2
        X, y = X[alive], (y[alive] == 1).astype(np.int8)            # class 0 H, 1 A
        return (np.ascontiguousarray(np.concatenate([X, X[:, perm] * signs]), np.float32),
                np.concatenate([y, 1 - y]), names)
    ys = y.copy()
    ys[y == 0], ys[y == 1] = 1, 0
    return (np.ascontiguousarray(np.concatenate([X, X[:, perm] * signs]), np.float32),
            np.concatenate([y, ys]), names)


def export(model, names, depth, leaf):
    t = model.tree_
    k = t.value.shape[2]
    counts = t.value[:, 0, :] * t.weighted_n_node_samples[:, None]
    keep, nd = [0], {0: 0}
    i = 0
    while i < len(keep):
        node = keep[i]
        if t.children_left[node] >= 0 and nd[node] < depth:
            for ch in (t.children_left[node], t.children_right[node]):
                nd[ch] = nd[node] + 1
                keep.append(ch)
        i += 1
    new = {old: j for j, old in enumerate(keep)}
    n = len(keep)
    feature = np.full(n, -1, np.int32)
    threshold = np.zeros(n, np.float64)
    left = np.full(n, -1, np.int32)
    right = np.full(n, -1, np.int32)
    missing_left = np.zeros(n, np.uint8)
    for old, j in new.items():
        if t.children_left[old] >= 0 and nd[old] < depth:
            feature[j] = t.feature[old]
            threshold[j] = t.threshold[old]
            left[j], right[j] = new[t.children_left[old]], new[t.children_right[old]]
            missing_left[j] = t.missing_go_to_left[old]
    c = counts[keep]
    prob = (c + ALPHA) / (c.sum(1, keepdims=True) + k * ALPHA)
    return dict(feature=feature, threshold=threshold, left=left, right=right,
                missing_left=missing_left, prob=prob, count=c, impurity=t.impurity[keep],
                names=np.array(names), depth=np.int32(depth), leaf=np.int32(leaf))


def job(args):
    kind, feats, leaf, gi, depths, out = args
    from sklearn.tree import DecisionTreeClassifier
    ms = matches()
    X, y, names = load(None if gi is None else ms[gi], kind, feats)
    model = DecisionTreeClassifier(max_depth=max(depths), min_samples_leaf=leaf, random_state=0)
    model.fit(X, y)
    for d in depths:
        folder = os.path.join(out, "{}_{}_L{}_d{}".format(kind, feats, leaf, d))
        os.makedirs(folder, exist_ok=True)
        name = "tree_full.npz" if gi is None else "fold_{}.npz".format(gi)
        np.savez(os.path.join(folder, name), **export(model, names, d, leaf))
    if gi is None:                                            # numpy walker self-check
        import tree_walk
        z = export(model, names, max(depths), leaf)
        sub = X[::50]
        c = z["count"][tree_walk.leaf_index(z, sub)]
        err = float(np.abs(c / c.sum(1, keepdims=True) - model.predict_proba(sub)).max())
        assert err < 1e-9, err
    print("{} {} L{} fold {}: {} rows, {} nodes".format(kind, feats, leaf, gi, len(y),
                                                        model.tree_.node_count), flush=True)
    return gi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", nargs="+", default=["dead", "team"])
    ap.add_argument("--feats", nargs="+", default=["ext"])
    ap.add_argument("--leaf", type=int, nargs="+", default=[200])
    ap.add_argument("--depths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="default: all + full")
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ARCH, "sweep"))
    a = ap.parse_args()
    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RAW, "*.npz")))
    with mp.get_context("spawn").Pool(a.jobs) as pool:          # feature cache first
        list(pool.imap_unordered(data_file, stems))
    folds = list(range(len(matches()))) if a.folds is None else a.folds
    folds = folds + ([] if a.no_full else [None])
    jobs = [(k, f, leaf, gi, a.depths, a.out) for k in a.kind for f in a.feats for leaf in a.leaf
            for gi in folds]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass


if __name__ == "__main__":
    main()
