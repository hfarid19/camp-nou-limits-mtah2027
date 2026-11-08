#!/usr/bin/env python3
"""Train the decision-tree evidence model and its leave-one-DFL-game-out folds.

    OMP_NUM_THREADS=2 python candidates/decision_tree/train.py [--leaf 200 2000] [--depths 3 5 8 12]
        [--jobs 2] [--out $MTAH_WORK/arch/decision_tree/sweep]

One scikit-learn DecisionTreeClassifier (gini, best splits, random_state 0) per (min_samples_leaf,
training set), predicting the per-frame label 0 H / 1 A / 2 D from LDE's ~190 features
(learned_dfl_evidence/features.py). Training data: $MTAH_WORK/ds_v2/ds/<match>_<seed>.npz
(make_dataset_v2.py: DFL games under the randomised degrade_v2, stride 5, seeds 41-43). Every row
is also added team-swapped (features.swap_index, H <-> A labels), so the tree sees both
orientations; inference additionally averages the swapped prediction, which makes H/A exactly
symmetric.

Training sets: fold gi = all games except index gi of sorted(os.listdir($MTAH_WORK/idsse/npz))
(the bench's game index), and "full" = all 7 games (the shipped model). Each tree is grown to the
largest depth once; the shallower trees are its top levels (a node at the cut depth becomes a leaf
with its own class counts), which is exactly what max_depth = d would grow, because the greedy
splits above depth d do not depend on max_depth.

Export (plain numpy, read by pipeline.py without scikit-learn): <out>/L<leaf>_d<depth>/
fold_tree_<gi>.npz and tree_full.npz with arrays feature, threshold, left, right, missing_left,
prob (Laplace-smoothed class frequencies, alpha 1), count, impurity, names, depth, leaf.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
DS = os.path.join(WORK, "ds_v2", "ds")
DFL_DIR = os.path.join(WORK, "idsse", "npz")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "learned_dfl_evidence"))
import features as fe  # noqa: E402

ALPHA = 1.0


def matches():
    return sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))


def load(exclude=None):
    X, y = [], []
    names = kinds = None
    for path in sorted(glob.glob(os.path.join(DS, "*.npz"))):
        match = os.path.basename(path).rsplit("_", 1)[0]
        if match == exclude:
            continue
        z = np.load(path)
        X.append(z["X"])
        y.append(z["y"].astype(np.int8))
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
    X, y = np.concatenate(X), np.concatenate(y)
    perm, signs = fe.swap_index(names, kinds)
    ys = y.copy()
    ys[y == 0], ys[y == 1] = 1, 0
    X = np.ascontiguousarray(np.concatenate([X, X[:, perm] * signs]), np.float32)
    return X, np.concatenate([y, ys]), names


def export(model, names, depth, leaf):
    """Top `depth` levels of a fitted sklearn tree as compact numpy arrays (BFS order)."""
    t = model.tree_
    counts = t.value[:, 0, :] * t.weighted_n_node_samples[:, None]
    keep, nd = [0], {0: 0}
    i = 0
    while i < len(keep):                                   # breadth-first, cut at depth
        node = keep[i]
        if t.children_left[node] >= 0 and nd[node] < depth:
            for ch in (t.children_left[node], t.children_right[node]):
                nd[ch] = nd[node] + 1
                keep.append(ch)
        i += 1
    new = {old: k for k, old in enumerate(keep)}
    n = len(keep)
    feature = np.full(n, -1, np.int32)
    threshold = np.zeros(n, np.float64)
    left = np.full(n, -1, np.int32)
    right = np.full(n, -1, np.int32)
    missing_left = np.zeros(n, np.uint8)
    for old, k in new.items():
        if t.children_left[old] >= 0 and nd[old] < depth:
            feature[k] = t.feature[old]
            threshold[k] = t.threshold[old]
            left[k], right[k] = new[t.children_left[old]], new[t.children_right[old]]
            missing_left[k] = t.missing_go_to_left[old]
    c = counts[keep]
    prob = (c + ALPHA) / (c.sum(1, keepdims=True) + 3 * ALPHA)
    return dict(feature=feature, threshold=threshold, left=left, right=right,
                missing_left=missing_left, prob=prob, count=c, impurity=t.impurity[keep],
                names=np.array(names), depth=np.int32(depth), leaf=np.int32(leaf))


def job(args):
    leaf, gi, depths, out = args
    from sklearn.tree import DecisionTreeClassifier
    ms = matches()
    X, y, names = load(None if gi is None else ms[gi])
    model = DecisionTreeClassifier(max_depth=max(depths), min_samples_leaf=leaf, random_state=0)
    model.fit(X, y)
    for d in depths:
        folder = os.path.join(out, "L{}_d{}".format(leaf, d))
        os.makedirs(folder, exist_ok=True)
        name = "tree_full.npz" if gi is None else "fold_tree_{}.npz".format(gi)
        np.savez(os.path.join(folder, name), **export(model, names, d, leaf))
    if gi is None:                                         # self-check of the numpy walker
        sys.path.insert(0, HERE)
        import tree_walk
        z = export(model, names, max(depths), leaf)
        sub = X[::50]
        c = z["count"][tree_walk.leaf_index(z, sub)]
        err = float(np.abs(c / c.sum(1, keepdims=True) - model.predict_proba(sub)).max())
        print("L{} full: numpy walker vs sklearn predict_proba max |diff| {:.2e} on {} rows "
              "({:.1%} with a NaN)".format(leaf, err, len(sub), np.isnan(sub).any(1).mean()), flush=True)
        assert err < 1e-9, err
    print("L{} fold {} done: {} nodes".format(leaf, gi, model.tree_.node_count), flush=True)
    return leaf, gi


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaf", type=int, nargs="+", default=[200, 2000])
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 5, 8, 12])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(WORK, "arch", "decision_tree", "sweep"))
    ap.add_argument("--full-only", action="store_true")
    a = ap.parse_args()
    folds = [None] if a.full_only else list(range(len(matches()))) + [None]
    jobs = [(leaf, gi, a.depths, a.out) for leaf in a.leaf for gi in folds]
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass


if __name__ == "__main__":
    main()
