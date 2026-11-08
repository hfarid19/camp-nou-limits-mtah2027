#!/usr/bin/env python3
"""Train the two-stage rule model (dead stage + team stage) and its leave-one-DFL-game-out folds.

    OMP_NUM_THREADS=2 python candidates/decision_tree_b/train.py [--grow 4] [--leaf 1000]
        [--out $MTAH_WORK/arch/decision_tree_b/sweep]

Two shallow scikit-learn DecisionTreeClassifiers (gini, best splits, random_state 0), each on a
short whitelist of readable LDE features (learned_dfl_evidence/features.py):

- dead stage: every training row, target "dead ball" (y == 2) against "in play".
- team stage: in-play rows only (y in {0, 1}), target home (y == 0) against away. Every row is
  also added team-swapped (features.swap_index, H <-> A), so the rules are symmetric; inference
  also averages the swapped prediction, which makes H/A exactly symmetric.

Training data: $MTAH_WORK/ds_v2/ds/<match>_<seed>.npz (make_dataset_v2.py: the 7 DFL games under
the randomised degrade_v2, seeds 41-43, never the bench's 2027-2029, stride 5). Fold gi = all
games except index gi of sorted(os.listdir($MTAH_WORK/idsse/npz)), "full" = all 7 (shipped).

Each stage is grown once to --grow levels; the depth-d model is its top d levels (identical to
max_depth = d up to tie-breaking, as in decision_tree/train.py). The thresholds are therefore
learned inside each fold; only the feature whitelist and min_samples_leaf were fixed by hand.

Export: <out>/D<dead depth>_T<team depth>/fold_rules_<gi>.npz and rules_full.npz, holding both
trees with the prefixes d_ / t_ (feature, threshold, left, right, missing_left, prob, count) and
the whitelists d_names / t_names (column names in the LDE feature matrix).
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

# Readable inputs only. Fixed by hand before any fold was trained (and before seeing a fold score).
DEAD_FEATURES = [
    "ball_seen",        # 1 if the ball is tracked in this frame
    "ball_gap",         # log1p(frames) length of the ball gap this frame sits in (0 if tracked)
    "ball_disp_b",      # metres the ball moved in the last 0.5 s (NaN if missing)
    "ball_speed",       # ball speed m/s (5-frame mean)
    "ball_out",         # metres the ball is outside the pitch
    "ball_out_m2.0",    # ... averaged over +-2 s
    "mean_speed",       # mean player speed m/s
    "mean_speed_m2.0",  # ... averaged over +-2 s
    "any_since_ctrl",   # log1p(frames) since any player last had the ball alone (1 m / 1.5 m)
    "dmin",             # metres from the ball to the nearest player
]
TEAM_FEATURES = [
    "lastctrl",         # +1 home / -1 away / 0 none: team of the last player alone on the ball
    "nextctrl",         # the same, looking forward (the next player alone on the ball)
    "H_since_ctrl", "A_since_ctrl",   # log1p(frames) since home / away last had it
    "H_until_ctrl", "A_until_ctrl",   # log1p(frames) until home / away next have it
    "diff_d1",          # home nearest-player distance minus away's (m; < 0: home closer)
    "diff_d1_m2.0",     # ... averaged over +-2 s
]
ALPHA = 1.0


def matches():
    return sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))


def load(exclude=None):
    X, y = [], []
    names = kinds = None
    for path in sorted(glob.glob(os.path.join(DS, "*_*.npz"))):
        match = os.path.basename(path).rsplit("_", 1)[0]
        if match == exclude:
            continue
        z = np.load(path)
        X.append(z["X"])
        y.append(z["y"].astype(np.int8))
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
    return np.concatenate(X), np.concatenate(y), names, kinds


def stage_data(X, y, names, kinds, stage, team_features=None):
    perm, signs = fe.swap_index(names, kinds)
    if stage == "d":
        cols = [names.index(c) for c in DEAD_FEATURES]
        return np.ascontiguousarray(X[:, cols], np.float32), (y == 2).astype(np.int8)
    cols = [names.index(c) for c in (team_features or TEAM_FEATURES)]
    alive = y < 2
    Xa, ya = X[alive], y[alive]
    Xs = (Xa[:, perm] * signs)[:, cols]
    Xt = np.ascontiguousarray(np.concatenate([Xa[:, cols], Xs]), np.float32)
    return Xt, np.concatenate([(ya == 0), (ya == 1)]).astype(np.int8)   # target: home


def export(model, depth):
    """Top `depth` levels of a fitted binary sklearn tree, BFS order, prob = P(class 1)."""
    t = model.tree_
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
    new = {old: k for k, old in enumerate(keep)}
    n = len(keep)
    out = dict(feature=np.full(n, -1, np.int32), threshold=np.zeros(n, np.float64),
               left=np.full(n, -1, np.int32), right=np.full(n, -1, np.int32),
               missing_left=np.zeros(n, np.uint8))
    for old, k in new.items():
        if t.children_left[old] >= 0 and nd[old] < depth:
            out["feature"][k] = t.feature[old]
            out["threshold"][k] = t.threshold[old]
            out["left"][k], out["right"][k] = new[t.children_left[old]], new[t.children_right[old]]
            out["missing_left"][k] = t.missing_go_to_left[old]
    c = counts[keep]
    out["count"] = c
    out["prob"] = (c[:, 1] + ALPHA) / (c.sum(1) + 2 * ALPHA)
    return out


def fit_stage(args):
    stage, gi, grow, leaf, team_all = args
    from sklearn.tree import DecisionTreeClassifier
    X, y, names, kinds = load(None if gi is None else matches()[gi])
    Xs, ys = stage_data(X, y, names, kinds, stage, names if team_all else None)
    model = DecisionTreeClassifier(max_depth=grow, min_samples_leaf=leaf, random_state=0)
    model.fit(Xs, ys)
    trees = {d: export(model, d) for d in range(1, grow + 1)}
    if gi is None:                                   # self-check of the numpy walker
        sys.path.insert(0, HERE)
        import tree_walk
        z = trees[grow]
        sub = Xs[::50]
        c = z["count"][tree_walk.leaf_index(z, sub)]
        err = float(np.abs(c[:, 1] / c.sum(1) - model.predict_proba(sub)[:, 1]).max())
        print("stage {} full: numpy walker vs sklearn max |diff| {:.1e} on {} rows".format(
            stage, err, len(sub)), flush=True)
        assert err < 1e-9, err
    print("stage {} fold {}: {} rows".format(stage, gi, len(ys)), flush=True)
    return stage, gi, trees, (names if (team_all and stage == "t") else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grow", type=int, default=4)
    ap.add_argument("--leaf", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--team-all", action="store_true",
                    help="team stage may split on any of the ~190 LDE features (no whitelist)")
    ap.add_argument("--team-grow", type=int, default=None, help="team stage depth (default --grow)")
    ap.add_argument("--out", default=os.path.join(WORK, "arch", "decision_tree_b", "sweep"))
    a = ap.parse_args()
    tgrow = a.team_grow or a.grow
    folds = list(range(len(matches()))) + [None]
    jobs = [(s, gi, a.grow if s == "d" else tgrow, a.leaf, a.team_all) for gi in folds for s in ("t", "d")]
    got, tnames = {}, TEAM_FEATURES
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for stage, gi, trees, nm in pool.imap_unordered(fit_stage, jobs):
            got[(stage, gi)] = trees
            tnames = nm or tnames
    for gi in folds:
        for dd in range(1, a.grow + 1):
            for td in range(1, tgrow + 1):
                folder = os.path.join(a.out, "D{}_T{}".format(dd, td))
                os.makedirs(folder, exist_ok=True)
                arr = {"d_" + k: v for k, v in got[("d", gi)][dd].items()}
                arr.update({"t_" + k: v for k, v in got[("t", gi)][td].items()})
                arr["d_names"], arr["t_names"] = np.array(DEAD_FEATURES), np.array(tnames)
                arr["leaf"] = np.int32(a.leaf)
                name = "rules_full.npz" if gi is None else "fold_rules_{}.npz".format(gi)
                np.savez(os.path.join(folder, name), **arr)


if __name__ == "__main__":
    main()
