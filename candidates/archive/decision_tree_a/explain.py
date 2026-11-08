#!/usr/bin/env python3
"""Print an exported tree as readable rules and its feature importances.

    python candidates/decision_tree_a/explain.py [tree_dead.npz | tree_team.npz] [--depth 4] [--top 20]

Rules: one line per node, indented by depth, "feature <= threshold" (NaN direction in brackets),
with the node's share of training rows and its class mix (dead tree: alive / D; team tree:
H / A, on alive rows incl. the team-swapped copies). Thresholds of the log1p
frame-count features (ball_since, ball_gap, *_since_ctrl, ...) are also shown in seconds.
Importance: total weighted gini decrease of the splits on each feature, normalised to 1 (the
same definition as scikit-learn's feature_importances_), over the whole exported tree.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FRAMES = ("ball_since", "ball_until", "ball_gap", "since_ctrl", "until_ctrl", "since_near", "until_near")
CLASSES = {3: ("H", "A", "D"), "dead": ("alive", "D"), "team": ("H", "A")}


def fmt_thr(name, thr):
    if any(k in name for k in LOG_FRAMES) and "_m" not in name and "_s" not in name.split("ctrl")[-1].split("near")[-1]:
        return "{:.3g} (= {:.2f} s)".format(thr, np.expm1(thr) / 25.0)
    return "{:.3g}".format(thr)


def rules(t, max_depth, node=0, depth=0, lines=None, prefix=""):
    lines = [] if lines is None else lines
    c = t["count"][node]
    total = t["count"][0].sum()
    mix = c / c.sum()
    labels = CLASSES[len(mix)] if t.get("kind") is None else CLASSES[str(t["kind"])]
    tag = "n={:5.1%}  ".format(c.sum() / total) + " ".join(
        "{} {:.2f}".format(k, v) for k, v in zip(labels, mix))
    f = int(t["feature"][node])
    if f < 0 or depth >= max_depth:
        lines.append("{}{} -> {}  [{}]".format("  " * depth, prefix, labels[int(np.argmax(mix))], tag))
        return lines
    name = str(t["names"][f])
    miss = "NaN left" if t["missing_left"][node] else "NaN right"
    lines.append("{}{}[{}]".format("  " * depth, prefix, tag))
    rules(t, max_depth, int(t["left"][node]), depth + 1, lines,
          "if {} <= {} ({}): ".format(name, fmt_thr(name, t["threshold"][node]), miss))
    rules(t, max_depth, int(t["right"][node]), depth + 1, lines,
          "if {} >  {}: ".format(name, fmt_thr(name, t["threshold"][node])))
    return lines


def importances(t):
    w = t["count"].sum(1)
    imp = np.zeros(len(t["names"]))
    for k in np.where(t["feature"] >= 0)[0]:
        lft, rgt = t["left"][k], t["right"][k]
        imp[t["feature"][k]] += (w[k] * t["impurity"][k] - w[lft] * t["impurity"][lft]
                                 - w[rgt] * t["impurity"][rgt])
    return imp / imp.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", nargs="?", default=os.path.join(HERE, "tree_team.npz"))
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    z = np.load(a.tree)
    t = {k: z[k] for k in z.files}
    t["kind"] = "dead" if "dead" in os.path.basename(a.tree) else "team"
    n_leaves = int((t["feature"] < 0).sum())
    print("tree: depth {}, min_samples_leaf {}, {} nodes, {} leaves, {} distinct features".format(
        int(t["depth"]), int(t["leaf"]), len(t["feature"]), n_leaves,
        len(set(t["feature"][t["feature"] >= 0].tolist()))))
    print("\n".join(rules(t, a.depth)))
    imp = importances(t)
    print("\nfeature importance (weighted gini decrease, sums to 1):")
    for i in np.argsort(-imp)[:a.top]:
        print("  {:22s} {:.3f}".format(str(t["names"][i]), imp[i]))


if __name__ == "__main__":
    main()
