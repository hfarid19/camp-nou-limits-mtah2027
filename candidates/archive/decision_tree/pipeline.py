"""decision_tree: one small, readable decision tree as the per-frame H/A/D evidence + LDE's decoder.

Evidence. A single scikit-learn DecisionTreeClassifier (gini, depth DEPTH, min_samples_leaf LEAF)
predicts the frame label H / A / D from learned_dfl_evidence's ~190 identity-free features
(features.py). It is trained by train.py on the open DFL games degraded with the randomised
degrade_v2 ($MTAH_WORK/ds_v2, make_dataset_v2.py), every row also added team-swapped. At inference
the prediction on the team-swapped features is averaged in (H <-> A), so H/A are exactly
symmetric. The tree is exported to tree.npz and walked here in plain numpy (tree_walk.py): no
scikit-learn, no torch, deterministic. tree.npz is DFL-trained, so it is loaded through
dfl_folds.resolve (the bench swaps in fold_tree_<gi>.npz, trained without game gi).

Decoder. LDE's round-3 decoder copied verbatim into decoder.py (Viterbi with switch / dead costs,
local-contrast team logit, gated long-gap dead term, gap-end / ball-out D snaps, shortest-first
merge, first-hit H<->A boundary snap since there is no hand-over model), with p_dead = P(D) and
p_home = P(H) / (P(H) + P(A)). Then control.fold(codes, CONTROL_S): every predicted possession
spell shorter than CONTROL_S seconds after the other team's possession is handed back (the
challenge's "controlled possession" rule, as in ensemble_ctrl; 0 disables it).

Selection: depth and min_samples_leaf by leave-one-game-out on the DFL bench (bench --dfl2,
ctrl_mean); see README.md.
"""

import hashlib
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
for _p in (ROOT, os.path.join(ROOT, "evaluation"), CAND):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import control     # noqa: E402
import dfl_folds   # noqa: E402

MODEL = os.path.join(HERE, "tree.npz")
CONTROL_S = 2.0


def _load(name, path):
    """Import a helper file under a unique module name (other candidates have their own
    features.py / decoder modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fe = _load("decision_tree_lde_features", os.path.join(CAND, "learned_dfl_evidence", "features.py"))
dec = _load("decision_tree_decoder", os.path.join(HERE, "decoder.py"))
tw = _load("decision_tree_walk", os.path.join(HERE, "tree_walk.py"))

_TREE = {}
_MEMO = {}


def load_tree(path=None):
    path = dfl_folds.resolve(path or MODEL)
    if path not in _TREE:
        with np.load(path) as z:
            t = {k: z[k] for k in z.files}
        t["names"] = [str(s) for s in t["names"]]
        _TREE[path] = t
    return _TREE[path]


def sym_proba(tree, F, names, kinds):
    """(N, 3) P(H), P(A), P(D): mean of the tree on F and on the team-swapped F (H <-> A)."""
    assert names == tree["names"], "feature list differs from the trained tree"
    perm, signs = fe.swap_index(names, kinds)
    p = tw.proba(tree, F)
    q = tw.proba(tree, F[:, perm] * signs)
    return 0.5 * (p + q[:, [1, 0, 2]])


def features(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    h = hashlib.sha256()                  # content, not id(): freed arrays can reuse an id
    for a in (players, ball, np.asarray(slot_team)):
        h.update(np.ascontiguousarray(a).tobytes())
    key = (h.hexdigest(), float(sign_half1), tuple(map(tuple, blocks)))
    if key not in _MEMO:
        _MEMO.clear()
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        _MEMO[key] = (F, names, kinds)
    return _MEMO[key]


def decode(p, F, names, long_gap, kick, blocks, n, fps, tags=(), team_names=("Home", "Away"),
           control_s=None):
    control_s = CONTROL_S if control_s is None else control_s
    p_home = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-12)
    aux = dict(F=F, names=names, p_change=None, long_gap=long_gap, kick_restarts=kick)
    codes = np.asarray(dec.decode(p[:, 2], p_home, aux, blocks, n, fps, tags, team_names), np.int8)
    if control_s:
        runs = [(int(first), int(last) + 1) for _, first, last in blocks]
        codes = control.fold(codes, control_s, runs, fps)
    return codes


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    F, fnames, kinds = features(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return sym_proba(load_tree(), F, fnames, kinds)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    F, fnames, kinds = features(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    p = sym_proba(load_tree(), F, fnames, kinds)
    long_gap = dec.long_gap_gated(players, ball, fps, dec.CFG["gap_s"], dec.CFG["gap_speed_ms"])
    kick = dec.kickoff_restarts(tags, blocks, ball, fps, dec.CFG["prekick_s"])
    return decode(p, F, fnames, long_gap, kick, blocks, len(ball), fps, tags, names)
