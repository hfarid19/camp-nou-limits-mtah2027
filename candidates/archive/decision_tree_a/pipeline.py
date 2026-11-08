"""decision_tree_a: two small readable decision trees (dead ball, team) + LDE's decoder.

Evidence. Two scikit-learn DecisionTreeClassifiers, exported to numpy and walked here without
scikit-learn (tree_walk.py), on learned_dfl_evidence's identity-free per-frame features plus
extra.py's window features for the team call:

- tree_dead.npz: P(dead), trained on every graded frame (DEAD depth, see README.md).
- tree_team.npz: P(home | alive), trained on alive frames only, every row also team-swapped.

Both are averaged with their prediction on the team-swapped features, so H/A are exactly
symmetric. Both are DFL-trained and loaded through dfl_folds.resolve (the bench swaps in
fold_dead_<gi>.npz / fold_team_<gi>.npz, trained without game gi).

Decoding. The team log-odds are smoothed with a centred moving mean of +-TEAM_HW frames (a tree's
leaf probabilities are piecewise constant and flip at single thresholds; the smoothing removes
that flicker before the Viterbi). Then LDE's round-3 decoder (decoder.py, verbatim copy) with the
CFG overrides below, and control.fold(codes, CONTROL_S) (the challenge's "controlled
possession" rule, as in ensemble_ctrl). All settings were selected on the DFL leave-one-game-out
bench (bench --dfl2, ctrl_mean); see README.md.
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

MODEL_DEAD = os.path.join(HERE, "tree_dead.npz")
MODEL_TEAM = os.path.join(HERE, "tree_team.npz")
TEAM_HW = 25          # frames: +-1 s moving mean of the team log-odds
DEAD_HW = 0           # frames: same for the dead log-odds (0 = off)
CFG = {}              # overrides of decoder.CFG
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


fe = _load("decision_tree_a_lde_features", os.path.join(CAND, "learned_dfl_evidence", "features.py"))
ex = _load("decision_tree_a_extra", os.path.join(HERE, "extra.py"))
dec = _load("decision_tree_a_decoder", os.path.join(HERE, "decoder.py"))
tw = _load("decision_tree_a_walk", os.path.join(HERE, "tree_walk.py"))

_TREE = {}
_MEMO = {}


def load_tree(path):
    path = dfl_folds.resolve(path)
    if path not in _TREE:
        with np.load(path) as z:
            t = {k: z[k] for k in z.files}
        t["names"] = [str(s) for s in t["names"]]
        _TREE[path] = t
    return _TREE[path]


def _fingerprint(*arrays):
    """Content hash: an id()-based key can be reused by a new array once the old one is freed
    (decision_tree's memo has that latent bug; bench.py happens to keep the old arrays alive)."""
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def features(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    key = (_fingerprint(players, slot_team, ball), float(sign_half1), tuple(map(tuple, blocks)),
           tuple(pitch), fps)
    if key not in _MEMO:
        _MEMO.clear()
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        E, en, ek = ex.extra(F, names, fps)
        _MEMO[key] = (np.concatenate([F, E], 1), names + en, kinds + ek, len(names))
    return _MEMO[key]


def sym_binary(tree, X, Xs, col, col_swapped):
    """Mean of the tree's leaf probability column `col` on X and `col_swapped` on the team-swapped
    Xs (the tree uses the first len(tree["names"]) feature columns)."""
    nf = len(tree["names"])
    p = tree["prob"][tw.leaf_index(tree, X[:, :nf]), col]
    q = tree["prob"][tw.leaf_index(tree, Xs[:, :nf]), col_swapped]
    return 0.5 * (p + q)


def smooth_logit(p, hw):
    if hw <= 0:
        return p
    q = np.clip(p, 1e-3, 1 - 1e-3)
    return 1.0 / (1.0 + np.exp(-fe.mmean(np.log(q / (1 - q)), -hw, hw)))


def evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    """p_dead, p_home (smoothed, as the decoder uses them), and the feature matrix."""
    X, names, kinds, nbase = features(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    dead, team = load_tree(MODEL_DEAD), load_tree(MODEL_TEAM)
    for t in (dead, team):
        assert t["names"] == names[:len(t["names"])], "feature list differs from the trained tree"
    perm, signs = fe.swap_index(names, kinds)
    Xs = X[:, perm] * signs
    p_dead = sym_binary(dead, X, Xs, 1, 1)        # class 1 = dead, also on swapped features
    p_home = sym_binary(team, X, Xs, 0, 1)        # class 0 = home; on swapped features class 1
    return smooth_logit(p_dead, DEAD_HW), smooth_logit(p_home, TEAM_HW), X[:, :nbase], names[:nbase]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    p_dead, p_home, _, _ = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], 1)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    p_dead, p_home, F, fnames = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    long_gap = dec.long_gap_gated(players, ball, fps, dec.CFG["gap_s"], dec.CFG["gap_speed_ms"])
    kick = dec.kickoff_restarts(tags, blocks, ball, fps, dec.CFG["prekick_s"])
    aux = dict(F=F, names=fnames, p_change=None, long_gap=long_gap, kick_restarts=kick)
    codes = np.asarray(dec.decode(p_dead, p_home, aux, blocks, len(ball), fps, tags, names, cfg=CFG),
                       np.int8)
    if CONTROL_S:
        runs = [(int(first), int(last) + 1) for _, first, last in blocks]
        codes = control.fold(codes, CONTROL_S, runs, fps)
    return codes
