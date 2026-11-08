"""decision_tree_b: a slide-sized, two-stage rule model + LDE's decoder + the control rule.

Evidence. Two shallow decision trees on a short whitelist of readable LDE features
(learned_dfl_evidence/features.py), trained by train.py on the open DFL games under the randomised
degrade_v2 ($MTAH_WORK/ds_v2):

- dead stage: P(dead) from ball / player-motion / time-since-control features;
- team stage: P(home | in play) from the last / next team alone on the ball and the nearest-player
  distances. Its input is also evaluated team-swapped and averaged (H <-> A), so H/A are exactly
  symmetric.

P(H), P(A), P(D) = (1 - Pd) Ph, (1 - Pd) (1 - Ph), Pd. Both trees are stored in rules.npz and
walked in plain numpy (tree_walk.py): no scikit-learn, deterministic. rules.npz is DFL-trained,
so it is loaded through dfl_folds.resolve (the bench swaps in fold_rules_<gi>.npz, trained
without game gi).

Decoder. LDE's round-3 decoder (decoder.py, a verbatim copy of decision_tree/decoder.py) with
p_dead = P(D) and p_home = P(H) / (P(H) + P(A)), then control.fold(codes, CONTROL_S = 2.0 s),
exactly as decision_tree/pipeline.py, so the two candidates differ only in the evidence model.
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

MODEL = os.path.join(HERE, "rules.npz")
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


fe = _load("decision_tree_b_lde_features", os.path.join(CAND, "learned_dfl_evidence", "features.py"))
dec = _load("decision_tree_b_decoder", os.path.join(HERE, "decoder.py"))
tw = _load("decision_tree_b_walk", os.path.join(HERE, "tree_walk.py"))

_RULES = {}
_MEMO = {}


def load_rules(path=None):
    path = dfl_folds.resolve(path or MODEL)
    if path not in _RULES:
        with np.load(path) as z:
            r = {k: z[k] for k in z.files}
        _RULES[path] = dict(d={k[2:]: v for k, v in r.items() if k.startswith("d_")},
                            t={k[2:]: v for k, v in r.items() if k.startswith("t_")})
        for s in ("d", "t"):
            _RULES[path][s]["names"] = [str(v) for v in _RULES[path][s]["names"]]
    return _RULES[path]


# Team scorecard inputs: every one is negated by a home <-> away swap, so the scorecard (no
# intercept) is exactly team-symmetric. Missing values count 0 (no evidence either way).
SCORECARD = [
    ("lastctrl", lambda c: c["lastctrl"], 1.0),                        # +1 home / -1 away had it last
    ("lastctrl_m2.0", lambda c: c["lastctrl_m2.0"], 1.0),              # ... share over +-2 s
    ("since_ctrl_gap", lambda c: c["A_since_ctrl"] - c["H_since_ctrl"], 5.0),  # > 0: home had it more recently (log frames)
    ("nearest_gap_m2.0", lambda c: c["diff_d1_m2.0"], 10.0),           # > 0: away's nearest player closer (m, 2 s mean)
    ("ball_xdir_pmf", lambda c: c["ball_xdir_pmf"], 20.0),             # < 0: ball travelling towards home's attacking goal (m)
    ("fwd_pmf", lambda c: c["diff_fwd_pmf"], 5.0),                     # change in home-minus-away forward speed (m/s)
    ("width_gap", lambda c: c["H_width"] - c["A_width"], 30.0),        # > 0: home is wider (m)
    ("spread_gap_m5.0", lambda c: c["diff_spread_m5.0"], 10.0),        # > 0: home more spread (m, 5 s mean)
]


def scorecard_x(F, names):
    c = {n: F[:, i].astype(np.float64) for i, n in enumerate(names)}
    cols = [np.clip(np.nan_to_num(f(c), nan=0.0), -lim, lim) for _, f, lim in SCORECARD]
    return np.stack(cols, 1)


def stage_probs(rules, F, names, kinds):
    """P(dead), P(home | in play) per frame; the team stage is averaged with its team swap."""
    col = {n: i for i, n in enumerate(names)}
    d, t = rules["d"], rules["t"]
    pd = d["prob"][tw.leaf_index(d, F[:, [col[n] for n in d["names"]]])]
    if "w" in t:                                       # scorecard team stage
        return pd, 1.0 / (1.0 + np.exp(-scorecard_x(F, names) @ t["w"]))
    perm, signs = fe.swap_index(names, kinds)
    tc = [col[n] for n in t["names"]]
    ph = t["prob"][tw.leaf_index(t, F[:, tc])]
    ph_sw = t["prob"][tw.leaf_index(t, (F[:, perm] * signs)[:, tc])]
    return pd, 0.5 * (ph + 1.0 - ph_sw)


def proba_from(rules, F, names, kinds):
    pd, ph = stage_probs(rules, F, names, kinds)
    return np.stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd], 1)


def features(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    # content fingerprint, not id(): a freed array's id can be reused by the next game's array
    key = (hashlib.sha1(np.ascontiguousarray(ball).tobytes()).hexdigest(),
           hashlib.sha1(np.ascontiguousarray(players).tobytes()).hexdigest(),
           len(ball), float(sign_half1), tuple(map(tuple, blocks)))
    if key not in _MEMO:
        _MEMO.clear()
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        _MEMO[key] = fe.compute(players, slot_team, ball, sign, pitch, fps)
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
    return proba_from(load_rules(), F, fnames, kinds)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    F, fnames, kinds = features(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    p = proba_from(load_rules(), F, fnames, kinds)
    long_gap = dec.long_gap_gated(players, ball, fps, dec.CFG["gap_s"], dec.CFG["gap_speed_ms"])
    kick = dec.kickoff_restarts(tags, blocks, ball, fps, dec.CFG["prekick_s"])
    return decode(p, F, fnames, long_gap, kick, blocks, len(ball), fps, tags, names)
