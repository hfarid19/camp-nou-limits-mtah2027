"""pitch_control: LDE's LightGBM frame evidence + pitch-control features, LDE's Viterbi decoder.

See README.md for the method, references, data, every variant tried and the rebuild commands.

Frame evidence: LightGBM boosters on LDE's 194 identity-free features (features.py, an unchanged
copy of learned_dfl_evidence/features.py) plus pitch-control features (pc.py: Spearman 2018
potential pitch control and Fernandez & Bornn 2018 influence control at the ball, along its
future path, around it and over the pitch, plus per-team time to intercept), trained on the open
DFL / IDSSE games degraded with LDE's domain randomisation (make_dataset.py, train.py).
Decoder: learned_dfl_evidence's round-3 decoder, imported from lde_pipeline.py (an unchanged
copy of learned_dfl_evidence/pipeline.py), with its CFG.

Each model's file name and feature list are in model_meta.json (names only, nothing learned);
the sha256 of every model file is asserted at load. Deterministic: num_threads=1.
"""

import hashlib
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fe        # noqa: E402
import pc                    # noqa: E402
import lde_pipeline as lp    # noqa: E402

warnings.filterwarnings("ignore")

# Development only: PC_MODEL_DIR points the pipeline at a scratch model set (README variants).
MODEL_DIR = os.environ.get("PC_MODEL_DIR", HERE)
if MODEL_DIR != HERE:
    print("[pitch_control] WARNING: models from PC_MODEL_DIR=" + MODEL_DIR, file=sys.stderr)

_M = {}


def _models():
    if not _M:
        import lightgbm as lgb
        with open(os.path.join(MODEL_DIR, "model_meta.json")) as h:
            meta = json.load(h)
        assert lgb.__version__ == meta["lgb_version"], "lightgbm version differs from training"
        for w in ("d", "ha", "c"):
            path = os.path.join(MODEL_DIR, meta[w]["file"])
            with open(path, "rb") as h:
                got = hashlib.sha256(h.read()).hexdigest()
            assert got == meta[w]["sha256"], "{} sha256 mismatch".format(path)
            _M[w] = lgb.Booster(model_file=path)
            assert _M[w].feature_name() == meta[w]["names"], "{} feature list".format(w)
        _M["meta"] = meta
    return _M


_CACHE = {}


def _key(players, ball, sign_half1, blocks):
    h = hashlib.sha1(np.ascontiguousarray(ball).tobytes())
    h.update(np.ascontiguousarray(players[::97]).tobytes())
    h.update(repr((float(sign_half1), list(blocks), players.shape)).encode())
    return h.hexdigest()


def frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    key = _key(players, ball, sign_half1, blocks)
    if key in _CACHE:
        return _CACHE[key]
    m = _models()
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    need_pc = any(n not in names for w in ("d", "ha", "c") for n in m["meta"][w]["names"])
    if need_pc:
        P, pnames, pkinds = pc.compute(players, slot_team, ball, sign, pitch, fps)
        allF = np.concatenate([F, P], 1)
        alln, allk = names + pnames, kinds + pkinds
    else:
        allF, alln, allk = F, names, kinds
    col = {n: i for i, n in enumerate(alln)}

    def run(w, symmetric):
        nm = m["meta"][w]["names"]
        ix = [col[n] for n in nm]
        X = allF[:, ix]
        if not symmetric:
            return m[w].predict(X, num_threads=1)
        perm, signs = fe.swap_index(nm, [allk[i] for i in ix])
        p1 = m[w].predict(X, num_threads=1)
        p2 = m[w].predict(X[:, perm] * signs, num_threads=1)
        return p1, p2

    p_dead = run("d", False)
    h1, h2 = run("ha", True)
    p_home = 0.5 * (h1 + 1.0 - h2)                       # exact H/A symmetry
    c1, c2 = run("c", True)
    aux = dict(F=F, names=names, p_change=0.5 * (c1 + c2), change_prior=0.04)
    aux["long_gap"] = lp.long_gap_gated(players, ball, fps, lp.CFG["gap_s"], lp.CFG["gap_speed_ms"])
    _CACHE.clear()
    _CACHE[key] = (p_dead, p_home, aux)
    return p_dead, p_home, aux


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    p_dead, p_home, aux = frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    aux = dict(aux, kick_restarts=lp.kickoff_restarts(tags, blocks, ball, fps, lp.CFG["prekick_s"]))
    return lp.decode(p_dead, p_home, aux, blocks, len(ball), fps, tags, names)


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    """The per-frame evidence the decoder works from: P(D) = the dead model with the long-gap
    and goal-tag log-odds terms, P(H | alive) = the team model after LDE's local-contrast and
    tag terms; P(H) = (1 - P(D)) P(H | alive), P(A) = (1 - P(D)) (1 - P(H | alive))."""
    p_dead, p_home, aux = frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    c = lp.CFG
    q = np.clip(p_home, 1e-6, 1 - 1e-6)
    lh = np.log(q / (1 - q))
    if c["contrast_beta"] > 0:
        cc = lh - lp.pp.moving_mean(lh, int(c["contrast_hw"]))
        cc = np.sign(cc) * np.maximum(np.abs(cc) - c["contrast_gate"], 0.0)
        lh = np.clip(lh + c["contrast_beta"] * cc, -np.log(999.0), np.log(999.0))
    pdd = np.clip(p_dead, 1e-3, 1 - 1e-3)
    ld = np.log(pdd / (1 - pdd)) + c["w_gap"] * aux["long_gap"]
    if len(tags):
        col = {k: i for i, k in enumerate(aux["names"])}
        raw_seen = aux["F"][:, col["ball_seen"]] > 0.5
        tt, td = lp.tag_priors(tags, names, len(ball), fps, raw_seen, c["goal_win_s"])
        lh = lh + c["w_tag_team"] * tt
        ld = ld + c["w_tag_goal"] * td
    ph = np.clip(1 / (1 + np.exp(-lh)), 1e-3, 1 - 1e-3)
    pd = np.clip(1 / (1 + np.exp(-ld)), 1e-3, 1 - 1e-3)
    out = np.stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd], 1)
    return out / out.sum(1, keepdims=True)
