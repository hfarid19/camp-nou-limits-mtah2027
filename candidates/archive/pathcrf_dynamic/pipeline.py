"""pathcrf_dynamic: linear-chain CRF with dynamic, feature-dependent transition potentials.

See README.md. Emissions reuse LDE's three LightGBM boosters (copied here: model_d / model_ha /
model_c, sha256-identical, so bench --dfl maps them to LDE's verified leave-one-game-out folds);
the CRF weights (crf.npz, trained by train_crf.py on out-of-fold LDE evidence over fresh DFL
degradations) are loaded through dfl_folds.resolve, so bench --dfl uses fold_crf_<gi>.npz.

predict       -> Viterbi MAP path of the CRF per graded half (+ optional LDE-style post chain)
predict_proba -> forward-backward posterior marginals P(H), P(A), P(D) of the same CRF
"""

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))          # candidates/ (dfl_folds) outside the bench
import lde_base              # noqa: E402   (LDE round 3 evidence + post-processing helpers)
import crf_features as cf    # noqa: E402
import crf                   # noqa: E402
import dfl_folds             # noqa: E402

H, A, D = 0, 1, 2
CFG = dict(
    switch_bias=0.0,     # decode-time offsets added to the learned log-potentials
    die_bias=0.0,
    revive_bias=0.0,
    decode=0,            # 0: Viterbi MAP path; 1: per-frame argmax of the posterior marginals
                         #    (MPM) followed by merge_short_v2
    post=0,              # 1: LDE's post chain (gap-end snap, D-start look-back, merge_short_v2,
                         #    hand-over peak snap) after the CRF Viterbi path
    min_poss_s=1.0,
    min_dead_s=1.0,
    refine_window_s=0.5,
    w_tag_team=0.1,      # match only (Smart Tagging), as LDE
    w_tag_goal=1.0,
)
CRF_PATH = os.path.join(HERE, "crf.npz")
if os.environ.get("PCD_SWEEP") == "1":       # tuning only; the default run ignores the environment
    for _k in CFG:
        if "PCD_" + _k.upper() in os.environ:
            CFG[_k] = float(os.environ["PCD_" + _k.upper()])
    CRF_PATH = os.environ.get("PCD_CRF", CRF_PATH)
    print("[pcd] sweep CFG {} crf {}".format(CFG, CRF_PATH), file=sys.stderr)
_W = {}
_CACHE = {}


def _weights():
    if not _W:
        path = dfl_folds.resolve(CRF_PATH)
        z = np.load(path)
        _W.update({k: z[k] for k in z.files})
    return _W


def _key(args):
    players, slot_team, ball = args[0], args[1], args[2]
    h = hashlib.sha1(np.ascontiguousarray(ball).tobytes())
    h.update(np.ascontiguousarray(players).tobytes())
    h.update(repr((args[3], list(args[4]), len(args[5]), tuple(args[6]))).encode())
    return h.hexdigest()


def _evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    k = _key((players, slot_team, ball, sign_half1, blocks, tags, names))
    if k in _CACHE:
        return _CACHE[k]
    _CACHE.clear()
    p_dead, p_home, aux = lde_base.frame_probs(players, slot_team, ball, sign_half1, blocks,
                                               pitch, fps)
    lg = lde_base.long_gap_gated(players, ball, fps, lde_base.CFG["gap_s"],
                                 lde_base.CFG["gap_speed_ms"])
    sig = cf.signals(p_dead, p_home, aux["p_change"], aux["change_prior"], aux["F"],
                     aux["names"], lg, lde_base.CFG)
    if len(tags):                                   # match only: LDE's Smart Tagging priors
        raw_seen = sig["ball_seen"] > 0.5
        tt, td = lde_base.tag_priors(tags, names, len(ball), fps, raw_seen,
                                     lde_base.CFG["goal_win_s"])
        sig["lh"] = np.clip(sig["lh"] + CFG["w_tag_team"] * tt, -cf.CLIP, cf.CLIP).astype(np.float32)
        sig["lr"] = np.clip(sig["lr"] + CFG["w_tag_team"] * tt, -cf.CLIP, cf.CLIP).astype(np.float32)
        sig["ld"] = np.clip(sig["ld"] + CFG["w_tag_goal"] * td, -cf.CLIP, cf.CLIP).astype(np.float32)
    f = cf.build(sig)
    U, P = crf.potentials(_weights(), f, dict(switch=CFG["switch_bias"], die=CFG["die_bias"],
                                              revive=CFG["revive_bias"]))
    out = dict(U=U, P=P, aux=aux, sig=sig,
               kick=lde_base.kickoff_restarts(tags, blocks, ball, fps, lde_base.CFG["prekick_s"]))
    _CACHE[k] = out
    return out


def _post(path, ev, first, last, fps):
    seg = slice(first, last + 1)
    sig = ev["sig"]
    raw_seen = sig["ball_seen"] > 0.5
    out = np.nan_to_num(sig["ball_out"], nan=0.0) > 0.5
    out = lde_base.pp.moving_mean(out.astype(float), 3) > 0.5
    gaps = lde_base.long_gaps(raw_seen, fps, lde_base.CFG["gap_s"])
    g = [(s0 - first, e0 - first) for s0, e0 in gaps if s0 >= first and e0 <= last + 1]
    path = lde_base.snap_gap_ends(path, g, fps, lde_base.CFG["snap_win_s"])
    path = lde_base.snap_dead_start_back(path, out[seg], fps, lde_base.CFG["out_lookback_s"],
                                         lde_base.CFG["keep_s"])
    P = dict(lde_base.pp.PARAMS, min_poss_s=CFG["min_poss_s"], min_dead_s=CFG["min_dead_s"])
    path = lde_base.merge_short_v2(path, P, fps)
    pc = ev["aux"]["p_change"]
    path = lde_base.snap_to_peak(path, pc[seg], fps, CFG["refine_window_s"])
    return path


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    ev = _evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    pred = np.full(len(ball), 3, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        if CFG["decode"] >= 0.5:
            path = crf.marginals(ev["U"][seg], ev["P"][seg]).argmax(1).astype(np.int8)
            path = lde_base.merge_short_v2(path, dict(min_poss_s=CFG["min_poss_s"],
                                                      min_dead_s=CFG["min_dead_s"]), fps)
        else:
            path = crf.viterbi(ev["U"][seg], ev["P"][seg])
        if CFG["post"]:
            path = _post(path, ev, first, last, fps)
        rs = ev["kick"].get(first)
        if rs is not None:                     # ball dead before the half's kick-off (match tags)
            path[:rs - first] = D
        pred[seg] = path
    return pred


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    ev = _evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    prob = np.full((len(ball), 3), 1.0 / 3.0)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        prob[seg] = crf.marginals(ev["U"][seg], ev["P"][seg])
    return prob
