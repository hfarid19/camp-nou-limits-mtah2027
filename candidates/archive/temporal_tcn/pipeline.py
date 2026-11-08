"""temporal_tcn: dilated temporal convolution network (TCN) over identity-free per-frame features,
trained on degraded open DFL tracking, decoded by LDE's Viterbi decoder. See README.md.

predict(...)       -> codes (0 H, 1 A, 2 D, 3 X)
predict_proba(...) -> (N, 3) P(H), P(A), P(D): the TCN's own per-frame softmax (mirror-averaged)
"""

import hashlib
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import possession_pipeline as pp  # noqa: E402
import features as fe             # noqa: E402
import tcn_np                     # noqa: E402
import lde_decoder as dec         # noqa: E402

warnings.filterwarnings("ignore")
H, A, D = 0, 1, 2
STRIDE = 5                        # the TCN runs at 5 fps; the 5 phases are interleaved back

WEIGHTS = os.path.join(HERE, "tcn_weights.npz")

CFG = dict(
    mode=0.0,            # 0: LDE decoder on the TCN evidence; 1: raw argmax + mode filter
    raw_win_s=1.0,       # (mode 1) centred majority window
    raw_merge=1.0,       # (mode 1) merge_short_v2 after the mode filter
    # LDE decoder settings (lde_decoder.CFG defaults unless overridden here)
    switch_cost=0.75,
    dead_cost=5.0,
    w_gap=3.0,
    contrast_beta=1.25,
    refine_window_s=0.5,
    min_poss_s=1.0,
    snap_mode=1.0,
    temp_team=1.0,       # team log-odds multiplier (calibration of the TCN softmax)
    temp_dead=1.0,       # dead log-odds multiplier
)
_OVR = sorted(k for k in CFG if "TCN_" + k.upper() in os.environ)
if os.environ.get("TCN_SWEEP") == "1":       # tuning only; ignored unless explicitly enabled
    for _k in _OVR:
        CFG[_k] = float(os.environ["TCN_" + _k.upper()])
    if _OVR:
        print("[tcn] WARNING: CFG overridden (TCN_SWEEP=1): " +
              ", ".join("{}={}".format(k, CFG[k]) for k in _OVR), file=sys.stderr)
elif _OVR:
    print("[tcn] WARNING: TCN_* variables set but ignored (TCN_SWEEP is not 1)", file=sys.stderr)

_W = {}
_CACHE = {}


def _weights():
    if "w" not in _W:
        _W["w"] = tcn_np.load(WEIGHTS)
    return _W["w"]


def _key(players, ball, blocks):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(players).tobytes())
    h.update(np.ascontiguousarray(ball).tobytes())
    h.update(repr(list(blocks)).encode())
    return h.hexdigest()


def tcn_probs(F, names, kinds, blocks, W):
    """(N, 3) mirror-averaged softmax and (N,) hand-over probability; 1/3 outside the blocks."""
    assert list(names) == [str(s) for s in W["names"]], "feature list differs from the model"
    n = len(F)
    perm, signs = fe.swap_index(names, kinds)
    Fm = F[:, perm] * np.asarray(signs, np.float32)
    P = np.full((n, 3), 1.0 / 3.0)
    pc = np.zeros(n)
    for _, first, last in blocks:
        for p in range(STRIDE):
            sl = slice(first + p, last + 1, STRIDE)
            if first + p > last:
                continue
            lo = tcn_np.forward(W, tcn_np.prep(F[sl], W["mu"], W["sd"], W["ind"])).astype(np.float64)
            lm = tcn_np.forward(W, tcn_np.prep(Fm[sl], W["mu"], W["sd"], W["ind"])).astype(np.float64)
            po = tcn_np.softmax3(lo[:, :3])
            pm_ = tcn_np.softmax3(lm[:, :3])[:, [1, 0, 2]]       # mirror's H is our A
            P[sl] = 0.5 * (po + pm_)
            pc[sl] = 0.5 * (1 / (1 + np.exp(-lo[:, 3])) + 1 / (1 + np.exp(-lm[:, 3])))
    return P, pc


def evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    key = _key(players, ball, blocks)
    if key not in _CACHE:
        _CACHE.clear()
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        W = _weights()
        P, pc = tcn_probs(F, names, kinds, blocks, W)
        _CACHE[key] = (P, pc, F, names, float(W["change_prior"]))
    return _CACHE[key]


def mode_filter(path, win):
    """Centred majority vote over win frames (ties -> current label)."""
    if win <= 1:
        return path.copy()
    hw = win // 2
    oh = np.stack([(path == c).astype(np.float64) for c in range(3)], 1)
    cs = np.concatenate([np.zeros((1, 3)), np.cumsum(oh, 0)])
    i = np.arange(len(path))
    lo, hi = np.clip(i - hw, 0, len(path)), np.clip(i + hw + 1, 0, len(path))
    cnt = cs[hi] - cs[lo] + 1e-9 * oh
    return np.argmax(cnt, 1).astype(np.int8)


def decode_probs(P, pc, F, names, change_prior, players, ball, blocks, tags, names_teams, fps,
                 cfg=None):
    c = dict(CFG, **(cfg or {}))
    n = len(P)
    if c["mode"] >= 0.5:
        pred = np.full(n, 3, np.int8)
        Pm = dict(pp.PARAMS, min_poss_s=c["min_poss_s"], min_dead_s=1.0)
        for _, first, last in blocks:
            seg = slice(first, last + 1)
            path = np.argmax(P[seg], 1).astype(np.int8)
            path = mode_filter(path, int(round(c["raw_win_s"] * fps)))
            if c["raw_merge"] >= 0.5:
                path = dec.merge_short_v2(path, Pm, fps)
            pred[seg] = path
        return pred
    pa = np.clip(P[:, 0] + P[:, 1], 1e-6, 1)
    p_home = np.clip(P[:, 0] / pa, 1e-6, 1 - 1e-6)
    p_dead = np.clip(P[:, 2], 1e-6, 1 - 1e-6)
    if c["temp_team"] != 1.0:
        p_home = 1 / (1 + np.exp(-c["temp_team"] * np.log(p_home / (1 - p_home))))
    if c["temp_dead"] != 1.0:
        p_dead = 1 / (1 + np.exp(-c["temp_dead"] * np.log(p_dead / (1 - p_dead))))
    aux = dict(F=F, names=names, p_change=pc, change_prior=change_prior)
    aux["long_gap"] = dec.long_gap_gated(players, ball, fps, dec.CFG["gap_s"],
                                         dec.CFG["gap_speed_ms"])
    aux["kick_restarts"] = dec.kickoff_restarts(tags, blocks, ball, fps, dec.CFG["prekick_s"])
    dcfg = {k: c[k] for k in ("switch_cost", "dead_cost", "w_gap", "contrast_beta",
                              "refine_window_s", "min_poss_s", "snap_mode")}
    return dec.decode(p_dead, p_home, aux, blocks, n, fps, tags, names_teams, cfg=dcfg)


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P, _, _, _, _ = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return P.astype(np.float64)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P, pc, F, fnames, prior = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return decode_probs(P, pc, F, fnames, prior, players, ball, blocks, tags, names, fps)
