"""nn_seg (round 5, R5-5): nn_pff's MS-TCN member + its segment head (train_seg.py), decoded into a partition.

The MS-TCN runs exactly as nn_pff/pipeline.py (class posteriors pc, boundary posteriors pb). The segment head
predicts, at every 5 fps step, the extent [t - s, t + e] of the spell containing t. Every frame casts a
start vote at t - s and an end vote at t + e, each weighted 1 / (s + e) (so one spell gives ~1 vote at each
end); votes are smoothed with a Gaussian (sigma VOTE_SIGMA frames, peak 1) and B = min(1, votes / 2) is a
boundary probability (a perfect boundary collects an end and a start vote). The ASRF decoder of
nn_pff/pipeline.py (CFG unchanged) then cuts at the peaks of 0.5 * (max(pb) + B) (pre-specified mix, MODE).
Team swap: segment extents are swap-invariant; the swapped run's log-extents are averaged in.
Numpy only, deterministic.
"""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("nn_pff_pipeline_for_seg", os.path.join(CAND, "nn_pff", "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
fe, tcn_np, pff_folds = _base.fe, _base.tcn_np, _base.pff_folds

MODEL_PATH = os.path.join(HERE, "seg.npz")
STRIDE = 5
VOTE_SIGMA = 5.0
MODE = os.environ.get("SEG_MODE", "mean")     # mean (pre-specified) | replace | off (experiments only)
_CACHE = {}


def _hidden(m, x):
    """backbone's last hidden layer (C, T) of tcn_np.Single m on normalised x (T, K)."""
    w = m.w
    h = tcn_np._conv(np.ascontiguousarray(x.T), w["backbone.inp.weight"], w["backbone.inp.bias"])
    for i in range(m.cfg["L"]):
        q = "backbone.layers.{}.".format(i)
        t = np.maximum(tcn_np._conv(h, w[q + "conv.weight"], w[q + "conv.bias"], 2 ** i), 0.0)
        h = h + tcn_np._conv(t, w[q + "out.weight"], w[q + "out.bias"])
    return h


def _seg(S, h):
    a = np.maximum(S["c1.weight"][:, :, 0] @ h + S["c1.bias"][:, None], 0.0)
    return np.clip(S["c2.weight"][:, :, 0] @ a + S["c2.bias"][:, None], -3.0, 7.0)     # (2, T) log steps


def outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    path = pff_folds.resolve(MODEL_PATH)
    key = _base._key(players, ball, blocks, path)
    if key in _CACHE:
        return _CACHE[key]
    z = np.load(path)
    S = {k[4:]: z[k].astype(np.float64) for k in z.files if k.startswith("seg/")}
    model = tcn_np.Model(path)
    m = model.members[0]
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    assert names == model.names
    perm, signs = fe.swap_index(names, kinds)
    n = len(ball)
    pc = np.full((n, 3), 1.0 / 3.0)
    pb = np.zeros((n, 2))
    V = np.zeros(n)
    rad = int(4 * VOTE_SIGMA)
    ker = np.exp(-0.5 * (np.arange(-rad, rad + 1) / VOTE_SIGMA) ** 2)
    for _, first, last in blocks:
        votes = np.zeros(last - first + 1)
        for r in range(STRIDE):
            fr = np.arange(first + r, last + 1, STRIDE)
            if len(fr) == 0:
                continue
            X = F[fr].astype(np.float64)
            pc[fr], pb[fr] = model.predict(X, perm, signs)
            o1 = _seg(S, _hidden(m, m.normalise(X)))
            o2 = _seg(S, _hidden(m, m.normalise(X[:, perm] * signs)))
            d = np.exp(0.5 * (o1 + o2)) * STRIDE                     # frames to the spell's start / end
            wgt = 1.0 / np.maximum(d[0] + d[1], 1.0)
            for pos in (fr - d[0], fr + d[1]):
                k = np.rint(pos).astype(int) - first
                ok = (k >= 0) & (k < len(votes))
                np.add.at(votes, k[ok], wgt[ok])
        V[first:last + 1] = np.convolve(votes, ker, mode="same")
    B = np.minimum(1.0, V / 2.0)
    _CACHE.clear()
    _CACHE[key] = (pc, pb, B, F, names)
    return _CACHE[key]


def boundary(pb, B):
    if MODE == "off":
        return pb
    b = B if MODE == "replace" else 0.5 * (pb.max(1) + B)
    return np.stack([b, b], 1)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pc, pb, B, F, fnames = outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return _base.decode(pc, boundary(pb, B), blocks, len(ball), fps, aux=_base.post_aux(F, fnames))


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pc = outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps)[0]
    return pc.astype(np.float64)


def boundary_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pc, pb, B, _, _ = outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return np.concatenate([pb, B[:, None]], 1).astype(np.float64)
