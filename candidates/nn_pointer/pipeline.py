"""nn_pointer: possessor-pointer network (round 5, R5-3; train_ptr.py, ptr_net.py) + the TCN member's LDE
decoder (nn_pff_tcn/pipeline.decode_probs, unchanged).

predict(...)       -> codes (0 H, 1 A, 2 D, 3 X)
predict_proba(...) -> (N, 3) P(H), P(A), P(D): the pointer's team marginals (sum of the pointer softmax over
                      each team's slots; dead class), team-swap symmetrised, 5 stride-5 phases interleaved
Weights: ptr.npz through pff_folds.resolve (DFL LOGO / PFF out-of-fold under the benches), read through
chunked_weights.source (the file, or its re-joined parts when it is stored split).
Forward pass: BACKEND (below; a member file may set it) or, if set, the environment variable MTAH_PTR_BACKEND:
    numpy    ptr_np.forward, float64 (the reference; the default here)
    torch    ptr_torch.forward: the same maths in PyTorch on the CPU, float64, in a worker process
    torch32  the same in float32 (experiments only)
    auto     torch if torch can be imported, else numpy
Feature building, the stride-5 phases, the team-swap average and the decoder are the same for every backend.
Deterministic.
"""

import hashlib
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path.insert(0, ROOT)
sys.path.insert(0, CAND)
sys.path.insert(0, os.path.join(CAND, "nn_pff_tcn"))
sys.path.insert(0, os.path.join(CAND, "ball_radar_pff"))
sys.path.insert(0, HERE)
import features as fe             # noqa: E402  (nn_pff_tcn/features.py = nn_pff's)
import pff_folds                  # noqa: E402  (nn_pff_tcn/pff_folds.py)
import radar_common as rc         # noqa: E402
import ptr_np                     # noqa: E402
import ptr_torch                  # noqa: E402  (client only: torch itself runs in a worker process)
import ballc                      # noqa: E402
import chunked_weights            # noqa: E402
import importlib.util             # noqa: E402

_spec = importlib.util.spec_from_file_location("nn_pff_tcn_pipeline_for_ptr", os.path.join(CAND, "nn_pff_tcn", "pipeline.py"))
_tcn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tcn)

warnings.filterwarnings("ignore")
STRIDE = 5
WEIGHTS = os.path.join(HERE, "ptr.npz")
BACKEND = "numpy"                                 # forward pass (see the docstring); MTAH_PTR_BACKEND overrides
MIX = float(os.environ.get("PTR_MIX", "-1"))      # experiments only: override cfg mix (direct-head share)
_CACHE = {}


def _key(players, ball, blocks, path):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(players).tobytes())
    h.update(np.ascontiguousarray(ball).tobytes())
    h.update(repr(list(blocks)).encode())
    h.update(path.encode())
    return h.hexdigest()


def _forward():
    """ptr_np.forward or the torch backend's equivalent (same signature and outputs)."""
    backend = ptr_torch.select(BACKEND, os.path.basename(WEIGHTS))
    if backend == "numpy":
        return ptr_np.forward
    dtype = "float64" if backend == "torch" else "float32"
    return lambda W, meta, xs, ms, xf, Xn: ptr_torch.forward(W, meta, xs, ms, xf, Xn, dtype)


def ptr_probs(players, slot_team, ball, sign_half1, blocks, F, names, kinds, W, meta):
    assert list(names) == meta["names"], "feature list differs from the model"
    forward = _forward()
    if MIX >= 0:
        meta = dict(meta, cfg=dict(meta["cfg"], mix=MIX))
    n = len(ball)
    sf = fe.frame_sign(n, sign_half1, blocks)
    perm, signs = fe.swap_index(names, kinds)
    perm, signs = np.asarray(perm), np.asarray(signs, np.float64)
    Xn = np.nan_to_num((F.astype(np.float64) - meta["mean"]) / meta["std"])
    Xm = Xn[:, perm] * signs
    P = np.full((n, 3), 1.0 / 3.0)
    pc = np.zeros(n)
    for _, first, last in blocks:
        used = ~np.isnan(players[first:last + 1, :, 0]).all(0)
        pl, st = players[:, used], np.asarray(slot_team)[used]
        for p in range(STRIDE):
            if first + p > last:
                continue
            fr = np.arange(first + p, last + 1, STRIDE)
            bc = meta["cfg"].get("ballc", 0)
            inp = (ballc.prepare if bc else rc.prepare)(pl, st, ball, sf, 25.0, fr)
            # the training inputs are stored as float16 (prep_ptr.py); round the same way
            xs = inp["xs"].astype(np.float16).astype(np.float64)
            xf = inp["xf"].astype(np.float16).astype(np.float64)
            ms = inp["ms"].astype(np.float64)
            o = forward(W, meta, xs, ms, xf, Xn[fr])
            if bc:
                xs2, xf2 = ballc.swap(xs, xf)
            else:
                xs2 = xs.copy()
                xs2[..., 0:7] *= -1.0
                xf2 = xf.copy()
                xf2[:, 0:2] *= -1.0
            m = forward(W, meta, xs2, ms, xf2, Xm[fr])
            P[fr] = 0.5 * (o["P"] + m["P"][:, [1, 0, 2]])
            pc[fr] = 0.5 * (o["ho"] + m["ho"])
    return P, pc


def evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    path = pff_folds.resolve(WEIGHTS)
    key = _key(players, ball, blocks, path)
    if key not in _CACHE:
        _CACHE.clear()
        W, meta = ptr_np.load(chunked_weights.source(path))
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        P, pc = ptr_probs(players, slot_team, ball, sign_half1, blocks, F, names, kinds, W, meta)
        _CACHE[key] = (P, pc, F, names, meta["change_prior"])
    return _CACHE[key]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P, _, _, _, _ = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return P.astype(np.float64)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P, pc, F, fnames, prior = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return _tcn.decode_probs(P, pc, F, fnames, prior, players, ball, blocks, tags, names, fps)
