"""nn_tsfm: MOMENT-1-small (pretrained time-series foundation model) fine-tuned for per-step
possession classification on nn_pff's dataset (48 PFF World Cup 2022 games + 7 DFL games), decoded
by nn_pff_tcn's LDE Viterbi decoder (code unchanged). See README.md.

predict(...)       -> codes (0 H, 1 A, 2 D, 3 X)
predict_proba(...) -> (N, 3) P(H), P(A), P(D): the model's per-frame softmax (mirror-averaged)

Network: candidates/nn_tsfm/model.py (training) / moment_np.py (numpy inference). Weights:
tsfm.npz through pff_folds.resolve (DFL LOGO / PFF out-of-fold under the benches). The 5 fps
phases are processed in windows of 512 steps (stride 256, central 256 kept) and interleaved back
to 25 fps, both orientations (team-swapped input) averaged.

Backend: numpy (default; the submission path). NN_TSFM_BACKEND=torch runs model.MomentTCN with the
same weights on NN_TSFM_DEVICE (cpu | mps) -- experiments only; verify_parity.py checks the two.
"""

import hashlib
import importlib.util
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path.insert(0, ROOT)
sys.path.insert(0, CAND)
sys.path.insert(0, HERE)
import features as fe             # noqa: E402  (copy of nn_pff_tcn/features.py)
import moment_np                  # noqa: E402

warnings.filterwarnings("ignore")
H, A, D = 0, 1, 2
STRIDE = 5
WIN, HOP = 512, 256
WEIGHTS = os.path.join(HERE, "tsfm.npz")
BACKEND = os.environ.get("NN_TSFM_BACKEND", "numpy")


def _load_tcn_pipeline():
    """nn_pff_tcn/pipeline.py under a private name: its decode_probs / CFG are the decoder."""
    p = os.path.join(CAND, "nn_pff_tcn", "pipeline.py")
    spec = importlib.util.spec_from_file_location("nn_tsfm_decoder_host", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(p))
    spec.loader.exec_module(mod)
    sys.path.remove(os.path.dirname(p))
    return mod


_TCN = _load_tcn_pipeline()
CFG = dict(_TCN.CFG)
_CACHE = {}
_TORCH = {}


def _weights():
    """Loaded on every evidence() call: under run_pff.py / bench --dfl the resolved fold file
    changes from game to game."""
    return moment_np.load(WEIGHTS)


def _torch_forward(W):
    import torch
    sys.path.insert(0, HERE)
    import model as M
    key = W["_path"] + str(W["spec"])
    if key not in _TORCH:
        _TORCH.clear()
        a = W["_arch"]
        m = M.MomentTCN(a["din"], a["t5"], Cm=a["Cm"], dp=a["dp"], Ch=a["Ch"], Cs=a["Cs"], Chh=a["Chh"],
                        dils=tuple(a["dils"]), pe=torch.from_numpy(W["pe"]), max_patches=len(W["pe"]))
        sd = {k: torch.from_numpy(W[k]) for k in m.state_dict() if k in W}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all("embed_tokens" in k for k in missing), (missing, unexpected)
        dev = os.environ.get("NN_TSFM_DEVICE", "cpu")
        m.to(dev).eval()
        _TORCH[key] = (m, dev)
    m, dev = _TORCH[key]

    def f(Z, valid):
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(Z.T)[None]).to(dev)
            v = torch.from_numpy(valid[None]).to(dev)
            return m(x, v)[0].T.cpu().numpy().astype(np.float64)
    return f


def windows(n):
    """[(start, out_lo, out_hi)] windows of WIN steps covering [0, n) with the central HOP kept."""
    if n <= WIN:
        return [(0, 0, n)]
    starts = list(range(0, n - WIN, HOP)) + [n - WIN]
    out = []
    for i, s in enumerate(starts):
        lo = 0 if i == 0 else s + (WIN - HOP) // 2
        hi = n if i == len(starts) - 1 else s + (WIN + HOP) // 2
        out.append((s, lo, hi))
    return out


def run_sequence(fwd, Z):
    """Z (n, Din) prepared 5 fps sequence -> (n, 4) logits, windowed."""
    n = len(Z)
    out = np.zeros((n, 4))
    if n <= WIN:
        pad = np.zeros((WIN, Z.shape[1]), np.float32)
        pad[:n] = Z
        valid = np.zeros(WIN, bool)
        valid[:n] = True
        out[:] = fwd(pad, valid)[:n]
        return out
    for s, lo, hi in windows(n):
        lg = fwd(Z[s:s + WIN], np.ones(WIN, bool))
        out[lo:hi] = lg[lo - s:hi - s]
    return out


def model_probs(F, names, kinds, blocks, W):
    """(N, 3) mirror-averaged softmax and (N,) hand-over probability; 1/3 outside the blocks."""
    assert list(names) == [str(s) for s in W["names"]], "feature list differs from the model"
    fwd = _torch_forward(W) if BACKEND == "torch" else (lambda Z, v: moment_np.forward(W, Z, v).astype(np.float64))
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
            lo = run_sequence(fwd, moment_np.prep(F[sl], W["mu"], W["sd"], W["ind"]))
            lm = run_sequence(fwd, moment_np.prep(Fm[sl], W["mu"], W["sd"], W["ind"]))
            po = moment_np.softmax3(lo[:, :3])
            pm_ = moment_np.softmax3(lm[:, :3])[:, [1, 0, 2]]       # mirror's H is our A
            P[sl] = 0.5 * (po + pm_)
            pc[sl] = 0.5 * (1 / (1 + np.exp(-lo[:, 3])) + 1 / (1 + np.exp(-lm[:, 3])))
    return P, pc


def _key(players, ball, blocks):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(players).tobytes())
    h.update(np.ascontiguousarray(ball).tobytes())
    h.update(repr(list(blocks)).encode())
    return h.hexdigest()


def evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    key = _key(players, ball, blocks)
    if key not in _CACHE:
        _CACHE.clear()
        sign = fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
        W = _weights()
        P, pc = model_probs(F, names, kinds, blocks, W)
        _CACHE[key] = (P, pc, F, names, float(W["change_prior"]))
    return _CACHE[key]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P, _, _, _, _ = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return P.astype(np.float64)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    P, pc, F, fnames, prior = evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return _TCN.decode_probs(P, pc, F, fnames, prior, players, ball, blocks, tags, names, fps, cfg=CFG)
