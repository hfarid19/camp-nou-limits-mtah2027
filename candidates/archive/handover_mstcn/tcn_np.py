"""Plain-numpy forward pass of the ASRF-style MS-TCN exported by train_tcn.py (no torch at
inference: torch and LightGBM bundle clashing libomp builds on macOS)."""
import json

import numpy as np


def _conv(x, W, b, d=1):
    """x (Cin, T), W (Cout, Cin, k), acausal 'same' zero padding, dilation d."""
    k = W.shape[2]
    if k == 1:
        return W[:, :, 0] @ x + b[:, None]
    T = x.shape[1]
    xp = np.zeros((x.shape[0], T + 2 * d), x.dtype)
    xp[:, d:d + T] = x
    out = W[:, :, 0] @ xp[:, 0:T]
    out += W[:, :, 1] @ xp[:, d:d + T]
    out += W[:, :, 2] @ xp[:, 2 * d:2 * d + T]
    out += b[:, None]
    return out


def _stage(w, pre, x, L, nheads):
    h = _conv(x, w[pre + "inp.weight"], w[pre + "inp.bias"])
    for i in range(L):
        q = "{}layers.{}.".format(pre, i)
        t = np.maximum(_conv(h, w[q + "conv.weight"], w[q + "conv.bias"], 2 ** i), 0.0)
        h = h + _conv(t, w[q + "out.weight"], w[q + "out.bias"])
    return [_conv(h, w["{}heads.{}.weight".format(pre, j)], w["{}heads.{}.bias".format(pre, j)])
            for j in range(nheads)]


def _softmax(z):
    z = z - z.max(0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(0, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Single(object):
    def __init__(self, z, pre=""):
        self.cfg = json.loads(str(z[pre + "cfg"]))
        self.mean = z[pre + "mean"].astype(np.float64)
        self.std = z[pre + "std"].astype(np.float64)
        self.names = [str(s) for s in z[pre + "names"]]
        n = len(pre) + 2
        self.w = {k[n:]: z[k].astype(np.float64) for k in z.files if k.startswith(pre + "w/")}

    def normalise(self, X):
        return np.nan_to_num((X - self.mean) / self.std)

    def forward(self, Xn):
        """Xn (T, K) normalised -> class probs (T, 3) of the last class stage and boundary
        probs (T, 2) (H<->A, alive<->dead) of the last boundary stage."""
        c = self.cfg
        x = np.ascontiguousarray(Xn.T)
        cl, bl = _stage(self.w, "backbone.", x, c["L"], 2)
        for s in range(c["S_c"]):
            cl = _stage(self.w, "cref.{}.".format(s), _softmax(cl), c["Lr"], 1)[0]
        for s in range(c["S_b"]):
            bl = _stage(self.w, "bref.{}.".format(s), _sigmoid(bl), c["Lr"], 1)[0]
        return _softmax(cl).T, _sigmoid(bl).T


class Model(object):
    """One exported model (fold_tcn_<gi>.npz) or a pack of several (model_tcn.npz: the 7
    leave-one-game-out models, keys m<i>/...), whose outputs are averaged."""

    def __init__(self, path):
        z = np.load(path)
        if "n_members" in z.files:
            self.members = [Single(z, "m{}/".format(i)) for i in range(int(z["n_members"]))]
        else:
            self.members = [Single(z)]
        self.names = self.members[0].names

    def predict(self, X, perm, signs):
        """X (T, K) raw features -> team-swap symmetrised class probs (T, 3), boundary (T, 2)."""
        pcs, pbs = [], []
        for m in self.members:
            c1, b1 = m.forward(m.normalise(X))
            c2, b2 = m.forward(m.normalise(X[:, perm] * signs))
            pcs.append(0.5 * (c1 + c2[:, [1, 0, 2]]))
            pbs.append(0.5 * (b1 + b2))
        return np.mean(pcs, 0), np.mean(pbs, 0)
