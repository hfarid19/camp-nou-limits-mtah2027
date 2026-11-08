"""Plain-numpy inference of the TCN exported by train_tcn.py (no torch at inference).

Weights are a DFL-trained file, so they are always opened through dfl_folds.resolve(): under
bench --dfl that returns the leave-one-game-out fold model that never saw the scored game.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # candidates/ (dfl_folds)
import dfl_folds  # noqa: E402


def load(path):
    z = np.load(dfl_folds.resolve(path), allow_pickle=False)
    W = {k: z[k] for k in z.files}
    W["_path"] = path
    return W


def prep(X, mu, sd, ind):
    miss = np.isnan(X)
    z = np.clip((X - mu) / sd, -6.0, 6.0)
    z[miss] = 0.0
    return np.concatenate([z, miss[:, ind].astype(np.float32)], 1).astype(np.float32)


def _conv(h, w, b, d):
    """Zero-padded dilated conv, torch Conv1d semantics: y[t] = sum_k w[:, :, k] h[t + (k-1) d]."""
    k = w.shape[2]
    y = h @ w[:, :, k // 2].T
    T = len(h)
    for j in range(k):
        off = (j - k // 2) * d
        if off == 0 or abs(off) >= T:
            continue
        if off < 0:
            y[-off:] += h[:off] @ w[:, :, j].T
        else:
            y[:-off] += h[off:] @ w[:, :, j].T
    return y + b


def forward(W, z):
    """z (T, Din) model input -> (T, 4) logits: H, A, D, hand-over."""
    x = z @ W["inp.weight"][:, :, 0].T + W["inp.bias"]
    for i, d in enumerate(W["dils"].tolist()):
        h = np.maximum(x, 0.0)
        h = np.maximum(_conv(h, W["blocks.{}.a.weight".format(i)], W["blocks.{}.a.bias".format(i)],
                             int(d)), 0.0)
        x = x + h @ W["blocks.{}.b.weight".format(i)][:, :, 0].T + W["blocks.{}.b.bias".format(i)]
    return np.maximum(x, 0.0) @ W["out.weight"][:, :, 0].T + W["out.bias"]


def softmax3(lg):
    m = lg.max(1, keepdims=True)
    e = np.exp(lg - m)
    return e / e.sum(1, keepdims=True)
