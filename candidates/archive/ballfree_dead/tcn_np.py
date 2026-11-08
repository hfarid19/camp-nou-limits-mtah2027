"""numpy inference of train_tcn.TCN (eval mode: dropout off), exactly its conv arithmetic."""
import numpy as np


def _conv(x, w, b, d):
    """x (T, Cin), w (Cout, Cin, k) torch layout, dilation d, zero padding d*(k//2)."""
    k = w.shape[2]
    T = len(x)
    out = np.zeros((T, w.shape[0]), np.float64) + b
    pad = d * (k // 2)
    xp = np.concatenate([np.zeros((pad, x.shape[1])), x, np.zeros((pad, x.shape[1]))])
    for j in range(k):
        out += xp[j * d:j * d + T] @ w[:, :, j].T.astype(np.float64)
    return out


def logits(z, X):
    """z: loaded npz arrays, X (T, K) base features at 5 Hz -> (T,) dead logits."""
    x = (X - z["med"]) / z["scale"]
    x = np.clip(np.nan_to_num(x, nan=0.0), -5, 5).astype(np.float64)
    h = _conv(x, z["inp.weight"], z["inp.bias"], 1)
    for i, d in enumerate(z["dil"]):
        c = _conv(np.maximum(h, 0), z["convs.{}.weight".format(i)], z["convs.{}.bias".format(i)], int(d))
        h = h + _conv(np.maximum(c, 0), z["mix.{}.weight".format(i)], z["mix.{}.bias".format(i)], 1)
    return _conv(np.maximum(h, 0), z["head.weight"], z["head.bias"], 1)[:, 0]
