"""Plain-numpy forward of the possessor-pointer network (ptr_net.PointerNet), float64, deterministic.

load(path) -> dict of float64 weights + meta.  forward(W, xs, ms, xf, Xn) -> dict(P (T, 3) P(H), P(A), P(D)
(the pointer's team marginals, mixed with the direct head if cfg mix > 0), ho (T,) hand-over prob,
g (T, S+1) pointer softmax). One contiguous sequence (one stride-5 phase of one half).
"""

import json

import numpy as np

TCHUNK = 2000


def load(path):
    z = np.load(path)
    W = {k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("w/")}
    meta = dict(cfg=json.loads(str(z["cfg"])), mean=z["mean"].astype(np.float64), std=z["std"].astype(np.float64),
                names=[str(s) for s in z["names"]], kinds=[str(s) for s in z["kinds"]],
                change_prior=float(z["change_prior"]))
    return W, meta


def _lin(W, k, x):
    return x @ W[k + ".weight"].T + W[k + ".bias"]


def _ln(W, k, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * W[k + ".weight"] + W[k + ".bias"]


def _conv(x, Wt, b, d=1):
    """x (..., Cin, T), Wt (Cout, Cin, k in {1, 3}), 'same' zero padding, dilation d -> (..., Cout, T)."""
    k = Wt.shape[2]
    if k == 1:
        return np.einsum("oi,...it->...ot", Wt[:, :, 0], x) + b[:, None]
    T = x.shape[-1]
    pad = np.zeros(x.shape[:-1] + (T + 2 * d,), x.dtype)
    pad[..., d:d + T] = x
    out = np.einsum("oi,...it->...ot", Wt[:, :, 0], pad[..., 0:T])
    out += np.einsum("oi,...it->...ot", Wt[:, :, 1], pad[..., d:d + T])
    out += np.einsum("oi,...it->...ot", Wt[:, :, 2], pad[..., 2 * d:2 * d + T])
    return out + b[:, None]


def _drl(W, pre, x, d):
    t = np.maximum(_conv(x, W[pre + "conv.weight"], W[pre + "conv.bias"], d), 0.0)
    return x + _conv(t, W[pre + "out.weight"], W[pre + "out.bias"])


def _sab(W, pre, e, ms, heads):
    T, S, D = e.shape
    dh = D // heads
    q = _lin(W, pre + "q", e).reshape(T, S, heads, dh).transpose(0, 2, 1, 3)
    k = _lin(W, pre + "k", e).reshape(T, S, heads, dh).transpose(0, 2, 1, 3)
    v = _lin(W, pre + "v", e).reshape(T, S, heads, dh).transpose(0, 2, 1, 3)
    sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    sc = np.where(ms[:, None, None, :] > 0, sc, -1e4)
    sc = sc - sc.max(-1, keepdims=True)
    a = np.exp(sc)
    a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(T, S, D)
    e = _ln(W, pre + "ln1", e + _lin(W, pre + "o", o))
    e = _ln(W, pre + "ln2", e + _lin(W, pre + "ff2", np.maximum(_lin(W, pre + "ff1", e), 0.0)))
    return e * ms[..., None]


def _pool(e, ms, team):
    outs = []
    for side in (team > 0, team < 0):
        mk = ms * side
        cnt = mk.sum(1, keepdims=True)
        mean = (e * mk[..., None]).sum(1) / np.maximum(cnt, 1.0)
        mx = np.where(mk[..., None] > 0, e, -1e4).max(1)
        mx = np.where(cnt > 0, mx, 0.0)
        outs += [mean, mx]
    return np.concatenate(outs, -1)


def _softmax(z, ax=-1):
    z = z - z.max(ax, keepdims=True)
    e = np.exp(z)
    return e / e.sum(ax, keepdims=True)


def forward(W, meta, xs, ms, xf, Xn):
    cfg = meta["cfg"]
    xs, ms, xf, Xn = (np.asarray(a, np.float64) for a in (xs, ms, xf, Xn))
    T, S, _ = xs.shape
    if cfg.get("noX", 0):
        Xn = np.zeros_like(Xn)
    team = xs[..., 4]
    es, pls = [], []
    for t0 in range(0, T, TCHUNK):
        sl = slice(t0, t0 + TCHUNK)
        e = np.maximum(_lin(W, "phi2", np.maximum(_lin(W, "phi1", xs[sl]), 0.0)), 0.0)
        for i in range(cfg["nsab"]):
            e = _sab(W, "sabs.{}.".format(i), e, ms[sl], cfg["heads"])
        es.append(e)
        pls.append(_pool(e, ms[sl], team[sl]))
    e, pl = np.concatenate(es), np.concatenate(pls)
    h = np.concatenate([pl, xf, Xn], -1).T                                   # (., T)
    c = _conv(h, W["fin.weight"], W["fin.bias"])
    for i in range(cfg["L"]):
        c = _drl(W, "tcn.{}.".format(i), c, 2 ** i)
    ct = c.T                                                                  # (T, C)
    u = _lin(W, "sin", np.concatenate([e, np.broadcast_to(ct[:, None, :], (T, S, ct.shape[1]))], -1))
    u = np.maximum(u, 0.0) * ms[..., None]
    u = np.ascontiguousarray(u.transpose(1, 2, 0))                            # (S, Dp, T)
    for i in range(cfg["Lp"]):
        u = _drl(W, "stcn.{}.".format(i), u, 2 ** i)
    u = u.transpose(2, 0, 1)                                                  # (T, S, Dp)
    lp = _lin(W, "plog", np.concatenate([u, xs], -1))[..., 0]
    lp = np.where(ms > 0, lp, -1e4)
    umax = np.where(ms[..., None] > 0, u, -1e4).max(1)
    umax = np.where(ms.sum(1, keepdims=True) > 0, umax, 0.0)
    ld = _lin(W, "dlog", np.concatenate([ct, umax], -1))
    g = _softmax(np.concatenate([lp, ld], -1))
    gs = g[:, :S] * (ms > 0)
    P = np.stack([(gs * (team > 0)).sum(1), (gs * (team < 0)).sum(1), g[:, S]], 1)
    if cfg.get("mix", 0.0) > 0:
        pd = _softmax(_conv(c, W["dir.weight"], W["dir.bias"]).T)
        P = (1 - cfg["mix"]) * P + cfg["mix"] * pd
    ho = 1.0 / (1.0 + np.exp(-_conv(c, W["ho.weight"], W["ho.bias"])[0]))
    return dict(P=P, ho=ho, g=g)
