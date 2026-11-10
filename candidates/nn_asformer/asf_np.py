"""Plain-numpy forward pass of the networks exported by train.py (model.py): no torch at inference.
Deterministic (float64; attention = matmul + softmax, convolutions as in handover_mstcn's tcn_np.py,
the SSM as the same chunked SSD algorithm the training code uses). Parity with torch is checked by
verify_parity.py (max |numpy - torch CPU| on probabilities, full games)."""

import json
import math

import numpy as np


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z):
    return np.logaddexp(0.0, z)


def _ln(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _lin(x, W, b):
    return x @ W.T + b


def _dconv(x, W, b, d):
    """x (T, C), W (Cout, Cin, 3), 'same' zero padding, dilation d -> (T, Cout)."""
    T = x.shape[0]
    xp = np.zeros((T + 2 * d, x.shape[1]), x.dtype)
    xp[d:d + T] = x
    out = xp[0:T] @ W[:, :, 0].T
    out += xp[d:d + T] @ W[:, :, 1].T
    out += xp[2 * d:2 * d + T] @ W[:, :, 2].T
    return out + b


def band_attention(q, k, v, w, bias):
    """q, k, v (h, T, dh); bias (h, 2w+1). Exact band |t - s| <= w, chunked over queries."""
    h, T, dh = q.shape
    c = max(w, 16)
    Tp = ((T + c - 1) // c) * c
    pad = Tp - T
    qp = np.concatenate([q, np.zeros((h, pad, dh))], 1) if pad else q
    kp = np.concatenate([np.zeros((h, w, dh)), k, np.zeros((h, w + pad, dh))], 1)
    vp = np.concatenate([np.zeros((h, w, dh)), v, np.zeros((h, w + pad, dh))], 1)
    nc = Tp // c
    i = np.arange(c)[:, None]
    j = np.arange(c + 2 * w)[None, :]
    rel = j - w - i
    bw = bias[:, np.clip(rel, -w, w) + w]                  # (h, c, c+2w)
    out = np.empty((h, Tp, dh))
    sc = 1.0 / math.sqrt(dh)
    for n in range(nc):
        qc = qp[:, n * c:(n + 1) * c]
        kw = kp[:, n * c:n * c + c + 2 * w]
        vw = vp[:, n * c:n * c + c + 2 * w]
        s = qc @ kw.transpose(0, 2, 1) * sc + bw
        kabs = n * c - w + j
        valid = (np.abs(rel) <= w) & (kabs >= 0) & (kabs < T)
        s = np.where(valid[None], s, -1e9)
        out[:, n * c:(n + 1) * c] = _softmax(s) @ vw
    return out[:, :T]


def global_attention(q, k, v, slopes, chunk=512):
    """ALiBi full attention, chunked over queries. q, k, v (h, T, dh); slopes (h,)."""
    h, T, dh = q.shape
    out = np.empty_like(q)
    sc = 1.0 / math.sqrt(dh)
    pos = np.arange(T)
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        dist = np.abs(pos[a:b, None] - pos[None, :]).astype(np.float64)
        s = q[:, a:b] @ k.transpose(0, 2, 1) * sc - slopes[:, None, None] * dist[None]
        out[:, a:b] = _softmax(s) @ v
    return out


def segsum_exp(a):
    L = a.shape[-1]
    cs = np.cumsum(a, -1)
    diff = cs[..., :, None] - cs[..., None, :]
    tri = np.tril(np.ones((L, L), bool))
    return np.exp(np.where(tri, diff, -1e9)) * tri


def ssd_scan(x, dt, A, Bm, Cm, L):
    """x (T, H, P), dt (T, H), A (H,), Bm / Cm (T, H, N) -> (T, H, P); same algorithm as model.ssd_scan."""
    T, H, P = x.shape
    N = Bm.shape[-1]
    Tp = ((T + L - 1) // L) * L
    if Tp != T:
        pd = Tp - T
        x = np.concatenate([x, np.zeros((pd, H, P))], 0)
        dt = np.concatenate([dt, np.zeros((pd, H))], 0)
        Bm = np.concatenate([Bm, np.zeros((pd, H, N))], 0)
        Cm = np.concatenate([Cm, np.zeros((pd, H, N))], 0)
    nc = Tp // L
    x = x.reshape(nc, L, H, P)
    dt = dt.reshape(nc, L, H)
    Bm = Bm.reshape(nc, L, H, N)
    Cm = Cm.reshape(nc, L, H, N)
    a = (dt * A).transpose(2, 0, 1)                         # (H, nc, L)
    M = segsum_exp(a)                                       # (H, nc, L, L)
    cs = np.cumsum(a, -1)
    xd = x * dt[..., None]
    CB = np.einsum("clhn,cshn->hcls", Cm, Bm) * M
    y = np.einsum("hcls,cshp->clhp", CB, xd)
    dec_end = np.exp(cs[..., -1:] - cs)
    st = np.einsum("hcl,clhn,clhp->chnp", dec_end, Bm, xd)
    tot = np.exp(cs[..., -1])                               # (H, nc)
    S = np.zeros((H, N, P))
    Sprev = np.empty((nc, H, N, P))
    for c in range(nc):
        Sprev[c] = S
        S = tot[:, c, None, None] * S + st[c]
    y = y + np.einsum("hcl,clhn,chnp->clhp", np.exp(cs), Cm, Sprev)
    return y.reshape(Tp, H, P)[:T]


class Net(object):
    def __init__(self, z, pre=""):
        self.cfg = json.loads(str(z[pre + "cfg"]))
        self.mean = z[pre + "mean"].astype(np.float64)
        self.std = z[pre + "std"].astype(np.float64)
        self.names = [str(s) for s in z[pre + "names"]]
        n = len(pre) + 2
        self.w = {k[n:]: z[k].astype(np.float64) for k in z.files if k.startswith(pre + "w/")}

    def normalise(self, X):
        return np.nan_to_num((X - self.mean) / self.std)

    def _attn(self, p, x, f, w, cross):
        W = self.w
        c = self.cfg
        h = c["heads"]
        src = f if cross else x
        T = x.shape[0]
        q = _lin(x, W[p + "q.weight"], W[p + "q.bias"]).reshape(T, h, -1).transpose(1, 0, 2)
        k = _lin(src, W[p + "k.weight"], W[p + "k.bias"]).reshape(T, h, -1).transpose(1, 0, 2)
        v = _lin(src, W[p + "v.weight"], W[p + "v.bias"]).reshape(T, h, -1).transpose(1, 0, 2)
        if c["arch"] == "global":
            out = global_attention(q, k, v, W[p + "slopes"])
        else:
            out = band_attention(q, k, v, w, W[p + "bias"])
        return _lin(out.transpose(1, 0, 2).reshape(T, -1), W[p + "o.weight"], W[p + "o.bias"])

    def _ssm(self, p, x):
        W = self.w
        c = self.cfg
        H, N = c["heads"], c["ssm_N"]
        di = c["d_cur"] * c["ssm_expand"]
        P = di // H
        T = x.shape[0]
        z = _lin(x, W[p + "inp.weight"], W[p + "inp.bias"])
        xs, g, rest = z[:, :di], z[:, di:2 * di], z[:, 2 * di:]
        xs = xs.reshape(T, H, P)
        y = 0
        m = 2 * N + H
        for kk in range(2):
            r = rest[:, kk * m:(kk + 1) * m]
            Bm, Cm, dt = r[:, :N], r[:, N:2 * N], r[:, 2 * N:]
            Bm = np.broadcast_to(Bm[:, None, :], (T, H, N))
            Cm = np.broadcast_to(Cm[:, None, :], (T, H, N))
            dt = _softplus(dt + W[p + "dt_bias"][kk])
            A = -np.exp(W[p + "A_log"][kk])
            if kk == 1:
                xk, dtk, Bk, Ck = xs[::-1], dt[::-1], Bm[::-1], Cm[::-1]
            else:
                xk, dtk, Bk, Ck = xs, dt, Bm, Cm
            yk = ssd_scan(xk, dtk, A, np.ascontiguousarray(Bk), np.ascontiguousarray(Ck), c["ssm_chunk"]) \
                + xk * W[p + "D"][kk][None, :, None]
            if kk == 1:
                yk = yk[::-1]
            y = y + yk
        y = y.reshape(T, di) * (g * _sigmoid(g))
        return _lin(y, W[p + "out.weight"], W[p + "out.bias"])

    def _block(self, p, i, x, f, cross):
        W = self.w
        c = self.cfg
        eps = c["ln_eps"]
        h = _ln(x, W[p + "ln1.weight"], W[p + "ln1.bias"], eps)
        dil = 2 ** (i % 12)
        h = _gelu(_dconv(h, W[p + "ff.conv.weight"], W[p + "ff.conv.bias"], dil))
        x = x + _lin(h, W[p + "ff.out.weight"], W[p + "ff.out.bias"])
        h = _ln(x, W[p + "ln2.weight"], W[p + "ln2.bias"], eps)
        if c["arch"] == "temba":
            m = self._ssm(p + "mix.", h)
        else:
            m = self._attn(p + "mix.", h, f, min(dil, c["w_max"]), cross)
        return x + W[p + "alpha"][0] * m

    def _stage(self, p, x, L, d, f=None, cross=False):
        W = self.w
        self.cfg["d_cur"] = d
        h = _lin(x, W[p + "inp.weight"], W[p + "inp.bias"])
        for i in range(L):
            h = self._block("{}blocks.{}.".format(p, i), i, h, f, cross)
        h = _ln(h, W[p + "ln.weight"], W[p + "ln.bias"], self.cfg["ln_eps"])
        return h, _lin(h, W[p + "head_c.weight"], W[p + "head_c.bias"]), _lin(h, W[p + "head_b.weight"], W[p + "head_b.bias"])

    def forward(self, Xn):
        """Xn (T, K) normalised -> class probs (T, 3), boundary probs (T, 2) of the last stage."""
        c = self.cfg
        f, cl, bl = self._stage("enc.", Xn, c["L"], c["d"])
        if "proj.weight" in self.w:
            f = _lin(f, self.w["proj.weight"], self.w["proj.bias"])
        dd, Ld = c["d_dec"] or c["d"], c["L_dec"] or c["L"]
        cross = c["arch"] in ("asformer", "global")
        for s in range(c["S"]):
            inp = np.concatenate([_softmax(cl), _sigmoid(bl)], -1)
            _, cl, bl = self._stage("dec.{}.".format(s), inp, Ld, dd, f, cross)
        return _softmax(cl), _sigmoid(bl)


class Model(object):
    """One exported model, or a pack (keys m<i>/..., n_members) whose outputs are averaged."""

    def __init__(self, path):
        z = np.load(path)
        if "n_members" in z.files:
            self.members = [Net(z, "m{}/".format(i)) for i in range(int(z["n_members"]))]
        else:
            self.members = [Net(z)]
        self.names = self.members[0].names

    def predict(self, X, perm, signs):
        """X (T, K) raw features -> team-swap symmetrised class probs (T, 3), boundary probs (T, 2)."""
        pcs, pbs = [], []
        for m in self.members:
            c1, b1 = m.forward(m.normalise(X))
            c2, b2 = m.forward(m.normalise(X[:, perm] * signs))
            pcs.append(0.5 * (c1 + c2[:, [1, 0, 2]]))
            pbs.append(0.5 * (b1 + b2))
        return np.mean(pcs, 0), np.mean(pbs, 0)
