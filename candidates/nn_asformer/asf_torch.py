"""Torch inference backend for nn_asformer exports, for SCREENING only (round 4, experiment 4).

The "global" (full-attention) arch costs ~50 min per match in numpy (asf_np), far over the
submission's inference budget, so it cannot ship; to screen it on PFF out-of-fold games quickly,
pipeline_torch.py swaps asf_np.Model for this class: same npz, same predict() API (team-swap mirror
average), model.Net in eval mode, float32, with the global attention computed in query chunks
(no (h, T, T) mask: memory O(chunk x T)). Device: $ASF_TORCH_DEVICE (default cpu).
Parity with asf_np: verify with `python asf_torch.py <npz>` (max |diff| on class probs).
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402

CHUNK = 2048


def _attn_forward(self, x, f=None):
    src = f if self.cross else x
    q, k, v = self._split(self.q(x)), self._split(self.k(src)), self._split(self.v(src))
    T = x.shape[1]
    if not self.glob:
        out = M.band_attention(q, k, v, self.w, self.bias, T)
    else:
        out = torch.empty_like(q)
        pos = torch.arange(T, device=x.device)
        for a in range(0, T, CHUNK):
            b = min(T, a + CHUNK)
            dist = (pos[a:b, None] - pos[None, :]).abs().float()
            mask = -self.slopes[:, None, None] * dist[None]
            out[:, :, a:b] = F.scaled_dot_product_attention(q[:, :, a:b], k, v, attn_mask=mask[None])
    return self.o(out.transpose(1, 2).reshape(x.shape[0], T, -1))


M.Attention.forward = _attn_forward


class _Net(object):
    def __init__(self, z, dev):
        cfg = dict(M.CFG, **json.loads(str(z["cfg"])))
        self.mean = z["mean"].astype(np.float64)
        self.std = z["std"].astype(np.float64)
        self.names = [str(s) for s in z["names"]]
        self.net = M.Net(len(self.names), cfg)
        self.net.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("w/")})
        self.net.eval().to(dev)
        self.dev = dev

    def forward(self, Xn):
        with torch.no_grad():
            cs, bs = self.net(torch.from_numpy(Xn.astype(np.float32))[None].to(self.dev))
            return (torch.softmax(cs[-1][0], -1).cpu().double().numpy(),
                    torch.sigmoid(bs[-1][0]).cpu().double().numpy())

    def normalise(self, X):
        return np.nan_to_num((X - self.mean) / self.std)


class Model(object):
    def __init__(self, path):
        z = np.load(path)
        assert "n_members" not in z.files
        self.members = [_Net(z, torch.device(os.environ.get("ASF_TORCH_DEVICE", "cpu")))]
        self.names = self.members[0].names

    def predict(self, X, perm, signs):
        m = self.members[0]
        c1, b1 = m.forward(m.normalise(X))
        c2, b2 = m.forward(m.normalise(X[:, perm] * signs))
        return 0.5 * (c1 + c2[:, [1, 0, 2]]), 0.5 * (b1 + b2)


if __name__ == "__main__":
    import time
    import asf_np
    p = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    rng = np.random.default_rng(0)
    a, b = asf_np.Model(p), Model(p)
    K = len(a.names)
    X = rng.normal(size=(T, K))
    perm, signs = np.arange(K), np.ones(K)
    t = time.time()
    pa = a.predict(X, perm, signs)
    t1 = time.time()
    pb = b.predict(X, perm, signs)
    t2 = time.time()
    print("T {} max |diff| class {:.2e} boundary {:.2e}; numpy {:.1f}s torch {:.1f}s".format(
        T, np.abs(pa[0] - pb[0]).max(), np.abs(pa[1] - pb[1]).max(), t1 - t, t2 - t1))
