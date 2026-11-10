"""Plain-numpy inference of the MOMENT-TCN exported by train.py (no torch at inference).

forward(W, Z, valid) -> (T, 4) logits for one window; Z (T, Din) prepared input (prep), T a
multiple of 8, valid (T,) bool. Mirrors model.MomentTCN.forward in eval mode: RevIN over the valid
steps, patch embedding + sinusoidal positions, T5 encoder (RMSNorm, relative-position-bucket
attention bias shared across layers, no attention scaling, gated GELU-new feed-forward), per-patch
fusion, x8 upsample, skip, dilated TCN head.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # candidates/ (dfl_folds)
sys.path.insert(0, HERE)
import pff_folds  # noqa: E402

PATCH = 8
REVIN_EPS = 1e-5
NORM_CLIP = 20.0
MASK_NEG = -1e9


def load(path):
    z = np.load(pff_folds.resolve(path), allow_pickle=False)
    W = {k: z[k] for k in z.files}
    W["_arch"] = json.loads(str(W["arch"]))
    W["_path"] = path
    return W


def prep(X, mu, sd, ind):
    miss = np.isnan(X)
    z = np.clip((X - mu) / sd, -6.0, 6.0)
    z[miss] = 0.0
    return np.concatenate([z, miss[:, ind].astype(np.float32)], 1).astype(np.float32)


def rms_norm(x, w, eps):
    v = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
    return (x / np.sqrt(v + eps)) * w


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def relative_buckets(n, num_buckets, max_distance):
    """T5 _relative_position_bucket, bidirectional -> (n, n) int buckets [query, key]."""
    ctx = np.arange(n)[:, None]
    mem = np.arange(n)[None, :]
    rel = mem - ctx
    nb = num_buckets // 2
    out = (rel > 0).astype(np.int64) * nb
    rel = np.abs(rel)
    max_exact = nb // 2
    small = rel < max_exact
    with np.errstate(divide="ignore"):
        large = max_exact + (np.log(np.maximum(rel, 1) / max_exact) / np.log(max_distance / max_exact)
                             * (nb - max_exact)).astype(np.int64)
    large = np.minimum(large, nb - 1)
    return out + np.where(small, rel, large)


def softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def t5_encoder(W, e, key_valid):
    """e (S, n, d) embeddings, key_valid (S, n) bool -> (S, n, d)."""
    t5 = W["_arch"]["t5"]
    S, n, d = e.shape
    Hh, dk, eps = t5["num_heads"], t5["d_kv"], t5["layer_norm_epsilon"]
    bk = relative_buckets(n, t5["relative_attention_num_buckets"], t5["relative_attention_max_distance"])
    bias = W["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"][bk]   # (n, n, H)
    bias = np.ascontiguousarray(bias.transpose(2, 0, 1))[None]                               # (1, H, n, n)
    mask = np.where(key_valid, 0.0, MASK_NEG).astype(np.float32)[:, None, None, :]            # (S, 1, 1, n)
    pb = bias + mask
    h = e.astype(np.float32)
    for i in range(t5["num_layers"]):
        p = "encoder.block.{}.layer.".format(i)
        x = rms_norm(h, W[p + "0.layer_norm.weight"], eps)
        q = (x @ W[p + "0.SelfAttention.q.weight"].T).reshape(S, n, Hh, dk).transpose(0, 2, 1, 3)
        k = (x @ W[p + "0.SelfAttention.k.weight"].T).reshape(S, n, Hh, dk).transpose(0, 2, 1, 3)
        v = (x @ W[p + "0.SelfAttention.v.weight"].T).reshape(S, n, Hh, dk).transpose(0, 2, 1, 3)
        sc = q @ k.transpose(0, 1, 3, 2) + pb                                                  # (S, H, n, n)
        a = softmax(sc) @ v                                                                    # (S, H, n, dk)
        a = a.transpose(0, 2, 1, 3).reshape(S, n, Hh * dk)
        h = h + a @ W[p + "0.SelfAttention.o.weight"].T
        x = rms_norm(h, W[p + "1.layer_norm.weight"], eps)
        g = gelu_new(x @ W[p + "1.DenseReluDense.wi_0.weight"].T) * (x @ W[p + "1.DenseReluDense.wi_1.weight"].T)
        h = h + g @ W[p + "1.DenseReluDense.wo.weight"].T
    return rms_norm(h, W["encoder.final_layer_norm.weight"], eps)


def _conv(h, w, b, d):
    """Zero-padded dilated conv, torch Conv1d semantics (= nn_pff_tcn/tcn_np._conv)."""
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


def forward(W, Z, valid=None):
    A = W["_arch"]
    T, Din = Z.shape
    assert T % PATCH == 0 and Din == A["din"], (Z.shape, A["din"])
    n, Cm, d = T // PATCH, A["Cm"], A["t5"]["d_model"]
    if valid is None:
        valid = np.ones(T, bool)
    Z = Z.astype(np.float32)
    s = Z @ W["reduce.weight"].T                                     # (T, Cm)
    vm = valid[:, None].astype(np.float32)
    cnt = max(float(vm.sum()), 1.0)
    mean = (s * vm).sum(0, keepdims=True) / cnt
    var = (((s - mean) * vm) ** 2).sum(0, keepdims=True) / cnt
    z = np.clip((s - mean) / np.sqrt(var + REVIN_EPS), -NORM_CLIP, NORM_CLIP) * vm
    p = z.T.reshape(Cm, n, PATCH)                                     # (Cm, n, 8)
    e = p @ W["value_embedding.weight"].T + W["pe"][:n][None]         # (Cm, n, d)
    pv = valid.reshape(n, PATCH).all(-1)
    h = t5_encoder(W, e, np.repeat(pv[None], Cm, 0))                  # (Cm, n, d)
    pr = h @ W["proj.weight"].T + W["proj.bias"]                      # (Cm, n, dp)
    f = pr.transpose(1, 0, 2).reshape(n, -1) @ W["fuse.weight"].T + W["fuse.bias"]   # (n, Ch)
    f = np.repeat(f, PATCH, 0)                                        # (T, Ch)
    sk = Z @ W["skip.weight"][:, :, 0].T + W["skip.bias"]
    x = np.concatenate([f, sk], 1) @ W["inp.weight"][:, :, 0].T + W["inp.bias"]
    for i, dd in enumerate(A["dils"]):
        hh = np.maximum(x, 0.0)
        hh = np.maximum(_conv(hh, W["blocks.{}.a.weight".format(i)], W["blocks.{}.a.bias".format(i)], int(dd)), 0.0)
        x = x + hh @ W["blocks.{}.b.weight".format(i)][:, :, 0].T + W["blocks.{}.b.bias".format(i)]
    return np.maximum(x, 0.0) @ W["out.weight"][:, :, 0].T + W["out.bias"]


def softmax3(lg):
    m = lg.max(1, keepdims=True)
    e = np.exp(lg - m)
    return e / e.sum(1, keepdims=True)
