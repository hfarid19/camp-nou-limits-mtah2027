"""Torch port of radar_common.forward (experiments only: GPU feature stage and GPU scoring of wide radars).

Opt-in. The shipped pipeline never imports this file: the submission's radar runs radar_common.forward (numpy).

    BRP_TORCH_DEVICE=cuda python candidates/ball_radar_pff/radar_torch.py <script.py> [args ...]

runs <script.py> (feat_data.py, run_pff.py, bench.py, ...) with radar_common.forward replaced by the port below.
float64 on cuda / cpu, float32 on mps (no float64 there). Same operations in the same order as the numpy forward;
the two GRUs run through torch.nn.GRU with the saved weights.

    python candidates/ball_radar_pff/radar_torch.py --check <radar.npz> [T S]

compares the port against numpy on random inputs and prints the largest differences.
"""

import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import radar_common as rc          # noqa: E402

_NUMPY_FORWARD = rc.forward
_CACHE = {}


def _device():
    return os.environ.get("BRP_TORCH_DEVICE", "cuda")


def _model(W, dev, dt):
    """Weights as tensors + the two bidirectional GRUs, cached per weight dict."""
    import torch
    key = (id(W), dev, str(dt))
    if key in _CACHE:
        return _CACHE[key]
    P = {k: torch.as_tensor(np.asarray(v), dtype=dt, device=dev) for k, v in W.items()}
    for name in ("pgru", "bgru"):
        wi = P[name + ".f.weight_ih"]
        gru = torch.nn.GRU(wi.shape[1], P[name + ".f.weight_hh"].shape[1], bidirectional=True).to(dev, dt)
        with torch.no_grad():
            for d, suf in (("f", ""), ("b", "_reverse")):
                for p in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(gru, p + "_l0" + suf).copy_(P["{}.{}.{}".format(name, d, p)])
        gru.eval()
        gru.flatten_parameters()
        P["__" + name] = gru
    _CACHE.clear()
    _CACHE[key] = P
    return P


def _lin(P, k, x):
    return x @ P[k + ".weight"].T + P[k + ".bias"]


def _layernorm(P, k, x):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / (var + 1e-5).sqrt() * P[k + ".weight"] + P[k + ".bias"]


def _pool(e, ms, team):
    import torch
    outs = []
    for side in (team > 0, team < 0):
        mk = ms * side
        cnt = mk.sum(1, keepdim=True)
        mean = (e * mk[..., None]).sum(1) / cnt.clamp(min=1.0)
        mx = torch.where(mk[..., None] > 0, e, torch.full_like(e, -1e4)).max(1).values
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        outs += [mean, mx]
    return torch.cat(outs, -1)


def _frame_stage(P, xs, ms, xf, heads):
    import torch
    T, S, _ = xs.shape
    team = xs[..., 4]
    e = torch.relu(_lin(P, "phi1", xs))
    e = torch.relu(_lin(P, "phi2", e))
    D = e.shape[-1]
    dh = D // heads
    q = _lin(P, "q", e).reshape(T, S, heads, dh).permute(0, 2, 1, 3)
    k_ = _lin(P, "k", e).reshape(T, S, heads, dh).permute(0, 2, 1, 3)
    v = _lin(P, "v", e).reshape(T, S, heads, dh).permute(0, 2, 1, 3)
    sc = q @ k_.transpose(-1, -2) / float(np.sqrt(dh))
    sc = torch.where(ms[:, None, None, :] > 0, sc, torch.full_like(sc, -1e4))
    sc = sc - sc.max(-1, keepdim=True).values
    a = torch.exp(sc)
    a = a / a.sum(-1, keepdim=True)
    o = (a @ v).permute(0, 2, 1, 3).reshape(T, S, D)
    e = _layernorm(P, "ln1", e + _lin(P, "o", o))
    e = _layernorm(P, "ln2", e + _lin(P, "ff2", torch.relu(_lin(P, "ff1", e))))
    e = e * ms[..., None]
    ctx = _pool(e, ms, team)
    ctx = torch.relu(_lin(P, "ctx", torch.cat([ctx, xf], -1)))
    return e, ctx


def forward(W, inp, heads=4, tchunk=2000, schunk=8):
    import torch
    dev = _device()
    dt = torch.float32 if dev == "mps" else torch.float64
    P = _model(W, dev, dt)
    with torch.no_grad():
        xs, ms, pos, xf = (torch.as_tensor(np.asarray(inp[k]), dtype=dt, device=dev) for k in ("xs", "ms", "pos", "xf"))
        T, S, _ = xs.shape
        team = xs[..., 4]
        park0 = os.environ.get("BRP_HP_CPU") == "1"
        es, ctxs = [], []
        for t0 in range(0, T, tchunk):
            e, c = _frame_stage(P, xs[t0:t0 + tchunk], ms[t0:t0 + tchunk], xf[t0:t0 + tchunk], heads)
            es.append(e.cpu() if park0 else e)
            ctxs.append(c)
        e, ctx = torch.cat(es), torch.cat(ctxs)
        # slot chunks, streamed: the (T, S, 2Hp) GRU output of a wide radar does not fit on the GPU next to its
        # concatenations, so chunks are parked on the CPU (BRP_HP_CPU=1) and the team pooling is accumulated.
        park = os.environ.get("BRP_HP_CPU") == "1"
        schunk = int(os.environ.get("BRP_SCHUNK", schunk))
        hps, lps, hmax = [], [], None
        for s0 in range(0, S, schunk):
            k = min(schunk, S - s0)
            Xp = torch.cat([e[:, s0:s0 + k].to(dev), ctx[:, None, :].expand(T, k, ctx.shape[-1])], -1)
            h = P["__pgru"](Xp)[0]
            del Xp
            lps.append(_lin(P, "plog", torch.cat([h, xs[:, s0:s0 + k]], -1))[..., 0])
            m = torch.where(ms[:, s0:s0 + k, None] > 0, h, torch.full_like(h, -1e4)).max(1).values
            hmax = m if hmax is None else torch.maximum(hmax, m)
            hps.append(h.cpu() if park else h)
            del h
        lp = torch.cat(lps, 1)
        lp = torch.where(ms > 0, lp, torch.full_like(lp, -1e4))
        hmax = torch.where(ms.sum(1, keepdim=True) > 0, hmax, torch.zeros_like(hmax))
        ld = _lin(P, "dlog", torch.cat([hmax, ctx], -1))
        logits = torch.cat([lp, ld], -1)
        logits = logits - logits.max(-1, keepdim=True).values
        g = torch.exp(logits)
        g = g / g.sum(-1, keepdim=True)
        gs = g[:, :S]
        acc = {}
        for i, s0 in enumerate(range(0, S, schunk)):
            k = min(schunk, S - s0)
            h = hps[i].to(dev)
            hps[i] = None
            xt = torch.cat([e[:, s0:s0 + k].to(dev), h, gs[:, s0:s0 + k, None]], -1)
            del h
            xt = torch.relu(_lin(P, "bt1", xt)) * ms[:, s0:s0 + k, None]
            for side in (1, -1):
                mk = ms[:, s0:s0 + k] * ((team[:, s0:s0 + k] > 0) if side > 0 else (team[:, s0:s0 + k] < 0))
                sm = (xt * mk[..., None]).sum(1)
                mx = torch.where(mk[..., None] > 0, xt, torch.full_like(xt, -1e4)).max(1).values
                cn = mk.sum(1, keepdim=True)
                if side in acc:
                    acc[side] = (acc[side][0] + sm, torch.maximum(acc[side][1], mx), acc[side][2] + cn)
                else:
                    acc[side] = (sm, mx, cn)
            del xt
        zs = []
        for side in (1, -1):
            sm, mx, cn = acc[side]
            zs += [sm / cn.clamp(min=1.0), torch.where(cn > 0, mx, torch.zeros_like(mx))]
        z = torch.cat(zs, -1)
        epos = (gs[..., None] * pos).sum(1)
        gh = (gs * (team > 0)).sum(1, keepdim=True)
        ga = (gs * (team < 0)).sum(1, keepdim=True)
        zin = torch.cat([z, epos, gh, ga, g[:, S:], ctx, xf], -1)
        zin = torch.relu(_lin(P, "bt2", zin))
        hb = P["__bgru"](zin[:, None, :])[0][:, 0]
        ball = xf[:, 0:2] + _lin(P, "ball", hb)
        tl = _lin(P, "team", hb)
        tl = tl - tl.max(-1, keepdim=True).values
        tp = torch.exp(tl)
        tp = tp / tp.sum(-1, keepdim=True)
        out = dict(ball=ball, g=g, team=tp, epos=epos, gh=gh[:, 0], ga=ga[:, 0])
        return {k: v.double().cpu().numpy() for k, v in out.items()}


def install():
    import torch  # noqa: F401  (before the script's folder joins sys.path: candidates/nn_pff/queue.py shadows stdlib queue)
    import torch.nn  # noqa: F401
    rc.forward = forward


def _check(path, T=3000, S=30):
    W = rc.load_weights(path)
    r = np.random.default_rng(0)
    xs = r.normal(size=(T, S, rc.FS)).astype(np.float32)
    xs[..., 4] = np.where(np.arange(S) < S // 2, 1.0, -1.0)
    ms = (r.random((T, S)) > 0.3).astype(np.float32)
    ms[:5] = 0.0
    xs[ms == 0] = 0.0
    inp = dict(xs=xs, ms=ms, pos=xs[..., 0:2].copy(), xf=r.normal(size=(T, rc.FF)).astype(np.float32))
    import time
    t0 = time.time()
    a = _NUMPY_FORWARD(W, inp)
    t1 = time.time()
    b = forward(W, inp)
    t2 = time.time()
    b = forward(W, inp)
    t3 = time.time()
    print("device {} numpy {:.1f}s torch {:.1f}s (second call {:.2f}s)".format(_device(), t1 - t0, t2 - t1, t3 - t2))
    for k in a:
        print("  {:5s} max |d| {:.3e}".format(k, float(np.abs(a[k] - b[k]).max())))


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        _check(sys.argv[2], *[int(v) for v in sys.argv[3:5]])
    else:
        install()
        script = os.path.abspath(sys.argv[1])
        sys.argv = sys.argv[1:]
        sys.path.insert(0, os.path.dirname(script))
        runpy.run_path(script, run_name="__main__")
