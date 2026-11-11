"""EXPERIMENTS ONLY: run the members' network arithmetic on the GPU (torch, float32) instead of numpy float64.

torch runs in a separate worker process: torch and LightGBM bundle different libomp builds on macOS and crash when
both are loaded in one process (the radar member uses LightGBM).

    import gpu_accel; gpu_accel.install()        # before the pipeline is loaded (see final_f4b/pipeline_gpu.py)

Replaces ptr_np.forward (the pointer member, more than half of the pipeline's time) by an op-for-op torch port of the
same function, reading the same weight dict. Outputs differ from the numpy path in the last float32 digits, so this is
never used by make_submission.py (the submission must be byte-identical on every machine). Check with
`python candidates/gpu_accel.py <ptr model.npz>` (random inputs, max |diff| vs numpy).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_T = {}          # id(W) -> {name: tensor}


def _device():
    import torch
    want = os.environ.get("GPU_ACCEL_DEVICE")
    if want:
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _weights(key, W, dev):
    import torch
    if key not in _T:
        _T[key] = {n: torch.from_numpy(np.asarray(a, np.float32)).to(dev) for n, a in W.items()
                   if isinstance(a, np.ndarray) and a.dtype.kind == "f"}
    return _T[key]


def _ptr_torch(key, W, cfg, xs, ms, xf, Xn):
    """Torch port of candidates/nn_pointer/ptr_np.forward (same maths, float32 on the GPU)."""
    import torch
    import torch.nn.functional as F
    dev = _device()
    Wt = _weights(key, W, dev)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a, np.float32))).to(dev)  # noqa: E731
    xs, ms, xf, Xn = t(xs), t(ms), t(xf), t(Xn)
    T, S, _ = xs.shape
    if cfg.get("noX", 0):
        Xn = torch.zeros_like(Xn)
    team = xs[..., 4]
    lin = lambda k, x: x @ Wt[k + ".weight"].T + Wt[k + ".bias"]  # noqa: E731

    def ln(k, x):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-5) * Wt[k + ".weight"] + Wt[k + ".bias"]

    def conv(x, k, d=1):                      # x (B, Cin, T)
        w = Wt[k + ".weight"]
        return F.conv1d(x, w, Wt[k + ".bias"], padding=(d if w.shape[2] == 3 else 0), dilation=(d if w.shape[2] == 3 else 1))

    def drl(pre, x, d):
        return x + conv(torch.relu(conv(x, pre + "conv", d)), pre + "out")

    heads = cfg["heads"]
    es, pls = [], []
    with torch.no_grad():
        for t0 in range(0, T, 2000):
            sl = slice(t0, t0 + 2000)
            e = torch.relu(lin("phi2", torch.relu(lin("phi1", xs[sl]))))
            m = ms[sl]
            for i in range(cfg["nsab"]):
                pre = "sabs.{}.".format(i)
                n, D = e.shape[0], e.shape[2]
                dh = D // heads
                q = lin(pre + "q", e).reshape(n, S, heads, dh).permute(0, 2, 1, 3)
                k = lin(pre + "k", e).reshape(n, S, heads, dh).permute(0, 2, 1, 3)
                v = lin(pre + "v", e).reshape(n, S, heads, dh).permute(0, 2, 1, 3)
                sc = q @ k.transpose(-1, -2) / float(np.sqrt(dh))
                sc = torch.where(m[:, None, None, :] > 0, sc, torch.full_like(sc, -1e4))
                a = torch.softmax(sc, -1)
                o = (a @ v).permute(0, 2, 1, 3).reshape(n, S, D)
                e = ln(pre + "ln1", e + lin(pre + "o", o))
                e = ln(pre + "ln2", e + lin(pre + "ff2", torch.relu(lin(pre + "ff1", e))))
                e = e * m[..., None]
            es.append(e)
            outs = []
            tm = team[sl]
            for side in (tm > 0, tm < 0):
                mk = m * side.float()
                cnt = mk.sum(1, keepdim=True)
                mean = (e * mk[..., None]).sum(1) / cnt.clamp(min=1.0)
                mx = torch.where(mk[..., None] > 0, e, torch.full_like(e, -1e4)).max(1).values
                mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
                outs += [mean, mx]
            pls.append(torch.cat(outs, -1))
        e, pl = torch.cat(es), torch.cat(pls)
        h = torch.cat([pl, xf, Xn], -1).T[None]                                  # (1, ., T)
        c = conv(h, "fin")
        for i in range(cfg["L"]):
            c = drl("tcn.{}.".format(i), c, 2 ** i)
        ct = c[0].T                                                               # (T, C)
        u = lin("sin", torch.cat([e, ct[:, None, :].expand(T, S, ct.shape[1])], -1))
        u = torch.relu(u) * ms[..., None]
        u = u.permute(1, 2, 0).contiguous()                                       # (S, Dp, T)
        for i in range(cfg["Lp"]):
            u = drl("stcn.{}.".format(i), u, 2 ** i)
        u = u.permute(2, 0, 1)                                                    # (T, S, Dp)
        lp = lin("plog", torch.cat([u, xs], -1))[..., 0]
        lp = torch.where(ms > 0, lp, torch.full_like(lp, -1e4))
        umax = torch.where(ms[..., None] > 0, u, torch.full_like(u, -1e4)).max(1).values
        umax = torch.where(ms.sum(1, keepdim=True) > 0, umax, torch.zeros_like(umax))
        ld = lin("dlog", torch.cat([ct, umax], -1))
        g = torch.softmax(torch.cat([lp, ld], -1), -1)
        gs = g[:, :S] * (ms > 0)
        P = torch.stack([(gs * (team > 0)).sum(1), (gs * (team < 0)).sum(1), g[:, S]], 1)
        if cfg.get("mix", 0.0) > 0:
            pd = torch.softmax(conv(c, "dir")[0].T, -1)
            P = (1 - cfg["mix"]) * P + cfg["mix"] * pd
        ho = torch.sigmoid(conv(c, "ho")[0, 0])
    return dict(P=P.cpu().numpy().astype(np.float64), ho=ho.cpu().numpy().astype(np.float64), g=g.cpu().numpy().astype(np.float64))


def _mstcn_torch(key, W, c, Xn):
    """Torch port of candidates/nn_pff/tcn_np.Single.forward (MS-TCN member)."""
    import torch
    import torch.nn.functional as F
    dev = _device()
    w = _weights(key, W, dev)

    def conv(x, name, d=1):
        wt = w[name + ".weight"]
        k3 = wt.shape[2] == 3
        return F.conv1d(x, wt, w[name + ".bias"], padding=(d if k3 else 0), dilation=(d if k3 else 1))

    def stage(pre, x, L, nheads):
        h = conv(x, pre + "inp")
        for i in range(L):
            q = "{}layers.{}.".format(pre, i)
            h = h + conv(torch.relu(conv(h, q + "conv", 2 ** i)), q + "out")
        return [conv(h, "{}heads.{}".format(pre, j)) for j in range(nheads)]

    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(Xn, np.float32).T)).to(dev)[None]
        cl, bl = stage("backbone.", x, c["L"], 2)
        for s in range(c["S_c"]):
            cl = stage("cref.{}.".format(s), torch.softmax(cl, 1), c["Lr"], 1)[0]
        for s in range(c["S_b"]):
            bl = stage("bref.{}.".format(s), torch.sigmoid(bl), c["Lr"], 1)[0]
        pc, pb = torch.softmax(cl, 1)[0].T, torch.sigmoid(bl)[0].T
    return pc.cpu().numpy().astype(np.float64), pb.cpu().numpy().astype(np.float64)


def _tcn_torch(key, W, dils, z):
    """Torch port of candidates/nn_pff_tcn/tcn_np.forward (TCN member)."""
    import torch
    import torch.nn.functional as F
    dev = _device()
    w = _weights(key, W, dev)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(z, np.float32).T)).to(dev)[None]
        x = F.conv1d(x, w["inp.weight"], w["inp.bias"])
        for i, d in enumerate(dils):
            wa = w["blocks.{}.a.weight".format(i)]
            d = int(d)
            h = torch.relu(F.conv1d(torch.relu(x), wa, w["blocks.{}.a.bias".format(i)], padding=d * (wa.shape[2] // 2), dilation=d))
            x = x + F.conv1d(h, w["blocks.{}.b.weight".format(i)], w["blocks.{}.b.bias".format(i)])
        out = F.conv1d(torch.relu(x), w["out.weight"], w["out.bias"])[0].T
    return out.cpu().numpy()


# ---- worker process (holds torch) and the main-process stubs --------------------------------------------------------
_FN = {"ptr": _ptr_torch, "mstcn": _mstcn_torch, "tcn": _tcn_torch}
_CONN = None
_SENT = set()
_KEEP = []          # keeps the weight dicts alive so id() keys stay unique


def _serve(conn):
    store = {}
    while True:
        msg = conn.recv()
        if msg is None:
            return
        kind, key, W, args = msg
        if W is not None:
            store[key] = W
        try:
            conn.send(("ok", _FN[kind](key, store[key], *args)))
        except Exception as e:      # noqa: BLE001
            conn.send(("err", repr(e)))


def _call(kind, W, *args):
    global _CONN
    if _CONN is None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        _CONN, child = ctx.Pipe()
        ctx.Process(target=_serve, args=(child,), daemon=True).start()
    key = (kind, id(W))
    send_w = None
    if key not in _SENT:
        _SENT.add(key)
        _KEEP.append(W)
        send_w = {n: a for n, a in W.items() if isinstance(a, np.ndarray) and a.dtype.kind == "f"}
    _CONN.send((kind, key, send_w, args))
    status, out = _CONN.recv()
    if status != "ok":
        raise RuntimeError("gpu_accel worker: " + out)
    return out


def ptr_forward(W, meta, xs, ms, xf, Xn):
    return _call("ptr", W, meta["cfg"], np.asarray(xs, np.float32), np.asarray(ms, np.float32),
                 np.asarray(xf, np.float32), np.asarray(Xn, np.float32))


def mstcn_forward(self, Xn):
    return _call("mstcn", self.w, self.cfg, np.asarray(Xn, np.float32))


def tcn_forward(W, z):
    return _call("tcn", W, [int(d) for d in np.asarray(W["dils"]).tolist()], np.asarray(z, np.float32))


def install():
    """Patch every loaded copy of the members' numpy forward modules (they are loaded under several names).
    Safe to call repeatedly; call it after the pipeline is loaded and again before predicting."""
    import gc
    import types
    n = 0
    for m in [o for o in gc.get_objects() if isinstance(o, types.ModuleType)]:
        d = m.__dict__ if isinstance(getattr(type(m), "__dict__", None), object) and hasattr(m, "__dict__") else {}
        f = d.get("__file__") or ""
        if not isinstance(f, str) or d.get("_gpu_accel", False):
            continue
        f = f.replace(os.sep, "/")
        if f.endswith("nn_pointer/ptr_np.py"):
            m.forward_numpy, m.forward = m.forward, ptr_forward
        elif f.endswith("nn_pff/tcn_np.py"):
            m.Single.forward_numpy, m.Single.forward = m.Single.forward, mstcn_forward
        elif f.endswith("nn_pff_tcn/tcn_np.py"):
            m.forward_numpy, m.forward = m.forward, tcn_forward
        else:
            continue
        m._gpu_accel = True
        n += 1
    return n


if __name__ == "__main__":
    import time
    sys.path.insert(0, os.path.join(HERE, "nn_pointer"))
    import ptr_np
    W, meta = ptr_np.load(sys.argv[1])
    cfg = meta["cfg"]
    rng = np.random.default_rng(0)
    T, S = 6000, 14
    fs, ff, K = W["phi1.weight"].shape[1], W["fin.weight"].shape[1] - 4 * cfg["D"] - len(meta["names"]), len(meta["names"])
    xs = rng.normal(size=(T, S, fs)); xs[..., 4] = np.where(np.arange(S) % 2 == 0, 1.0, -1.0)
    ms = (rng.random((T, S)) > 0.3).astype(np.float64)
    xf, Xn = rng.normal(size=(T, ff)), rng.normal(size=(T, K))
    t0 = time.time(); a = ptr_np.forward(W, meta, xs, ms, xf, Xn); t1 = time.time()
    b = ptr_forward(W, meta, xs, ms, xf, Xn); t2 = time.time(); b = ptr_forward(W, meta, xs, ms, xf, Xn); t3 = time.time()
    print("device", _device(), "numpy {:.1f}s  gpu first {:.1f}s  gpu again {:.2f}s".format(t1 - t0, t2 - t1, t3 - t2))
    print({k: float(np.abs(a[k] - b[k]).max()) for k in a})
