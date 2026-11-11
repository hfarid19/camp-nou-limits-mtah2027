"""PyTorch-CPU forward of the possessor-pointer network: the maths of ptr_np.forward, op for op, in torch.

forward(W, meta, xs, ms, xf, Xn) has ptr_np.forward's signature and returns the same dict (P, ho, g as float64
numpy arrays). Every step mirrors ptr_np.py: the same chunking of the set-attention blocks (TCHUNK frames), x @ W.T
+ b for the linear layers, layer norm, softmax and sigmoid written out as in numpy, and each convolution as the same
1 or 3 shifted matrix products on a zero-padded copy (no torch convolution kernels, so no NNPACK / oneDNN algorithm
choice is involved). Precision: float64 (default; equal to the numpy path to ~1e-15) or float32 (experiments).

torch runs in a separate worker process (`python -I ptr_torch.py --worker`, started on the first call, numpy +
torch only): on macOS, torch and LightGBM (the radar member) bundle different OpenMP runtimes and a process that
uses both hangs or crashes. The worker:
  - sets torch.set_num_threads(MTAH_TORCH_THREADS, else OMP_NUM_THREADS, else 1) and one inter-op thread;
  - disables oneDNN (torch.backends.mkldnn.enabled = False) and calls torch.use_deterministic_algorithms(True);
  - uses the CPU only (no MPS / CUDA);
  - receives the weights once per weight dict (it keeps only the latest), then one pickled request per forward
    call, over its stdin / stdout.
The pipeline's process never imports torch. select(default) picks the backend (MTAH_PTR_BACKEND overrides the
member's default) and says once, on stderr, which one runs.
"""

import os
import pickle
import subprocess
import sys

import numpy as np

TCHUNK = 2000          # = ptr_np.TCHUNK


# ---- the forward pass (runs inside the worker) ----------------------------------------------------------------------
def _forward(torch, Wt, cfg, xs, ms, xf, Xn, dtype):
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dtype)      # noqa: E731
    xs, ms, xf, Xn = t(xs), t(ms), t(xf), t(Xn)
    T, S, _ = xs.shape
    if cfg.get("noX", 0):
        Xn = torch.zeros_like(Xn)
    team = xs[..., 4]
    neg = torch.tensor(-1e4, dtype=dtype)
    zero = torch.tensor(0.0, dtype=dtype)

    def lin(k, x):
        return x @ Wt[k + ".weight"].T + Wt[k + ".bias"]

    def ln(k, x):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-5) * Wt[k + ".weight"] + Wt[k + ".bias"]

    def conv(x, k, d=1):
        """x (..., Cin, T), 'same' zero padding, dilation d -> (..., Cout, T); as ptr_np._conv."""
        w, b = Wt[k + ".weight"], Wt[k + ".bias"]
        if w.shape[2] == 1:
            return torch.matmul(w[:, :, 0], x) + b[:, None]
        n = x.shape[-1]
        pad = torch.zeros(x.shape[:-1] + (n + 2 * d,), dtype=dtype)
        pad[..., d:d + n] = x
        out = torch.matmul(w[:, :, 0], pad[..., 0:n])
        out += torch.matmul(w[:, :, 1], pad[..., d:d + n])
        out += torch.matmul(w[:, :, 2], pad[..., 2 * d:2 * d + n])
        return out + b[:, None]

    def drl(pre, x, d):
        return x + conv(torch.clamp_min(conv(x, pre + "conv", d), 0.0), pre + "out")

    def softmax(z):
        z = z - z.max(-1, keepdim=True).values
        e = torch.exp(z)
        return e / e.sum(-1, keepdim=True)

    def sab(pre, e, m, heads):
        n, s, D = e.shape
        dh = D // heads
        q = lin(pre + "q", e).reshape(n, s, heads, dh).permute(0, 2, 1, 3)
        k = lin(pre + "k", e).reshape(n, s, heads, dh).permute(0, 2, 1, 3)
        v = lin(pre + "v", e).reshape(n, s, heads, dh).permute(0, 2, 1, 3)
        sc = q @ k.transpose(-1, -2) / float(np.sqrt(dh))
        sc = torch.where(m[:, None, None, :] > 0, sc, neg)
        sc = sc - sc.max(-1, keepdim=True).values
        a = torch.exp(sc)
        a = a / a.sum(-1, keepdim=True)
        o = (a @ v).permute(0, 2, 1, 3).reshape(n, s, D)
        e = ln(pre + "ln1", e + lin(pre + "o", o))
        e = ln(pre + "ln2", e + lin(pre + "ff2", torch.clamp_min(lin(pre + "ff1", e), 0.0)))
        return e * m[..., None]

    def pool(e, m, tm):
        outs = []
        for side in (tm > 0, tm < 0):
            mk = m * side.to(dtype)
            cnt = mk.sum(1, keepdim=True)
            mean = (e * mk[..., None]).sum(1) / torch.clamp_min(cnt, 1.0)
            mx = torch.where(mk[..., None] > 0, e, neg).max(1).values
            mx = torch.where(cnt > 0, mx, zero)
            outs += [mean, mx]
        return torch.cat(outs, -1)

    with torch.no_grad():
        es, pls = [], []
        for t0 in range(0, T, TCHUNK):
            sl = slice(t0, t0 + TCHUNK)
            e = torch.clamp_min(lin("phi2", torch.clamp_min(lin("phi1", xs[sl]), 0.0)), 0.0)
            for i in range(cfg["nsab"]):
                e = sab("sabs.{}.".format(i), e, ms[sl], cfg["heads"])
            es.append(e)
            pls.append(pool(e, ms[sl], team[sl]))
        e, pl = torch.cat(es), torch.cat(pls)
        h = torch.cat([pl, xf, Xn], -1).T                                            # (., T)
        c = conv(h, "fin")
        for i in range(cfg["L"]):
            c = drl("tcn.{}.".format(i), c, 2 ** i)
        ct = c.T                                                                      # (T, C)
        u = lin("sin", torch.cat([e, ct[:, None, :].expand(T, S, ct.shape[1])], -1))
        u = torch.clamp_min(u, 0.0) * ms[..., None]
        u = u.permute(1, 2, 0).contiguous()                                           # (S, Dp, T)
        for i in range(cfg["Lp"]):
            u = drl("stcn.{}.".format(i), u, 2 ** i)
        u = u.permute(2, 0, 1)                                                        # (T, S, Dp)
        lp = lin("plog", torch.cat([u, xs], -1))[..., 0]
        lp = torch.where(ms > 0, lp, neg)
        umax = torch.where(ms[..., None] > 0, u, neg).max(1).values
        umax = torch.where(ms.sum(1, keepdim=True) > 0, umax, zero)
        ld = lin("dlog", torch.cat([ct, umax], -1))
        g = softmax(torch.cat([lp, ld], -1))
        gs = g[:, :S] * (ms > 0).to(dtype)
        P = torch.stack([(gs * (team > 0).to(dtype)).sum(1), (gs * (team < 0).to(dtype)).sum(1), g[:, S]], 1)
        if cfg.get("mix", 0.0) > 0:
            pd = softmax(conv(c, "dir").T)
            P = (1 - cfg["mix"]) * P + cfg["mix"] * pd
        ho = 1.0 / (1.0 + torch.exp(-conv(c, "ho")[0]))
    f64 = lambda a: a.numpy().astype(np.float64)      # noqa: E731
    return dict(P=f64(P), ho=f64(ho), g=f64(g))


def _threads():
    for var in ("MTAH_TORCH_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(var, "").strip()
        if v:
            return max(1, int(v))
    return 1


def _worker():
    """Worker loop: requests are pickled tuples on stdin, replies pickled tuples on stdout."""
    rd, wr = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                                    # keep stray prints off the reply channel
    try:
        import torch
    except Exception as e:                                     # noqa: BLE001
        pickle.dump(("err", "import torch failed: {!r}".format(e)), wr, protocol=4)
        wr.flush()
        return
    torch.set_num_threads(_threads())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    config = str(torch.__config__.show())
    # plain str / int / bool only: unpickling a torch object (torch.__version__ is a TorchVersion) would import
    # torch into the pipeline's process
    info = dict(torch=str(torch.__version__), threads=int(torch.get_num_threads()),
                mkldnn=bool(torch.backends.mkldnn.enabled),
                blas=config.split("BLAS_INFO=")[1].split(",")[0] if "BLAS_INFO=" in config else "?")
    pickle.dump(("ok", info), wr, protocol=4)
    wr.flush()
    dtypes = {"float64": torch.float64, "float32": torch.float32}
    store = {}
    while True:
        try:
            msg = pickle.load(rd)
        except EOFError:
            return
        if msg is None:
            return
        try:
            kind = msg[0]
            if kind == "weights":
                _, key, W, dt = msg
                store.clear()
                store[key] = {n: torch.from_numpy(np.ascontiguousarray(a)).to(dtypes[dt]) for n, a in W.items()}
                out = None
            elif kind == "forward":
                _, key, dt, cfg, xs, ms, xf, Xn = msg
                out = _forward(torch, store[key], cfg, xs, ms, xf, Xn, dtypes[dt])
            else:
                raise ValueError(kind)
            pickle.dump(("ok", out), wr, protocol=4)
        except Exception as e:                                 # noqa: BLE001
            pickle.dump(("err", repr(e)), wr, protocol=4)
        wr.flush()


# ---- client side (the pipeline's process; never imports torch) -----------------------------------------------------
class Worker(object):
    def __init__(self):
        env = dict(os.environ)
        self.proc = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "--worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.sent = {}
        status, self.info = self._recv()
        if status != "ok":
            self.close()
            raise RuntimeError("ptr_torch worker: " + str(self.info))

    def _recv(self):
        had_torch = "torch" in sys.modules
        try:
            out = pickle.load(self.proc.stdout)
        except EOFError:
            raise RuntimeError("ptr_torch worker exited (code {})".format(self.proc.wait()))
        if "torch" in sys.modules and not had_torch:
            raise RuntimeError("ptr_torch: a worker reply imported torch into the pipeline's process")
        return out

    def call(self, msg):
        pickle.dump(msg, self.proc.stdin, protocol=4)
        self.proc.stdin.flush()
        status, out = self._recv()
        if status != "ok":
            raise RuntimeError("ptr_torch worker: " + str(out))
        return out

    def forward(self, W, meta, xs, ms, xf, Xn, dtype="float64"):
        key = (id(W), dtype)
        if self.sent.get(key) is not W:                    # keep W referenced so its id stays unique
            self.call(("weights", key, {n: np.asarray(a, np.float64) for n, a in W.items()}, dtype))
            self.sent = {key: W}                           # the worker keeps only the latest weights too
        a = [np.asarray(v, np.float64) for v in (xs, ms, xf, Xn)]
        return self.call(("forward", key, dtype, meta["cfg"]) + tuple(a))

    def close(self):
        if self.proc.poll() is None:
            try:
                pickle.dump(None, self.proc.stdin, protocol=4)
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self.proc.wait()


_WORKER = None


def worker():
    global _WORKER
    if _WORKER is None:
        _WORKER = Worker()
        import atexit
        atexit.register(_WORKER.close)
    return _WORKER


def forward(W, meta, xs, ms, xf, Xn, dtype="float64"):
    return worker().forward(W, meta, xs, ms, xf, Xn, dtype)


BACKENDS = ("numpy", "torch", "torch32", "auto")
_SAID = set()


def select(default="numpy", label="pointer"):
    """-> "numpy", "torch" (float64) or "torch32". MTAH_PTR_BACKEND, if set, overrides default; "auto" = torch when
    the worker can import torch, else numpy. Prints the choice once per (label, choice) on stderr."""
    env = os.environ.get("MTAH_PTR_BACKEND", "").strip().lower()
    want = env or default
    if want not in BACKENDS:
        raise ValueError("MTAH_PTR_BACKEND / BACKEND must be one of {}, not {!r}".format(BACKENDS, want))
    why = "MTAH_PTR_BACKEND={}".format(env) if env else "default {}".format(default)
    if want == "auto":
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            want, why = "numpy", why + "; torch is not installed"
        else:
            try:
                worker()
                want = "torch"
            except Exception as e:                         # noqa: BLE001
                want, why = "numpy", why + "; the torch worker failed: {}".format(e)
    if want == "numpy":
        msg = "numpy float64 (reference path)"
    else:
        i = worker().info
        msg = "PyTorch {} on the CPU, {}, {} thread{}, oneDNN {}, BLAS {} (worker process)".format(
            i["torch"], "float64" if want == "torch" else "float32", i["threads"], "" if i["threads"] == 1 else "s",
            "on" if i["mkldnn"] else "off", i["blas"])
    if (label, want) not in _SAID:
        _SAID.add((label, want))
        sys.stderr.write("{}: forward pass on {} [{}]\n".format(label, msg, why))
        sys.stderr.flush()
    return want


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    _worker()
