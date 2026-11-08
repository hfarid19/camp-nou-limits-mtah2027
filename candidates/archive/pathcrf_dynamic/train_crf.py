#!/usr/bin/env python3
"""Train the dynamic-transition CRF of crf.py by CRF negative log-likelihood (numpy + L-BFGS).

    python -I candidates/pathcrf_dynamic/train_crf.py --data DIR --folds-out DIR --out crf.npz
                                     [--fold GI] [--hidden 0] [--tau 0] [--l2 1e-5] [--win 750]
                                     [--iters 150]

Data: prep.py's out-of-fold LDE signals on fresh DFL degradations (full 25 fps). Each graded
half is cut into non-overlapping windows of --win frames (30 s; PathCRF trains on 10 s windows of
its 5 fps data). Objective per window: log Z (forward algorithm) - S(gold path), summed over all
windows and divided by the number of frames, + l2 * |theta|^2 (all head weights).

The gradient is exact: forward-backward node / edge marginals give dNLL/dU_t = mu_t - 1[y_t] and
dNLL/dpsi_t(i,j) = xi_t(i,j) - 1[y_{t-1}=i, y_t=j], back-propagated through the potential heads
(linear, or linear + one tanh layer with --hidden). Optimiser: scipy L-BFGS-B, fixed iteration
budget, deterministic (fixed init seed). Runs entirely in numpy; no torch.

Leave-one-game-out: --fold gi trains on the six other DFL games and writes
<folds-out>/fold_crf_<gi>.npz (+ its held-out NLL); without --fold, missing folds are trained
first, then the full model (--out) on all seven games.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crf_features as cf  # noqa: E402
import crf                 # noqa: E402

H, A, D = 0, 1, 2
# head name -> (input groups, init {column: weight})
HEADS = dict(u=(("ut",), {0: 0.05, 1: 0.1}), d=(("ud",), {0: -0.5, 1: 0.15}),
             s=(("ssym", "santi"), {0: -4.0}), x=(("die",), {0: -6.0}),
             r=(("rsym", "ranti"), {0: -6.0}))


def load(data_dir, win, seeds=None):
    rows = []
    for p in sorted(glob.glob(os.path.join(data_dir, "DFL-MAT-*_*.npz"))):
        if seeds and int(p[:-4].rsplit("_", 1)[1]) not in seeds:
            continue
        z = np.load(p)
        sig = {k: z[k] for k in z.files if k not in ("codes", "blocks", "gi")}
        f = cf.build(sig)
        codes = z["codes"]
        gi = int(z["gi"])
        for _, first, last in z["blocks"]:
            for s in range(first, last + 2 - win, win):
                y = codes[s:s + win]
                if (y > 2).any():
                    continue
                rows.append((gi, {k: v[s:s + win] for k, v in f.items()}, y))
    g = np.array([r[0] for r in rows])
    F = {k: np.stack([r[1][k] for r in rows]).astype(np.float64) for k in cf.GROUPS}
    Y = np.stack([r[2] for r in rows]).astype(np.int64)
    return g, F, Y


class Model(object):
    def __init__(self, dims, hidden, seed=0, tau=0.0, init_lde=False):
        self.hidden = hidden
        self.tau = tau
        self.shapes = []
        rng = np.random.default_rng(seed)
        init = []
        for name, (groups, w0) in HEADS.items():
            if tau > 0 and name in ("u", "d"):
                continue
            k = sum(dims[gname] for gname in groups)
            w = np.zeros(k)
            for i, v in w0.items():
                w[i] = v
            if init_lde:            # LDE's tuned constant costs (switch 0.75, dead 5 at tau 0.15)
                w[0] = {"s": -0.75, "x": -5.0, "r": -5.0}.get(name, w[0])
            self.shapes.append(("w_" + name, (k,)))
            init.append(w)
            if hidden and name in ("s", "x", "r"):
                self.shapes += [("W_" + name, (k, hidden)), ("b_" + name, (hidden,)),
                                ("v_" + name, (hidden,))]
                init += [rng.normal(0, 0.3, (k, hidden)).ravel(), np.zeros(hidden),
                         np.zeros(hidden)]
        self.theta0 = np.concatenate(init)

    def unpack(self, theta):
        out, i = {}, 0
        for name, shape in self.shapes:
            n = int(np.prod(shape))
            out[name] = theta[i:i + n].reshape(shape)
            i += n
        return out

    def pack(self, grads):
        return np.concatenate([np.asarray(grads[name]).ravel() for name, _ in self.shapes])


def head_fwd(W, name, x):
    y = x @ W["w_" + name]
    cache = None
    if ("W_" + name) in W:
        h = np.tanh(x @ W["W_" + name] + W["b_" + name])
        y = y + h @ W["v_" + name]
        cache = h
    return y, cache


def head_bwd(W, name, x, cache, g, grads):
    k = x.shape[-1]
    grads["w_" + name] = grads.get("w_" + name, 0) + (x.reshape(-1, k).T @ g.ravel())
    if cache is not None:
        hd = cache.shape[-1]
        hh = cache.reshape(-1, hd)
        grads["v_" + name] = grads.get("v_" + name, 0) + hh.T @ g.ravel()
        dz = (g.ravel()[:, None] * W["v_" + name][None, :]) * (1 - hh ** 2)
        grads["W_" + name] = grads.get("W_" + name, 0) + x.reshape(-1, k).T @ dz
        grads["b_" + name] = grads.get("b_" + name, 0) + dz.sum(0)


def _lse(a, axis):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def objective(theta, model, F, Y, l2, need_grad=True, aug=None, smm=None):
    """CRF NLL per frame (aug None) or, with aug = (Uaug, Paug) cost-augmentation arrays, the
    structured hinge max_y' [S(y') + Delta(y, y')] - S(y) per frame (margin-rescaled SSVM).
    smm = (Uaug, Paug): softmax-margin (Gimpel & Smith 2010), log sum_y' exp(S(y') + Delta) - S(y),
    i.e. the CRF NLL with a cost-augmented partition function (smooth, convex for linear heads)."""
    W = model.unpack(theta)
    if model.tau > 0:
        B0, T0 = Y.shape
        U = crf.lde_unary(F["lde"].reshape(B0 * T0, 2), model.tau).reshape(B0, T0, 3)
    else:
        ut, ud = F["ut"], F["ud"]
        tp, cp = head_fwd(W, "u", ut)
        tn, cn = head_fwd(W, "u", -ut)
        T = tp - tn
        Dd, cd = head_fwd(W, "d", ud)
        sig = 1 / (1 + np.exp(-T))
        U = np.stack([-np.logaddexp(0, -T), -np.logaddexp(0, T), Dd], -1)
    xH = np.concatenate([F["ssym"], F["santi"]], -1)
    xA = np.concatenate([F["ssym"], -F["santi"]], -1)
    rH_x = np.concatenate([F["rsym"], F["ranti"]], -1)
    rA_x = np.concatenate([F["rsym"], -F["ranti"]], -1)
    sH, csH = head_fwd(W, "s", xH)
    sA, csA = head_fwd(W, "s", xA)
    die, cdie = head_fwd(W, "x", F["die"])
    rH, crH = head_fwd(W, "r", rH_x)
    rA, crA = head_fwd(W, "r", rA_x)
    B, Tn = Y.shape
    P = np.zeros((B, Tn, 3, 3))
    P[:, :, A, H] = sH
    P[:, :, H, A] = sA
    P[:, :, H, D] = die
    P[:, :, A, D] = die
    P[:, :, D, H] = rH
    P[:, :, D, A] = rA
    bi = np.arange(B)[:, None]
    ti = np.arange(Tn)[None, :]
    n = B * Tn
    if aug is not None:
        yh = batch_viterbi(U + aug[0], P + aug[1])
        gold = U[bi, ti, Y].sum(1) + P[bi, ti[:, 1:], Y[:, :-1], Y[:, 1:]].sum(1)
        pred = ((U + aug[0])[bi, ti, yh].sum(1)
                + (P + aug[1])[bi, ti[:, 1:], yh[:, :-1], yh[:, 1:]].sum(1))
        loss = float((pred - gold).sum() / n)
        if not need_grad:
            return loss
        gU = np.zeros_like(U)
        gP = np.zeros_like(P)
        np.add.at(gU, (np.broadcast_to(bi, Y.shape), np.broadcast_to(ti, Y.shape), yh), 1.0)
        np.add.at(gU, (np.broadcast_to(bi, Y.shape), np.broadcast_to(ti, Y.shape), Y), -1.0)
        bb, tt = np.broadcast_to(bi, (B, Tn - 1)), np.broadcast_to(ti[:, 1:], (B, Tn - 1))
        np.add.at(gP, (bb, tt, yh[:, :-1], yh[:, 1:]), 1.0)
        np.add.at(gP, (bb, tt, Y[:, :-1], Y[:, 1:]), -1.0)
        return loss + l2 * float((theta ** 2).sum()), _backprop(
            W, model, theta, l2, gU / n, gP / n, locals())
    U0, P0 = U, P
    if smm is not None:
        U, P = U + smm[0], P + smm[1]
    alpha = np.empty((B, Tn, 3))
    alpha[:, 0] = U[:, 0]
    for t in range(1, Tn):
        alpha[:, t] = _lse(alpha[:, t - 1, :, None] + P[:, t], 1) + U[:, t]
    logz = _lse(alpha[:, -1], 1)
    gold = U0[bi, ti, Y].sum(1) + P0[bi, ti[:, 1:], Y[:, :-1], Y[:, 1:]].sum(1)
    nll = float((logz - gold).sum() / n)
    reg = l2 * float((theta ** 2).sum())
    if not need_grad:
        return nll
    beta = np.zeros((B, Tn, 3))
    for t in range(Tn - 1, 0, -1):
        beta[:, t - 1] = _lse(P[:, t] + (U[:, t] + beta[:, t])[:, None, :], 2)
    mu = np.exp(alpha + beta - logz[:, None, None])
    xi = np.exp(alpha[:, :-1, :, None] + P[:, 1:] + (U[:, 1:] + beta[:, 1:])[:, :, None, :]
                - logz[:, None, None, None])
    gU = mu
    gU[bi, ti, Y] -= 1.0
    gP = np.zeros_like(P)
    gP[:, 1:] = xi
    gP[bi, ti[:, 1:], Y[:, :-1], Y[:, 1:]] -= 1.0
    gU /= n
    gP /= n
    return nll + reg, _backprop(W, model, theta, l2, gU, gP, locals())


def _backprop(W, model, theta, l2, gU, gP, v):
    F = v["F"]
    grads = {}
    if model.tau <= 0:
        ut, ud, sig, cp, cn, cd = v["ut"], v["ud"], v["sig"], v["cp"], v["cn"], v["cd"]
        gT = gU[..., H] * (1 - sig) - gU[..., A] * sig
        head_bwd(W, "u", ut, cp, gT, grads)
        head_bwd(W, "u", -ut, cn, -gT, grads)
        head_bwd(W, "d", ud, cd, gU[..., D], grads)
    head_bwd(W, "s", v["xH"], v["csH"], gP[:, :, A, H], grads)
    head_bwd(W, "s", v["xA"], v["csA"], gP[:, :, H, A], grads)
    head_bwd(W, "x", F["die"], v["cdie"], gP[:, :, H, D] + gP[:, :, A, D], grads)
    head_bwd(W, "r", v["rH_x"], v["crH"], gP[:, :, D, H], grads)
    head_bwd(W, "r", v["rA_x"], v["crA"], gP[:, :, D, A], grads)
    return model.pack(grads) + 2 * l2 * theta


def batch_viterbi(U, P):
    B, Tn, _ = U.shape
    bp = np.zeros((B, Tn, 3), np.int8)
    delta = U[:, 0].copy()
    for t in range(1, Tn):
        cand = delta[:, :, None] + P[:, t]
        bp[:, t] = cand.argmax(1)
        delta = cand.max(1) + U[:, t]
    y = np.empty((B, Tn), np.int64)
    y[:, -1] = delta.argmax(1)
    b = np.arange(B)
    for t in range(Tn - 1, 0, -1):
        y[:, t - 1] = bp[b, t, y[:, t]]
    return y


def augmentation(Y, h, l0, c_fp, k):
    """Cost-augmentation arrays for the structured hinge.
    Unary: spell-normalised Hamming, weight h * (1 + l0 / len(gold run containing t)) on every
    wrong label (so every gold spell, short or long, carries at least h * l0 of loss: spellF1
    counts spells, not frames). Pairwise: c_fp on a change i -> j at t unless the gold path has
    the same change within +-k frames (a 1-frame offset is free in spellF1)."""
    B, Tn = Y.shape
    Ua = np.zeros((B, Tn, 3))
    Pa = np.zeros((B, Tn, 3, 3))
    for b in range(B):
        y = Y[b]
        ch = np.concatenate(([0], np.where(np.diff(y) != 0)[0] + 1, [Tn]))
        w = np.empty(Tn)
        for s0, e0 in zip(ch[:-1], ch[1:]):
            w[s0:e0] = h * (1.0 + l0 / (e0 - s0))
        Ua[b] = w[:, None]
        Ua[b, np.arange(Tn), y] = 0.0
        zone = np.zeros((Tn, 3, 3), bool)
        for t in ch[1:-1]:
            zone[max(1, t - k):t + k + 1, y[t - 1], y[t]] = True
        pen = np.full((Tn, 3, 3), c_fp)
        pen[:, [0, 1, 2], [0, 1, 2]] = 0.0
        pen[zone] = 0.0
        pen[0] = 0.0
        Pa[b] = pen
    return Ua, Pa


def subset(F, Y, m):
    return {k: v[m] for k, v in F.items()}, Y[m]


def fit_ssvm(F, Y, dims, a, tag=""):
    """Structured hinge (margin rescaling) by full-batch subgradient Adam, Polyak-averaged over
    the second half of the iterations (deterministic)."""
    model = Model(dims, a.hidden, tau=a.tau, init_lde=a.init_lde)
    aug = augmentation(Y, a.h, a.l0, a.cfp, a.k)
    th = model.theta0.copy()
    m1, m2 = np.zeros_like(th), np.zeros_like(th)
    avg, navg = np.zeros_like(th), 0
    t0 = time.time()
    for it in range(1, a.iters + 1):
        f, g = objective(th, model, F, Y, a.l2, aug=aug)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        lr = a.lr / np.sqrt(it / 10.0 + 1.0)
        th = th - lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(m2 / (1 - 0.999 ** it)) + 1e-8)
        if it > a.iters // 2:
            avg += th
            navg += 1
        if it % 25 == 0:
            print("  {} iter {} hinge {:.5f} {:.0f}s".format(tag, it, f, time.time() - t0),
                  flush=True)
    return model, avg / navg


def fit(F, Y, dims, a, tag=""):
    if a.loss == "ssvm":
        return fit_ssvm(F, Y, dims, a, tag)
    model = Model(dims, a.hidden, tau=a.tau, init_lde=a.init_lde)
    smm = augmentation(Y, a.h, a.l0, a.cfp, a.k) if a.loss == "smm" else None
    t0 = time.time()
    it = [0]

    def cb(theta):
        it[0] += 1
        if it[0] % 25 == 0:
            print("  {} iter {} {:.0f}s".format(tag, it[0], time.time() - t0), flush=True)
    res = minimize(objective, model.theta0, args=(model, F, Y, a.l2, True, None, smm), jac=True,
                   method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=a.iters, gtol=1e-10, ftol=1e-13))
    print("  {} {} iters, objective {:.5f}, {}".format(tag, res.nit, res.fun, res.message),
          flush=True)
    return model, res.x


def export(model, theta, path, **extra):
    W = model.unpack(theta)
    np.savez(path, **{k: np.asarray(v, np.float64) for k, v in W.items()},
             hidden=model.hidden, tau=model.tau, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--win", type=int, default=750)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="> 0: fixed LDE unary at this temperature, learn only the transitions")
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--init-lde", action="store_true",
                    help="start the pairwise biases at LDE's tuned constant costs")
    ap.add_argument("--loss", default="nll", choices=["nll", "ssvm", "smm"])
    ap.add_argument("--h", type=float, default=0.05, help="ssvm: Hamming weight per frame")
    ap.add_argument("--l0", type=float, default=75.0, help="ssvm: per-spell Hamming (frames)")
    ap.add_argument("--cfp", type=float, default=1.0, help="ssvm: cost of a spurious change")
    ap.add_argument("--k", type=int, default=12, help="ssvm: change tolerance (frames)")
    ap.add_argument("--lr", type=float, default=0.05, help="ssvm: Adam step")
    ap.add_argument("--seeds", default="", help="subset of degradation seeds (comma list)")
    a = ap.parse_args()
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    g, F, Y = load(a.data, a.win, seeds)
    dims = {k: F[k].shape[-1] for k in cf.GROUPS}
    print("windows", len(Y), "per game", np.bincount(g).tolist(), "dims", dims,
          "load {:.0f}s".format(time.time() - t0), flush=True)
    os.makedirs(a.folds_out, exist_ok=True)
    games = sorted(set(g.tolist()))
    todo = [a.fold] if a.fold >= 0 else games
    for gi in todo:
        path = os.path.join(a.folds_out, "fold_crf_{}.npz".format(gi))
        if a.fold < 0 and os.path.isfile(path):
            continue
        tr, va = subset(F, Y, g != gi), subset(F, Y, g == gi)
        model, theta = fit(*tr, dims, a, tag="fold {}".format(gi))
        vnll = objective(theta, model, *va, 0.0, False)     # CRF NLL of the potentials
        export(model, theta, path, val_nll=vnll)
        print("fold {} val nll {:.5f} ({:.0f}s)".format(gi, vnll, time.time() - t0), flush=True)
    if a.fold >= 0:
        return
    vals = [float(np.load(os.path.join(a.folds_out, "fold_crf_{}.npz".format(gi)))["val_nll"])
            for gi in games]
    print("LOGO held-out NLL per frame", [round(v, 5) for v in vals], "mean",
          round(float(np.mean(vals)), 5))
    model, theta = fit(F, Y, dims, a, tag="full")
    export(model, theta, a.out, logo_nll=np.mean(vals))
    for k, v in model.unpack(theta).items():
        if k.startswith("w_"):
            print(k, np.round(v, 3).tolist())
    print("done {:.0f}s".format(time.time() - t0), json.dumps(vars(a)))


if __name__ == "__main__":
    main()
