"""Linear-chain CRF over (H, A, D) with per-frame, feature-dependent potentials (numpy inference).

Score of a label path y (PathCRF eq.: S = sum_t f_t(y_t) + sum_{t>=1} psi_t(y_{t-1}, y_t)):

  unary   (fixed-unary variant, W["tau"] > 0: LDE's emission tau * log-probabilities, see
          lde_unary; only the pairwise heads are learned)
          f_t(H) = log sigmoid(T_t), f_t(A) = log sigmoid(-T_t), f_t(D) = Dd_t
          T_t  = h_u(ut_t) - h_u(-ut_t)           (antisymmetric: swapping teams flips T)
          Dd_t = h_d(ud_t)
  pairwise psi_t(H,H) = psi_t(A,A) = psi_t(D,D) = 0
          psi_t(A,H) = h_s([ssym_t,  santi_t])    psi_t(H,A) = h_s([ssym_t, -santi_t])
          psi_t(H,D) = psi_t(A,D) = h_x(die_t)
          psi_t(D,H) = h_r([rsym_t,  ranti_t])    psi_t(D,A) = h_r([rsym_t, -ranti_t])

Each head h(x) = w . x + v . tanh(W x + b) (v empty for the linear CRF). The weights come from
train_crf.py (final model: structured hinge / SSVM; CRF NLL and softmax-margin were also tried),
stored in an .npz.
Decoding: Viterbi (MAP path) and forward-backward posterior marginals (predict_proba).
"""

import math

import numpy as np

H, A, D = 0, 1, 2
HEADS = dict(u="ut", d="ud", s=("ssym", "santi"), x="die", r=("rsym", "ranti"))


def _head(W, name, x):
    y = x @ W["w_" + name]
    if ("W_" + name) in W and W["W_" + name].size:
        y = y + np.tanh(x @ W["W_" + name] + W["b_" + name]) @ W["v_" + name]
    return y


def lde_unary(lde, tau):
    """LDE's Viterbi emission: tau * (log pH + log(1-pD), log pA + log(1-pD), log pD), clipped
    probabilities 1e-3 as LDE's decode does."""
    lh, ld = lde[:, 0].astype(np.float64), lde[:, 1].astype(np.float64)
    c = np.log(999.0)
    lh, ld = np.clip(lh, -c, c), np.clip(ld, -c, c)
    alive = -np.logaddexp(0, ld)
    return tau * np.stack([-np.logaddexp(0, -lh) + alive, -np.logaddexp(0, lh) + alive,
                           -np.logaddexp(0, -ld)], 1)


def potentials(W, f, psi_bias=None):
    """-> U (N, 3) unary, P (N, 3, 3) pairwise (P[t, i, j]: y[t-1] = i -> y[t] = j)."""
    if float(W.get("tau", 0)) > 0:             # fixed unary: LDE's emission at temperature tau
        U = lde_unary(f["lde"], float(W["tau"]))
    else:
        ut, ud = f["ut"].astype(np.float64), f["ud"].astype(np.float64)
        T = _head(W, "u", ut) - _head(W, "u", -ut)
        Dd = _head(W, "d", ud)
        U = np.stack([-np.logaddexp(0, -T), -np.logaddexp(0, T), Dd], 1)
    ss, sa = f["ssym"].astype(np.float64), f["santi"].astype(np.float64)
    rs, ra = f["rsym"].astype(np.float64), f["ranti"].astype(np.float64)
    toH = _head(W, "s", np.concatenate([ss, sa], 1))
    toA = _head(W, "s", np.concatenate([ss, -sa], 1))
    die = _head(W, "x", f["die"].astype(np.float64))
    rH = _head(W, "r", np.concatenate([rs, ra], 1))
    rA = _head(W, "r", np.concatenate([rs, -ra], 1))
    b = psi_bias or {}
    P = np.zeros((len(U), 3, 3))
    P[:, A, H] = toH + b.get("switch", 0.0)
    P[:, H, A] = toA + b.get("switch", 0.0)
    P[:, H, D] = P[:, A, D] = die + b.get("die", 0.0)
    P[:, D, H] = rH + b.get("revive", 0.0)
    P[:, D, A] = rA + b.get("revive", 0.0)
    return U, np.clip(P, -60, 60)


def viterbi(U, P):
    T = len(U)
    back = np.zeros((T, 3), np.int8)
    u = U.tolist()
    p = P.reshape(T, 9).tolist()
    s = list(u[0])
    for t in range(1, T):
        q = p[t]
        e = u[t]
        ns = [0.0, 0.0, 0.0]
        bk = [0, 0, 0]
        for j in range(3):
            a0, a1, a2 = s[0] + q[j], s[1] + q[3 + j], s[2] + q[6 + j]
            if a0 >= a1 and a0 >= a2:
                ns[j], bk[j] = a0, 0
            elif a1 >= a2:
                ns[j], bk[j] = a1, 1
            else:
                ns[j], bk[j] = a2, 2
            ns[j] += e[j]
        back[t] = bk
        s = ns
    path = np.empty(T, np.int8)
    path[-1] = int(np.argmax(s))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def marginals(U, P):
    """Forward-backward posterior marginals (N, 3), scaled in probability space."""
    T = len(U)
    um = U.max(1, keepdims=True)
    E = np.exp(U - um).tolist()
    M = np.exp(P).reshape(T, 9).tolist()
    fw = [None] * T
    a = E[0]
    z = sum(a)
    a = [v / z for v in a]
    fw[0] = a
    for t in range(1, T):
        m, e = M[t], E[t]
        b = [(a[0] * m[j] + a[1] * m[3 + j] + a[2] * m[6 + j]) * e[j] for j in range(3)]
        z = b[0] + b[1] + b[2]
        a = [v / z for v in b]
        fw[t] = a
    post = np.empty((T, 3))
    bw = [1.0, 1.0, 1.0]
    post[-1] = fw[-1]
    for t in range(T - 1, 0, -1):
        m, e = M[t], E[t]
        x = [bw[j] * e[j] for j in range(3)]
        nb = [m[3 * i] * x[0] + m[3 * i + 1] * x[1] + m[3 * i + 2] * x[2] for i in range(3)]
        z = nb[0] + nb[1] + nb[2]
        bw = [v / z for v in nb]
        f = fw[t - 1]
        q = [f[i] * bw[i] for i in range(3)]
        s = q[0] + q[1] + q[2]
        post[t - 1] = [v / s for v in q]
    return post
