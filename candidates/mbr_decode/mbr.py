"""Minimum-Bayes-risk (expected spellF1) decoding of an ensemble's per-frame outputs. numpy only.

Pieces
------
spells(codes, runs)          possession spells (maximal H or A runs inside each graded run), as arrays.
spell_util(cand, samp)       per candidate spell: the fraction of the metric's IoU thresholds (0.3 ... 0.7)
                             at which it is matched to a spell of one sample. Matching = mutual best IoU,
                             same team (the metric's greedy one-to-one match; identical at IoU >= 0.5).
                             mean over thresholds of F1 = 2 * sum(util) / (n_pred + n_gt), so the expected
                             spellF1 of a candidate is a ratio of sums over its spells.
ffbs(...)                    posterior samples of a 3-label HMM (optionally r sub-states per label = a
                             negative-binomial duration) on the pooled member probabilities, tempered by
                             alpha, at a frame stride; forward filtering, backward sampling, fixed seed.
select_windows(...)          the decoder. Windows = maximal stretches between frames where EVERY candidate
                             says D (so no candidate spell crosses a window edge). In each window pick the
                             candidate with the highest  2 * U_kw - F * n_kw  (U = mean utility vs the
                             samples, n = number of spells), F = the current expected spellF1 (Dinkelbach
                             iteration for the ratio); the base candidate (index 0) keeps the window unless
                             another beats it by more than `margin`.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])


def runs_of(blocks):
    return [(int(f), int(l) + 1) for _, f, l in blocks]


def spells(codes, runs):
    """-> (start, stop, team) int arrays of the H / A runs inside each graded run, sorted."""
    S, E, K = [], [], []
    for a, b in runs:
        w = codes[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.concatenate(([0], br))
        en = np.concatenate((br, [len(w)]))
        lab = w[st]
        keep = lab <= A
        S.append(st[keep] + a)
        E.append(en[keep] + a)
        K.append(lab[keep])
    if not S:
        z = np.zeros(0, np.int64)
        return z, z, z
    return (np.concatenate(S).astype(np.int64), np.concatenate(E).astype(np.int64),
            np.concatenate(K).astype(np.int64))


def _pairs(cs, ce, ss, se):
    """All (i, j) with candidate spell i overlapping sample spell j (both lists sorted, disjoint)."""
    lo = np.searchsorted(se, cs, "right")
    hi = np.searchsorted(ss, ce, "left")
    w = hi - lo
    if not len(w) or w.max() <= 0:
        z = np.zeros(0, np.int64)
        return z, z
    I, J = [], []
    idx = np.arange(len(cs))
    for off in range(int(w.max())):
        m = w > off
        I.append(idx[m])
        J.append(lo[m] + off)
    return np.concatenate(I), np.concatenate(J)


def spell_util(cand, samp, ious=IOUS):
    """cand, samp: spells() triples. -> util per candidate spell (fraction of IoU thresholds matched)."""
    cs, ce, ck = cand
    ss, se, sk = samp
    u = np.zeros(len(cs))
    if not len(cs) or not len(ss):
        return u
    i, j = _pairs(cs, ce, ss, se)
    if not len(i):
        return u
    m = ck[i] == sk[j]
    i, j = i[m], j[m]
    inter = np.minimum(ce[i], se[j]) - np.maximum(cs[i], ss[j])
    union = np.maximum(ce[i], se[j]) - np.minimum(cs[i], ss[j])
    iou = inter / union.astype(float)
    bc = np.zeros(len(cs))
    np.maximum.at(bc, i, iou)
    bs = np.zeros(len(ss))
    np.maximum.at(bs, j, iou)
    mut = (iou >= bc[i]) & (iou >= bs[j])
    best = np.zeros(len(cs))
    np.maximum.at(best, i[mut], iou[mut])
    return (best[:, None] >= ious[None, :]).mean(1)


# ------------------------------------------------------------------------------------------------
# posterior sampling

def _logsumexp(x, axis):
    m = x.max(axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def hmm_matrices(rho, r, p_ha=0.5):
    """Log transition matrix of 3 labels x r sub-states. rho = exit probability per step from the last
    sub-state of a label; exits go to the other alive team with p_ha (from H / A) or to D, and from D
    equally to H / A."""
    S = 3 * r
    T = np.zeros((S, S))
    for lab in range(3):
        for q in range(r):
            s = lab * r + q
            if q < r - 1:
                T[s, s + 1] = 1.0
                continue
            T[s, s] = 1.0 - rho
            if lab == D:
                dest = {H: 0.5, A: 0.5}
            else:
                dest = {1 - lab: p_ha, D: 1.0 - p_ha}
            for l2, p in dest.items():
                T[s, l2 * r] += rho * p
    with np.errstate(divide="ignore"):
        return np.log(T)


def ffbs(P, a, b, alpha, rho, r, stride, n, rng, p_ha=0.5):
    """n posterior label paths (n, b - a) int8 for the frames [a, b) of one graded run."""
    L = np.log(np.clip(P[a:b], 1e-6, 1.0))
    m = (b - a + stride - 1) // stride
    pad = m * stride - (b - a)
    if pad:
        L = np.concatenate([L, np.repeat(L[-1:], pad, 0)])
    E = alpha * L.reshape(m, stride, 3).sum(1)                   # (m, 3) tempered log-likelihood
    E = np.repeat(E, r, axis=1)                                  # sub-states share their label's emission
    logT = hmm_matrices(rho, r, p_ha)
    S = 3 * r
    init = np.full(S, -np.inf)
    init[np.arange(3) * r] = np.log(1.0 / 3)
    F = np.empty((m, S))
    F[0] = init + E[0]
    for t in range(1, m):
        F[t] = _logsumexp(F[t - 1][:, None] + logT, 0) + E[t]
    z = np.empty((n, m), np.int64)
    u = rng.random((n, m))
    w = F[-1] - F[-1].max()
    p = np.exp(w)
    c = np.cumsum(p / p.sum())
    z[:, -1] = np.minimum(np.searchsorted(c, u[:, -1]), S - 1)
    for t in range(m - 2, -1, -1):
        w = F[t][None, :] + logT[:, z[:, t + 1]].T               # (n, S)
        w = w - w.max(1, keepdims=True)
        p = np.exp(w)
        c = np.cumsum(p, 1)
        z[:, t] = np.minimum((c < (u[:, t] * c[:, -1])[:, None]).sum(1), S - 1)
    lab = (z // r).astype(np.int8)
    return np.repeat(lab, stride, axis=1)[:, :b - a]


def sample_paths(P, blocks, n_frames, alpha, rho, r, stride, n, seed, p_ha=0.5):
    rng = np.random.default_rng(seed)
    out = np.full((n, n_frames), X, np.int8)
    for a, b in runs_of(blocks):
        out[:, a:b] = ffbs(P, a, b, alpha, rho, r, stride, n, rng, p_ha)
    return out


# ------------------------------------------------------------------------------------------------
# window selection

def windows(cands, runs):
    """Window id per frame (-1 outside windows): maximal stretches inside a run with at least one
    candidate not D. -> (wid array, number of windows)."""
    n = len(cands[0])
    alive = np.zeros(n, bool)
    for c in cands:
        alive |= (c == H) | (c == A)
    wid = np.full(n, -1, np.int64)
    k = 0
    for a, b in runs:
        w = alive[a:b]
        if not w.any():
            continue
        e = np.diff(np.concatenate(([0], w.astype(np.int8), [0])))
        st = np.flatnonzero(e == 1)
        en = np.flatnonzero(e == -1)
        for s0, e0 in zip(st, en):
            wid[a + s0:a + e0] = k
            k += 1
    return wid, k


def utilities(cands, samples, runs, weights=None):
    """-> list over candidates of (spells, mean utility per spell vs the samples), mean n_gt."""
    samp_sp = [spells(s, runs) for s in samples]
    w = np.ones(len(samples)) / len(samples) if weights is None else np.asarray(weights) / np.sum(weights)
    out = []
    for c in cands:
        sp = spells(c, runs)
        U = np.zeros(len(sp[0]))
        for wk, ss in zip(w, samp_sp):
            U += wk * spell_util(sp, ss)
        out.append((sp, U))
    ng = float(np.sum([wk * len(ss[0]) for wk, ss in zip(w, samp_sp)]))
    return out, ng


def select_windows(cands, samples, runs, margin=0.0, iters=4, weights=None, util=None):
    """cands[0] is the base. -> (decoded codes, info dict)."""
    util, ng = util or utilities(cands, samples, runs, weights)
    wid, nw = windows(cands, runs)
    K = len(cands)
    Uw = np.zeros((K, nw))
    Nw = np.zeros((K, nw))
    for k, (sp, U) in enumerate(util):
        w = wid[sp[0]]
        np.add.at(Uw[k], w, U)
        np.add.at(Nw[k], w, 1)
    choice = np.zeros(nw, np.int64)
    F = 2 * Uw[0].sum() / max(Nw[0].sum() + ng, 1e-9)
    F0 = F
    for _ in range(iters):
        G = 2 * Uw - F * Nw
        G[1:] -= margin
        choice = np.argmax(G, 0)
        Usum = Uw[choice, np.arange(nw)].sum()
        Nsum = Nw[choice, np.arange(nw)].sum()
        Fn = 2 * Usum / max(Nsum + ng, 1e-9)
        if abs(Fn - F) < 1e-6:
            F = Fn
            break
        F = Fn
    out = cands[0].copy()
    for k in range(1, K):
        sel = np.isin(wid, np.flatnonzero(choice == k))
        out[sel] = cands[k][sel]
    return out, dict(F0=F0, F=F, n_windows=nw, n_switched=int((choice != 0).sum()),
                     choice_hist=np.bincount(choice, minlength=K))


def expected_f1(codes, samples, runs, weights=None):
    sp = spells(codes, runs)
    w = np.ones(len(samples)) / len(samples) if weights is None else np.asarray(weights) / np.sum(weights)
    f = 0.0
    for wk, s in zip(w, samples):
        ss = spells(s, runs)
        f += wk * 2 * spell_util(sp, ss).sum() / max(len(sp[0]) + len(ss[0]), 1)
    return f
