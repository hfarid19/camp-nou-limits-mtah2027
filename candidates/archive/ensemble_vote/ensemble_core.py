"""Vote rules shared by the live pipelines and the offline fit (tuning/fit.py).

hard_vote   every member's own per-frame label votes (weight w_m, default 1); the label with the most
            votes wins. Fixed tie-break: each member's vote also carries a tiny bonus eps * (K - r)
            for its rank r in the member list (ordered by DFL LOGO spellF1, best first), so a tie goes
            to the label backed by the best-ranked member among the tied voters. Optional smoothing
            afterwards: learned_dfl_evidence's merge_short_v2 (shortest offending segment first,
            H/A/D below min_poss_s / min_dead_s merged into their neighbours).
soft_vote   weighted pooling of members' P(H, A, D): arithmetic mean, or geometric (log-linear) mean
            renormalised; then ONE decode with learned_dfl_evidence's decode() (3-state Viterbi with
            its switch / dead costs, gap-end and out snaps, merge_short_v2, hand-over peak snap with
            LDE's model_c, kick-off D) on p_dead = P(D), p_home = P(H) / (P(H) + P(A)).
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
EPS_RANK = 1e-3


def block_mask(blocks, n):
    m = np.zeros(n, bool)
    for _, first, last in blocks:
        m[first:last + 1] = True
    return m


def vote_shares(codes_list, weights=None):
    K = len(codes_list)
    w = np.ones(K) if weights is None else np.asarray(weights, float)
    n = len(codes_list[0])
    V = np.zeros((n, 3))
    for k, c in enumerate(codes_list):
        c = np.asarray(c)
        for lab in (H, A, D):
            V[:, lab] += w[k] * (c == lab)
    return V


def hard_vote(codes_list, blocks, lde, fps=25, weights=None, smooth=None):
    """codes_list ordered best member first. smooth: None or dict(min_poss_s=, min_dead_s=)."""
    K = len(codes_list)
    w = np.ones(K) if weights is None else np.asarray(weights, float)
    wt = w + EPS_RANK * (K - np.arange(K))
    V = vote_shares(codes_list, wt)
    lab = np.argmax(V, 1).astype(np.int8)
    n = len(lab)
    out = np.full(n, X, np.int8)
    for _, first, last in blocks:
        path = lab[first:last + 1].copy()
        if smooth:
            P = dict(min_poss_s=smooth["min_poss_s"], min_dead_s=smooth["min_dead_s"])
            path = lde.merge_short_v2(path, P, fps)
        out[first:last + 1] = path
    return out


def vote_proba(codes_list, blocks, weights=None):
    V = vote_shares(codes_list, weights)
    s = V.sum(1, keepdims=True)
    P = np.where(s > 0, V / np.maximum(s, 1e-12), 1.0 / 3)
    return P


def pool(probas, weights=None, mode="arith"):
    K = len(probas)
    w = np.ones(K) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    if mode == "arith":
        P = sum(w[k] * np.asarray(probas[k], np.float64) for k in range(K))
    elif mode == "geo":
        L = sum(w[k] * np.log(np.clip(np.asarray(probas[k], np.float64), 1e-4, 1.0))
                for k in range(K))
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
    else:
        raise ValueError(mode)
    P = np.nan_to_num(P, nan=1.0 / 3)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-12)


SIDE_NAMES = ["ball_seen", "ball_out", "H_d1", "A_d1"]


def soft_decode(P, side, blocks, n, fps, lde, cfg=None):
    """One LDE decode of the pooled probabilities. side: learned_dfl_evidence's side information
    (members._run_lde). tags are not passed again: members' probabilities already carry them."""
    F = np.stack([np.asarray(side[k], np.float64) for k in SIDE_NAMES], 1)
    kr = {int(a): int(b) for a, b in zip(side["kick_first"], side["kick_frame"])}
    aux = dict(F=F, names=list(SIDE_NAMES), p_change=np.asarray(side["p_change"], np.float64),
               change_prior=0.04, long_gap=np.asarray(side["long_gap"], np.float64),
               kick_restarts=kr)
    p_dead = np.clip(P[:, D], 1e-6, 1 - 1e-6)
    alive = np.maximum(P[:, H] + P[:, A], 1e-12)
    p_home = np.clip(P[:, H] / alive, 1e-6, 1 - 1e-6)
    return lde.decode(p_dead, p_home, aux, blocks, n, fps, (), ("Home", "Away"), cfg=cfg)
