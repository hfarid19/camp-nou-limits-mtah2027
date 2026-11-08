"""semimarkov_lattice: explicit-duration (semi-Markov) decoding over a touch lattice.

See README.md for the method, references, variants and scores. In short:

  evidence   learned_dfl_evidence's per-frame LightGBM dead / home / hand-over probabilities
             (lde_base.py is a verbatim copy of that candidate's pipeline; the model files are
             copies of its models, so its leave-one-game-out fold models apply unchanged);
  lattice    candidate boundary frames: hand-over detector peaks, dead-model flips, raw-ball gap
             edges, ball kinematic events (direction / speed changes), nearest-player distance
             minima, ball-out flips, plus a coarse fallback grid;
  decoder    exact segmental DP (Viterbi + forward-backward) over segments between lattice
             points with 4 states H, A, D|H, D|A; a segment scores its summed tempered frame
             evidence + a learned duration log-pmf (possession spell / stoppage, KDE from the DFL
             tracking flags, one table per leave-one-game-out fold, durations.npz loaded through
             dfl_folds.resolve) + learned transition log-probs (hand-over vs stoppage, restart
             by the other team vs the same) + a boundary term from the hand-over detector;
  proba      segmental forward-backward posterior marginals P(H), P(A), P(D) per frame.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # candidates/ (dfl_folds)
import possession_pipeline as pp  # noqa: E402
import lde_base as L              # noqa: E402
import dfl_folds                  # noqa: E402

H, A, D = 0, 1, 2

CFG = dict(
    tau=0.3,           # per-frame evidence temperature (LDE: 0.15)
    w_dur=1.0,         # weight of the duration log-pmf
    w_trans=1.0,       # weight of the transition log-probs
    bonus_poss=1.0,    # per-segment log bonus for a possession spell
    bonus_dead=0.0,    # per-segment log bonus for a stoppage
    gamma=0.6,         # hand-over detector log-odds on H<->A boundaries
    prior_k=0.0,       # subtract prior_k * log class prior per frame (posterior -> likelihood)
    self_pen=12.0,     # cost of a same-state continuation (only for segments longer than DMAX)
    dmax_poss_s=150.0,
    dmax_dead_s=400.0,
    grid_s=0.5,        # fallback lattice grid
    use_touch=1,       # add touch-lattice points
    pc_thr=0.05,       # hand-over peak threshold
    min_sep=2,         # lattice points closer than this (frames) are merged
    refine=0,          # 0 none; 1 frame-level evidence refine of every boundary within +-refine_w
    refine_w=12,
    post_lde=1,        # LDE's D-edge snaps + hand-over peak snap after decoding
    snap_peak_s=0.5,
    merge_min_s=0.0,   # merge_short_v2 with this minimum (0 = off)
    post_tau=0.2,      # extra temperature on the forward-backward scores (marginals only)
    contrast_beta=1.0, # < 0: LDE's team logit as is (gate 1.25); >= 0: recompute the contrast
    contrast_hw=100,
    contrast_gate=0.25,
    delta=0.0,         # weight of the dead-logit jump on alive <-> D boundaries
    delta_w=12,
    delta_clip=6.0,
    dur_kind=0,        # 0 learned KDE log-pmf; 1 geometric with the same mean (ablation = HMM)
    ctx=0,             # 1: context-dependent possession durations (10-state model)
)


# ------------------------------------------------------------------------------------------
# evidence (learned_dfl_evidence's models and evidence construction)
# ------------------------------------------------------------------------------------------

def evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    p_dead, p_home, aux = L.frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    aux["long_gap"] = L.long_gap_gated(players, ball, fps, L.CFG["gap_s"], L.CFG["gap_speed_ms"])
    c = L.CFG
    n = len(ball)
    q = np.clip(p_home, 1e-6, 1 - 1e-6)
    l6 = np.log(q / (1 - q))
    cc = l6 - pp.moving_mean(l6, int(c["contrast_hw"]))
    cc = np.sign(cc) * np.maximum(np.abs(cc) - c["contrast_gate"], 0.0)
    lh = np.clip(l6 + c["contrast_beta"] * cc, -np.log(999.0), np.log(999.0))
    pdd = np.clip(p_dead, 1e-3, 1 - 1e-3)
    ld = np.log(pdd / (1 - pdd)) + c["w_gap"] * aux["long_gap"]
    F, nm = aux["F"], aux["names"]
    col = {k: i for i, k in enumerate(nm)}
    raw_seen = F[:, col["ball_seen"]] > 0.5
    out = np.nan_to_num(F[:, col["ball_out"]], nan=0.0) > 0.5
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    tag_team = np.zeros(n)
    if len(tags):
        tt, td = L.tag_priors(tags, names, n, fps, raw_seen, c["goal_win_s"])
        tag_team = c["w_tag_team"] * tt
        lh = lh + tag_team
        ld = ld + c["w_tag_goal"] * td
    kick = L.kickoff_restarts(tags, blocks, ball, fps, c["prekick_s"])
    ev = dict(lh=lh, l6=l6, tag_team=tag_team, ld=ld, pc=aux["p_change"], raw_seen=raw_seen, out=out,
              d_home=F[:, col["H_d1"]].astype(np.float64), d_away=F[:, col["A_d1"]].astype(np.float64),
              ball=np.asarray(ball, np.float64), change_prior=np.float64(aux["change_prior"]),
              kick=np.array(sorted(kick.items()), np.int64).reshape(-1, 2))
    return ev


def team_logit(ev, cfg):
    """LDE's local-contrast team logit (default) or a re-parameterised one from the raw logit."""
    if cfg["contrast_beta"] < 0 or "l6" not in ev:
        return ev["lh"]
    l6 = ev["l6"]
    cc = l6 - pp.moving_mean(l6, int(cfg["contrast_hw"]))
    cc = np.sign(cc) * np.maximum(np.abs(cc) - cfg["contrast_gate"], 0.0)
    return (np.clip(l6 + cfg["contrast_beta"] * cc, -np.log(999.0), np.log(999.0))
            + ev.get("tag_team", 0.0))


def frame_scores(ev, cfg):
    ph = np.clip(1 / (1 + np.exp(-team_logit(ev, cfg))), 1e-3, 1 - 1e-3)
    pd = np.clip(1 / (1 + np.exp(-ev["ld"])), 1e-3, 1 - 1e-3)
    E = np.stack([np.log(ph) + np.log(1 - pd), np.log(1 - ph) + np.log(1 - pd), np.log(pd)], 1)
    if cfg["prior_k"]:
        E = E - cfg["prior_k"] * np.log(np.array([0.276, 0.276, 0.448]))
    return E


# ------------------------------------------------------------------------------------------
# touch lattice
# ------------------------------------------------------------------------------------------

def local_max(x, k):
    n = len(x)
    ok = np.isfinite(x)
    xm = np.where(ok, x, -np.inf)
    m = np.ones(n, bool) & ok
    for s in range(1, k + 1):
        m[s:] &= xm[s:] >= xm[:-s]
        m[:-s] &= xm[:-s] > xm[s:]
    return m


def touch_points(ev, lo, hi, fps, cfg):
    """Candidate boundary frames in [lo, hi) (absolute), each a likely touch / state change."""
    sl = slice(lo, hi)
    pts = []
    pc = ev["pc"][sl]
    pts.append(np.where(local_max(pc, 3) & (pc > cfg["pc_thr"]))[0])
    ld = pp.moving_mean(ev["ld"][sl], 2)
    pts.append(np.where(np.diff(np.sign(ld)) != 0)[0] + 1)                 # dead-model flips
    dld = np.abs(np.r_[0, ld[4:] - ld[:-4], 0, 0, 0])
    pts.append(np.where(local_max(dld, 5) & (dld > 1.0))[0])
    seen = ev["raw_seen"][sl]
    s_, e_ = pp.runs_of(~seen)
    keep = (e_ - s_) >= 5
    pts.append(s_[keep])
    pts.append(e_[keep])
    out = ev["out"][sl].astype(np.int8)
    pts.append(np.where(np.diff(out) != 0)[0] + 1)
    b = ev["ball"][sl]
    bx = pp.moving_mean(b[:, 0], 1)
    by = pp.moving_mean(b[:, 1], 1)
    bx[~seen] = np.nan
    by[~seen] = np.nan
    k = 3
    vx0 = np.full(len(bx), np.nan)
    vy0 = np.full(len(bx), np.nan)
    vx1 = np.full(len(bx), np.nan)
    vy1 = np.full(len(bx), np.nan)
    vx0[k:] = (bx[k:] - bx[:-k]) * fps / k
    vy0[k:] = (by[k:] - by[:-k]) * fps / k
    vx1[:-k] = (bx[k:] - bx[:-k]) * fps / k
    vy1[:-k] = (by[k:] - by[:-k]) * fps / k
    s0 = np.hypot(vx0, vy0)
    s1 = np.hypot(vx1, vy1)
    dv = np.hypot(vx1 - vx0, vy1 - vy0)            # velocity change magnitude (m/s)
    kin = np.where(np.isfinite(dv) & (np.maximum(s0, s1) > 1.5), dv, np.nan)
    pts.append(np.where(local_max(kin, 4) & (kin > 3.0))[0])
    dmin = np.fmin(ev["d_home"][sl], ev["d_away"][sl])
    nd = -dmin
    pts.append(np.where(local_max(nd, 5) & (dmin < 2.0))[0])
    near_h = ev["d_home"][sl] < ev["d_away"][sl]
    pts.append(np.where(np.diff(near_h.astype(np.int8)) != 0)[0] + 1)
    p = np.concatenate([np.asarray(x, np.int64) for x in pts]) + lo
    return p[(p > lo) & (p < hi)]


def lattice(ev, first, last, fps, cfg):
    lo, hi = first, last + 1
    g = max(1, int(round(cfg["grid_s"] * fps)))
    pts = [np.arange(lo, hi, g), np.array([lo, hi])]
    if cfg["use_touch"]:
        pts.append(touch_points(ev, lo, hi, fps, cfg))
    p = np.unique(np.concatenate(pts))
    if cfg["min_sep"] > 1 and len(p) > 2:            # thin: keep a point only if far enough
        keep = [p[0]]
        for x in p[1:-1]:
            if x - keep[-1] >= cfg["min_sep"]:
                keep.append(x)
        if hi - keep[-1] < cfg["min_sep"] and len(keep) > 1:
            keep.pop()
        keep.append(hi)
        p = np.array(keep)
    return p


# ------------------------------------------------------------------------------------------
# semi-Markov DP
# ------------------------------------------------------------------------------------------

def load_table():
    z = np.load(dfl_folds.resolve(os.path.join(HERE, "durations.npz")))
    return {k: z[k] for k in z.files}


def build_model(tab, cfg):
    """State space, duration log-pmf rows, transition log-probs, initial log-probs.

    ctx 0: 4 states H, A, D|H, D|A (one possession duration pmf).
    ctx 1: 10 states: possession (team, entry D / hand-over, exit D / hand-over) with the duration
           pmf of that context (a possession entered from a restart is rarely < 1 s, one that ends
           in a stoppage often is), plus D|H, D|A. Returns (cls, rows, T, logfirst, ho) where ho
           marks H<->A transitions (the hand-over detector boundary term applies to them)."""
    fps = int(tab["fps"])
    lp = np.asarray(tab["logpmf"], float)
    dmax = lp.shape[1]
    ph = float(tab["p_handover"])
    pr = float(tab["p_restart_other"])
    pf = np.asarray(tab["p_first"], float)
    pf_poss, pf_dead = (pf[0] + pf[1]) / 2, pf[2] / 2
    wd, wt = cfg["w_dur"], cfg["w_trans"]
    if not cfg["ctx"]:
        cls = np.array([H, A, D, D])
        rows = [lp[0], lp[0], lp[1], lp[1]]
        T = np.full((4, 4), -np.inf)
        T[H, A] = T[A, H] = np.log(ph)
        T[H, 2] = T[A, 3] = np.log(1 - ph)
        T[2, H] = T[3, A] = np.log(1 - pr)
        T[2, A] = T[3, H] = np.log(pr)
        first = np.log([pf_poss, pf_poss, pf_dead, pf_dead])
    else:
        lc = np.asarray(tab["logpmf_ctx"], float)
        pxd = np.asarray(tab["p_exit_dead"], float)      # P(exit D | entry D), P(exit D | entry X)
        cls, rows = [], []
        for t in (H, A):
            for e in (0, 1):
                for x in (0, 1):
                    cls.append(t)
                    rows.append(lc[2 * e + x])
        cls += [D, D]
        rows += [lp[1], lp[1]]
        cls = np.array(cls)
        S = len(cls)
        T = np.full((S, S), -np.inf)
        first = np.full(S, -np.inf)
        pex = lambda e, x: np.log(pxd[e] if x == 0 else 1 - pxd[e])   # noqa: E731
        for t in (H, A):
            for e in (0, 1):
                for x in (0, 1):
                    i = t * 4 + e * 2 + x
                    if x == 0:
                        T[i, 8 + t] = 0.0
                    else:
                        for x2 in (0, 1):
                            T[i, (1 - t) * 4 + 2 + x2] = pex(1, x2)
            for t2 in (H, A):
                for x2 in (0, 1):
                    T[8 + t, t2 * 4 + x2] = np.log(1 - pr if t2 == t else pr) + pex(0, x2)
                    first[t2 * 4 + x2] = np.log(pf_poss) + pex(0, x2)
            first[8 + t] = np.log(pf_dead)
    if cfg["dur_kind"] == 1:                         # ablation: geometric durations, same mean
        dd = np.arange(1, dmax + 1)
        geo = []
        for r in rows:
            mu = float((np.exp(r) * dd).sum())
            geo.append(-np.log(mu) + (dd - 1) * np.log1p(-1.0 / mu))
        rows = geo
    S = len(cls)
    durlp = np.full((S, dmax + 1), -np.inf)
    for s_ in range(S):
        if cls[s_] == D:
            durlp[s_, 1:] = wd * rows[s_] + cfg["bonus_dead"]
            durlp[s_, int(cfg["dmax_dead_s"] * fps) + 1:] = -np.inf
        else:
            durlp[s_, 1:] = wd * rows[s_] + cfg["bonus_poss"]
            durlp[s_, int(cfg["dmax_poss_s"] * fps) + 1:] = -np.inf
    T = wt * T
    for s_ in range(S):
        T[s_, s_] = -cfg["self_pen"]
    ho = (cls[:, None] < 2) & (cls[None, :] < 2) & (cls[:, None] != cls[None, :])
    Dlim = min(dmax, int(max(cfg["dmax_poss_s"], cfg["dmax_dead_s"]) * fps))
    return cls, durlp, T, wt * first, ho, Dlim


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def semimarkov(E, fr, ev_c, tab, cfg, want_post=True, ev_d=None):
    """E (T, 3) frame scores of one block, fr (L,) lattice cut frames relative to the block
    (fr[0] = 0, fr[-1] = T). Returns the MAP class per lattice cell and (optionally) the
    posterior class marginals per cell."""
    cls, durlp, T, logfirst, ho, Dlim = build_model(tab, cfg)
    S = len(cls)
    tau = cfg["tau"]
    C = np.vstack([np.zeros((1, 3)), np.cumsum(E, 0)]) * tau
    CP = C[fr][:, cls]                                   # (L, S)
    Lp = len(fr)
    bnd = np.zeros((Lp, S, S))
    if ev_c is not None and cfg["gamma"]:
        bnd += cfg["gamma"] * ev_c[:, None, None] * ho[None]
    if ev_d is not None and cfg["delta"]:
        into_d = (cls[:, None] < 2) & (cls[None, :] == D)
        out_d = (cls[:, None] == D) & (cls[None, :] < 2)
        bnd += cfg["delta"] * (ev_d[:, None, None] * into_d[None] - ev_d[:, None, None] * out_d[None])
    starts_lo = np.searchsorted(fr, fr - Dlim)           # earliest start point per end point
    ar = np.arange(S)

    def seg_scores(j, idx):
        d = fr[j] - fr[idx]
        return (CP[j][None, :] - CP[idx]) + durlp[:, d].T  # (W, S)

    # ---- Viterbi
    V = np.full((Lp, S), -np.inf)
    Ein = np.full((Lp, S), -np.inf)
    Ein[0] = logfirst
    bp_i = np.zeros((Lp, S), np.int64)
    bp_s = np.zeros((Lp, S), np.int8)
    for j in range(1, Lp):
        idx = np.arange(starts_lo[j], j)
        sc = Ein[idx] + seg_scores(j, idx)
        a = sc.argmax(0)
        V[j] = sc[a, ar]
        bp_i[j] = idx[a]
        if j < Lp - 1:
            cand = V[j][:, None] + T + bnd[j]            # (from, to)
            bp_s[j] = cand.argmax(0)
            Ein[j] = cand.max(0)
    state = np.zeros(Lp - 1, np.int8)
    j, s = Lp - 1, int(V[Lp - 1].argmax())
    while j > 0:
        i = bp_i[j, s]
        state[i:j] = s
        s_prev = bp_s[i, s] if i > 0 else s
        j, s = i, int(s_prev)
    if not want_post:
        return cls[state], None
    # ---- forward-backward (scores optionally flattened by post_tau)
    pt = cfg["post_tau"]
    Af = np.full((Lp, S), -np.inf)
    Fin = np.full((Lp, S), -np.inf)
    Fin[0] = logfirst * pt
    Tp, bndp = T * pt, bnd * pt
    for j in range(1, Lp):
        idx = np.arange(starts_lo[j], j)
        Af[j] = _lse(Fin[idx] + pt * seg_scores(j, idx), 0)
        if j < Lp - 1:
            Fin[j] = _lse(Af[j][:, None] + Tp + bndp[j], 0)
    Z = _lse(Af[Lp - 1], 0)
    ends_hi = np.searchsorted(fr, fr + Dlim, side="right")
    Bout = np.full((Lp, S), -np.inf)
    Bout[Lp - 1] = 0.0
    Bin = np.full((Lp, S), -np.inf)
    for i in range(Lp - 2, -1, -1):
        jdx = np.arange(i + 1, min(ends_hi[i], Lp))
        d = fr[jdx] - fr[i]
        sc = pt * ((CP[jdx] - CP[i][None, :]) + durlp[:, d].T)
        Bin[i] = _lse(sc + Bout[jdx], 0)
        if i > 0:
            Bout[i] = _lse(Tp + bndp[i] + Bin[i][None, :], 1)
    diff = np.zeros((Lp, 3))
    onehot = np.eye(3)[cls]
    for j in range(1, Lp):
        idx = np.arange(starts_lo[j], j)
        w = np.exp(Fin[idx] + pt * seg_scores(j, idx) + Bout[j][None, :] - Z)   # (W, S)
        wc = w @ onehot
        np.add.at(diff, idx, wc)
        diff[j] -= wc.sum(0)
    post = np.cumsum(diff, 0)[:-1]
    post = np.clip(post, 0, None)
    post /= np.maximum(post.sum(1, keepdims=True), 1e-12)
    return cls[state], post


# ------------------------------------------------------------------------------------------
# decoding
# ------------------------------------------------------------------------------------------

def refine_boundaries(path, E, w):
    """Move each boundary to the frame within +-w maximising the evidence of the two sides."""
    path = path.copy()
    segs = pp.segments(path)
    C = np.vstack([np.zeros((1, 3)), np.cumsum(E, 0)])
    for i in range(1, len(segs)):
        s0, _, a = segs[i - 1]
        s, e, b = segs[i]
        lo, hi = max(s0 + 1, s - w), min(e - 1, s + w)
        if hi <= lo:
            continue
        cand = np.arange(lo, hi + 1)
        gain = C[cand, a] - C[cand, b]
        t = int(cand[gain.argmax()])
        if t < s:
            path[t:s] = b
        elif t > s:
            path[s:t] = a
    return path


def decode_ev(ev, blocks, n, fps, tab, cfg=None, want_post=True):
    c = dict(CFG, **(cfg or {}))
    E = frame_scores(ev, c)
    pc = np.clip(ev["pc"], 1e-3, 1 - 1e-3)
    prior = float(ev["change_prior"])
    cl = np.log(pc / (1 - pc)) - np.log(prior / (1 - prior))
    cl = np.maximum.reduce([np.r_[cl[k:], np.full(k, -9.0)] if k >= 0 else np.r_[np.full(-k, -9.0), cl[:k]]
                            for k in (-2, -1, 0, 1, 2)])
    w = int(c["delta_w"])                      # dead-logit jump across a boundary (clipped)
    ldc = np.vstack([[0.0], np.cumsum(ev["ld"])[:, None]])[:, 0]
    t = np.arange(n)
    after = (ldc[np.minimum(t + w, n)] - ldc[t]) / np.maximum(np.minimum(t + w, n) - t, 1)
    before = (ldc[t] - ldc[np.maximum(t - w, 0)]) / np.maximum(t - np.maximum(t - w, 0), 1)
    jd = np.clip(after - before, -c["delta_clip"], c["delta_clip"])
    pred = np.full(n, 3, np.int8)
    proba = np.full((n, 3), 1.0 / 3)
    kick = {int(a): int(b) for a, b in ev["kick"]}
    out = ev["out"]
    gaps_all = L.long_gaps(ev["raw_seen"], fps, L.CFG["gap_s"])
    P = dict(pp.PARAMS, min_poss_s=c["merge_min_s"], min_dead_s=c["merge_min_s"])
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        fr = lattice(ev, first, last, fps, c)
        st, post = semimarkov(E[seg], fr - first, cl[np.minimum(fr, n - 1)], tab, c, want_post,
                              jd[np.minimum(fr, n - 1)])
        rel = fr - first
        path = np.empty(last + 1 - first, np.int8)
        cls = st
        for k in range(len(rel) - 1):
            path[rel[k]:rel[k + 1]] = cls[k]
            if post is not None:
                proba[first + rel[k]:first + rel[k + 1]] = post[k]
        if c["refine"]:
            path = refine_boundaries(path, E[seg], int(c["refine_w"]))
        if c["post_lde"]:
            g = [(s0 - first, e0 - first) for s0, e0 in gaps_all if s0 >= first and e0 <= last + 1]
            path = L.snap_gap_ends(path, g, fps, L.CFG["snap_win_s"])
            path = L.snap_dead_start_back(path, out[seg], fps, L.CFG["out_lookback_s"], L.CFG["keep_s"])
        if c["merge_min_s"] > 0:
            path = L.merge_short_v2(path, P, fps)
        if c["post_lde"] and c["snap_peak_s"] > 0:
            path = L.snap_to_peak(path, ev["pc"][seg], fps, c["snap_peak_s"])
        rs = kick.get(first)
        if rs is not None:
            path[:rs - first] = D
            proba[first:rs] = (0.0, 0.0, 1.0)
        pred[seg] = path
    return pred, proba


_CACHE = {}


def _ev_cached(args):
    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(args[2]).tobytes()).hexdigest() + str(args[4])
    if _CACHE.get("key") != key:
        _CACHE.clear()
        _CACHE["key"] = key
        _CACHE["ev"] = evidence(*args)
        _CACHE["tab"] = load_table()
    return _CACHE["ev"], _CACHE["tab"]


def _run(args):
    key = "res"
    if key not in _CACHE or _CACHE.get("res_key") != _CACHE.get("key"):
        ev, tab = _ev_cached(args)
        _CACHE["res"] = decode_ev(ev, args[4], len(args[2]), args[8], tab)
        _CACHE["res_key"] = _CACHE["key"]
    return _CACHE["res"]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    args = (players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    _ev_cached(args)
    return _run(args)[0].copy()


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    args = (players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    _ev_cached(args)
    return _run(args)[1].copy()
