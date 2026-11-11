"""nn_pff: MS-TCN (ASRF: class + boundary branches) trained on 48 PFF World Cup 2022 games (touch
labels) + the 7 DFL games (raw flag), degraded with randomised degrade_v2 / degrade_v4; see
README.md. Network and decoder code = handover_mstcn_v3 (tcn_np.py byte-identical, decoder
unchanged); weights from train_mstcn.py.

predict(...)       -> (N,) codes 0 H / 1 A / 2 D / 3 X
predict_proba(...) -> (N, 3) P(H), P(A), P(D): final-stage class posteriors (team-swap
                      symmetrised, the five 5 fps phases interleaved back to 25 fps)

Weights: mstcn.npz, loaded through pff_folds.resolve (bench --dfl: the DFL leave-one-game-out
model; run_pff.py: the PFF out-of-fold model). Inference is plain numpy, deterministic.
"""

import hashlib
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))      # candidates/ (dfl_folds)
sys.path.insert(0, HERE)
import possession_pipeline as pp  # noqa: E402
import features as fe             # noqa: E402
import tcn_np                     # noqa: E402
import pff_folds                  # noqa: E402
import chunked_weights            # noqa: E402  (candidates/chunked_weights.py)

warnings.filterwarnings("ignore")
H, A, D = 0, 1, 2
MODEL_PATH = os.path.join(HERE, "mstcn.npz")
STRIDE = 5

CFG = dict(
    decoder="asrf",      # "viterbi" (dynamic switch costs from the boundary branch) or "asrf"
    tau=1.0,             # temperature on the per-frame class log-probabilities
    sw0=6.0,             # H<->A base switch cost (log units)
    gamma_ha=1.0,        # weight of logit(b_ha) on the H<->A switch cost
    dc0=6.0,             # alive<->D base cost
    gamma_d=1.0,         # weight of logit(b_d) on the alive<->D cost
    c_min=0.0,
    b_prior=0.05,        # logit(b) is taken relative to this prior
    # asrf decoder
    b_thr=0.2,           # boundary peak threshold (H<->A, or max of both when b_thr_d < 0)
    b_thr_d=-1.0,        # alive<->dead boundary peak threshold (< 0: one threshold on max)
    peak_win_s=0.4,      # a peak is the max within +-this window
    flip_thr=0.5,        # boundary-forced hand-over (0 = off): min peak height at both cuts
    flip_q=0.3,          # ... min share of the other team's alive mass in the segment
    flip_len_s=2.0,      # ... max segment length
    logmean=0.0,         # 1: label a segment by its mean log prob instead of mean prob
    b_smooth=0.0,        # moving-mean half width (frames) on the boundary probs before peaks
    contrast_beta=1.25,   # LDE local-contrast team logit (0 = off)
    contrast_hw=100,
    contrast_gate=1.25,
    post_d=0.0,          # 1: LDE/v2 dead-edge post chain (gap-end snap, out look-back)
    gap_s=3.0,
    snap_win_s=1.0,
    out_lookback_s=6.0,
    keep_s=0.5,
    min_poss_s=0.0,      # merge_short_v2 minima (0 = off)
    min_dead_s=0.0,
)

_CACHE = {}


def _key(players, ball, blocks, path):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ball).tobytes())
    h.update(np.ascontiguousarray(players[::50]).tobytes())
    h.update(repr(blocks).encode())
    h.update(path.encode())
    return h.hexdigest()


def tcn_outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps):
    """-> pc (N, 3) class probs, pb (N, 2) boundary probs (H<->A, alive<->dead), F, names."""
    path = pff_folds.resolve(MODEL_PATH)
    key = _key(players, ball, blocks, path)
    if key in _CACHE:
        return _CACHE[key]
    model = tcn_np.Model(chunked_weights.source(path))      # the file, or its re-joined parts if stored split
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    assert names == model.names, "feature list differs from the trained model"
    perm, signs = fe.swap_index(names, kinds)
    n = len(ball)
    pc = np.full((n, 3), 1.0 / 3.0)
    pb = np.zeros((n, 2))
    for _, first, last in blocks:
        for r in range(STRIDE):
            fr = np.arange(first + r, last + 1, STRIDE)
            if len(fr) == 0:
                continue
            pc[fr], pb[fr] = model.predict(F[fr].astype(np.float64), perm, signs)
    _CACHE.clear()
    _CACHE[key] = (pc, pb, F, names)
    return _CACHE[key]


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def viterbi_dyn(E, c_ha, c_d):
    """3-state Viterbi (H, A, D), per-frame H<->A cost c_ha[t] and alive<->D cost c_d[t]
    (cost of entering the state at frame t)."""
    T = len(E)
    back = np.zeros((T, 3), np.int8)
    e = E.tolist()
    cha = c_ha.tolist()
    cd = c_d.tolist()
    sH, sA, sD = e[0]
    for t in range(1, T):
        c, d = cha[t], cd[t]
        eh, ea, ed = e[t]
        a, b, x = sH, sA - c, sD - d
        if a >= b and a >= x:
            nH, bh = a, 0
        elif b >= x:
            nH, bh = b, 1
        else:
            nH, bh = x, 2
        a, b, x = sH - c, sA, sD - d
        if b >= a and b >= x:
            nA, ba = b, 1
        elif a >= x:
            nA, ba = a, 0
        else:
            nA, ba = x, 2
        a, b, x = sH - d, sA - d, sD
        if x >= a and x >= b:
            nD, bd = x, 2
        elif a >= b:
            nD, bd = a, 0
        else:
            nD, bd = b, 1
        back[t] = (bh, ba, bd)
        sH, sA, sD = nH + eh, nA + ea, nD + ed
    path = np.empty(T, np.int8)
    path[-1] = int(np.argmax([sH, sA, sD]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def peaks(b, thr, w):
    """Local maxima of b above thr (max within +-w frames; first of a plateau)."""
    n = len(b)
    if n == 0:
        return np.zeros(0, int)
    from numpy.lib.stride_tricks import sliding_window_view
    bp = np.concatenate([np.full(w, -1.0), b, np.full(w, -1.0)])
    mx = sliding_window_view(bp, 2 * w + 1).max(1)
    cand = np.where((b >= thr) & (b >= mx))[0]
    if len(cand) == 0:
        return cand
    keep = [cand[0]]
    for t in cand[1:]:
        if t - keep[-1] > w:
            keep.append(t)
    return np.array(keep)


def asrf_segment(pc, pb, thr, w, thr_d=None, logmean=False, flip_thr=None, flip_q=0.3,
                 flip_len=50):
    """ASRF refinement: cut at boundary peaks, label each segment by its mean class prob."""
    T = len(pc)
    if thr_d is None:
        cuts = peaks(pb.max(1), thr, w)
    else:
        cuts = np.union1d(peaks(pb[:, 0], thr, w), peaks(pb[:, 1], thr_d, w))
    edges = np.concatenate(([0], cuts[(cuts > 0) & (cuts < T)], [T]))
    edges = np.unique(edges)
    path = np.empty(T, np.int8)
    labs, means = [], []
    for s, e in zip(edges[:-1], edges[1:]):
        if logmean:
            m = np.log(np.clip(pc[s:e], 1e-6, 1)).mean(0)
        else:
            m = pc[s:e].mean(0)
        labs.append(int(np.argmax(m)))
        means.append(m)
        path[s:e] = labs[-1]
    if flip_thr is not None:
        # boundary-forced hand-over: a short alive segment cut out by two strong H<->A peaks but
        # labelled like both neighbours flips to the other team if that team keeps >= flip_q of
        # the alive mass there
        hb = pb[:, 0]
        for i in range(1, len(labs) - 1):
            s, e = edges[i], edges[i + 1]
            if labs[i] == D or labs[i - 1] != labs[i] or labs[i + 1] != labs[i]:
                continue
            if e - s > flip_len or min(hb[s], hb[e]) < flip_thr:
                continue
            m = means[i]
            q = m[1 - labs[i]] / max(m[0] + m[1], 1e-9)
            if q >= flip_q:
                path[s:e] = 1 - labs[i]
    return path


def snap_gap_ends(path, gaps, fps, win_s):
    """(from learned_dfl_evidence / v2) Each D end -> end of the nearest long raw-ball gap within
    win_s, either way (if that gap ends after the D spell starts)."""
    path = path.copy()
    if not gaps:
        return path
    w = int(win_s * fps)
    ge = np.array([g[1] for g in gaps])
    for s, e, lab in pp.segments(path):
        if lab != D or e >= len(path):
            continue
        j = int(np.argmin(np.abs(ge - e)))
        if abs(ge[j] - e) <= w and ge[j] > s:
            t = int(ge[j])
            if t > e:
                path[e:t] = D
            elif t < e:
                path[t:e] = path[e]
    return path


def snap_dead_start_back(path, out, fps, lookback_s, keep_s):
    """(from learned_dfl_evidence / v2) alive -> D change: start D at the first ball-out frame in
    the previous lookback_s, staying keep_s after the possession's start."""
    path = path.copy()
    segs = pp.segments(path)
    lb, keep = int(lookback_s * fps), int(keep_s * fps)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, _, l = segs[i]
        if l != D or l0 == D:
            continue
        lo = max(s0 + keep, s - lb)
        if lo >= s:
            continue
        hits = np.where(out[lo:s + fps])[0]
        if len(hits):
            t = lo + int(hits[0])
            if t < s:
                path[t:s] = D
    return path


def contrast_team(pc, hw, beta, gate):
    """LDE's local-contrast team logit applied to the TCN's P(H | alive)."""
    alive = pc[:, 0] + pc[:, 1]
    q = np.clip(pc[:, 0] / np.maximum(alive, 1e-9), 1e-6, 1 - 1e-6)
    l = np.log(q / (1 - q))
    cc = l - pp.moving_mean(l, int(hw))
    cc = np.sign(cc) * np.maximum(np.abs(cc) - gate, 0.0)
    q2 = 1 / (1 + np.exp(-(l + beta * cc)))
    out = pc.copy()
    out[:, 0] = alive * q2
    out[:, 1] = alive * (1 - q2)
    return out


def merge_short_v2(path, min_poss, min_dead):
    """Shortest offending segment first, recomputed after every merge (from learned_dfl_evidence)."""
    min_len = {H: min_poss, A: min_poss, D: min_dead}
    path = path.copy()
    while True:
        segs = pp.segments(path)
        if len(segs) == 1:
            break
        cand = sorted((e - s, i) for i, (s, e, lab) in enumerate(segs) if e - s < min_len[lab])
        done = False
        for _, i in cand:
            s, e, lab = segs[i]
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                path[s:e] = prev
            elif lab == D or prev is None or nxt is None:
                continue
            elif prev != D and nxt != D:
                path[s:e] = nxt
            else:
                path[s:e] = nxt if prev == D else prev
            done = True
            break
        if not done:
            break
    return path


def decode(pc, pb, blocks, n, fps, cfg=None, aux=None):
    """aux: dict(raw_seen=(N,) bool, out=(N,) bool) for the dead-edge post chain."""
    c = dict(CFG, **(cfg or {}))
    pred = np.full(n, 3, np.int8)
    if c["b_smooth"] > 0:
        pb = np.stack([pp.moving_mean(pb[:, k], int(c["b_smooth"])) for k in (0, 1)], 1)
    if c["contrast_beta"] > 0:
        pc = contrast_team(pc, c["contrast_hw"], c["contrast_beta"], c["contrast_gate"])
    lp = np.log(np.clip(pc, 1e-6, 1.0))
    lprior = _logit(np.array(c["b_prior"]))
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        if c["decoder"] == "asrf":
            path = asrf_segment(pc[seg], pb[seg], c["b_thr"], max(1, int(c["peak_win_s"] * fps)),
                                None if c["b_thr_d"] < 0 else c["b_thr_d"], c["logmean"] > 0,
                                None if c["flip_thr"] <= 0 else c["flip_thr"], c["flip_q"],
                                int(c["flip_len_s"] * fps))
        else:
            E = c["tau"] * lp[seg]
            c_ha = np.maximum(c["c_min"], c["sw0"] - c["gamma_ha"] * (_logit(pb[seg, 0]) - lprior))
            c_d = np.maximum(c["c_min"], c["dc0"] - c["gamma_d"] * (_logit(pb[seg, 1]) - lprior))
            path = viterbi_dyn(E, c_ha, c_d)
        if c["post_d"] > 0 and aux is not None:
            gaps = [(s0 - first, e0 - first) for s0, e0 in zip(*pp.runs_of(~aux["raw_seen"]))
                    if e0 - s0 > c["gap_s"] * fps and s0 >= first and e0 <= last + 1]
            path = snap_gap_ends(path, gaps, fps, c["snap_win_s"])
            path = snap_dead_start_back(path, aux["out"][seg], fps, c["out_lookback_s"], c["keep_s"])
        if c["min_poss_s"] > 0 or c["min_dead_s"] > 0:
            path = merge_short_v2(path, int(c["min_poss_s"] * fps), int(c["min_dead_s"] * fps))
        pred[seg] = path
    return pred


def post_aux(F, names):
    col = {k: i for i, k in enumerate(names)}
    raw_seen = F[:, col["ball_seen"]] > 0.5
    out = np.nan_to_num(F[:, col["ball_out"]], nan=0.0) > 0.5
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    return dict(raw_seen=raw_seen, out=out)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pc, pb, F, fnames = tcn_outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return decode(pc, pb, blocks, len(ball), fps, aux=post_aux(F, fnames))


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pc, _, _, _ = tcn_outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return pc.astype(np.float64)


def boundary_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    """(N, 2) boundary probabilities (H<->A, alive<->dead) of the last boundary stage (offline
    decoder tuning; not used by the bench)."""
    _, pb, _, _ = tcn_outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    return pb.astype(np.float64)
