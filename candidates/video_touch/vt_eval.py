"""Touch-spotting evaluation: temporal NMS + one-to-one greedy matching at a frame tolerance."""
import numpy as np


def nms_peaks(p, thr, radius):
    """Local maxima of p above thr, suppressing others within +-radius frames (greedy by score)."""
    idx = np.where(p >= thr)[0]
    if len(idx) == 0:
        return np.zeros(0, int)
    order = idx[np.argsort(-p[idx], kind="stable")]
    taken = np.zeros(len(p), bool)
    keep = []
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        taken[max(0, i - radius):i + radius + 1] = True
    return np.sort(np.array(keep, int))


def match(pred, gt, tol):
    """Greedy one-to-one matching, nearest first. Returns (tp, pairs list of (pi, gi))."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    if len(pred) == 0 or len(gt) == 0:
        return 0, []
    cand = []
    j0 = 0
    for pi, t in enumerate(pred):
        lo, hi = np.searchsorted(gt, t - tol), np.searchsorted(gt, t + tol, side="right")
        for gi in range(lo, hi):
            cand.append((abs(int(gt[gi]) - int(t)), pi, gi))
    cand.sort()
    up, ug, pairs = set(), set(), []
    for d, pi, gi in cand:
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); pairs.append((pi, gi))
    return len(pairs), pairs


def pr(pred, gt, tol):
    tp, _ = match(pred, gt, tol)
    P = tp / max(len(pred), 1)
    R = tp / max(len(gt), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def average_precision(p, gt, tol, radius, mask=None):
    """AP of NMS peaks (thr 0) ranked by score; mask restricts the frames considered."""
    q = p.copy()
    if mask is not None:
        q = np.where(mask, q, -1)
    peaks = nms_peaks(q, 0.0, radius)
    sc = q[peaks]
    order = np.argsort(-sc, kind="stable")
    peaks, sc = peaks[order], sc[order]
    # incremental greedy matching by score order (standard spotting AP)
    gt = np.asarray(gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(peaks))
    for k, t in enumerate(peaks):
        lo, hi = np.searchsorted(gt, t - tol), np.searchsorted(gt, t + tol, side="right")
        best, bd = -1, 1e9
        for gi in range(lo, hi):
            if not used[gi] and abs(gt[gi] - t) < bd:
                best, bd = gi, abs(gt[gi] - t)
        if best >= 0:
            used[best] = True; tp[k] = 1
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(tp) + 1)
    rec = ctp / max(len(gt), 1)
    ap = 0.0
    prev_r = 0.0
    # monotone precision envelope
    env = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    for r, pp in zip(rec, env):
        ap += (r - prev_r) * pp
        prev_r = r
    return ap
