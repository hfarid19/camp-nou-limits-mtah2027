"""Shared pieces of the touch-ceiling simulation: paths, the 48 selection games, the targets, a fast
spellF1 that is bit-identical to possession_metrics.spell_scores, and v6's PFF out-of-fold codes.

Nothing here reads evaluation/gt, evaluation/reports or the 16 PFF holdout games.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND, ROOT,
                os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
PFF = os.path.join(WORK, "pff_wc2022")
ARCH_NN = os.path.join(WORK, "arch", "nn_pff")
OUT = os.path.join(WORK, "arch", "touch_ceiling")
H, A, D, X = pm.H, pm.A, pm.D, pm.X
FPS = 25
MEMBERS = ("t5c", "m5c", "radar_pff")          # v6 tie-break order (probs/nn_pff/<name>, out-of-fold)
V6_RULE = (1.0, 1.25)                           # merge_short_v2 1 s, control.fold 1.25 s
TARGETS = ("raw", "T0.5", "T1")                 # fine3 = mean of the three


def selection_games():
    f = json.load(open(os.path.join(ARCH_NN, "selection_folds.json")))["folds"]
    return sorted(g for fold in f for g in fold)


_T = {}


def targets():
    """{game: (blocks, {raw, T0.5, T1, ...})} -- vote_offline's cache (selection games only)."""
    if not _T:
        _T.update(pickle.load(open(os.path.join(ARCH_NN, "vote_cache", "pff_targets.pkl"), "rb")))
        assert sorted(_T) == selection_games()
    return _T


# ------------------------------------------------------------------------------------------------
# fast spellF1 (same definition as pm.spell_scores on the abstention-filled codes)
# ------------------------------------------------------------------------------------------------

def spells(codes, runs):
    """(start, stop, code) arrays of maximal H / A runs inside each graded run."""
    S, E, K = [], [], []
    for a, b in runs:
        w = codes[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.concatenate(([0], br))
        en = np.concatenate((br, [len(w)]))
        k = w[st]
        m = (k == H) | (k == A)
        S.append(st[m] + a)
        E.append(en[m] + a)
        K.append(k[m])
    return np.concatenate(S), np.concatenate(E), np.concatenate(K)


def gt_spells(gt):
    runs = pm.valid_runs(gt)
    return runs, spells(gt, runs)


def spell_f1(gsp, pred):
    """Mean F1 over pm.SPELL_IOUS; gsp = gt_spells(gt). pred must have no X on graded frames
    (fill_abstentions is applied if it has)."""
    runs, (gs, ge, gk) = gsp
    if (pred == X).any():
        pred = pm.fill_abstentions(pred, runs)
    ps, pe, pk = spells(pred, runs)
    ng, npd = len(gs), len(ps)
    # every same-team overlapping pair: pred spells are sorted and disjoint, so for gt spell i the
    # overlapping preds are j in [lo_i, hi_i)
    lo = np.searchsorted(pe, gs, side="right")
    hi = np.searchsorted(ps, ge, side="left")
    cnt = np.maximum(hi - lo, 0)
    if cnt.sum() == 0:
        return 0.0
    gi = np.repeat(np.arange(ng), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pj = np.repeat(lo, cnt) + off
    keep = gk[gi] == pk[pj]
    gi, pj = gi[keep], pj[keep]
    inter = np.minimum(ge[gi], pe[pj]) - np.maximum(gs[gi], ps[pj])
    union = np.maximum(ge[gi], pe[pj]) - np.minimum(gs[gi], ps[pj])
    iou = inter / union.astype(float)
    order = np.lexsort((pj, gi, -iou))
    iou, gi, pj = iou[order].tolist(), gi[order].tolist(), pj[order].tolist()
    f1s = []
    for th in pm.SPELL_IOUS:
        mg, mp = set(), set()
        for v, i, j in zip(iou, gi, pj):
            if v < th:
                break
            if i in mg or j in mp:
                continue
            mg.add(i)
            mp.add(j)
        tp = len(mg)
        p = tp / float(npd) if npd else 0.0
        r = tp / float(ng) if ng else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


# ------------------------------------------------------------------------------------------------
# v6 out-of-fold codes on PFF (shipped rule), cached
# ------------------------------------------------------------------------------------------------

def member_codes(game):
    return [np.load(os.path.join(WORK, "probs", "nn_pff", m, "pff2_{}_2027_codes.npy".format(game))).astype(np.int8)
            for m in MEMBERS]


def v6_codes(game):
    d = os.path.join(OUT, "v6_codes")
    f = os.path.join(d, game + ".npy")
    if not os.path.isfile(f):
        if os.path.join(CAND, "nn_pff") not in sys.path:
            sys.path.append(os.path.join(CAND, "nn_pff"))   # appended: nn_pff/queue.py must not shadow queue
        import vote_offline
        os.makedirs(d, exist_ok=True)
        blocks = targets()[game][0]
        v = vote_offline.vote(member_codes(game), blocks, *V6_RULE)
        np.save(f, np.asarray(v, np.int8))
    return np.load(f)
