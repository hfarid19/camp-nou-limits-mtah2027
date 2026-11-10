"""nn_pff training sequences + the R5-4 auxiliary targets (make_aux.py), aligned row by row.

load(ds_dir, spec) = nn_pff/data.load (same files, folds, holdout refusals, sequence splits) and each
sequence also carries aux = float32 (10, T): rows 0-7 the regression targets A (NaN = none), row 8 the
restart team (0 H / 1 A / -1 none), row 9 the degraded-ball-missing flag M. Nothing changes for nn_pff.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
if os.path.join(CAND, "nn_pff") not in sys.path:
    sys.path.append(os.path.join(CAND, "nn_pff"))   # appended: nn_pff/queue.py would shadow the stdlib
import data as D  # noqa: E402

ANTI = (0, 2)          # aux regression channels negated by the team swap
NREG = 8


def load(ds_dir, spec, dfl_fold_s=0.0, pff_fold_s=0.0, aux_dir=None):
    hold = D.holdout()
    if D.allow_holdout():
        if spec.startswith("pff"):
            raise ValueError("NN_PFF_ALLOW_HOLDOUT=1: only the full and dfl<g> specs (no PFF folds)")
        hold = set()
    shold = D.skc_holdout()
    ex = D.excluded(spec)
    base = os.path.dirname(os.path.abspath(ds_dir))
    raw_dir, cache = os.path.join(base, "raw"), os.path.join(base, "tgt")
    aux_dir = aux_dir or os.path.join(base, "aux")
    seqs, games = [], set()
    names = kinds = None
    for p in sorted(glob.glob(os.path.join(ds_dir, "*_*.npz"))):
        g = D.game_of(p)
        assert g[4:] not in hold or not g.startswith("PFF-"), "holdout game in the dataset: " + p
        assert g not in shold, "SkillCorner / FOOTPASS holdout game in the dataset: " + p
        if g in ex:
            continue
        fs = dfl_fold_s if g.startswith("DFL") else pff_fold_s
        y, bha, bd, yc, idx, half = D.targets(p, os.path.join(raw_dir, os.path.basename(p)), cache, fs)
        z = np.load(p)
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
        X = z["X"]
        az = np.load(os.path.join(aux_dir, os.path.basename(p)))
        assert np.array_equal(az["idx"], idx), p
        aux = np.concatenate([az["A"].T, az["R"][None].astype(np.float32), az["M"][None].astype(np.float32)], 0)
        brk = np.concatenate(([True], (np.diff(idx) != 5) | (np.diff(half) != 0)))
        sid = np.cumsum(brk) - 1
        for s in np.unique(sid):
            m = sid == s
            if m.sum() < 64:
                continue
            seqs.append((g, X[m].astype(np.float32), y[m].astype(np.int64),
                         np.stack([bha[m], bd[m], yc[m]], 0), aux[:, m].astype(np.float32)))
        games.add(g)
    return seqs, names, kinds, sorted(games)
