"""Training sequences for nn_pff (no torch here): ds_pff feature files -> per-half 5 fps sequences.

Each ds file (make_dataset.py) is one degradation of one game, one stride-5 phase. Per file we
derive, at its idx, the class target y (0 H, 1 A, 2 D, 3 ungraded) and the two soft boundary
targets of handover_mstcn_v2/prep_data.py (Gaussian, sigma 2.5 frames, around every direct H<->A
change and every alive<->dead change of the full-resolution codes), from the raw file's codes.
Targets are cached in <ds>/../tgt/<file> (small).

Fold specs (game-grouped; a model never sees the game it scores):
    pff<k>   all training data except the 12 games of PFF selection fold k (+ all 7 DFL)
    dfl<g>   all training data except DFL game g (index in the sorted DFL list, as bench.py)
    full     all training data (48 PFF selection games + 7 DFL)
The 16 PFF holdout games are never in the dataset (make_dataset.py refuses them; asserted again).
Round-4 opt-in (the holdout was scored once for v6 and is spent): with NN_PFF_ALLOW_HOLDOUT=1 a
dataset folder may hold the 16 holdout games (make_dataset.py --allow-holdout) and the `full` and
dfl<g> specs then train on all 64 PFF games; pff<k> specs are refused in that mode (no PFF
out-of-fold score exists for such models). The trainers record the mode in every weight file
(`pff_holdout_trained`, and `train_games` lists the games). Without the variable nothing changes.
SkillCorner A-League training games (SKC-*, $MTAH_WORK/skc_aleague) may be linked into a dataset
folder; they are in every spec's training set (never scored on these benches); the 5 SKC holdout
games are refused; the same for FOOTPASS (FP-*, $MTAH_WORK/footpass) and its 13 holdout games.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path[:0] = [os.path.join(ROOT, "evaluation"), CAND, HERE]
WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SIGMA = 2.5
ARCH = os.path.join(WORK, "arch", "nn_pff")


def holdout():
    return set(json.load(open(os.path.join(ARCH, "holdout_games.json")))["games"])


def allow_holdout():
    """Explicit opt-in (env NN_PFF_ALLOW_HOLDOUT=1): train on the 16 spent PFF holdout games too."""
    return os.environ.get("NN_PFF_ALLOW_HOLDOUT", "") == "1"


def skc_holdout():
    """Holdout games of the extra datasets (SkillCorner A-League, FOOTPASS): never trained on."""
    out = set()
    for d in ("skc_aleague", "footpass"):
        p = os.path.join(WORK, d, "holdout_games.json")
        if os.path.isfile(p):
            out |= set(json.load(open(p))["holdout"])
    return out


def pff_folds():
    return json.load(open(os.path.join(ARCH, "selection_folds.json")))["folds"]


def dfl_matches():
    return sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "idsse", "npz")) if f.endswith(".npz"))


def game_of(fname):
    """'PFF-3812_41.npz' -> 'PFF-3812'; 'DFL-MAT-J03WMX_41.npz' -> 'DFL-MAT-J03WMX'."""
    return os.path.basename(fname)[:-4].rsplit("_", 1)[0]


def excluded(spec):
    """Game names left out of training by a fold spec."""
    if spec == "full":
        return set()
    if spec.startswith("pff"):
        return {"PFF-" + g for g in pff_folds()[int(spec[3:])]}
    if spec.startswith("dfl"):
        return {dfl_matches()[int(spec[3:])]}
    raise ValueError(spec)


def change_positions(codes):
    a, b = codes[:-1], codes[1:]
    ha = np.where((a != b) & (a < 2) & (b < 2))[0] + 0.5
    dd = np.where((a != b) & (((a == 2) & (b < 2)) | ((a < 2) & (b == 2))))[0] + 0.5
    return ha, dd


def change_target(codes, hw=4):
    """1 within +-hw frames of a direct H <-> A change (= learned_dfl_evidence make_dataset)."""
    c = np.zeros(len(codes), bool)
    t = np.where((codes[1:] != codes[:-1]) & (codes[1:] < 2) & (codes[:-1] < 2))[0] + 1
    for k in range(-hw, hw + 1):
        c[np.clip(t + k, 0, len(c) - 1)] = True
    return c.astype(np.float32)


def soft_target(frames, pos, sigma=SIGMA):
    if len(pos) == 0:
        return np.zeros(len(frames), np.float32)
    j = np.clip(np.searchsorted(pos, frames), 1, max(len(pos) - 1, 1))
    if len(pos) == 1:
        d = np.abs(frames - pos[0])
    else:
        d = np.minimum(np.abs(frames - pos[j - 1]), np.abs(frames - pos[j]))
    return np.exp(-d ** 2 / (2 * sigma ** 2)).astype(np.float32)


def targets(ds_path, raw_path, cache_dir, fold_s=0.0):
    tag = "" if fold_s <= 0 else "_f{:g}".format(fold_s)
    cp = os.path.join(cache_dir, os.path.basename(ds_path)[:-4] + tag + ".npz")
    if os.path.isfile(cp):
        z = np.load(cp)
        if "yc" in z.files:
            return z["y"], z["bha"], z["bd"], z["yc"], z["idx"], z["half"]
    z = np.load(ds_path)
    idx = z["idx"].astype(np.int64)
    rz = np.load(raw_path)
    codes = rz["codes"].astype(np.int8)
    half = np.zeros(len(codes), np.int8)
    for h, a, b in rz["blocks"]:
        half[int(a):int(b) + 1] = int(h)
    assert np.array_equal(codes[idx], z["y"]), ds_path
    if fold_s > 0:
        import control
        import possession_metrics as pm
        codes = control.fold(codes, fold_s, pm.valid_runs(codes), 25)
    ha, dd = change_positions(codes)
    fr = idx.astype(np.float64)
    y, bha, bd = codes[idx], soft_target(fr, ha), soft_target(fr, dd)
    yc = change_target(codes)[idx]
    half = half[idx]
    assert (half > 0).all(), ds_path
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cp[:-4] + ".tmp{}.npz".format(os.getpid())
    np.savez(tmp, y=y, bha=bha, bd=bd, yc=yc, idx=idx, half=half)
    os.replace(tmp, cp)
    return y, bha, bd, yc, idx, half


def load(ds_dir, spec, dfl_fold_s=0.0, pff_fold_s=0.0, max_files=None):
    """-> (seqs, names, kinds, train_games); seqs = list of (game, X float32 (T, K), y int64 (T,),
    b float32 (3, T): soft H<->A boundary, soft alive<->dead boundary, hand-over target yc), one per
    half of each training file."""
    hold = holdout()
    if allow_holdout():
        if spec.startswith("pff"):
            raise ValueError("NN_PFF_ALLOW_HOLDOUT=1: only the full and dfl<g> specs (no PFF folds)")
        hold = set()
    shold = skc_holdout()
    ex = excluded(spec)
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(ds_dir)), "raw")
    cache = os.path.join(os.path.dirname(os.path.abspath(ds_dir)), "tgt")
    seqs, games = [], set()
    names = kinds = None
    files = sorted(glob.glob(os.path.join(ds_dir, "*_*.npz")))
    if max_files:
        files = files[:max_files]
    for p in files:
        g = game_of(p)
        assert g[4:] not in hold or not g.startswith("PFF-"), "holdout game in the dataset: " + p
        assert g not in shold, "SkillCorner / FOOTPASS holdout game in the dataset: " + p
        if g in ex:
            continue
        fs = dfl_fold_s if g.startswith("DFL") else pff_fold_s
        y, bha, bd, yc, idx, half = targets(p, os.path.join(raw_dir, os.path.basename(p)), cache, fs)
        z = np.load(p)
        names, kinds = [str(s) for s in z["names"]], [str(s) for s in z["kinds"]]
        X = z["X"]
        # sequences: break at a half change / hole in the stride-5 grid (= prep_data.py)
        brk = np.concatenate(([True], (np.diff(idx) != 5) | (np.diff(half) != 0)))
        sid = np.cumsum(brk) - 1
        for s in np.unique(sid):
            m = sid == s
            if m.sum() < 64:
                continue
            seqs.append((g, X[m].astype(np.float32), y[m].astype(np.int64), np.stack([bha[m], bd[m], yc[m]], 0)))
        games.add(g)
    return seqs, names, kinds, sorted(games)
