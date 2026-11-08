"""DFL LOGO selection helper for temporal_tcn (replicates bench --dfl offline: same caches, fold
model gi for game gi). Final numbers always come from bench.py itself.

  python oof.py probs <folds_dir> <kind> <tag>      -> OOF P/pc per game into probs_<tag>/
  python oof.py sweep <tag> key=v1,v2 key2=...      -> grid over decoder CFG, spellF1 per point
"""
import itertools
import os
import sys
import time

import numpy as np

REPO = "/path/to/more-than-a-hack-2027"
sys.path.insert(0, os.path.join(REPO, "candidates"))
sys.path.insert(0, os.path.join(REPO, "candidates", "temporal_tcn"))
os.chdir(REPO)
import bench  # noqa: E402
import pipeline as tp  # noqa: E402
import tcn_np  # noqa: E402
import features as fe  # noqa: E402
import possession_metrics as pm  # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

OUT = os.path.dirname(os.path.abspath(__file__))
FCACHE = os.path.join(OUT, "fcache")
MATCHES = sorted(f[:-4] for f in os.listdir(bench.DFL_DIR) if f.endswith(".npz"))


def game(gi):
    m = MATCHES[gi]
    slots, slot_team, ball, gt, blocks, sign = bench.dfl_prepared(m, 2027)
    os.makedirs(FCACHE, exist_ok=True)
    fp = os.path.join(FCACHE, m + ".npy")
    if not os.path.isfile(fp):
        fs = fe.frame_sign(len(ball), sign, blocks)
        F, names, kinds = fe.compute(slots, slot_team, ball, fs, bench.tsd.PITCH, bench.tsd.FPS)
        np.save(fp, F)
    F = np.load(fp)
    return slots, slot_team, ball, gt, blocks, sign, F


def probs(folds_dir, kind, tag):
    od = os.path.join(OUT, "probs_" + tag)
    os.makedirs(od, exist_ok=True)
    names, kinds = None, None
    for gi in range(len(MATCHES)):
        if os.path.isfile(os.path.join(od, MATCHES[gi] + ".npz")) or \
                not os.path.isfile(os.path.join(folds_dir, "fold_{}_{}.npz".format(kind, gi))):
            continue
        t0 = time.time()
        slots, slot_team, ball, gt, blocks, sign, F = game(gi)
        W = np.load(os.path.join(folds_dir, "fold_{}_{}.npz".format(kind, gi)))
        W = {k: W[k] for k in W.files}
        names = [str(s) for s in W["names"]]
        kinds = [str(s) for s in W["kinds"]]
        P, pc = tp.tcn_probs(F, names, kinds, blocks, W)
        np.savez(os.path.join(od, MATCHES[gi] + ".npz"), P=P, pc=pc, prior=W["change_prior"], names=W["names"])
        print(gi, MATCHES[gi], "{:.0f}s".format(time.time() - t0), flush=True)


def score(tag, cfg, verbose=False):
    od = os.path.join(OUT, "probs_" + tag)
    rows = []
    for gi in range(len(MATCHES)):
        if not os.path.isfile(os.path.join(od, MATCHES[gi] + ".npz")):
            continue
        slots, slot_team, ball, gt, blocks, sign, F = game(gi)
        z = np.load(os.path.join(od, MATCHES[gi] + ".npz"))
        names = [str(x) for x in z["names"]]
        pred = tp.decode_probs(z["P"], z["pc"], F, names, float(z["prior"]), slots, ball, blocks,
                               [], ("Home", "Away"), 25, cfg)
        r = pm.evaluate(gt, pred, bootstrap_n=0)
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
        if verbose:
            print("  {} spell {:.4f} macro {:.4f} {}/{}".format(MATCHES[gi], *rows[-1]))
    a = np.array(rows)
    return a[:, 0].mean(), a[:, 1].mean(), int(a[:, 2].sum()), int(a[:, 3].sum())


if __name__ == "__main__":
    if sys.argv[1] == "probs":
        probs(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "sweep":
        tag = sys.argv[2]
        grid = []
        for a in sys.argv[3:]:
            k, v = a.split("=")
            grid.append([(k, float(x)) for x in v.split(",")])
        for combo in itertools.product(*grid):
            cfg = dict(combo)
            s, m, npred, ngt = score(tag, cfg, verbose=len(sys.argv) == 3)
            print("{}  spellF1 {:.4f} macroF1 {:.4f} poss {}/{}".format(cfg, s, m, npred, ngt),
                  flush=True)


def lde_probs():
    """LDE's evidence with its verified LOGO fold models, in this file's P format (probs_lde)."""
    import lightgbm as lgb
    S = WORK
    L = os.path.join(REPO, "candidates", "learned_dfl_evidence")
    od = os.path.join(OUT, "probs_lde")
    os.makedirs(od, exist_ok=True)
    full = {w: lgb.Booster(model_file=os.path.join(L, "model_{}.txt".format(w))).num_trees()
            for w in ("d", "ha", "c")}
    src = {"d": S + "/verify_r2d/folds", "ha": S + "/verify_r1/folds", "c": S + "/verify_r2d/folds"}
    import json
    meta = json.load(open(os.path.join(L, "model_meta.json")))
    names, kinds = meta["names"], meta["kinds"]
    perm, signs = fe.swap_index(names, kinds)
    for gi in range(len(MATCHES)):
        slots, slot_team, ball, gt, blocks, sign, F = game(gi)
        m = {w: lgb.Booster(model_file=os.path.join(src[w], "fold_{}_{}.txt".format(w, gi)))
             for w in src}
        Fs = F[:, perm] * signs
        pd_ = m["d"].predict(F, num_iteration=full["d"], num_threads=1)
        ph = 0.5 * (m["ha"].predict(F, num_iteration=full["ha"], num_threads=1)
                    + 1 - m["ha"].predict(Fs, num_iteration=full["ha"], num_threads=1))
        pc = 0.5 * (m["c"].predict(F, num_iteration=full["c"], num_threads=1)
                    + m["c"].predict(Fs, num_iteration=full["c"], num_threads=1))
        P = np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], 1)
        np.savez(os.path.join(od, MATCHES[gi] + ".npz"), P=P, pc=pc,
                 prior=np.array(meta.get("change_prior", 0.04)), names=np.array(names))
        print(gi, flush=True)


if __name__ == "__main__" and sys.argv[1] == "lde":
    lde_probs()
