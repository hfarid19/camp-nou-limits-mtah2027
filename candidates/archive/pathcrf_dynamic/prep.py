#!/usr/bin/env python3
"""Build the CRF training sequences: out-of-fold LDE evidence on fresh DFL degradations.

    python -I candidates/pathcrf_dynamic/prep.py --out DIR --seeds 61,62 [--jobs 2]

For every DFL game gi (index in sorted(os.listdir(idsse npz))) and every seed: one domain-
randomised degradation (dfl_degrade.degrade_dfl, i.e. LDE's make_dataset recipe, seeds that LDE
and the bench never used), then LDE's three boosters evaluated with the LEAVE-ONE-GAME-OUT fold
models of game gi (so the evidence on game gi never comes from a model that saw gi, exactly as on
the bench), cut to the full models' number of trees as bench.py does. Saved per game x seed, at
full 25 fps: the signals of crf_features.signals(), tracking-flag labels and graded blocks.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lightgbm as lgb      # noqa: E402
import lde_base             # noqa: E402
import crf_features as cf   # noqa: E402
import dfl_degrade as dd    # noqa: E402
import features as fe       # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

SCRATCH = WORK
DFL_DIR = os.path.join(SCRATCH, "idsse", "npz")
FOLDS = os.path.join(SCRATCH, "folds_verified.json")


class Cut(object):
    def __init__(self, fold, rounds):
        self.fold, self.rounds = fold, rounds

    def predict(self, X, **kw):
        kw["num_iteration"] = self.rounds
        return self.fold.predict(X, **kw)


def fold_models(gi):
    import hashlib
    folds = json.load(open(FOLDS))
    m = {}
    for w in ("d", "ha", "c"):
        path = os.path.join(HERE, "model_{}.txt".format(w))
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        folder, kind = folds[next(k for k in folds if sha.startswith(k))]
        full = lgb.Booster(model_file=path)
        m[w] = Cut(lgb.Booster(model_file=os.path.join(folder, "fold_{}_{}.txt".format(kind, gi))),
                   full.num_trees())
    m["meta"] = json.load(open(os.path.join(HERE, "model_meta.json")))
    return m


def job(args):
    gi, match, seed, out_dir = args
    out = os.path.join(out_dir, "{}_{}.npz".format(match, seed))
    if os.path.isfile(out):
        return out
    t0 = time.time()
    g = dd.load_game(os.path.join(DFL_DIR, match + ".npz"))
    slots, slot_team, ball, codes, blocks, sign, info = dd.degrade_dfl(g, seed, dd.game_key(match))
    m = fold_models(gi)
    p_dead, p_home, aux = lde_base.frame_probs(slots, slot_team, ball, sign, blocks, dd.PITCH,
                                               dd.FPS, models=m)
    lg = lde_base.long_gap_gated(slots, ball, dd.FPS, lde_base.CFG["gap_s"],
                                 lde_base.CFG["gap_speed_ms"])
    sig = cf.signals(p_dead, p_home, aux["p_change"], aux["change_prior"], aux["F"], aux["names"],
                     lg, lde_base.CFG)
    np.savez_compressed(out, codes=codes, blocks=np.array(blocks), gi=gi, **sig)
    print("{} seed {} gi {}: {} frames, {:.0f}s {}".format(match, seed, gi, len(codes),
                                                         time.time() - t0, info), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="61,62")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    matches = sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))
    jobs = [(gi, m, int(s), a.out) for s in a.seeds.split(",") for gi, m in enumerate(matches)]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for r in pool.imap_unordered(job, jobs):
            pass


if __name__ == "__main__":
    main()
