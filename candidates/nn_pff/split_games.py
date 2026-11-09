#!/usr/bin/env python3
"""Fix the PFF game split BEFORE any training (run once, 2026-11-08).

    python -I candidates/nn_pff/split_games.py

16 of the 64 PFF games are the final holdout (never trained, tuned or selected on; scored once at
the very end for the final pick). The other 48 are cut into 4 game-grouped folds of 12 for
selection (each selection game is scored by a model that never saw it). numpy default_rng(SEED)
permutation of the sorted game ids: the first 16 are the holdout, the next 48 in order of the
permutation go to folds 0..3 (12 each).
"""
import glob
import json
import os

import numpy as np

WORK = os.environ.get("MTAH_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "work")
SEED = 20261108
OUT = os.path.join(WORK, "arch", "nn_pff")


def main():
    games = sorted(os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(WORK, "pff_wc2022", "npz", "PFF-*.npz")))
    assert len(games) == 64, len(games)
    perm = np.random.default_rng(SEED).permutation(len(games))
    hold = sorted(games[i] for i in perm[:16])
    sel = [games[i] for i in perm[16:]]
    folds = [sorted(sel[12 * k:12 * (k + 1)]) for k in range(4)]
    os.makedirs(OUT, exist_ok=True)
    for name, obj in (("holdout_games.json", dict(seed=SEED, games=hold)),
                      ("selection_folds.json", dict(seed=SEED, folds=folds))):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            assert json.load(open(p)) == obj, "split already fixed and differs: " + p
        else:
            json.dump(obj, open(p, "w"), indent=1)
        print(p, obj)


if __name__ == "__main__":
    main()
