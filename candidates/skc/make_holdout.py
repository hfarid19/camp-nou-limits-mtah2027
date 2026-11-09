#!/usr/bin/env python3
"""Fixed 25% holdout of an ingested dataset's games (numpy seed 2027), written once.

    python -I candidates/skc/make_holdout.py skc_aleague SKC-      # -> $MTAH_WORK/skc_aleague/holdout_games.json
    python -I candidates/skc/make_holdout.py soccertrack ST-

Trainers must never train on (or select hyper-parameters with) the holdout games; they are the
dataset's final test games. The file is never overwritten.
"""

import glob
import json
import os
import sys

import numpy as np

WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "work")
SEED = 2027


def main():
    ds, prefix = sys.argv[1], sys.argv[2]
    out = os.path.join(WORK, ds, "holdout_games.json")
    if os.path.exists(out):
        print(open(out).read())
        return
    games = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(WORK, ds, "npz", prefix + "*.npz")))
    k = int(round(0.25 * len(games)))
    hold = sorted(np.random.default_rng(SEED).choice(games, k, replace=False).tolist())
    rec = dict(dataset=ds, seed=SEED, rule="numpy default_rng(2027).choice(sorted games, round(0.25 n), replace=False)",
               holdout=hold, train=[g for g in games if g not in hold])
    with open(out, "w") as h:
        json.dump(rec, h, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
