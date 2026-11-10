#!/usr/bin/env python3
"""bench.py --dfl2 with degrade_v4 (dead-ball model refitted to the match): the "dfl4" bench.

    python candidates/ensemble_ctrl_v5b/bench_v4.py candidates/<name>/pipeline.py \
        --folds $MTAH_WORK/arch/ensemble_ctrl_v2/folds.json [--save-probs DIR] [--params JSON]

Reuses bench.bench_dfl unchanged (leave-one-game-out model swapping, 3 seeds, ctrl readings);
only the degradation differs: bench.dfl2_prepared is wrapped so that version 4 builds the game
with degrade_v4.degrade_v4 and caches it under the distinct prefix dfl4_<match>_<seed>.npz (or
dfl4<tag>_ with --cache-tag). Saved codes are named dfl4_<match>_<seed>_codes.npy, so
score_sel2.py <dir> --split dfl4 scores them against the same label readings as dfl2.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
for p in (ROOT, os.path.join(ROOT, "evaluation"), CAND):
    if p not in sys.path:
        sys.path.insert(0, p)
import bench                      # noqa: E402
import degrade_v4                 # noqa: E402
import possession_metrics as pm   # noqa: E402

_orig_prepared = bench.dfl2_prepared
PARAMS = {}
CACHE_TAG = ""


def dfl4_prepared(match, seed, version=2):
    if version != 4:
        return _orig_prepared(match, seed, version)
    os.makedirs(bench.CACHE_DIR, exist_ok=True)
    path = os.path.join(bench.CACHE_DIR, "dfl4{}_{}_{}.npz".format(CACHE_TAG, match, seed))
    if not os.path.isfile(path):
        z = np.load(os.path.join(bench.DFL_DIR, match + ".npz"))
        status, poss, half = z["status"], z["poss"], z["half"]
        gt = np.full(len(status), pm.X, np.int8)
        gt[(status == 1) & (poss == 1)] = pm.H
        gt[(status == 1) & (poss == 2)] = pm.A
        gt[status == 0] = pm.D
        blocks = []
        for h in (1, 2):
            idx = np.where((half == h) & (gt != pm.X))[0]
            if len(idx):
                blocks.append((len(blocks) + 1, int(idx[0]), int(idx[-1])))
        players, ball = z["players"].astype(np.float64), z["ball"].astype(np.float64)
        team = z["team_of_player"].astype(int)
        slots, slot_team, dball = degrade_v4.degrade_v4(players, team, ball, status, gt != pm.X,
                                                        np.random.default_rng(seed), PARAMS)
        sign = bench.tsd.attack_sign(bench.tsd.Game(match, players, team, ball, []), blocks)
        np.savez(path, slots=slots, slot_team=slot_team, ball=dball, gt=gt,
                 blocks=np.array(blocks), sign=sign)
    z = np.load(path)
    return (z["slots"], z["slot_team"], z["ball"], z["gt"],
            [tuple(int(v) for v in b) for b in z["blocks"]], float(z["sign"]))


bench.dfl2_prepared = dfl4_prepared


def main():
    global CACHE_TAG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--save-probs", default=None, metavar="DIR")
    ap.add_argument("--ctrl", type=float, default=0.0)
    ap.add_argument("--params", default=None, help="JSON file of degrade_v4 knob overrides")
    ap.add_argument("--cache-tag", default="", help="suffix of the dfl4 cache prefix (with --params)")
    args = ap.parse_args()
    if args.params:
        PARAMS.update(json.load(open(args.params)))
    CACHE_TAG = args.cache_tag
    folds = json.load(open(args.folds)) if args.folds else {}
    print("{} on DFL (degrade_v4{}) x seeds {}".format(args.candidate, CACHE_TAG, bench.DFL2_SEEDS))
    rows = bench.bench_dfl(args.candidate, folds, args.folds, args.save_probs, v2=4, ctrl=args.ctrl)
    result = dict(candidate=args.candidate, split="dfl4" + CACHE_TAG,
                  spellF1_mean=round(float(np.mean([r["spellF1"] for r in rows])), 4),
                  macroF1_mean=round(float(np.mean([r["macroF1"] for r in rows])), 4), rows=rows)
    games = sorted({r["game"] for r in rows})
    per_game = [np.mean([r["spellF1"] for r in rows if r["game"] == g]) for g in games]
    result["spellF1_se"] = round(float(np.std(per_game, ddof=1) / np.sqrt(len(games))), 4)
    result["ctrl_mean"] = round(float(np.mean([r["ctrl_mean"] for r in rows])), 4)
    print("ctrl_mean {}".format(result["ctrl_mean"]))
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
