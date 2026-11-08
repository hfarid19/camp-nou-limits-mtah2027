#!/usr/bin/env python3
"""Robustness diagnostic: leave-one-DFL-game-out spellF1 of the full decoder (not a selection target).

    python -I candidates/learned_dfl_evidence/dfl_logo_eval.py [--seed 21] [--cfg '{"switch_cost": 1}']

For every DFL game: degrade it with a seed that was never used for training (21), run the fold
models that were trained WITHOUT that game (train.py writes them to <cache>/folds), decode with
pipeline.decode and score with evaluation/possession_metrics.evaluate against the game's own
tracking flags (BallStatus / BallPossession). Only the scorer is imported from evaluation/.
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm   # noqa: E402
import make_dataset as md         # noqa: E402
import pipeline as lp             # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=md.DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--games", default="")
    ap.add_argument("--fold-dirs", default="{}",
                    help='per-model fold dirs, e.g. {"ha": "<r1 cache>/folds"}; default <cache>/folds')
    ap.add_argument("--degrade", default="{}",
                    help='make_dataset settings for the eval degradation, e.g. {"BLACKOUT_P": 0.5}')
    ap.add_argument("--meta", default=os.path.join(HERE, "model_meta.json"),
                    help="model_meta.json giving each fold model's boosting rounds")
    ap.add_argument("--long-gap", action="store_true", help="also apply the long-gap dead term")
    args = ap.parse_args()
    fold_dirs = json.loads(args.fold_dirs)
    for k, v in json.loads(args.degrade).items():
        assert hasattr(md, k), k
        setattr(md, k, v)
    import lightgbm as lgb
    cfgs = json.loads(args.cfg)
    cfgs = cfgs if isinstance(cfgs, list) else [cfgs]
    meta = json.load(open(args.meta))
    rows = {i: [] for i in range(len(cfgs))}
    for gi, match in enumerate(meta["matches"]):
        if args.games and match not in args.games.split(","):
            continue
        g = md.load_game(os.path.join(args.cache, "npz", match + ".npz"))
        slots, slot_team, ball, codes, blocks, sign, _ = md.degrade_dfl(g, args.seed,
                                                                       md.game_key(match))
        models = dict(meta=meta)
        for w in ("d", "ha", "c"):
            path = os.path.join(fold_dirs.get(w, os.path.join(args.cache, "folds")),
                                "fold_{}_{}.txt".format(w, gi))
            models[w] = None
            if os.path.isfile(path) and w in meta:
                b = lgb.Booster(model_file=path)
                models[w] = _Trunc(b, meta[w]["rounds"])
        p_dead, p_home, aux = lp.frame_probs(slots, slot_team, ball, sign, blocks, md.PITCH,
                                             md.FPS, models=models)
        if args.long_gap:
            aux["long_gap"] = lp.long_gap_gated(slots, ball, md.FPS, lp.CFG["gap_s"],
                                                lp.CFG["gap_speed_ms"])
        for i, cfg in enumerate(cfgs):
            pred = lp.decode(p_dead, p_home, aux, blocks, len(codes), md.FPS, cfg=cfg)
            r = pm.evaluate(codes, pred, bootstrap_n=0)
            rows[i].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                            r["n_gt_spells"]))
            print("{} cfg {}: spellF1 {:.4f} macroF1 {:.4f} poss {}/{}".format(
                match, i, *rows[i][-1]), flush=True)
    for i, cfg in enumerate(cfgs):
        a = np.array(rows[i])
        print("DFL-LOGO cfg {} {}: spellF1 {:.4f} macroF1 {:.4f}".format(
            i, cfg, a[:, 0].mean(), a[:, 1].mean()))


class _Trunc(object):
    def __init__(self, booster, rounds):
        self.b, self.k = booster, rounds

    def predict(self, X, num_threads=1):
        return self.b.predict(X, num_iteration=self.k, num_threads=num_threads)


if __name__ == "__main__":
    main()
