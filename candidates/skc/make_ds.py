#!/usr/bin/env python3
"""Training files for the nn trainers from an ingested dataset, in ds_pff's exact format.

    python -I candidates/skc/make_ds.py skc_aleague SKC- [--seeds 41 42 44] [--jobs 2]
    python -I candidates/skc/make_ds.py soccertrack ST-  [--seeds 41 42 44] [--jobs 2]

Writes $MTAH_WORK/<dataset>/ds/{ds,raw}/<PREFIX><game>_<seed>.npz with candidates/nn_pff/make_dataset.job
(unchanged: the same randomised degrade_v2 (seeds 41, 42) / degrade_v4 (seed 44) draws, AR(1) ball offset,
LDE features on a stride-5 phase, kick-off attack-sign rule, labels from the npz's status / poss), for
the dataset's TRAIN games only: the games in $MTAH_WORK/<dataset>/holdout_games.json are refused.

To include them in a ds_pff-style training set, hard-link them next to the existing files, e.g.
    cp -al $MTAH_WORK/ds_pff $MTAH_WORK/ds_pff_skc
    ln $MTAH_WORK/skc_aleague/ds/ds/SKC-*  $MTAH_WORK/ds_pff_skc/ds/
    ln $MTAH_WORK/skc_aleague/ds/raw/SKC-* $MTAH_WORK/ds_pff_skc/raw/
and train with --ds $MTAH_WORK/ds_pff_skc/ds. nn_pff/data.py treats any non-"DFL" game like a PFF game
(pff_fold_s target fold); SKC-/ST- games are never in a PFF or DFL selection fold, so they are in the
training set of every fold and never scored by the OOF selection.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
WORK = os.environ.get("MTAH_WORK") or os.path.join(os.path.dirname(CAND), "external", "work")
_MD = {}


def _job(args):
    # candidates/nn_pff/make_dataset.py, loaded by path under its own name: nn_pff/ must not go on
    # sys.path (its queue.py shadows the stdlib), and it imports learned_dfl_evidence's make_dataset.
    if "m" not in _MD:
        spec = importlib.util.spec_from_file_location("nn_pff_make_dataset",
                                                      os.path.join(CAND, "nn_pff", "make_dataset.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MD["m"] = m
    return _MD["m"].job(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("prefix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 44])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    base = os.path.join(WORK, a.dataset)
    hold = json.load(open(os.path.join(base, "holdout_games.json")))
    train = hold["train"]
    assert not set(train) & set(hold["holdout"])
    out = os.path.join(base, "ds")
    for sub in ("ds", "raw"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    jobs = [(os.path.join(base, "npz", g + ".npz"), g, s, out) for g in train for s in a.seeds]
    with open(os.path.join(out, "config.json"), "w") as h:
        json.dump(dict(dataset=a.dataset, games=train, seeds=a.seeds, builder="candidates/nn_pff/make_dataset.job",
                       v4_seeds=[44]), h, indent=1)
    if a.jobs <= 1:
        for j in jobs:
            print(_job(j), flush=True)
        return
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(_job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
