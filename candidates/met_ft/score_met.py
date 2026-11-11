#!/usr/bin/env python3
"""MET-FT: score one member model on Metrica sample games exactly as bench.py's Metrica bench feeds it (bench.prepared:
tsd.degrade, seeds 2027 / 2028), against both label sets: strict (Metrica-rule controlled possession,
metrica_rules.strict_labels: PRIMARY, what MET-FT trains on) and practice (tsd.practice_labels: the bench's labels).

    NN_PFF_MODEL=<model.npz> python -I candidates/met_ft/score_met.py <pipeline_cfg.py> <name> --games G1 G2 G3

Member alone = the pipeline's own predict() codes. Codes / probs -> $MTAH_WORK/probs/met_ft/<name>/met_<game>_<seed>[_codes].npy,
rows -> $MTAH_WORK/arch/r5_arch/met_ft/scores/<name>.jsonl. --check-dir DIR compares the codes with an earlier
bench run's dev_<game>_<seed>_codes.npy (reproduction check).
"""

import argparse
import json
import multiprocessing.pool    # noqa: F401
import multiprocessing.queues  # noqa: F401
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path.insert(0, CAND)
import bench                       # noqa: E402
sys.path.append(os.path.join(CAND, "control_events"))   # appended: never shadows a pipeline's own modules
import metrica_rules as mr         # noqa: E402

pm, tsd = bench.pm, bench.tsd
GAMES = {"G1": "Sample_Game_1", "G2": "Sample_Game_2", "G3": "Sample_Game_3"}
WORK = bench.WORK
SEEDS = [2027, 2028]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("name")
    ap.add_argument("--games", nargs="+", default=["G1", "G2", "G3"])
    ap.add_argument("--check-dir", default=None)
    a = ap.parse_args()
    for k in ("BENCH_DFL_FOLD", "BENCH_DFL_FOLDS", "NN_PFF_FOLD"):
        assert k not in os.environ, k
    out = os.path.join(WORK, "probs", "met_ft", a.name)
    os.makedirs(out, exist_ok=True)
    sd = os.path.join(WORK, "arch", "r5_arch", "met_ft", "scores")
    os.makedirs(sd, exist_ok=True)
    module = bench.load_candidate(a.pipeline)
    for g in a.games:
        sample = GAMES[g]
        folder = os.path.join(bench.SAMPLE_DIR, sample)
        game = tsd.load_epts_game(folder) if sample == "Sample_Game_3" else tsd.load_csv_game(folder)
        prac, blocks = tsd.practice_labels(game)                 # = met_data.labels
        strict = mr.strict_labels(game, prac, blocks)
        for seed in SEEDS:
            slots, slot_team, ball, gt, bl, sign = bench.prepared(sample, seed)
            assert np.array_equal(gt, prac) and list(bl) == list(blocks)
            args = (slots, slot_team, ball, sign, bl, [], ("Home", "Away"), tsd.PITCH, tsd.FPS)
            t0 = time.time()
            pred = np.asarray(module.predict(*args), np.int8)
            prob = np.asarray(module.predict_proba(*args), np.float32)
            np.save(os.path.join(out, "met_{}_{}_codes.npy".format(sample, seed)), pred)
            np.save(os.path.join(out, "met_{}_{}.npy".format(sample, seed)), prob)
            rs = pm.evaluate(strict, pred, bootstrap_n=0)
            rp = pm.evaluate(prac, pred, bootstrap_n=0)
            mins = sum(int(l) - int(f) + 1 for _, f, l in bl) / 25.0 / 60.0
            row = dict(name=a.name, model=os.environ.get("NN_PFF_MODEL"), game=g, seed=seed,
                       strict=round(rs["spell_f1_mean"], 5), practice=round(rp["spell_f1_mean"], 5),
                       macro_strict=round(rs["macro_f1"], 5), n_pred=rs["n_pred_spells"], n_strict=rs["n_gt_spells"],
                       n_practice=rp["n_gt_spells"], p90=round(rs["n_pred_spells"] / mins * 90, 1),
                       p90_strict=round(rs["n_gt_spells"] / mins * 90, 1), secs=round(time.time() - t0, 1))
            if a.check_dir:
                f = os.path.join(a.check_dir, "dev_{}_{}_codes.npy".format(sample, seed))
                if os.path.isfile(f):
                    row["same_as_bench"] = bool(np.array_equal(np.load(f).astype(np.int8), pred))
            print(json.dumps(row), flush=True)
            with open(os.path.join(sd, a.name + ".jsonl"), "a") as h:
                h.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
